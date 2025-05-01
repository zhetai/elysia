from typing import List
from logging import Logger

from rich import print
from rich.panel import Panel

import dspy

from elysia.dspy_additions.environment_of_thought import EnvironmentOfThought
from elysia.util.reference import create_reference
from elysia.objects import Branch, Reasoning, Response, Status, Tool, Warning
from elysia.tools.retrieval.objects import Aggregation
from elysia.tools.retrieval.prompt_templates import AggregationPrompt
from elysia.tools.retrieval.util import execute_weaviate_aggregation
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager
from elysia.util.objects import TrainingUpdate, TreeUpdate
from elysia.util.parsing import format_aggregation_response


class Aggregate(Tool):
    def __init__(
        self,
        logger: Logger | None = None,
        **kwargs,
    ):
        self.logger = logger

        super().__init__(
            name="aggregate",
            description="""
            Perform functions such as counting, averaging, summing, etc. on the data, grouping by some property and returning metrics/statistics about different categories.
            Or providing a high level overview of the dataset.
            Note this does not include viewing the data, only performing operations to find quantities and summary statistics.
            You will be given information about the collection, such as the fields and types of the data.
            Then, aggregate over different categories of the collection, with operations: top_occurences, count, sum, average, min, max, median, mode, group_by.
            Without viewing the data.
            """,
            status="Aggregating...",
            inputs={
                "collection_names": {
                    "description": "the names of the collections that are most relevant to the aggregation. If you are unsure, give all collections.",
                    "type": "list",
                    "default": [],
                },
            },
            end=False,
        )

    def _find_previous_aggregations(
        self, environment: dict, collection_names: list[str]
    ):
        previous_aggregations = {}
        for collection_name in collection_names:
            if (
                "aggregate" in environment
                and collection_name in environment["aggregate"]
            ):
                for aggregation in environment["aggregate"][collection_name]:
                    if "code" in aggregation and "text" in aggregation["code"]:
                        if collection_name in previous_aggregations:
                            previous_aggregations[collection_name].append(
                                aggregation["code"]["text"]
                            )
                        else:
                            previous_aggregations[collection_name] = [
                                aggregation["code"]["text"]
                            ]

        return previous_aggregations

    async def _handle_generic_error(
        self,
        e: Exception,
        collection_name: str,
        user_prompt: str,
    ):
        metadata = {
            "collection_name": collection_name,
            "error_message": str(e),
            "impossible": user_prompt,
        }

        if self.logger:
            self.logger.warning(
                f"Oops! Something went wrong during the LLM aggregation. Continuing..."
            )
            self.logger.debug(f"The error was: {str(e)}")

        yield Aggregation([], metadata)
        yield Warning(
            f"Oops! Something went wrong during the LLM aggregation. Continuing..."
        )

    async def _handle_aggregation_error(
        self,
        e: Exception,
        collection_name: str,
        user_prompt: str,
        aggregation_output: dict | List[dict] | None = None,
    ):
        metadata = {
            "collection_name": collection_name,
            "assertion_error_message": str(e),
            "impossible": user_prompt,
        }
        if aggregation_output is not None:
            metadata["aggregation_output"] = aggregation_output

        if self.logger:
            self.logger.warning(
                f"Oops! Something went wrong when executing the aggregation. Continuing..."
            )
            self.logger.debug(f"The error was: {str(e)}")

        yield Aggregation([], metadata)
        yield Warning(
            f"Oops! Something went wrong when executing the aggregation. Continuing..."
        )

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        client_manager: ClientManager,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        **kwargs,
    ):
        # Set up the branch spanning off the node
        Branch(
            {
                "name": "Aggregation Generator",
                "description": "Generate summary statistics of the data via an LLM aggregation query.",
            }
        )

        if self.logger:
            self.logger.debug("Aggregation Tool called!")
            self.logger.debug("Arguments received:")
            self.logger.debug(f"{inputs}")

        # Set up the dspy model
        aggregate_generator = EnvironmentOfThought(AggregationPrompt)
        aggregate_generator = dspy.asyncify(aggregate_generator)

        # Get the collections from the inputs
        collection_names = inputs["collection_names"]

        if self.logger:
            self.logger.info(f"Collection names received: {collection_names}")

        # Get some metadata about the collection
        previous_aggregations = self._find_previous_aggregations(
            tree_data.environment.environment, collection_names
        )

        # Get schemas for these collections
        schemas = tree_data.output_collection_metadata(with_mappings=False)
        schemas = {
            collection_name: schemas[collection_name]
            for collection_name in collection_names
        }

        # Generate query with LLM
        try:
            aggregation = await aggregate_generator(
                user_prompt=tree_data.user_prompt,
                reference=create_reference(),
                style=tree_data.atlas.style,
                agent_description=tree_data.atlas.agent_description,
                end_goal=tree_data.atlas.end_goal,
                conversation_history=tree_data.conversation_history,
                environment=tree_data.environment.to_json(),
                available_collections=collection_names,
                collection_schemas=schemas,
                tasks_completed=tree_data.tasks_completed_string(),
                previous_aggregation_queries=previous_aggregations,
                lm=complex_lm,
            )
        except Exception as e:
            for collection_name in collection_names:
                async for yield_object in self._handle_generic_error(
                    e,
                    collection_name,
                    tree_data.user_prompt,
                ):
                    yield yield_object

        # Yield results to front end
        yield Response(text=aggregation.message_update)
        yield Reasoning(reasoning=aggregation.reasoning)
        yield TrainingUpdate(
            model="aggregate",
            inputs=tree_data.to_json(),
            outputs=aggregation.__dict__["_store"],
        )

        # Return if model deems aggregation impossible
        if aggregation.impossible or aggregation.aggregation_queries is None:
            if self.logger:
                self.logger.info(
                    "Model judged this aggregation to be impossible. Returning to the decision tree..."
                )
            for collection_name in collection_names:
                metadata = {
                    "collection_name": collection_name,
                    "impossible": tree_data.user_prompt,
                    "impossible_reasoning": aggregation.reasoning,
                    "aggregation_output": (
                        aggregation.aggregation_queries.model_dump()
                        if aggregation.aggregation_queries is not None
                        else None
                    ),
                }
                yield Aggregation([], metadata)

            yield TreeUpdate(
                from_node="aggregate",
                to_node="aggregate_executor",
                reasoning=aggregation.reasoning,
                last_in_branch=True,
            )
            return

        # Go through each response
        for i, aggregation_output in enumerate(aggregation.aggregation_queries):

            # Execute query within Weaviate
            async with client_manager.connect_to_async_client() as client:
                try:
                    responses, code_strings = await execute_weaviate_aggregation(
                        client, aggregation_output
                    )
                except Exception as e:
                    for collection_name in collection_names:
                        async for yield_object in self._handle_aggregation_error(
                            e,
                            collection_name,
                            tree_data.user_prompt,
                            aggregation_output.model_dump(),
                        ):
                            yield yield_object
                    continue  # continue to next aggregation output

            yield Status(
                f"Aggregated over {', '.join(aggregation_output.target_collections)}"
            )

            for k, (collection_name, response) in enumerate(
                zip(aggregation_output.target_collections, responses)
            ):
                if self.logger and self.logger.level <= 20:
                    print(
                        Panel.fit(
                            code_strings[k],
                            title=f"{collection_name} (Weaviate Aggregation Query)",
                            border_style="yellow",
                            padding=(1, 1),
                        )
                    )

                objects = [{collection_name: format_aggregation_response(response)}]
                metadata = {
                    "collection_name": collection_name,
                    "aggregation_output": aggregation_output.model_dump(),
                    "code": {
                        "language": "python",
                        "title": "Aggregation",
                        "text": code_strings[k],
                    },
                }

                yield Aggregation(objects, metadata)

        if self.logger:
            self.logger.debug("Aggregation Tool finished!")


# async def main():
#     from elysia.objects import Result, Text, Update
#     from elysia.tree.tree import Tree
#     from elysia.util.client import ClientManager

#     # init tree
#     client_manager = ClientManager()
#     await client_manager.start_clients()
#     tree = Tree(
#         collection_names=["ecommerce", "weather", "weaviate_documentation"], debug=True
#     )
#     await tree.async_init(client_manager)

#     tree.tree_data.user_prompt = (
#         "what is the average price of trousers in the ecommerce collection?"
#     )
#     aggregate = Aggregate(verbosity=2)
#     async for result in aggregate(
#         base_lm=tree.base_lm,
#         complex_lm=tree.complex_lm,
#         inputs={"collection_names": ["ecommerce", "weather", "weaviate_documentation"]},
#         tree_data=tree.tree_data,
#         client_manager=client_manager,
#     ):
#         if isinstance(result, Result):
#             print(await result.to_frontend("1", "2", "3"))
#         elif isinstance(result, Update):
#             print(await result.to_frontend("2", "3"))
#         elif isinstance(result, Text):
#             print(result.text)


# if __name__ == "__main__":
#     import asyncio

#     asyncio.run(main())
