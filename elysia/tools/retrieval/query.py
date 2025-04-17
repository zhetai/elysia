from typing import List
from logging import Logger


from rich import print
from rich.panel import Panel

import dspy

from elysia.dspy_additions.environment_of_thought import EnvironmentOfThought
from elysia.util.reference import create_reference
from elysia.objects import Branch, Reasoning, Response, Status, Tool, Warning
from elysia.tools.retrieval.chunk import AsyncCollectionChunker
from elysia.tools.retrieval.objects import (
    BoringGenericRetrieval,
    ConversationRetrieval,
    DocumentRetrieval,
    EcommerceRetrieval,
    EpicGenericRetrieval,
    MessageRetrieval,
    TicketRetrieval,
)
from elysia.tools.retrieval.prompt_templates import (
    ObjectSummaryPrompt,
    QueryCreatorPrompt,
)
from elysia.tools.retrieval.util import execute_weaviate_query
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager
from elysia.util.objects import TrainingUpdate, TreeUpdate


class Query(Tool):
    def __init__(self, logger: Logger | None = None, **kwargs):
        super().__init__(
            name="query",
            description="""
            Query the knowledge base based on searching with a specific query.
            This views the data specifically, and returns the data that matches the query.
            You will be given information about the collection, such as the fields and types of the data.
            Then, query with semantic search, keyword search, or a combination of both.
            """,
            status="Querying...",
            inputs={
                "collection_names": {
                    "description": "the names of the collections that are most relevant to the query. If you are unsure, give all collections.",
                    "type": "list",
                    "default": [],
                },
            },
            end=False,
        )

        self.logger = logger
        self.retrieval_map = {
            "conversation": ConversationRetrieval,
            "message": MessageRetrieval,
            "ticket": TicketRetrieval,
            "ecommerce": EcommerceRetrieval,
            "epic_generic": EpicGenericRetrieval,
            "boring_generic": BoringGenericRetrieval,
            "document": DocumentRetrieval,
        }

    def _find_previous_queries(self, environment: dict, collection_names: list[str]):
        previous_queries = {}
        # create a dict of collection_name: previous_queries
        for collection_name in collection_names:
            # check if a query has happened at all for this collection
            if "query" in environment and collection_name in environment["query"]:
                for query in environment["query"][collection_name]:
                    if "query_output" in query["metadata"]:
                        if collection_name in previous_queries:
                            previous_queries[collection_name].append(
                                query["metadata"]["query_output"]
                            )
                        else:
                            previous_queries[collection_name] = [
                                query["metadata"]["query_output"]
                            ]

        return previous_queries

    def _evaluate_content_field(self, display_type: str, mappings: dict) -> str:
        """
        Return the original object field if its mapped to 'content', otherwise return an empty string.
        If no content field mapping exists in `mappings`, it will return an empty string anyway (by default).
        """

        if display_type == "conversation":
            return mappings[display_type]["content"]
        elif display_type == "message":
            return mappings[display_type]["content"]
        elif display_type == "ticket":
            return mappings[display_type]["content"]
        elif display_type == "ecommerce":
            return mappings[display_type]["description"]
        elif display_type == "epic_generic":
            return mappings[display_type]["content"]
        elif display_type == "document":
            return mappings[display_type]["content"]
        else:
            return ""

    def _evaluate_needs_chunking(
        self,
        display_type: str,
        query_type: str,
        schema: dict,
        threshold: int = 1000,  # number of characters to be considered large enough to chunk
    ) -> bool:
        mapping = schema["mappings"]
        content_field = self._evaluate_content_field(display_type, mapping)

        return (
            content_field != ""
            and schema["fields"][content_field]["mean"] > threshold
            and query_type != "filter_only"
            and display_type
            == "document"  # for the moment, the only type we chunk is document
        )

    def _handle_generic_error(
        self,
        e: Exception,
        collection_name: str,
        user_prompt: str,
        query_output: dict | List[dict] | None = None,
    ):
        metadata = {
            "collection_name": collection_name,
            "error_message": str(e),
            "impossible": user_prompt,
        }

        if query_output is not None:
            metadata["query_output"] = query_output

        return BoringGenericRetrieval([], metadata)

    def _handle_assertion_error(
        self,
        e: Exception,
        collection_name: str,
        user_prompt: str,
        query_output: dict | List[dict] | None = None,
    ):
        metadata = {
            "collection_name": collection_name,
            "assertion_error_message": str(e),
            "impossible": user_prompt,
        }
        if query_output is not None:
            metadata["query_output"] = query_output

        return BoringGenericRetrieval([], metadata)

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        """
        Can perform three main functions:
        1. Query the knowledge base with Weaviate using hybrid, semantic, or keyword search, applying filters and more.
        2. Automatically chunk collections that are too large to vectorise effectively
        3. Summarise retrieved objects if requested
        Chunking creates a parallel quantized collection of chunks with cross references to the original collection.
        When filtering on the parallel chunked collection, filters are applied to the original collection by cross referencing.
        """

        # Create branches
        Branch(
            {
                "name": "Query Executor",
                "description": "Write and execute a query via an LLM.",
            }
        )
        Branch(
            {
                "name": "Chunker",
                "description": "Chunk collections that are too large to vectorise effectively.",
            }
        )
        Branch(
            {
                "name": "Object Summariser",
                "description": "Summarise retrieved objects.",
            }
        )

        if self.logger:
            self.logger.debug("Query Tool called!")
            self.logger.debug(f"Inputs: {inputs}")

        # Get the collections
        collection_names = [c.lower() for c in inputs["collection_names"]]

        if self.logger:
            self.logger.debug(f"Collection names received: {collection_names}")

        # Set up the dspy model
        query_generator = EnvironmentOfThought(QueryCreatorPrompt)
        query_generator = dspy.asyncify(query_generator)

        # Get some metadata about the collection
        previous_queries = self._find_previous_queries(
            tree_data.environment.environment,
            tree_data.collection_data.collection_names,
        )

        if self.logger:
            self.logger.debug(f"Previous queries: {previous_queries}")

        # Get schemas for these collections
        schemas = tree_data.output_collection_metadata(with_mappings=True)
        schemas = {
            collection_name: schemas[collection_name]
            for collection_name in collection_names
        }

        schemas_without_mappings = tree_data.output_collection_metadata(
            with_mappings=False
        )
        schemas_without_mappings = {
            collection_name: schemas_without_mappings[collection_name]
            for collection_name in collection_names
        }

        # get display types
        display_types = tree_data.output_collection_return_types()
        display_types = {
            collection_name: display_types[collection_name]
            for collection_name in collection_names
        }

        # Generate query with LLM
        try:
            query = await query_generator(
                user_prompt=tree_data.user_prompt,
                reference=create_reference(),
                conversation_history=tree_data.conversation_history,
                environment=tree_data.environment.to_json(),
                collection_schemas=schemas_without_mappings,
                tasks_completed=tree_data.tasks_completed_string(),
                previous_queries=previous_queries,
                collection_display_types=display_types,
                lm=complex_lm,
            )
        except Exception as e:
            for collection_name in collection_names:
                yield self._handle_generic_error(
                    e, collection_name, tree_data.user_prompt
                )

            if self.logger:
                self.logger.warning(
                    f"Oops! Something went wrong when creating the query. Continuing..."
                )
                self.logger.debug(
                    f"The error was during the generation of the query: {str(e)}"
                )
            yield Warning(
                f"Oops! Something went wrong when creating the query. Continuing..."
            )
            return  # exit here because this will break the whole file

        # Yield results to front end
        yield Response(text=query.message_update)
        yield Reasoning(reasoning=query.reasoning)
        yield TrainingUpdate(
            model="query",
            inputs=tree_data.to_json(),
            outputs=query.__dict__["_store"],
        )

        # Return if model deems query impossible
        if query.impossible or query.query_output is None:

            for collection_name in collection_names:
                metadata = {
                    "collection_name": collection_name,
                    "impossible": tree_data.user_prompt,
                    "impossible_reasoning": query.reasoning,
                    "query_output": (
                        query.query_output.model_dump()
                        if query.query_output is not None
                        else None
                    ),
                }
                yield BoringGenericRetrieval([], metadata)

            if self.logger:
                self.logger.warning(
                    f"Model judged query to be impossible. Returning to the decision tree..."
                )

            yield TreeUpdate(
                from_node="query",
                to_node="query_executor",
                reasoning=query.reasoning,
                last_in_branch=True,
            )
            return

        # make sure all collection names are lower case
        for q in query.query_output:
            q.target_collections = [c.lower() for c in q.target_collections]

        # Create a list of keys to iterate over to avoid changing dict size during iteration
        keys_to_process = list(query.data_display.keys())
        for key in keys_to_process:
            if key != key.lower():
                query.data_display[key.lower()] = query.data_display[key]
                del query.data_display[key]

        yield TreeUpdate(
            from_node="query",
            to_node="query_executor",
            reasoning=query.reasoning,
            last_in_branch=not any(
                self._evaluate_needs_chunking(
                    query.data_display[collection_name].display_type,
                    current_query_output.search_type,
                    schemas[collection_name],
                )
                for current_query_output in query.query_output
                for collection_name in current_query_output.target_collections
            ),
        )

        chunking_status_sent = False
        summarise_status_sent = False

        # Go through outputs for each unique query
        for i, query_output in enumerate(query.query_output):
            collection_names = query_output.target_collections.copy()

            for collection_name in collection_names:
                # Get display types
                display_type = query.data_display[collection_name].display_type

                if self.logger:
                    self.logger.debug(
                        f"Display type for collection {collection_name}: {display_type}"
                    )

                # Evaluate if this collection/query needs chunking
                needs_chunking = self._evaluate_needs_chunking(
                    display_type,
                    query_output.search_type,
                    schemas[collection_name],
                )
                content_field = self._evaluate_content_field(
                    display_type, schemas[collection_name]["mappings"]
                )  # used for chunking

                if needs_chunking:

                    if not chunking_status_sent:
                        yield Status("Chunking collections...")
                        yield TreeUpdate(
                            from_node="query_executor",
                            to_node="chunker",
                            reasoning="Chunking collections because they are too large to vectorise effectively.",
                            last_in_branch=not any(
                                query.data_display[x].summarise_items
                                for x in query.data_display
                            ),
                        )
                        chunking_status_sent = True

                    if self.logger:
                        self.logger.debug(f"Chunking {collection_name}")

                    # set up chunking (create reference in this collection)
                    collection_chunker = AsyncCollectionChunker(collection_name)
                    await collection_chunker.create_chunked_reference(
                        content_field, client_manager
                    )

                    # create a copy of the query output
                    query_output_copy = query_output.model_copy(deep=True)

                    # update the limit to be larger
                    query_output_copy.limit = query_output.limit * 3

                    # update the collection to be the unchunked collection ONLY
                    query_output_copy.target_collections = [collection_name]

                    # run this augmented query for the unchunked collection
                    async with client_manager.connect_to_async_client() as client:
                        unchunked_response, _ = await execute_weaviate_query(
                            client,
                            query_output_copy,
                            reference_property="isChunked",
                        )

                    # chunk the unchunked response
                    yield Status(
                        f"Chunking {len(unchunked_response[0].objects)} objects in {collection_name}..."
                    )
                    await collection_chunker(
                        unchunked_response[0], content_field, client_manager
                    )

                    # modify the original query output to be the chunked collection
                    for j, c in enumerate(query_output.target_collections):
                        if c == collection_name:
                            query_output.target_collections[j] = (
                                f"ELYSIA_CHUNKED_{collection_name}__"
                            )

                    if self.logger:
                        self.logger.debug(
                            f"Chunked {collection_name} and updated query output to {query_output.target_collections}"
                        )

            # Execute query within Weaviate
            async with client_manager.connect_to_async_client() as client:
                try:
                    responses, code_strings = await execute_weaviate_query(
                        client, query_output
                    )
                except Exception as e:
                    for collection_name in collection_names:
                        yield self._handle_assertion_error(
                            e,
                            collection_name,
                            tree_data.user_prompt,
                            query_output.model_dump(),
                        )

                    if self.logger:
                        self.logger.warning(
                            f"Something went wrong with the LLM creating the query! Continuing..."
                        )
                        self.logger.debug(
                            f"The assertion error was during the execution of the query: {str(e)}"
                        )
                    yield Warning(
                        f"Something went wrong with the LLM creating the query! Continuing..."
                    )
                    # dont exit because can try again for next collection etc
                    continue

                yield Status(
                    f"Retrieved {sum(len(x.objects) for x in responses)} objects from {len(collection_names)} collections..."
                )

            for k, (collection_name, response) in enumerate(
                zip(collection_names, responses)
            ):
                if self.logger and self.logger.level <= 20:
                    print(
                        Panel.fit(
                            code_strings[k],
                            title=f"{collection_name} (Weaviate Query)",
                            border_style="yellow",
                            padding=(1, 1),
                        )
                    )

                # Get display type and summarise items bool
                display_type = query.data_display[collection_name].display_type
                summarise_items = query.data_display[collection_name].summarise_items

                # Get the objects from the response
                objects = []
                for obj in response.objects:
                    objects.append({k: v for k, v in obj.properties.items()})
                    objects[-1]["uuid"] = obj.uuid.hex

                # Write various metadata for LLM parsing
                metadata = {
                    "collection_name": collection_name,
                    "display_type": display_type,
                    "summarise_items": summarise_items,
                    "query_text": query_output.search_query,
                    "query_type": query_output.search_type,
                    "chunked": self._evaluate_needs_chunking(
                        display_type,
                        query_output.search_type,
                        schemas[collection_name],
                    ),
                    "query_output": query_output.model_dump(),
                    "code": {
                        "language": "python",
                        "title": "Query",
                        "text": code_strings[k],
                    },
                }

                # Create the retrieval object
                output = self.retrieval_map[display_type](
                    objects,
                    metadata,
                    mapping=schemas[collection_name]["mappings"][display_type],
                )

                # If the display type is document or conversation, initialise the object (requires some async operations)
                if display_type == "document" or display_type == "conversation":
                    await output.async_init(client_manager)

                if summarise_items:
                    if self.logger:
                        self.logger.debug(
                            f"{collection_name} will have itemised summaries."
                        )

                    object_summariser = dspy.ChainOfThought(ObjectSummaryPrompt)
                    object_summariser = dspy.asyncify(object_summariser)
                    yield Status(
                        f"Summarising {len(objects)} objects from {collection_name}..."
                    )

                    summariser = await object_summariser(
                        objects=output.to_json(), lm=base_lm
                    )

                    if self.logger:
                        self.logger.debug(f"Summarisation complete.")

                    if not summarise_status_sent:
                        yield TreeUpdate(
                            from_node="query_executor",
                            to_node="object_summariser",
                            reasoning=summariser.reasoning,
                            last_in_branch=True,
                        )
                        summarise_status_sent = True

                    output.add_summaries(summariser.summaries)
                else:
                    output.add_summaries([])

                # Yield the output object
                yield output

        if not chunking_status_sent:
            yield TreeUpdate(
                from_node="query_executor",
                to_node="chunker",
                reasoning="This step was skipped because no chunking was needed for the collections queried.",
                last_in_branch=True,
            )

        if not summarise_status_sent:
            yield TreeUpdate(
                from_node="query_executor",
                to_node="object_summariser",
                reasoning="This step was skipped because the LLM determined that no output types were summaries.",
                last_in_branch=True,
            )

        if self.logger:
            self.logger.debug("Query Tool finished!")


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
#         "what is HNSW? search only where the categories field is 'Documentation'"
#     )
#     query = Query(verbosity=2)
#     async for result in query(
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
