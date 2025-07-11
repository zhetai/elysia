from typing import AsyncGenerator
from logging import Logger
from pydantic import BaseModel, Field

from rich import print
from rich.panel import Panel

import dspy

from elysia.util.elysia_chain_of_thought import ElysiaChainOfThought
from elysia.objects import Response, Status, Tool, Error, Result, Return
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
    QueryCreatorPrompt,
    SimpleQueryCreatorPrompt,
)
from elysia.tools.retrieval.util import execute_weaviate_query
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager
from elysia.util.objects import TrainingUpdate, TreeUpdate, FewShotExamples


class Query(Tool):
    """
    Elysia Query Tool
    Queries the knowledge base with Weaviate using hybrid, semantic, or keyword search, applying filters and more.
    These are dynamically decided by an LLM, based on the user prompt and the collections available.

    Additionally, this Tool can perform the following actions:
    - Chunk collections that are too large to vectorise effectively
    - Dynamically return the retrieved objects to the frontend according to the display type decided by the LLM
    - Set a flag to summarise retrieved objects in the tree

    For a version of this tool that does not perform any of these actions, see the SimpleQuery tool.
    """

    def __init__(
        self,
        logger: Logger | None = None,
        summariser_in_tree: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            name="query",
            description="""
            Retrieves and displays specific data entries from the collections.
            Then, query with semantic search, keyword search, or a combination of both.
            Queries can be filtered, sorted, and more.
            Retrieving and displaying specific data entries rather than performing calculations or summaries.
            Do not use 'query' as a preliminary filtering step when 'aggregate' can achieve the same result more efficiently (if 'aggregate' is available).
            """,
            status="Querying...",
            inputs={
                "collection_names": {
                    "description": "the names of the collections that are most relevant to the query. If you are unsure, give all collections.",
                    "type": list[str],
                    "default": [],
                },
            },
            end=False,
        )

        self.logger = logger
        self.summariser_in_tree = summariser_in_tree
        # print("LOGGER LEVEL: ", self.logger.level)
        self.retrieval_map = {
            "conversation": ConversationRetrieval,
            "message": MessageRetrieval,
            "ticket": TicketRetrieval,
            "ecommerce": EcommerceRetrieval,
            "epic_generic": EpicGenericRetrieval,
            "boring_generic": BoringGenericRetrieval,
            "document": DocumentRetrieval,
        }

    async def is_tool_available(
        self,
        tree_data: TreeData,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager,
    ) -> bool:
        """
        Only available when there is a Weaviate connection.
        If this tool is not available, inform the user that they need to set the WCD_URL and WCD_API_KEY in the settings.
        """
        return client_manager.is_client

    def _find_previous_queries(
        self, environment: dict, collection_names: list[str]
    ) -> dict:
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
        threshold: int = 400,  # number of tokens to be considered large enough to chunk
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

    def _fix_collection_names(
        self, collection_names: list[str], schemas: dict
    ) -> list[str]:
        return [
            correct_collection_name
            for correct_collection_name in list(schemas.keys())
            if correct_collection_name.lower()
            in [collection_name.lower() for collection_name in collection_names]
        ]

    def _fix_collection_names_in_dict(
        self, collection_dict: dict, schemas: dict
    ) -> dict:
        true_collection_names = list(schemas.keys())
        truth_map = {k.lower(): k for k in true_collection_names}
        return {truth_map.get(k.lower(), k): v for k, v in collection_dict.items()}

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager,
        **kwargs,
    ) -> AsyncGenerator[Return | TreeUpdate | Error | TrainingUpdate, None]:
        """
        Can perform three main functions:
        1. Query the knowledge base with Weaviate using hybrid, semantic, or keyword search, applying filters and more.
        2. Automatically chunk collections that are too large to vectorise effectively
        3. Summarise retrieved objects if requested
        Chunking creates a parallel quantized collection of chunks with cross references to the original collection.
        When filtering on the parallel chunked collection, filters are applied to the original collection by cross referencing.
        """

        if self.logger:
            self.logger.debug("Query Tool called!")
            self.logger.debug(f"Inputs: {inputs}")

        # Get the collection schemas (truth)
        schemas = tree_data.output_collection_metadata(with_mappings=True)

        # Find matching collection names (to match case sensitivity)
        if inputs["collection_names"] == [] or inputs["collection_names"] is None:
            collection_names = list(schemas.keys())
        else:
            collection_names = self._fix_collection_names(
                inputs["collection_names"], schemas
            )

        # Subset the schemas to just the collection names
        schemas = {
            collection_name: schemas[collection_name]
            for collection_name in collection_names
        }
        if self.logger:
            self.logger.debug(f"Collection names received: {collection_names}")

        # Set up the dspy model
        query_generator = ElysiaChainOfThought(
            QueryCreatorPrompt,
            tree_data=tree_data,
            environment=True,
            collection_schemas=True,
            tasks_completed=True,
            message_update=True,
            collection_names=collection_names,
        )

        # Get some metadata about the collection
        previous_queries = self._find_previous_queries(
            tree_data.environment.environment,
            tree_data.collection_data.collection_names,
        )

        if self.logger:
            self.logger.debug(f"Previous queries: {previous_queries}")

        # get display types
        display_types = tree_data.output_collection_return_types()
        display_types = {
            collection_name: display_types[collection_name]
            for collection_name in collection_names
        }  # subset to just our collection names

        # get searchable fields
        searchable_fields = {
            collection_name: [
                named_vector
                for named_vector in schemas[collection_name]["named_vectors"]
                if schemas[collection_name]["named_vectors"][named_vector]["enabled"]
            ]
            for collection_name in collection_names
        }

        # Generate query with LLM
        try:
            if tree_data.settings.USE_FEEDBACK:
                query, example_uuids = (
                    await query_generator.aforward_with_feedback_examples(
                        feedback_model="query",
                        client_manager=client_manager,
                        base_lm=base_lm,
                        complex_lm=complex_lm,
                        available_collections=collection_names,
                        previous_queries=previous_queries,
                        collection_display_types=display_types,
                        searchable_fields=searchable_fields,
                        num_base_lm_examples=6,
                        return_example_uuids=True,
                    )
                )
            else:
                query = await query_generator.aforward(
                    lm=complex_lm,
                    available_collections=collection_names,
                    previous_queries=previous_queries,
                    collection_display_types=display_types,
                    searchable_fields=searchable_fields,
                )

        except Exception as e:
            yield Error(str(e))
            return

        if self.logger and query.query_output is not None:
            self.logger.debug(f"Query: {query.query_output.query_outputs}")

        # Yield results to front end
        yield Response(text=query.message_update)
        yield TrainingUpdate(
            module_name="query",
            inputs={
                "available_collections": collection_names,
                "previous_queries": previous_queries,
                "collection_display_types": display_types,
                "searchable_fields": searchable_fields,
                **tree_data.to_json(),
            },
            outputs=query.__dict__["_store"],
        )
        if tree_data.settings.USE_FEEDBACK:
            yield FewShotExamples(uuids=example_uuids)

        # if self.summariser_in_tree:
        #     yield TreeUpdate(
        #         from_node="query",
        #         to_node="summarise_items",
        #         reasoning=query.reasoning,
        #         last_in_branch=(
        #             query.impossible
        #             or query.query_output is None
        #             or not any(
        #                 self._evaluate_needs_chunking(
        #                     query.data_display[collection_name].display_type,
        #                     current_query_output.search_type,
        #                     schemas[collection_name],
        #                 )
        #                 for current_query_output in query.query_output.query_outputs
        #                 for collection_name in current_query_output.target_collections
        #             )
        #         ),
        #     )

        # Return if model deems query impossible
        if (
            query.impossible
            or query.query_output is None
            or all(q is None for q in query.query_output.query_outputs)
            or len(query.query_output.query_outputs) == 0
        ):

            for collection_name in collection_names:
                metadata = {
                    "collection_name": collection_name,
                    "impossible": tree_data.user_prompt,
                    "impossible_reasoning": query.reasoning,
                    "query_output": (
                        query.query_output.query_outputs.model_dump()
                        if query.query_output is not None
                        else None
                    ),
                }
                yield BoringGenericRetrieval([], metadata)

            if self.logger:
                self.logger.warning(
                    f"Model judged query to be impossible. Returning to the decision tree..."
                )

            # if self.summariser_in_tree:
            #     yield TreeUpdate(
            #         from_node="query",
            #         to_node="summarise_items",
            #         reasoning=query.reasoning,
            #         last_in_branch=True,
            #     )
            #     return

        # extract and error handle the query output
        # TODO: replace with assertions when they're released
        for current_query_output in query.query_output.query_outputs:

            current_query_output.target_collections = self._fix_collection_names(
                current_query_output.target_collections, schemas
            )

            for collection_name in current_query_output.target_collections:
                if collection_name not in collection_names:
                    yield Error(
                        f"Collection {collection_name} in target_collections field of the query output, "
                        "but not found in the available collections. "
                        "Make sure you are using the correct collection names."
                    )
                    return

        query.data_display = self._fix_collection_names_in_dict(
            query.data_display, schemas
        )
        query.fields_to_search = self._fix_collection_names_in_dict(
            query.fields_to_search, schemas
        )

        for collection_name in query.data_display:

            if collection_name not in collection_names:
                yield Error(
                    f"Collection {collection_name} in data_display keys, "
                    "but not found in the available collections. "
                    "Make sure you are using the correct collection names."
                )
                return

        for collection_name in query.fields_to_search:
            if collection_name not in collection_names:
                yield Error(
                    f"Collection {collection_name} in fields_to_search keys, "
                    "but not found in the available collections. "
                    "Make sure you are using the correct collection names."
                )
                return

        chunking_status_sent = False

        # Go through outputs for each unique query
        for i, query_output in enumerate(query.query_output.query_outputs):
            collection_names = query_output.target_collections.copy()

            for collection_name in collection_names:

                display_type = query.data_display[collection_name].display_type

                if self.logger:
                    self.logger.debug(
                        f"Display type for collection {collection_name}: {display_type}"
                    )
                    self.logger.debug(
                        f"Fields to search for collection {collection_name}: {query.fields_to_search[collection_name]}"
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

                    # if not chunking_status_sent:
                    #     yield Status("Chunking collections...")
                    #     yield TreeUpdate(
                    #         from_node="query_executor",
                    #         to_node="chunker",
                    #         reasoning="Chunking collections because they are too large to vectorise effectively.",
                    #         last_in_branch=not any(
                    #             query.data_display[collection_name_0].summarise_items
                    #             for collection_name_0 in query.data_display
                    #         ),
                    #     )
                    #     chunking_status_sent = True

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
                        try:
                            unchunked_response, _ = await execute_weaviate_query(
                                client,
                                query_output_copy,
                                reference_property="isChunked",
                                named_vector_fields=query.fields_to_search,
                            )
                        except Exception as e:
                            yield Error(str(e))
                            continue

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
                                f"ELYSIA_CHUNKED_{collection_name.lower()}__"
                            )
                    query.fields_to_search[
                        f"ELYSIA_CHUNKED_{collection_name.lower()}__"
                    ] = None

                    if self.logger:
                        self.logger.debug(
                            f"Chunked {collection_name} and updated query output to {query_output.target_collections}"
                        )

            # Execute query within Weaviate
            async with client_manager.connect_to_async_client() as client:
                try:
                    responses, code_strings = await execute_weaviate_query(
                        client,
                        query_output,
                        named_vector_fields=query.fields_to_search,
                    )
                except Exception as e:
                    for collection_name in collection_names:
                        yield Error(str(e))
                    continue  # don't exit, try again for next query_output (might not error)

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
                    objects[-1]["uuid"] = str(obj.uuid)

                # Write various metadata for LLM parsing
                metadata = {
                    "collection_name": collection_name,
                    "display_type": display_type,
                    "needs_summarising": summarise_items,
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

                if summarise_items and self.summariser_in_tree:
                    if (
                        "items_to_summarise"
                        not in tree_data.environment.hidden_environment
                    ):
                        tree_data.environment.hidden_environment[
                            "items_to_summarise"
                        ] = []
                    tree_data.environment.hidden_environment[
                        "items_to_summarise"
                    ].append(output)
                else:
                    yield output

        if self.logger:
            self.logger.debug("Query Tool finished!")


class SimpleQuery(Tool):
    """
    Simplified Elysia Query Tool
    Queries the knowledge base with Weaviate using hybrid, semantic, or keyword search, applying filters and more.
    These are dynamically decided by an LLM, based on the user prompt and the collections available.

    This tool has none of the additional functionality of the Query tool.
    So the results will be returned as they are, with no display type mapping, summarisation, or chunking.
    """

    def __init__(
        self,
        logger: Logger | None = None,
        result_name: str = "Retrieval",
        **kwargs,
    ):
        super().__init__(
            name="query",
            description="""
            Retrieves and displays specific data entries from the collections.
            Then, query with semantic search, keyword search, or a combination of both.
            Queries can be filtered, sorted, and more.
            Retrieving and displaying specific data entries rather than performing calculations or summaries.
            Do not use 'query' as a preliminary filtering step when 'aggregate' can achieve the same result more efficiently (if 'aggregate' is available).
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
        self.result_name = result_name

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

    def _fix_collection_names(self, collection_names: list[str], schemas: dict):
        return [
            correct_collection_name
            for correct_collection_name in list(schemas.keys())
            if correct_collection_name.lower()
            in [collection_name.lower() for collection_name in collection_names]
        ]

    def _fix_collection_names_in_dict(
        self, collection_dict: dict, schemas: dict
    ) -> dict:
        true_collection_names = list(schemas.keys())
        truth_map = {k.lower(): k for k in true_collection_names}
        return {truth_map.get(k.lower(), k): v for k, v in collection_dict.items()}

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager,
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

        if self.logger:
            self.logger.debug("Query Tool called!")
            self.logger.debug(f"Inputs: {inputs}")

        # Get the collection schemas (truth)
        schemas = tree_data.output_collection_metadata(with_mappings=True)

        # Find matching collection names (to match case sensitivity)
        if inputs["collection_names"] == [] or inputs["collection_names"] is None:
            collection_names = list(schemas.keys())
        else:
            collection_names = self._fix_collection_names(
                inputs["collection_names"], schemas
            )

        # Subset the schemas to just the collection names
        schemas = {
            collection_name: schemas[collection_name]
            for collection_name in collection_names
        }
        if self.logger:
            self.logger.debug(f"Collection names received: {collection_names}")

        # Set up the dspy model
        query_generator = ElysiaChainOfThought(
            SimpleQueryCreatorPrompt,
            tree_data=tree_data,
            environment=True,
            collection_schemas=True,
            tasks_completed=True,
            message_update=True,
            collection_names=collection_names,
        )

        # Get some metadata about the collection
        previous_queries = self._find_previous_queries(
            tree_data.environment.environment,
            tree_data.collection_data.collection_names,
        )

        if self.logger:
            self.logger.debug(f"Previous queries: {previous_queries}")

        # get searchable fields
        searchable_fields = {
            collection_name: [
                named_vector
                for named_vector in schemas[collection_name]["named_vectors"]
                if schemas[collection_name]["named_vectors"][named_vector]["enabled"]
            ]
            for collection_name in collection_names
        }

        # Generate query with LLM
        try:
            if tree_data.settings.USE_FEEDBACK:
                query, example_uuids = (
                    await query_generator.aforward_with_feedback_examples(
                        feedback_model="simple_query",
                        client_manager=client_manager,
                        base_lm=base_lm,
                        complex_lm=complex_lm,
                        available_collections=collection_names,
                        previous_queries=previous_queries,
                        searchable_fields=searchable_fields,
                        num_base_lm_examples=6,
                        return_example_uuids=True,
                    )
                )
            else:
                query = await query_generator.aforward(
                    lm=complex_lm,
                    available_collections=collection_names,
                    previous_queries=previous_queries,
                    searchable_fields=searchable_fields,
                )

        except Exception as e:
            yield Error(str(e))
            return

        if self.logger and query.query_output is not None:
            self.logger.debug(f"Query: {query.query_output.query_outputs}")

        # Yield results to front end
        yield Response(text=query.message_update)
        yield TrainingUpdate(
            module_name="simple_query",
            inputs={
                "available_collections": collection_names,
                "previous_queries": previous_queries,
                "searchable_fields": searchable_fields,
                **tree_data.to_json(),
            },
            outputs=query.__dict__["_store"],
        )
        if tree_data.settings.USE_FEEDBACK:
            yield FewShotExamples(uuids=example_uuids)

        # Return if model deems query impossible
        if (
            query.impossible
            or query.query_output is None
            or all(q is None for q in query.query_output.query_outputs)
            or len(query.query_output.query_outputs) == 0
        ):

            for collection_name in collection_names:
                metadata = {
                    "collection_name": collection_name,
                    "impossible": tree_data.user_prompt,
                    "impossible_reasoning": query.reasoning,
                    "query_output": (
                        query.query_output.query_outputs.model_dump()
                        if query.query_output is not None
                        else None
                    ),
                }
                yield BoringGenericRetrieval([], metadata)

            if self.logger:
                self.logger.warning(
                    f"Model judged query to be impossible. Returning to the decision tree..."
                )

        # extract and error handle the query output
        # TODO: replace with assertions when they're released
        for current_query_output in query.query_output.query_outputs:

            current_query_output.target_collections = self._fix_collection_names(
                current_query_output.target_collections, schemas
            )

            for collection_name in current_query_output.target_collections:
                if collection_name not in collection_names:
                    yield Error(
                        f"Collection {collection_name} in target_collections field of the query output, "
                        "but not found in the available collections. "
                        "Make sure you are using the correct collection names."
                    )
                    return

        query.fields_to_search = self._fix_collection_names_in_dict(
            query.fields_to_search, schemas
        )

        for collection_name in query.fields_to_search:
            if collection_name not in collection_names:
                yield Error(
                    f"Collection {collection_name} in fields_to_search keys, "
                    "but not found in the available collections. "
                    "Make sure you are using the correct collection names."
                )
                return

        # Go through outputs for each unique query
        for i, query_output in enumerate(query.query_output.query_outputs):
            collection_names = query_output.target_collections.copy()

            # Execute query within Weaviate
            async with client_manager.connect_to_async_client() as client:
                try:
                    responses, code_strings = await execute_weaviate_query(
                        client,
                        query_output,
                        named_vector_fields=query.fields_to_search,
                    )
                except Exception as e:
                    for collection_name in collection_names:
                        yield Error(
                            f"Query failed for collection {collection_name}: {str(e)}. Query was: {query_output.model_dump()}"
                        )
                    continue  # don't exit, try again for next query_output (might not error)

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

                # Get the objects from the response
                objects = []
                for obj in response.objects:
                    objects.append({k: v for k, v in obj.properties.items()})
                    objects[-1]["uuid"] = str(obj.uuid)

                # Write various metadata for LLM parsing
                metadata = {
                    "collection_name": collection_name,
                    "query_text": query_output.search_query,
                    "query_type": query_output.search_type,
                    "query_output": query_output.model_dump(),
                    "code": {
                        "language": "python",
                        "title": "Query",
                        "text": code_strings[k],
                    },
                }

                # Create the retrieval object
                yield Result(
                    objects=objects,
                    metadata=metadata,
                    mapping=None,
                    name=self.result_name,
                    payload_type=collection_name,
                    llm_message=(
                        f"Executed query on collection {collection_name}. "
                        f"Returned {len(objects)} objects."
                    ),
                    unmapped_keys=["uuid", "_REF_ID"],
                )
