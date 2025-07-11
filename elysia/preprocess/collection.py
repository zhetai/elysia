import random
import dspy
from rich.progress import Progress
from typing import AsyncGenerator

# Weaviate
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.query import Metrics
from weaviate.collections import CollectionAsync

from elysia.config import nlp, Settings, load_base_lm
from elysia.config import settings as environment_settings

from elysia.util import return_types as rt

from elysia.preprocess.prompt_templates import (
    CollectionSummariserPrompt,
    DataMappingPrompt,
    ReturnTypePrompt,
)
from elysia.util.collection import async_get_collection_data_types
from elysia.util.async_util import asyncio_run
from elysia.util.parsing import format_dict_to_serialisable
from elysia.util.client import ClientManager


class ProcessUpdate:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    async def to_frontend(self, progress: float, error: str = "", **kwargs):
        return {
            "type": "update" if progress != 1 else "completed",
            "collection_name": self.collection_name,
            "progress": progress,
            "error": error,
        }

    async def __call__(self, *args, **kwargs):
        return await self.to_frontend(*args, **kwargs)


class CollectionPreprocessor:
    """
    Preprocess a collection, obtain a LLM-generated summary of the collection,
    a set of statistics for each field (such as unique categories), and a set of mappings
    from the fields to the frontend-specific fields in Elysia.

    In order:

    1. Evaluate all the data fields and groups/statistics of the data fields as a whole
    2. Write a summary of the collection via an LLM
    3. Evaluate what return types are available for this collection
    4. For each data field in the collection, evaluate what corresponding entry goes to what field in the return type (mapping)
    5. Save as a ELYSIA_METADATA_{collection_name}__ collection
    """

    def __init__(
        self,
        threshold_for_missing_fields: float = 0.6,
        settings: Settings = environment_settings,
    ):
        """
        Args:
            threshold_for_missing_fields (float): A value between 0 and 1.
                If the number of missing fields in the data mapping is greater than this threshold,
                the data mapping will not be considered as a potential mapping.
                In essence, when this percentage of fields are missing from the mapping, it is not a good mapping and ignored.
            settings (Settings): The settings to use for the LLM.
                Default to the global `settings` object in `elysia.config`.
        """
        self.collection_summariser_prompt = dspy.ChainOfThought(
            CollectionSummariserPrompt
        )
        self.return_type_prompt = dspy.ChainOfThought(ReturnTypePrompt)
        self.data_mapping_prompt = dspy.ChainOfThought(DataMappingPrompt)

        self.threshold_for_missing_fields = threshold_for_missing_fields
        self.lm = load_base_lm(settings)
        self.logger = settings.logger

    async def _summarise_collection(
        self,
        properties: dict,
        subset_objects: list[dict],
        len_collection: int,
    ) -> tuple[str, dict]:

        prediction = await self.collection_summariser_prompt.aforward(
            data_sample=subset_objects,
            data_fields=list(properties.keys()),
            sample_size=f"""
            {len(subset_objects)} out of a total of {len_collection}.
            You are seeing {round(len(subset_objects)/len_collection*100, 2)}% of the objects.
            """,
            lm=self.lm,
        )

        summary_concat = ""
        for sentence in [
            prediction.overall_summary,
            prediction.relationships,
            prediction.structure,
            prediction.irregularities,
        ]:
            if (
                sentence.endswith(".")
                or sentence.endswith("?")
                or sentence.endswith("!")
                or sentence.endswith("\n")
            ):
                summary_concat += f"{sentence} "
            else:
                summary_concat += f"{sentence}."

        return summary_concat, prediction.field_descriptions

    async def _evaluate_field_statistics(
        self, collection, properties: dict, property: str, full_response=None
    ) -> dict:
        out = {}
        out["type"] = properties[property]

        # Number (summary statistics)
        if properties[property] == "number":
            response = await collection.aggregate.over_all(
                return_metrics=[
                    Metrics(property).number(mean=True, maximum=True, minimum=True)
                ]
            )
            out["range"] = [
                response.properties[property].minimum,
                response.properties[property].maximum,
            ]
            out["mean"] = response.properties[property].mean
            out["groups"] = []

        # Text (grouping + lengths)
        elif properties[property] == "text":

            # TODO: this returns EVERY SINGLE text value in the collection,
            # need to make this more efficient, is it possible to specify in metadata to not return the text values?
            response = await collection.aggregate.over_all(
                total_count=True, group_by=GroupByAggregate(prop=property)
            )
            groups = [group.grouped_by.value for group in response.groups]

            if len(groups) < 30:
                out["groups"] = groups
            else:
                out["groups"] = []

            if full_response is not None:
                # For text, we want to evaluate the length of the text in tokens (use spacy)
                lengths = []
                for obj in full_response.objects:
                    if property in obj.properties and isinstance(
                        obj.properties[property], str
                    ):
                        lengths.append(len(nlp(obj.properties[property])))

                out["range"] = [min(lengths), max(lengths)]
                out["mean"] = sum(lengths) / len(lengths)

            else:
                out["range"] = []
                out["mean"] = -9999999

        # Boolean (grouping + mean)
        elif properties[property] == "boolean":
            response = await collection.aggregate.over_all(
                return_metrics=[Metrics(property).boolean(percentage_true=True)]
            )
            out["groups"] = [True, False]
            out["mean"] = response.properties[property].percentage_true
            out["range"] = [0, 1]

        # Date (summary statistics)
        elif properties[property] == "date":
            response = await collection.aggregate.over_all(
                return_metrics=[
                    Metrics(property).date_(median=True, minimum=True, maximum=True)
                ]
            )
            out["range"] = [
                response.properties[property].minimum,
                response.properties[property].maximum,
            ]
            out["mean"] = response.properties[property].median

        # List (lengths)
        elif properties[property].endswith("[]"):
            if full_response is not None:
                lengths = [
                    len(obj.properties[property]) for obj in full_response.objects
                ]

                out["range"] = [min(lengths), max(lengths)]
                out["mean"] = sum(lengths) / len(lengths)
                out["groups"] = []

        else:
            out["range"] = []
            out["mean"] = -9999999
            out["groups"] = []

        return out

    async def _evaluate_return_types(
        self, collection_summary: str, data_fields: dict, example_objects: list[dict]
    ) -> list[str]:

        prediction = await self.return_type_prompt.aforward(
            collection_summary=collection_summary,
            data_fields=data_fields,
            example_objects=example_objects,
            possible_return_types=rt.specific_return_types,
            lm=self.lm,
        )
        return_types = prediction.return_types

        if return_types == []:
            return_types = ["epic_generic"]

        return return_types

    async def _define_mappings(
        self,
        input_fields: list,
        output_fields: list,
        properties: dict,
        collection_information: dict,
        example_objects: list[dict],
    ) -> dict:

        prediction = await self.data_mapping_prompt.aforward(
            input_data_fields=input_fields,
            output_data_fields=output_fields,
            input_data_types=properties,
            collection_information=collection_information,
            example_objects=example_objects,
            lm=self.lm,
        )

        return prediction.field_mapping

    async def _evaluate_index_properties(self, collection: CollectionAsync) -> dict:
        schema_info = await collection.config.get()

        index_properties = {
            "isNullIndexed": schema_info.inverted_index_config.index_null_state,
            "isLengthIndexed": schema_info.inverted_index_config.index_property_length,
            "isTimestampIndexed": schema_info.inverted_index_config.index_timestamps,
        }
        return index_properties

    async def _find_named_vectors(self, collection: CollectionAsync):
        schema_info = await collection.config.get()
        if not schema_info.vector_config:
            return []
        else:
            return {
                vector: {
                    "source_properties": schema_info.vector_config[
                        vector
                    ].vectorizer.source_properties,
                    "enabled": True,
                    "description": "",
                }
                for vector in schema_info.vector_config
            }

    async def __call__(
        self,
        collection_name: str,
        client_manager: ClientManager,
        min_sample_size: int = 10,
        max_sample_size: int = 20,
        num_sample_tokens: int = 30000,
        force: bool = False,
    ) -> AsyncGenerator[dict, None]:
        """
        Run the preprocessor for a single collection.

        Args:
            collection_name (str): The name of the collection to preprocess.
            client_manager (ClientManager): The client manager to use.
            min_sample_size (int): The minimum number of objects to sample from the collection to evaluate the statistics/summary. Optional, defaults to 10.
            max_sample_size (int): The maximum number of objects to sample from the collection to evaluate the statistics/summary. Optional, defaults to 20.
            num_sample_tokens (int): The number of tokens to approximately sample from the collection to evaluate the summary.
                The preprocessor will aim to use this many tokens in the sample objects to evaluate the summary.
                But will not exceed the maximum number of objects specified by `max_sample_size`, and always use at least `min_sample_size` objects.
            force (bool): Whether to force the preprocessor to run even if the collection already exists. Optional, defaults to False.
        """

        async with client_manager.connect_to_async_client() as client:
            if not await client.collections.exists(collection_name):
                raise Exception(f"Collection {collection_name} does not exist!")

            if (
                not await client.collections.exists(
                    f"ELYSIA_METADATA_{collection_name.lower()}__"
                )
                or force
            ):
                # Start saving the updates
                self.process_update = ProcessUpdate(collection_name)
                total = len(rt.specific_return_types) + 1 + 1
                progress = 0.0

                # Get the collection and its properties
                try:
                    collection = client.collections.get(collection_name)
                    properties = await async_get_collection_data_types(
                        client, collection_name
                    )
                except Exception as e:
                    yield await self.process_update(
                        progress=0,
                        error=f"Error getting collection properties: {str(e)}",
                    )
                    return

                # get number of items in collection
                agg = await collection.aggregate.over_all(total_count=True)
                len_collection: int = agg.total_count  # type: ignore

                # Randomly sample sample_size objects for the summary
                indices = random.sample(
                    range(len_collection),
                    max(min(max_sample_size, len_collection), 1),
                )

                # Get first object to estimate token count
                obj = await collection.query.fetch_objects(limit=1, offset=indices[0])
                token_count_0 = len(nlp(str(obj.objects[0].properties)))
                subset_objects: list[dict] = [obj.objects[0].properties]  # type: ignore

                # Get number of objects to sample to get close to num_sample_tokens
                num_sample_objects = max(
                    min_sample_size, num_sample_tokens // token_count_0
                )

                for index in indices[1:num_sample_objects]:
                    obj = await collection.query.fetch_objects(limit=1, offset=index)
                    subset_objects.append(obj.objects[0].properties)  # type: ignore

                # Estimate number of tokens
                self.logger.debug(
                    f"Estimated token count of sample: {token_count_0*len(subset_objects)}"
                )
                self.logger.debug(f"Number of objects in sample: {len(subset_objects)}")

                # Summarise the collection using LLM
                try:
                    summary, field_descriptions = await self._summarise_collection(
                        properties,
                        subset_objects,
                        len_collection,
                    )
                except Exception as e:
                    yield await self.process_update(
                        progress=0, error=f"Error summarising collection: {str(e)}"
                    )
                    return

                yield await self.process_update(progress=1 / float(total))

                full_response = await collection.query.fetch_objects(
                    limit=min(len_collection, max_sample_size)
                )

                # Get some example objects
                example_objects_response = await collection.query.fetch_objects(limit=3)
                example_objects: list[dict] = [
                    obj.properties for obj in example_objects_response.objects  # type: ignore
                ]

                # Initialise the output
                out = {
                    "name": collection_name,
                    "length": len_collection,
                    "summary": summary,
                    "index_properties": await self._evaluate_index_properties(
                        collection
                    ),
                    "named_vectors": await self._find_named_vectors(collection),
                    "fields": {},
                    "mappings": {},
                }

                try:
                    # Evaluate the summary statistics of each field
                    for property in properties:
                        out["fields"][property] = await self._evaluate_field_statistics(
                            collection, properties, property, full_response
                        )
                        if property in field_descriptions:
                            out["fields"][property]["description"] = field_descriptions[
                                property
                            ]
                        else:
                            out["fields"][property]["description"] = ""
                except Exception as e:
                    yield await self.process_update(
                        progress=1 / float(total),
                        error=f"Error evaluating field statistics: {str(e)}",
                    )
                    return

                # Evaluate the return types
                try:
                    return_types = await self._evaluate_return_types(
                        summary, properties, example_objects
                    )
                except Exception as e:
                    yield await self.process_update(
                        progress=2 / float(total),
                        error=f"Error evaluating return types: {str(e)}",
                    )
                    return

                yield await self.process_update(progress=2 / float(total))
                progress = 2 / float(total)
                remaining_progress = 1 - progress
                num_remaining = len(return_types)

                # Define the mappings
                mappings = {}
                for return_type in return_types:
                    fields = rt.types_dict[return_type]

                    try:
                        mapping = await self._define_mappings(
                            input_fields=list(fields.keys()),
                            output_fields=list(properties.keys()),
                            properties=properties,
                            collection_information=out,
                            example_objects=example_objects,
                        )
                    except Exception as e:
                        yield await self.process_update(
                            progress=2 / float(total),
                            error=f"Error defining mappings: {str(e)}",
                        )
                        return

                    # remove any extra fields the model may have added
                    mapping = {
                        k: v for k, v in mapping.items() if k in list(fields.keys())
                    }

                    progress += remaining_progress / num_remaining
                    yield await self.process_update(progress=min(progress, 0.99))

                    mappings[return_type] = mapping

                # Check across all mappings how many missing fields there are
                new_return_types = []
                for return_type in return_types:
                    num_missing = sum(
                        [m == "" for m in list(mappings[return_type].values())]
                    )
                    if (
                        num_missing / len(mappings[return_type].keys())
                        < self.threshold_for_missing_fields
                    ):
                        new_return_types.append(return_type)

                # check for no return types
                if len(new_return_types) == 0:
                    if "epic_generic" in return_types:
                        new_return_types = ["boring_generic"]
                        out["mappings"]["boring_generic"] = {
                            p: p for p in properties.keys()
                        }
                    else:
                        new_return_types = ["epic_generic"]

                        # and re-map for generic
                        try:
                            mapping = await self._define_mappings(
                                input_fields=list(rt.epic_generic.keys()),
                                output_fields=list(properties.keys()),
                                properties=properties,
                                collection_information=out,
                                example_objects=example_objects,
                            )
                        except Exception as e:
                            yield await self.process_update(
                                progress=min(progress, 0.99),
                                error=f"Error defining generic mappings: {str(e)}",
                            )
                            return

                        # and if this one fails, set to boring_generic
                        num_missing = sum([m == "" for m in list(mapping.values())])
                        if (
                            num_missing / len(mapping.keys())
                            >= self.threshold_for_missing_fields
                        ):
                            # boring_generic
                            out["mappings"]["boring_generic"] = {
                                p: p for p in properties.keys()
                            }
                        else:
                            out["mappings"]["epic_generic"] = mapping

                else:
                    out["mappings"] = {
                        return_type: mappings[return_type]
                        for return_type in new_return_types
                    }

                try:
                    # Save to a collection
                    if await client.collections.exists(
                        f"ELYSIA_METADATA_{collection_name.lower()}__"
                    ):
                        await client.collections.delete(
                            f"ELYSIA_METADATA_{collection_name.lower()}__"
                        )
                except Exception as e:
                    yield await self.process_update(
                        progress=min(progress, 0.99),
                        error=f"Error deleting existing metadata collection: {str(e)}",
                    )
                    return

                try:
                    metadata_collection = await client.collections.create(
                        f"ELYSIA_METADATA_{collection_name.lower()}__"
                    )
                    await metadata_collection.data.insert(out)
                except Exception as e:
                    yield await self.process_update(
                        progress=min(progress, 0.99),
                        error=f"Error saving metadata to collection: {str(e)}",
                    )
                    return

                yield await self.process_update(progress=1)

            else:
                self.logger.info(
                    f"Preprocessed collection for {collection_name} already exists!"
                )


async def preprocess_async(
    collection_names: list[str],
    client_manager: ClientManager | None = None,
    min_sample_size: int = 10,
    max_sample_size: int = 20,
    num_sample_tokens: int = 30000,
    settings: Settings = environment_settings,
    force: bool = False,
):
    """
    Asynchronous version of `preprocess`.

    Args:
        collection_names (list[str]): The names of the collections to preprocess.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.
        min_sample_size (int): The minimum number of objects to sample from the collection to evaluate the statistics/summary. Optional, defaults to 10.
        max_sample_size (int): The maximum number of objects to sample from the collection to evaluate the statistics/summary. Optional, defaults to 20.
        num_sample_tokens (int): The maximum number of tokens in the sample objects used to evaluate the summary. Optional, defaults to 30000.
        settings (Settings): The settings to use. Optional, defaults to the environment variables/configured settings.
        force (bool): Whether to force the preprocessor to run even if the collection already exists. Optional, defaults to False.
    """
    if client_manager is None:
        client_manager = ClientManager(
            wcd_url=settings.WCD_URL, wcd_api_key=settings.WCD_API_KEY
        )

    try:
        preprocessor = CollectionPreprocessor(settings=settings)

        if settings.LOGGING_LEVEL_INT > 20:
            for collection_name in collection_names:
                async for result in preprocessor(
                    collection_name=collection_name,
                    client_manager=client_manager,
                    min_sample_size=min_sample_size,
                    max_sample_size=max_sample_size,
                    num_sample_tokens=num_sample_tokens,
                    force=force,
                ):
                    if (
                        result is not None
                        and "error" in result
                        and result["error"] != ""
                    ):
                        raise Exception(result["error"])

        else:
            for collection_name in collection_names:
                with Progress() as progress:
                    task = progress.add_task(
                        f"Preprocessing {collection_name}", total=1
                    )
                    async for result in preprocessor(
                        collection_name=collection_name,
                        client_manager=client_manager,
                        min_sample_size=min_sample_size,
                        max_sample_size=max_sample_size,
                        num_sample_tokens=num_sample_tokens,
                        force=force,
                    ):
                        if result is not None and "progress" in result:
                            progress.update(task, completed=result["progress"])
                        elif (
                            result is not None
                            and "error" in result
                            and result["error"] != ""
                        ):
                            raise Exception(result["error"])
    finally:
        await client_manager.close_clients()


def preprocess(
    collection_names: list[str],
    client_manager: ClientManager | None = None,
    min_sample_size: int = 5,
    max_sample_size: int = 100,
    num_sample_tokens: int = 30000,
    settings: Settings = environment_settings,
    force: bool = False,
):
    """
    Preprocess a collection, obtain a LLM-generated summary of the collection,
    a set of statistics for each field (such as unique categories), and a set of mappings
    from the fields to the frontend-specific fields in Elysia.

    In order:
    1. Evaluate all the data fields and groups/statistics of the data fields as a whole
    2. Write a summary of the collection via an LLM
    3. Evaluate what return types are available for this collection
    4. For each data field in the collection, evaluate what corresponding entry goes to what field in the return type (mapping)
    5. Save as a ELYSIA_METADATA_{collection_name}__ collection

    Depending on the size of objects in the collection, you can choose the minimum and maximum sample size,
    which will be used to create a sample of objects for the LLM to create a collection summary.
    If your objects are particularly large, you can set the sample size to be smaller, to use less tokens and speed up the LLM processing.
    If your objects are small, you can set the sample size to be larger, to get a more accurate summary.
    This is a trade-off between speed/compute and accuracy.

    But note that the pre-processing step only needs to be done once for each collection.
    The output of this function is cached, so that if you run it again, it will not re-process the collection (unless the force flag is set to True).

    This function saves the output to a collection called ELYSIA_METADATA_{collection_name}__, which is automatically called by Elysia.
    This is saved to whatever Weaviate cluster URL/API key you have configured, or in your environment variables.
    You can change this by setting the `wcd_url` and `wcd_api_key` in the settings, and pass this Settings object to this function.

    Args:
        collection_names (list[str]): The names of the collections to preprocess.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.
        min_sample_size (int): The minimum number of objects to sample from the collection to evaluate the statistics/summary. Optional, defaults to 10.
        max_sample_size (int): The maximum number of objects to sample from the collection to evaluate the statistics/summary. Optional, defaults to 20.
        num_sample_tokens (int): The maximum number of tokens in the sample objects used to evaluate the summary. Optional, defaults to 30000.
        settings (Settings): The settings to use. Optional, defaults to the environment variables/configured settings.
        force (bool): Whether to force the preprocessor to run even if the collection already exists. Optional, defaults to False.
    """

    asyncio_run(
        preprocess_async(
            collection_names,
            client_manager,
            min_sample_size,
            max_sample_size,
            num_sample_tokens,
            settings,
            force,
        )
    )


def preprocessed_collection_exists(
    collection_name: str, client_manager: ClientManager | None = None
) -> bool:
    """
    Check if the preprocessed collection exists in the Weaviate cluster.
    This function simply checks if the cached preprocessed metadata exists in the Weaviate cluster.
    It does so by checking if the collection ELYSIA_METADATA_{collection_name.lower()}__ exists.

    Args:
        collection_name (str): The name of the collection to check.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.

    Returns:
        bool: True if the collection exists, False otherwise.
    """
    if client_manager is None:
        client_manager = ClientManager()

    with client_manager.connect_to_client() as client:
        exists = client.collections.exists(
            f"ELYSIA_METADATA_{collection_name.lower()}__"
        )
    return exists


def delete_preprocessed_collection(
    collection_name: str, client_manager: ClientManager | None = None
):
    """
    Delete the preprocessed collection from the Weaviate cluster.
    This function simply deletes the cached preprocessed metadata from the Weaviate cluster.
    It does so by deleting the collection ELYSIA_METADATA_{collection_name.lower()}__.

    Args:
        collection_name (str): The name of the collection to delete the preprocessed metadata for.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.
    """
    if client_manager is None:
        client_manager = ClientManager()

    with client_manager.connect_to_client() as client:
        if client.collections.exists(f"ELYSIA_METADATA_{collection_name.lower()}__"):
            client.collections.delete(f"ELYSIA_METADATA_{collection_name.lower()}__")


async def edit_preprocessed_collection(
    collection_name: str,
    client_manager: ClientManager | None = None,
    named_vectors: list[dict] | None = None,
    summary: str | None = None,
    mappings: dict[str, dict[str, str]] | None = None,
    fields: list[dict[str, str] | None] | None = None,
):
    """
    Edit a preprocessed collection.
    This function allows you to edit the named vectors, summary, mappings, and fields of a preprocessed collection.
    It does so by updating the ELYSIA_METADATA_{collection_name.lower()}__ collection.

    Args:
        collection_name (str): The name of the collection to edit.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.
        named_vectors (list[dict]): The named vectors to update. This has fields "name", "enabled", and "description".
            The "name" is used to identify the named vector to change (the name will not change).
            Set "enabled" to True/False to enable/disable the named vector.
            Set "description" to describe the named vector.
            The description of named vectors is not automatically generated by the LLM.
            Any named vectors that are not provided will not be updated.
            If None or not provided, the named vectors will not be updated.
        summary (str): The summary to update.
            The summary is a short description of the collection, generated by the LLM.
            This will replace the existing summary of the collection.
            If None or not provided, the summary will not be updated.
        mappings (dict): The mappings to update.
            The mappings are what the frontend will use to display the collection, and the associated fields.
            I.e., which fields correspond to which output fields on the frontend.
            The keys of the outer level of the dictionary are the mapping names, the values are dictionaries with the mappings.
            The inner dictionary has the keys as the collection fields, and the values as the frontend fields.
            If None or not provided, the mappings will not be updated.
        fields (list[dict]): The fields to update.
            Each element in the list is a dictionary with the following fields:
            - "name": The name of the field. (This is used to identify the field to change, the name will not change).
            - "description": The description of the field to update.
            Any fields that are not provided will not be updated.
            If None or not provided, the fields will not be updated.
    """

    if client_manager is None:
        client_manager = ClientManager()
        close_clients_after_completion = True
    else:
        close_clients_after_completion = False

    async with client_manager.connect_to_async_client() as client:
        metadata_name = f"ELYSIA_METADATA_{collection_name.lower()}__"

        # check if the collection itself exists
        if not await client.collections.exists(collection_name.lower()):
            raise Exception(f"Collection {collection_name} does not exist")

        # check if the metadata collection exists
        if not await client.collections.exists(metadata_name):
            raise Exception(f"Metadata collection for {collection_name} does not exist")

        else:
            metadata_collection = client.collections.get(metadata_name)
            metadata = await metadata_collection.query.fetch_objects(limit=1)
            uuid = metadata.objects[0].uuid
            properties: dict = metadata.objects[0].properties  # type: ignore

        # update the named vectors
        if named_vectors is not None:
            for named_vector in named_vectors:

                if named_vector["enabled"] is not None:
                    properties["named_vectors"][named_vector["name"]]["enabled"] = (
                        named_vector["enabled"]
                    )

                if named_vector["description"] is not None:
                    properties["named_vectors"][named_vector["name"]]["description"] = (
                        named_vector["description"]
                    )

        # update the summary
        if summary is not None:
            properties["summary"] = summary

        # update the mappings
        if mappings is not None:
            properties["mappings"] = mappings

        # update the fields
        if fields is not None:
            for field in fields:
                if field is not None:
                    properties["fields"][field["name"]]["description"] = field[
                        "description"
                    ]

        format_dict_to_serialisable(properties)

        # update the collection
        await metadata_collection.data.update(uuid=uuid, properties=properties)

    if close_clients_after_completion:
        await client_manager.close_clients()

    return properties
