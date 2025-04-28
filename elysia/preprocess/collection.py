import random
import dspy
from rich.progress import Progress
from logging import Logger

# Weaviate
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.query import Metrics

from elysia.config import nlp
from elysia.globals import return_types as rt
from elysia.preprocess.prompt_executors import (
    CollectionSummariserExecutor,
    DataMappingExecutor,
    ReturnTypeExecutor,
)
from elysia.util.client import ClientManager
from elysia.util.collection import async_get_collection_data_types
from elysia.util.async_util import asyncio_run

from elysia.config import settings as environment_settings


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
        lm: str = dspy.LM(model="claude-3-5-haiku-latest"),
        logger: Logger = environment_settings.logger,
    ):
        self.collection_summariser_executor = CollectionSummariserExecutor()
        self.data_mapping_executor = DataMappingExecutor()
        self.return_type_executor = ReturnTypeExecutor()

        self.collection_summariser_executor = dspy.asyncify(
            self.collection_summariser_executor
        )
        self.data_mapping_executor = dspy.asyncify(self.data_mapping_executor)
        self.return_type_executor = dspy.asyncify(self.return_type_executor)

        self.threshold_for_missing_fields = threshold_for_missing_fields
        self.lm = lm
        self.logger = logger

    async def _summarise_collection(
        self,
        collection,
        properties: dict,
        summary_sample_size: int = 200,
        len_collection: int = 1000,
    ):
        # Randomly sample sample_size objects for the summary
        indices = random.sample(range(len_collection), summary_sample_size)
        subset_objects = []

        counter = 0
        async for item in collection.iterator():
            counter += 1
            if counter in indices:
                subset_objects.append(item.properties)

        # Summarise the collection
        summary = await self.collection_summariser_executor(
            data=subset_objects, data_fields=list(properties.keys()), lm=self.lm
        )
        return summary

    async def _evaluate_field_statistics(
        self, collection, properties: dict, property: str, full_response=None
    ):
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
                lengths = [
                    len(nlp(obj.properties[property])) for obj in full_response.objects
                ]

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
    ):
        return_types = await self.return_type_executor(
            collection_summary=collection_summary,
            data_fields=data_fields,
            example_objects=example_objects,
            possible_return_types=rt.specific_return_types,
            lm=self.lm,
        )

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
    ):
        mapping, mapper, error_message = await self.data_mapping_executor(
            input_data_fields=input_fields,
            output_data_fields=output_fields,
            input_data_types=properties,
            collection_information=collection_information,
            example_objects=example_objects,
            lm=self.lm,
        )

        return mapping, error_message

    # TODO: this needs to go over to the async client manager
    async def __call__(
        self,
        collection_name: str,
        client_manager: ClientManager,
        manageable_sample_size: int = 100,
        summary_sample_size: int = 50,
        force: bool = False,
    ):
        async with client_manager.connect_to_async_client() as client:
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
                error = ""
                # Get the collection and its properties
                try:
                    collection = client.collections.get(collection_name)
                    properties = await async_get_collection_data_types(
                        client, collection_name
                    )
                except Exception as e:
                    yield await self.process_update(progress=0, error=str(e))
                    return

                # get number of items in collection
                agg = await collection.aggregate.over_all(total_count=True)
                len_collection = agg.total_count

                # Summarise the collection using LLM
                try:
                    summary = await self._summarise_collection(
                        collection,
                        properties,
                        min(summary_sample_size, len_collection),
                        len_collection,
                    )
                except Exception as e:
                    error = str(e)
                    yield await self.process_update(progress=0, error=error)
                    return

                yield await self.process_update(progress=1 / float(total))

                # Evaluate if the collection is manageable, if not, we will not fetch all objects
                collection_is_manageable = len_collection < manageable_sample_size

                # If the collection is manageable, fetch all objects
                if collection_is_manageable:
                    full_response = await collection.query.fetch_objects(
                        limit=len_collection
                    )
                else:
                    full_response = None

                # Get some example objects
                example_objects = await collection.query.fetch_objects(limit=3)
                example_objects = [obj.properties for obj in example_objects.objects]

                # Initialise the output
                out = {
                    "name": collection_name,
                    "length": len_collection,
                    "summary": summary,
                    "fields": {},
                    "mappings": {},
                }

                try:
                    # Evaluate the summary statistics of each field
                    for property in properties:
                        out["fields"][property] = await self._evaluate_field_statistics(
                            collection, properties, property, full_response
                        )
                except Exception as e:
                    yield await self.process_update(
                        progress=1 / float(total), error=str(e)
                    )
                    return

                # Evaluate the return types
                return_types = await self._evaluate_return_types(
                    summary, properties, example_objects
                )

                yield await self.process_update(progress=2 / float(total))
                progress = 2 / float(total)
                remaining_progress = 1 - progress
                num_remaining = len(return_types)

                # Define the mappings
                mappings = {}
                for return_type in return_types:
                    fields = rt.types_dict[return_type]

                    mapping, error_message = await self._define_mappings(
                        input_fields=list(fields.keys()),
                        output_fields=list(properties.keys()),
                        properties=properties,
                        collection_information=out,
                        example_objects=example_objects,
                    )

                    if error_message != "":
                        yield await self.process_update(
                            progress=progress, error=error_message
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
                        mapping, error_message = await self._define_mappings(
                            input_fields=list(rt.epic_generic.keys()),
                            output_fields=list(properties.keys()),
                            properties=properties,
                            collection_information=out,
                            example_objects=example_objects,
                        )

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

                # Save to a collection
                if await client.collections.exists(
                    f"ELYSIA_METADATA_{collection_name.lower()}__"
                ):
                    await client.collections.delete(
                        f"ELYSIA_METADATA_{collection_name.lower()}__"
                    )

                metadata_collection = await client.collections.create(
                    f"ELYSIA_METADATA_{collection_name.lower()}__"
                )
                await metadata_collection.data.insert(out)

                yield await self.process_update(progress=1)
            else:
                self.logger.info(f"Collection {collection_name} already exists!")


def preprocess(
    collection_names: list[str],
    client_manager: ClientManager | None = None,
    manageable_sample_size: int = 100,
    summary_sample_size: int = 50,
    verbose: bool = False,
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

    Args:
        collection_name (str): The name of the collection to preprocess.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables.
        manageable_sample_size (int): The number of objects to sample from the collection to evaluate the statistics.
        summary_sample_size (int): The number of objects to sample from the collection to write a summary.
        verbose (bool): Whether to print the progress of the preprocessor.
        force (bool): Whether to force the preprocessor to run even if the collection already exists.
    """

    if client_manager is None:
        client_manager = ClientManager()

    preprocessor = CollectionPreprocessor()

    async def run(collection_name: str):
        async for result in preprocessor(
            collection_name,
            client_manager,
            manageable_sample_size,
            summary_sample_size,
            force,
        ):
            pass

    async def run_with_live(collection_name: str):
        with Progress() as progress:
            task = progress.add_task(f"Preprocessing {collection_name}", total=1)
            async for result in preprocessor(
                collection_name,
                client_manager,
                manageable_sample_size,
                summary_sample_size,
                force,
            ):
                if result is not None and "progress" in result:
                    progress.update(task, completed=result["progress"])

    if verbose:
        for collection_name in collection_names:
            asyncio_run(run_with_live(collection_name))
    else:
        for collection_name in collection_names:
            asyncio_run(run(collection_name))
