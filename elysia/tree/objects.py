from typing import Any, List
from logging import Logger
from rich import print

from elysia.objects import Result
from elysia.util.client import ClientManager

# Util
from elysia.util.parsing import format_dict_to_serialisable, remove_whitespace


# -- Environment --
class Environment:
    """
    Store of all objects across different types of queries and responses.

    The environment is formatted as follows:
    ```python
    {
        "function_name": {
            "result_name": [
                {
                    "metadata": dict,
                    "objects": list[dict],
                },
                ...
            ]
        }
    }
    ```

    Where `"function_name"` is the name of the function that the result belongs to,
    e.g. if the result came from a tool called `"query"`, then `"function_name"` is `"query"`.

    `"result_name"` is the name of the Result object, e.g. if the result is a list of objects, then `"result_name"` is `"objects"`.

    `"metadata"` is the metadata of the result, e.g. the time taken to retrieve the result, the query used, etc.


    """

    def __init__(self, environment: dict[str, dict[str, List[Result]]] = {}):
        self.environment = environment

        self.environment["SelfInfo"] = {}
        self.environment["SelfInfo"]["generic"] = [
            {
                "name": "Elysia",
                "description": "An agentic RAG service in Weaviate.",
                "purpose": remove_whitespace(
                    """Elysia is an agentic retrieval augmented generation (RAG) service, where users can query from Weaviate collections,
                and the assistant will retrieve the most relevant information and answer the user's question. This includes a variety
                of different ways to query, such as by filtering, sorting, querying multiple collections, and providing summaries
                and textual responses.

                Elysia will dynamically display retrieved objects from the collections in the frontend.
                Elysia works via a tree-based approach, where the user's question is used to generate a tree of potential
                queries to retrieve the most relevant information.
                Each end of the tree connects to a separate agent that will perform a specific task, such as retrieval, aggregation, or generation, or more.

                The tree itself has decision nodes that determine the next step in the query.
                The decision nodes are decided via a decision-agent, which decides the task.

                The agents communicate via a series of different prompts, which are stored in the prompt-library.
                The decision-agent prompts are designed to be as general as possible, so that they can be used for a variety of different tasks.
                Some of these variables include conversation history, retrieved objects, the user's original question, train of thought via model reasoning, and more.

                The backend of Elysia is built with a range of libraries. Elysia is built with a lot of base python, and was constructed from the ground up to keep it as modular as possible.
                The LLM components are built with DSPy, a library for training and running LLMs.
                You are likely to be using a trained prompt, trained with DSPy's MIPRO or Bootstrap methods.
                This improves the quality of the responses by optimising the language of the prompt itself, as well as some examples for few-shot learning.
                """
                ),
            }
        ]

    def add(self, result: Result, function_name: str):
        """
        Adds a result to the environment.
        Is called automatically by the tree when a result is returned from an agent.

        Args:
            result (Result): The result to add to the environment.
            function_name (str): The name of the function that the result belongs to.
        """
        objects = result.to_json()
        name = result.name

        if function_name not in self.environment:
            self.environment[function_name] = {}

        if name not in self.environment[function_name]:
            self.environment[function_name][name] = []

        if len(objects) > 0:
            self.environment[function_name][name].append(
                {
                    "metadata": result.metadata,
                    "objects": [],
                }
            )

            for i, obj in enumerate(objects):
                # check if the object is already in the environment
                obj_found = False
                where_obj = None
                for env_item in self.environment[function_name][name]:
                    if obj in env_item["objects"]:
                        obj_found = True
                        where_obj = env_item["objects"].index(obj)
                        _REF_ID = env_item["objects"][where_obj]["_REF_ID"]
                        break

                if obj_found:
                    self.environment[function_name][name][-1]["objects"].append(
                        {
                            "object_info": f"This object is a repeat of {_REF_ID}, so properties are omitted.",
                        }
                    )
                else:
                    _REF_ID = f"{function_name}_{name}_{len(self.environment[function_name][name])}_{i}"
                    self.environment[function_name][name][-1]["objects"].append(
                        {
                            "_REF_ID": _REF_ID,
                            **obj,
                        }
                    )

    def to_json(self):
        """
        Converts the environment to a JSON serialisable format.
        Used to access specific objects from the environment.
        """
        return self.environment


# -- Collection Data --
class CollectionData:

    def __init__(self, collection_names: list[str], logger: Logger | None = None):
        self.collection_names = collection_names
        self.metadata = {}
        self.logger = logger

    async def set_collection_names(
        self, collection_names: list[str], client_manager: ClientManager
    ):
        temp_metadata = {}

        # check if any of these collections exist in the full metadata (cached)
        collections_to_get = []
        for collection_name in collection_names:
            if collection_name not in self.metadata:
                collections_to_get.append(collection_name)

        self.removed_collections = []
        self.incorrect_collections = []

        async with client_manager.connect_to_async_client() as client:
            for collection_name in collections_to_get:
                metadata_name = f"ELYSIA_METADATA_{collection_name.lower()}__"

                # check if the collection itself exists
                if not await client.collections.exists(collection_name.lower()):
                    self.incorrect_collections.append(collection_name)

                # check if the metadata collection exists
                elif await client.collections.exists(metadata_name):
                    metadata_collection = client.collections.get(metadata_name)
                    metadata = await metadata_collection.query.fetch_objects(limit=1)
                    properties = metadata.objects[0].properties
                    format_dict_to_serialisable(properties)
                    temp_metadata[collection_name] = properties

                # if the metadata collection does not exist, it is not preprocessed
                else:
                    self.removed_collections.append(collection_name)

        if len(self.removed_collections) > 0 and self.logger:
            self.logger.warning(
                "The following collections have not been pre-processed for Elysia. "
                f"{self.removed_collections}. "
                "Ignoring these collections for now."
            )

        if len(self.incorrect_collections) > 0 and self.logger:
            self.logger.warning(
                "The following collections cannot be found in this Weaviate cluster. "
                f"{self.incorrect_collections}. "
                "These are being ignored for now. Please check that the collection names are correct."
            )

        self.collection_names = [
            collection_name
            for collection_name in collection_names
            if collection_name not in self.removed_collections
            and collection_name not in self.incorrect_collections
        ]

        # add to cached full metadata
        self.metadata = {
            **self.metadata,
            **{
                collection_name: temp_metadata[collection_name]
                for collection_name in temp_metadata
                if collection_name not in self.metadata
            },
        }

        return self.collection_names

    def output_full_metadata(
        self, collection_names: list[str] | None = None, with_mappings: bool = False
    ):
        if collection_names is None:
            collection_names = self.collection_names

        if with_mappings:
            return {
                collection_name: self.metadata[collection_name]
                for collection_name in collection_names
            }
        else:
            return {
                collection_name: {
                    k: v
                    for k, v in self.metadata[collection_name].items()
                    if k != "mappings"
                }
                for collection_name in collection_names
            }

    def output_collection_summaries(self, collection_names: list[str] | None = None):
        if collection_names is None:
            return {
                collection_name: self.metadata[collection_name]["summary"]
                for collection_name in self.collection_names
            }
        else:
            return {
                collection_name: self.metadata[collection_name]["summary"]
                for collection_name in collection_names
            }

    def output_mapping_lists(self):
        return {
            collection_name: list(self.metadata[collection_name]["mappings"].keys())
            for collection_name in self.collection_names
        }

    def output_mappings(self):
        return {
            collection_name: self.metadata[collection_name]["mappings"]
            for collection_name in self.collection_names
        }


# -- Prompt Input Data Collection Objects --
class TreeData:
    """
    Store of data across the tree.
    This includes things like conversation history, actions, decisions, etc.
    These data are given to ALL agents, so every agent is aware of the stage of the decision processes.

    This also contains functions that process the data into an LLM friendly format,
    such as a string with extra description.
    E.g. the number of trees completed is converted into a string (i/N)
         and additional warnings if i is close to N.
    """

    def __init__(
        self,
        collection_data: CollectionData,
        user_prompt: str = "",
        conversation_history: list[dict] = [],
        environment: Environment = Environment(),
        tasks_completed: list[dict] = [],
        num_trees_completed: int = 0,
        recursion_limit: int = 3,
    ):
        """
        Args:
            collection_data (CollectionData): The collection data to use for the tree, described in the CollectionData class.
            user_prompt (str): The user's prompt.
            conversation_history (list[dict]): The conversation history stored in the current tree, of the form:
                ```python
                [
                    {
                        "role": "user" | "assistant",
                        "content": str,
                    },
                    ...
                ]
                ```
            environment (Environment): The environment, described in the Environment class.
            tasks_completed (list[dict]): The tasks completed as a list of dictionaries.
                This is separate from the environment, as it separates what tasks were completed in each prompt in which order.
            num_trees_completed (int): The current level of the decision tree, how many iterations have been completed so far.
            recursion_limit (int): The maximum number of iterations allowed in the decision tree.
        """
        # -- Base Data --
        self.user_prompt = user_prompt
        self.conversation_history = conversation_history
        self.environment = environment
        self.tasks_completed = tasks_completed
        self.num_trees_completed = num_trees_completed
        self.recursion_limit = recursion_limit

        # -- Collection Data --
        self.collection_data = collection_data
        self.collection_names = []

    def to_json(self):
        return {k: v for k, v in self.__dict__.items()}

    def set_property(self, property: str, value: Any):
        self.__dict__[property] = value

    def update_string(self, property: str, value: str):
        if property not in self.__dict__:
            self.__dict__[property] = ""
        self.__dict__[property] += value

    def update_list(self, property: str, value: Any):
        if property not in self.__dict__:
            self.__dict__[property] = []
        self.__dict__[property].append(value)

    def update_dict(self, property: str, key: str, value: Any):
        if property not in self.__dict__:
            self.__dict__[property] = {}
        self.__dict__[property][key] = value

    def delete_from_dict(self, property: str, key: str):
        if property in self.__dict__ and key in self.__dict__[property]:
            del self.__dict__[property][key]

    def soft_reset(self):
        self.previous_reasoning = {}

    def _update_task(self, task_dict, key, value):
        if value is not None:
            if key in task_dict:
                # If key already exists, append to it
                if isinstance(value, str):
                    task_dict[key] += "\n" + value
                elif isinstance(value, int) or isinstance(value, float):
                    task_dict[key] += value
                elif isinstance(value, list):
                    task_dict[key].extend(value)
                elif isinstance(value, dict):
                    task_dict[key].update(value)
                elif isinstance(value, bool):
                    task_dict[key] = value
            elif key not in task_dict:
                # If key does not exist, create it
                task_dict[key] = value

    def update_tasks_completed(
        self, prompt: str, task: str, num_trees_completed: int, **kwargs
    ):
        # search to see if the current prompt already has an entry for this task
        prompt_found = False
        task_found = False
        iteration_found = False

        for i, task_prompt in enumerate(self.tasks_completed):
            if task_prompt["prompt"] == prompt:
                prompt_found = True
                for j, task_j in enumerate(task_prompt["task"]):
                    if task_j["task"] == task:
                        task_found = True
                        task_i = j  # position of the task in the prompt

                    if task_j["iteration"] == num_trees_completed:
                        iteration_found = True

        # If the prompt is not found, add it to the list
        if not prompt_found:
            self.tasks_completed.append({"prompt": prompt, "task": [{}]})
            self.tasks_completed[-1]["task"][0]["task"] = task
            self.tasks_completed[-1]["task"][0]["iteration"] = num_trees_completed
            for kwarg in kwargs:
                self._update_task(
                    self.tasks_completed[-1]["task"][0], kwarg, kwargs[kwarg]
                )
            return

        # If the prompt is found but the task is not, add it to the list
        if prompt_found and not task_found:
            self.tasks_completed[-1]["task"].append({})
            self.tasks_completed[-1]["task"][-1]["task"] = task
            self.tasks_completed[-1]["task"][-1]["iteration"] = num_trees_completed
            for kwarg in kwargs:
                self._update_task(
                    self.tasks_completed[-1]["task"][-1], kwarg, kwargs[kwarg]
                )
            return

        # task already exists in this query, but the iteration is new
        if prompt_found and task_found and not iteration_found:
            self.tasks_completed[-1]["task"].append({})
            self.tasks_completed[-1]["task"][-1]["task"] = task
            self.tasks_completed[-1]["task"][-1]["iteration"] = num_trees_completed
            for kwarg in kwargs:
                self._update_task(
                    self.tasks_completed[-1]["task"][-1], kwarg, kwargs[kwarg]
                )
            return

        # If the prompt is found and the task is found, update the task
        if prompt_found and task_found:
            for kwarg in kwargs:
                self._update_task(
                    self.tasks_completed[-1]["task"][task_i], kwarg, kwargs[kwarg]
                )

    def tasks_completed_string(self):
        """
        Output a nicely formatted string of the tasks completed so far, designed to be used in the LLM prompt.
        This is where the outputs of the `llm_message` fields are displayed.
        You can use this if you are interfacing with LLMs in tools, to help it understand the context of the tasks completed so far.

        Returns:
            str: A separated and formatted string of the tasks completed so far in an LLM-parseable format.
        """
        out = ""
        for j, task_prompt in enumerate(self.tasks_completed):
            out += f"<prompt_{j+1}>\n"
            out += f"Prompt: {task_prompt['prompt']}\n"

            for i, task in enumerate(task_prompt["task"]):
                out += f"<task_{i+1}>\n"

                if "action" in task and task["action"]:
                    out += f"Completed action: {task['task']}\n"
                else:
                    out += f"Chosen subcategory: {task['task']}\n"

                for key in task:
                    if key != "task" and key != "action":
                        out += f"{key.capitalize()}: {task[key]}\n"

                out += f"</task_{i+1}>\n"
            out += f"</prompt_{j+1}>\n"

        return out

    async def set_collection_names(
        self, collection_names: list[str], client_manager: ClientManager
    ):
        self.collection_names = await self.collection_data.set_collection_names(
            collection_names, client_manager
        )
        return self.collection_names

    def tree_count_string(self):
        out = f"{self.num_trees_completed+1}/{self.recursion_limit}"
        if self.num_trees_completed == self.recursion_limit - 1:
            out += " (this is the last decision you can make before being cut off)"
        if self.num_trees_completed >= self.recursion_limit:
            out += " (recursion limit reached, write your full chat response accordingly - the decision process has been cut short, and it is likely the user's question has not been fully answered and you either haven't been able to do it or it was impossible)"
        return out

    def output_collection_metadata(self, with_mappings: bool = False):
        """
        Outputs the full metadata for the given collection names.

        Args:
            with_mappings (bool): Whether to output the mappings for the collections as well as the other metadata.

        Returns:
            dict: A dictionary of collection names to their metadata.
                The metadata are of the form:
                ```python
                {
                    collection_name_1: {
                        # summary statistics of each field in the collection
                        "fields": dict = {
                            field_name_1: dict = {
                                "range": list[float],
                                "type": str,
                                "groups": list[str],
                                "mean": float
                            },
                            field_name_2: dict,
                            ...
                        },

                        # mapping_1, mapping_2 etc refer to frontend-specific types that the AI has deemed appropriate for this data
                        # then the dict is to map the frontend fields to the data fields
                        "mappings": dict = {
                            mapping_1: dict,
                            mapping_2: dict,
                            ...,
                        },

                        # number of items in collection (float but just for consistency)
                        "length": float,

                        # AI generated summary of the dataset
                        "summary": str,

                        # name of collection
                        "name": str

                    },
                    ...
                }
                ```
                If `with_mappings` is `False`, then the mappings are not included.
                Each key in the outer level dictionary is a collection name.

        """
        return self.collection_data.output_full_metadata(
            self.collection_names, with_mappings
        )

    def output_collection_return_types(self):
        """
        Outputs the return types for the collections in the tree data.
        Essentially, this is a list of the keys that can be used to map the objects to the frontend.

        Returns:
            dict: A dictionary of collection names to their return types.
                ```python
                {
                    collection_name_1: list[str],
                    collection_name_2: list[str],
                    ...,
                }
                ```
                Each of these lists is a list of the keys that can be used to map the objects to the frontend.
        """
        collection_return_types = self.collection_data.output_mapping_lists()
        return {
            collection_name: collection_return_types[collection_name]
            for collection_name in self.collection_names
        }

    def to_json(self):
        return {
            "user_prompt": self.user_prompt,
            "conversation_history": self.conversation_history,
            "environment": self.environment.to_json(),
            "tasks_completed": self.tasks_completed,
            "num_trees_completed": self.num_trees_completed,
        }
