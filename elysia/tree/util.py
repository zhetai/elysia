import uuid
from typing import Any

# dspy requires a 'base' LM but this should not be used
import dspy
from pympler import asizeof
from logging import Logger

from weaviate.util import generate_uuid5
from weaviate.classes.query import MetadataQuery, Sort, Filter

from elysia.objects import (
    Response,
    Return,
    Tool,
    Update,
    Status,
)

# Objects
from elysia.tree.objects import TreeData
from elysia.util.objects import TrainingUpdate, TreeUpdate, FewShotExamples
from elysia.util.elysia_chain_of_thought import ElysiaChainOfThought
from elysia.util.reference import create_reference
from elysia.util.parsing import format_datetime
from elysia.util.client import ClientManager
from elysia.tree.prompt_templates import (
    construct_decision_prompt,
    FollowUpSuggestionsPrompt,
    TitleCreatorPrompt,
)
from elysia.tools.text.prompt_templates import TextResponsePrompt

# Settings
from elysia.config import Settings, load_base_lm, load_complex_lm
from elysia.util.retrieve_feedback import retrieve_feedback


class ForcedTextResponse(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="final_text_response",
            description="",
            status="Writing response...",
            inputs={},
            end=True,
        )

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        text_response = ElysiaChainOfThought(
            TextResponsePrompt,
            tree_data=tree_data,
            environment=True,
            tasks_completed=True,
            message_update=False,
        )

        output = await text_response.aforward(
            lm=base_lm,
        )

        yield Response(text=output.response)


class RecursionLimitException(Exception):
    pass


class Decision:
    def __init__(
        self,
        function_name: str,
        function_inputs: dict,
        reasoning: str,
        impossible: bool,
        end_actions: bool,
        last_in_tree: bool = False,
    ):
        self.function_name = function_name
        self.function_inputs = function_inputs
        self.reasoning = reasoning
        self.impossible = impossible
        self.end_actions = end_actions
        self.last_in_tree = last_in_tree


class DecisionNode:
    def __init__(
        self,
        id: str,
        instruction: str,
        options: dict[str, dict[str, str | bool | Tool | None]],
        root: bool = False,
        logger: Logger = None,
        use_elysia_collections: bool = True,
    ):
        self.id = id
        self.instruction = instruction
        self.options = options
        self.root = root
        self.logger = logger
        self.use_elysia_collections = use_elysia_collections

    def _get_options(self):
        return self.options

    def add_option(
        self,
        id: str,
        description: str,
        inputs: dict,
        action: Tool | None = None,
        end: bool = True,
        status: str = "",
        next: str | None = None,
    ):
        if status == "":
            status = f"Running {id}..."

        self.options[id] = {
            "description": description,
            "inputs": inputs,
            "action": action,
            "end": end,
            "status": status,
            "next": next,
        }

    def remove_option(self, id: str):
        if id in self.options:
            del self.options[id]

    def _options_to_json(self, available_tools: list[str]):
        """
        Options that get shown to the LLM.
        Remove any that are empty branches.
        """
        out = {}
        for node in self.options:
            if node not in available_tools:  # empty branch
                continue

            out[node] = {
                "function_name": node,
                "description": self.options[node]["description"],
                "inputs": self.options[node]["inputs"],
            }
        return out

    def _unavailable_options_to_json(self, unavailable_tools: list[tuple[str, str]]):
        """
        Options unavailable at this time and why.
        """
        out = {}
        for tool, reason in unavailable_tools:
            if reason == "":
                reason = "No reason provided."
            out[tool] = {
                "function_name": tool,
                "available_at": reason,
            }
        return out

    def decide_from_route(self, route: str):
        """
        Decide from a route.
        """
        possible_nodes = self._get_options()

        next_route = route[0]
        if next_route not in possible_nodes:
            raise Exception(
                f"Next node in training route ({next_route}) not in possible nodes ({possible_nodes})"
            )

        route = route[1:]
        completed = len(route) == 0

        return (
            Decision(
                function_name=next_route,
                reasoning=f"Decided to run {next_route} from route {route}",
                impossible=False,
                function_inputs={},
                end_actions=completed,
            ),
            route,
        )

    async def load_model_from_examples(
        self,
        decision_executor: dspy.Module,
        client_manager: ClientManager,
        user_prompt: str,
    ) -> dspy.Module:

        examples = await retrieve_feedback(client_manager, user_prompt, "decision")

        if len(examples) == 0:
            return decision_executor

        optimizer = dspy.LabeledFewShot(k=10)
        compiled_executor = optimizer.compile(decision_executor, trainset=examples)
        return compiled_executor

    async def __call__(
        self,
        tree_data: TreeData,
        base_lm: dspy.LM = None,
        complex_lm: dspy.LM = None,
        available_tools: list[str] = [],
        unavailable_tools: list[str] = [],
        successive_actions: dict = {},
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        available_options = self._options_to_json(available_tools)
        unavailable_options = self._unavailable_options_to_json(unavailable_tools)

        if len(available_options) == 0:
            raise RuntimeError(
                "No available tools to call! Make sure you have added some tools to the tree. "
                "Or the .is_tool_available() method is returning True for at least one tool."
            )

        if self.logger:
            self.logger.debug(f"Available options: {list(available_options.keys())}")

        one_choice = (
            all(option["inputs"] == {} for option in available_options.values())
            and len(available_options) == 1
        )

        for option in available_options:
            if available_options[option]["inputs"] == {}:
                available_options[option][
                    "inputs"
                ] = "No inputs are needed for this function."

        if not one_choice:
            decision_executor = ElysiaChainOfThought(
                construct_decision_prompt(available_tools),
                tree_data=tree_data,
                environment=True,
                collection_schemas=self.use_elysia_collections,
                tasks_completed=True,
                message_update=True,
            )

            if tree_data.settings.USE_FEEDBACK:
                output, uuids = await decision_executor.aforward_with_feedback_examples(
                    feedback_model="decision",
                    client_manager=client_manager,
                    base_lm=base_lm,
                    complex_lm=complex_lm,
                    instruction=self.instruction,
                    tree_count=tree_data.tree_count_string(),
                    available_actions=available_options,
                    unavailable_actions=unavailable_options,
                    successive_actions=successive_actions,
                    num_base_lm_examples=3,
                    return_example_uuids=True,
                )
            else:
                output = await decision_executor.aforward(
                    instruction=self.instruction,
                    tree_count=tree_data.tree_count_string(),
                    available_actions=available_options,
                    unavailable_actions=unavailable_options,
                    successive_actions=successive_actions,
                    lm=base_lm,
                )

            decision = Decision(
                output.function_name,
                output.function_inputs,
                output.reasoning,
                output.impossible,
                output.end_actions and self.options[output.function_name]["end"],
            )

            results = [
                TrainingUpdate(
                    module_name="decision",
                    inputs=tree_data.to_json(),
                    outputs={k: v for k, v in output.__dict__["_store"].items()},
                ),
                TreeUpdate(
                    from_node=self.id,
                    to_node=output.function_name,
                    reasoning=output.reasoning,
                    last_in_branch=True,
                ),
                Status(self.options[output.function_name]["status"]),
            ]

            if output.function_name != "text_response":
                results.append(Response(output.message_update))

            if tree_data.settings.USE_FEEDBACK and len(uuids) > 0:
                results.append(FewShotExamples(uuids))

        else:
            decision = Decision(
                function_name=list(available_options.keys())[0],
                reasoning=f"Only one option available: {list(available_options.keys())[0]} (and no function inputs are needed).",
                impossible=False,
                function_inputs={},
                end_actions=(
                    self.options[list(available_options.keys())[0]]["end"]
                    and self.options[list(available_options.keys())[0]]["next"] is None
                ),
            )

            results = [
                TreeUpdate(
                    from_node=self.id,
                    to_node=list(available_options.keys())[0],
                    reasoning=decision.reasoning,
                    last_in_branch=True,
                ),
                Status(self.options[list(available_options.keys())[0]]["status"]),
            ]

        return decision, results

    def detailed_memory_usage(self) -> dict:
        """
        Returns a detailed breakdown of memory usage for all objects in the DecisionNode.

        Returns:
            dict: Dictionary containing memory sizes (in bytes) for each component
        """
        memory_usage = {}

        # Core attributes
        memory_usage["id"] = asizeof.asizeof(self.id)
        memory_usage["instruction"] = asizeof.asizeof(self.instruction)
        memory_usage["root"] = asizeof.asizeof(self.root)

        # Decision-related data - break down each option
        memory_usage["options"] = {}
        for option_id, option in self.options.items():
            memory_usage["options"][option_id] = {
                "description": asizeof.asizeof(option["description"]),
                "inputs": asizeof.asizeof(option["inputs"]),
                "action": asizeof.asizeof(option["action"]),
                "end": asizeof.asizeof(option["end"]),
                "status": asizeof.asizeof(option["status"]),
                "next": asizeof.asizeof(option["next"]),
            }

        # Calculate total
        memory_usage["total"] = sum(
            (
                v
                if isinstance(v, int)
                else (
                    sum(
                        (
                            sv
                            if isinstance(sv, int)
                            else sum(sv.values()) if isinstance(sv, dict) else 0
                        )
                        for sv in v.values()
                    )
                    if isinstance(v, dict)
                    else 0
                )
            )
            for v in memory_usage.values()
        )

        return memory_usage


class TreeReturner:
    """
    Class to parse the output of the tree to the frontend.
    """

    def __init__(
        self,
        user_id: str,
        conversation_id: str,
        tree_index: int = 0,
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.tree_index = tree_index
        self.store = []

    def set_tree_index(self, tree_index: int):
        self.tree_index = tree_index

    def clear_store(self):
        self.store = []

    def add_prompt(self, prompt: str, query_id: str):
        self.store.append(
            {
                "type": "user_prompt",
                "id": str(uuid.uuid4()),
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "payload": {
                    "prompt": prompt,
                },
            }
        )

    async def __call__(
        self,
        result: Return | TreeUpdate,
        query_id: str,
        last_in_tree: bool = False,
    ):
        if isinstance(result, Update):
            payload = await result.to_frontend(
                self.user_id, self.conversation_id, query_id
            )
            self.store.append(payload)
            return payload

        if isinstance(result, Return):
            payload = await result.to_frontend(
                self.user_id, self.conversation_id, query_id
            )
            self.store.append(payload)
            return payload

        if isinstance(result, TreeUpdate):
            payload = await result.to_frontend(
                self.user_id,
                self.conversation_id,
                query_id,
                self.tree_index,
                last_in_tree,
            )
            self.store.append(payload)
            return payload


async def create_conversation_title(conversation: list[dict], lm: dspy.LM):
    title_creator = dspy.Predict(TitleCreatorPrompt)
    title = await title_creator.aforward(
        conversation=conversation,
        lm=lm,
    )
    return title.title


async def get_follow_up_suggestions(
    tree_data: TreeData,
    current_suggestions: list[str],
    lm: dspy.LM,
    num_suggestions: int = 2,
    context: str | None = None,
):

    # load dspy model for suggestor
    follow_up_suggestor = dspy.Predict(FollowUpSuggestionsPrompt)

    if context is None:
        context = (
            "System context: You are an agentic RAG service querying or aggregating information from Weaviate collections via filtering, "
            "sorting, multi-collection queries, summary statistics. "
            "Uses tree-based approach with specialized agents and decision nodes for dynamic information retrieval, "
            "summary generation, and real-time results display while maintaining natural conversation flow. "
            "Create questions that are natural follow-ups to the user's prompt, "
            "which they may find interesting or create relevant insights to the already retrieved data. "
            "Or, questions which span across other collections, but are still relevant to the user's prompt."
        )

    # get prediction
    try:
        prediction = await follow_up_suggestor.aforward(
            user_prompt=tree_data.user_prompt,
            reference=create_reference(),
            conversation_history=tree_data.conversation_history,
            environment=tree_data.environment.environment,
            data_information=tree_data.output_collection_metadata(with_mappings=False),
            old_suggestions=current_suggestions,
            context=context,
            num_suggestions=num_suggestions,
            lm=lm,
        )
        suggestions = prediction.suggestions
    except Exception as e:
        suggestions = []

    return suggestions


async def get_saved_trees_weaviate(
    collection_name: str,
    client_manager: ClientManager | None = None,
    user_id: str | None = None,
):
    """
    Get all saved trees from a Weaviate collection.

    Args:
        collection_name (str): The name of the collection to get the trees from.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created from environment variables.
        user_id (str): The user ID to get the trees from.
            If not provided, the trees will be retrieved from the collection without any filters.

    Returns:
        dict: A dictionary of tree UUIDs and their titles.
    """
    if client_manager is None:
        client_manager = ClientManager()
        close_after_use = True
    else:
        close_after_use = False

    if user_id is not None:
        user_id_filter = Filter.by_property("user_id").equal(user_id)
    else:
        user_id_filter = None

    async with client_manager.connect_to_async_client() as client:

        if not await client.collections.exists(collection_name):
            return {}

        collection = client.collections.get(collection_name)

        len_collection = (
            await collection.aggregate.over_all(
                total_count=True,
                filters=user_id_filter,
            )
        ).total_count

        response = await collection.query.fetch_objects(
            limit=len_collection,
            sort=Sort.by_update_time(ascending=False),
            return_metadata=MetadataQuery(last_update_time=True),
            filters=user_id_filter,
        )

    if close_after_use:
        client_manager.close_clients()

    trees = {
        obj.properties["conversation_id"]: {
            "title": obj.properties["title"],
            "last_update_time": format_datetime(obj.metadata.last_update_time),
        }
        for obj in response.objects
    }

    return trees


async def delete_tree_from_weaviate(
    conversation_id: str,
    collection_name: str,
    client_manager: ClientManager | None = None,
):
    if client_manager is None:
        client_manager = ClientManager()

    async with client_manager.connect_to_async_client() as client:
        collection = client.collections.get(collection_name)
        uuid = generate_uuid5(conversation_id)
        if await collection.data.exists(uuid):
            await collection.data.delete_by_id(uuid)
