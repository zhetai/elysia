import ast
import asyncio
import inspect
import json
import os
import time
from copy import deepcopy
from typing import Any, Dict, List

# dspy requires a 'base' LM but this should not be used
import dspy
import nest_asyncio
from dspy import LM
from pympler import asizeof
from rich import print
from rich.console import Console
from rich.live import Live
from rich.text import Text as RichText
from rich.panel import Panel

import logging
from logging import Logger

# Weaviate
from elysia.config import Settings
from elysia.config import settings as environment_settings

# async dspy

# globals
from elysia.objects import (
    Completed,
    Response,
    Result,
    Return,
    Text,
    Tool,
    Update,
    Warning,
    Status,
)
from elysia.tools.retrieval.aggregate import Aggregate

# Actions
# from elysia.tools.querying.query import AgenticQuery
from elysia.tools.retrieval.query import Query
from elysia.tools.text.text import FakeTextResponse, Summarizer, TextResponse

# Objects
from elysia.tree.objects import CollectionData, TreeData

# Decision Prompt executors
from elysia.tree.prompt_executors import DecisionExecutor
from elysia.util.client import ClientManager
from elysia.config import (
    check_base_lm_settings,
    check_complex_lm_settings,
    load_base_lm,
    load_complex_lm,
)
from elysia.util.objects import LMTimer, Timer, TrainingUpdate, TreeUpdate

# Util
from elysia.util.parsing import remove_whitespace
from elysia.util.collection_metadata import retrieve_all_collection_names


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
        options: list[dict[str, str]],
        root: bool = False,
        model_filepath: str = "",
        load_dspy_model: bool = True,
        logger: Logger = None,
    ):
        self.id = id
        self.instruction = instruction
        self.options = options
        self.root = root
        self.load_dspy_model = load_dspy_model
        self.logger = logger
        self.model_filepath = model_filepath

    def _load_model(self, executor):
        if (
            self.load_dspy_model
            and self.model_filepath is not None
            and os.path.exists(self.model_filepath)
        ):
            try:
                executor.load(self.model_filepath)
                if self.logger:
                    self.logger.info(
                        f"[green]Loaded Decision model[/green] at [italic magenta]{self.model_filepath}[/italic magenta]"
                    )
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Failed to load Decision model at {self.model_filepath}"
                    )

        return executor

    def _get_options(self):
        out = []
        for option in self.options:
            if not self.options[option]["rule"]:
                out.append(option)
        return out

    def add_option(
        self,
        id: str,
        description: str,
        inputs: dict,
        action: Tool | None = None,
        end: bool = True,
        status: str = "",
        next: str | None = None,
        rule: bool = False,
    ):
        if status == "":
            status = f"Running {id}..."

        self.options[id] = {
            "description": description,
            "inputs": inputs,
            "action": action,
            "end": rule or end,
            "status": status,
            "next": next,
            "rule": rule,
        }

    def remove_option(self, id: str):
        if id in self.options:
            del self.options[id]

    def _options_to_json(self, empty_branches: list[str] = []):
        """
        Options that get shown to the LLM.
        Remove any that are empty branches.
        """
        out = {}
        for node in self.options:
            if node in empty_branches:  # empty branch
                continue

            out[node] = {
                "function_name": node,
                "description": self.options[node]["description"],
                "inputs": self.options[node]["inputs"],
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

    async def __call__(
        self,
        tree_data: TreeData,
        lm: dspy.LM = None,
        **kwargs,
    ):
        decision_executor = DecisionExecutor(self._get_options())
        # decision_executor = self._load_model(decision_executor)
        decision_executor = dspy.asyncify(decision_executor)

        output = await decision_executor(
            user_prompt=tree_data.user_prompt,
            conversation_history=tree_data.conversation_history,
            tasks_completed=tree_data.tasks_completed_string(),
            instruction=self.instruction,
            collection_information=tree_data.output_collection_metadata(
                with_mappings=False
            ),
            tree_count=tree_data.tree_count_string(),
            available_actions=self._options_to_json(),
            environment=tree_data.environment.to_json(),
            lm=lm,
        )

        yield Decision(
            output.function_name,
            output.function_inputs,
            output.reasoning,
            output.impossible,
            output.end_actions and self.options[output.function_name]["end"],
        )

        yield TrainingUpdate(
            model="decision",
            inputs=tree_data.to_json(),
            outputs={k: v for k, v in output.__dict__["_store"].items()},
        )

        yield TreeUpdate(
            from_node=self.id,
            to_node=output.function_name,
            reasoning=output.reasoning,
            last_in_branch=True,
        )

        yield Status(self.options[output.function_name]["status"])

        if output.function_name != "text_response":
            yield Response(output.message_update)

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
        memory_usage["model_filepath"] = asizeof.asizeof(self.model_filepath)
        memory_usage["load_dspy_model"] = asizeof.asizeof(self.load_dspy_model)

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
                "rule": asizeof.asizeof(option["rule"]),
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


# TODO: move these methods to the main tree, perhaps?
#       and each of these methods could do the auxiliary tasks e.g. updating environment, conversation history, etc.
# TODO: but could we also clean up the tree by moving things to different classes?
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

    def set_tree_index(self, tree_index: int):
        self.tree_index = tree_index

    async def __call__(
        self,
        result: Return | TreeUpdate,
        query_id: str,
        last_in_tree: bool = False,
    ):
        if isinstance(result, Update):
            return await result.to_frontend(self.conversation_id, query_id)

        if isinstance(result, Return):
            return await result.to_frontend(
                self.user_id, self.conversation_id, query_id
            )

        if isinstance(result, TreeUpdate):
            return await result.to_frontend(
                self.user_id,
                self.conversation_id,
                query_id,
                self.tree_index,
                last_in_tree,
            )


class BranchVisitor(ast.NodeVisitor):
    def __init__(self):
        self.branches = []

    def _evaluate_constant(self, node):
        """Convert AST Constant node to its actual value."""
        return node.value

    def _evaluate_dict(self, node):
        """Convert AST Dict node to a regular Python dictionary."""
        if isinstance(node, ast.Dict):
            return {
                self._evaluate_constant(key): self._evaluate_constant(value)
                for key, value in zip(node.keys, node.values)
            }
        return None

    def visit_Call(self, node):
        # Check if the call is creating a Branch object
        if isinstance(node.func, ast.Name) and node.func.id == "Branch":
            # Extract the updates dictionary from the Branch constructor
            if node.args:
                dict_node = node.args[0]
                evaluated_dict = self._evaluate_dict(dict_node)
                if evaluated_dict:
                    self.branches.append(evaluated_dict)
        self.generic_visit(node)


class Tree:
    def __init__(
        self,
        user_id: str = "1",
        conversation_id: str = "1",
        branch_initialisation: str = "default",
        dspy_initialisation: str = "mipro",
        load_dspy_model: bool = True,
        debug: bool = False,
        settings: Settings = environment_settings,
    ):
        # Define base variables of the tree
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.load_dspy_model = load_dspy_model
        self.settings = settings

        # keep track of the number of trees completed
        self.num_trees_completed = 0

        # Initialise some tree variables
        self.decision_nodes: dict[str, DecisionNode] = {}
        self.decision_history = []
        self.tree_index = -1
        self.base_lm_timer = LMTimer("base_lm")
        self.complex_lm_timer = LMTimer("complex_lm")
        self.suggestions = []
        self.actions_called = {}
        self.query_id_to_prompt = {}
        self.prompt_to_query_id = {}

        # -- Initialise the tree
        self.set_debug(debug)
        self.base_lm = self.settings.BASE_MODEL_LM
        self.complex_lm = self.settings.COMPLEX_MODEL_LM

        self.tools = {}
        self.set_dspy_initialisation(dspy_initialisation)
        self.set_branch_initialisation(branch_initialisation)

        self.tools["final_text_response"] = TextResponse()

        # some variables for storing feedback
        self.action_information = []
        self.history = {}
        self.training_updates = []

        # -- Get the root node and construct the tree
        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

        # initialise the returner (for frontend)
        self.returner = TreeReturner(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
        )

        # Define the inputs to prompts
        self.tree_data = TreeData(
            collection_data=CollectionData(
                collection_names=[], logger=self.settings.logger
            ),
            recursion_limit=5,
        )

        # Print the tree if required
        self.settings.logger.info("Initialised tree with the following decision nodes:")
        for decision_node in self.decision_nodes.values():
            self.settings.logger.info(
                f"  - [magenta]{decision_node.id}[/magenta]: {list(decision_node.options.keys())}"
            )

    def default_init(self):
        self.add_branch(
            root=True,
            branch_id="base",
            instruction="""
            Choose a base-level task based on the user's prompt and available information.
            You can search, which includes aggregating or querying information - this should be used if the user needs (more) information.
            You can end the conversation by choosing text response, or summarise some retrieved information.
            Base your decision on what information is available and what the user is asking for - you can search multiple times if needed,
            but you should not search if you have already found all the information you need.
            """,
            status="Choosing a base-level task...",
        )

        self.add_tool(branch_id="base", tool=Summarizer)

        self.add_tool(branch_id="base", tool=FakeTextResponse)

        self.add_branch(
            root=False,
            branch_id="search",
            from_branch_id="base",
            instruction="""
            Choose between querying the knowledge base via semantic/keyword search, or aggregating information by performing operations, on the knowledge base.
            Querying is when the user is looking for specific information related to the content of the dataset, requiring a specific search query. This is for retrieving specific information via a _query_, similar to a search engine.
            Aggregating is when the user is looking for a specific operations on the dataset, such as summary statistics of the quantity of some items. Aggregation can also include grouping information by some property and returning statistics about the groups.
            """,
            description=f"""
            Search the knowledge base. This should be used when the user is lacking information for this particular prompt. This retrieves information only and provides no output to the user except the information.
            Choose to query (semantic or keyword search on a knowledge base), or aggregate information (calculate properties/summary statistics/averages and operations on the knowledge bases).
            """,
            status="Searching the knowledge base...",
        )

        self.add_tool(branch_id="search", tool=Query)

        self.add_tool(branch_id="search", tool=Aggregate)

    def one_branch_init(self):
        self.add_branch(
            root=True,
            branch_id="base",
            instruction="""
            Choose a base-level task based on the user's prompt and available information.
            Decide based on the tools you have available as well as their descriptions.
            Read them thoroughly and match the actions to the user prompt.
            """,
            status="Choosing a base-level task...",
        )
        self.add_tool(branch_id="base", tool=Summarizer)

        self.add_tool(branch_id="base", tool=FakeTextResponse)

        self.add_tool(branch_id="base", tool=Query)

        self.add_tool(branch_id="base", tool=Aggregate)

    def clear_tree(self):
        self.decision_nodes = {}

    def set_branch_initialisation(self, initialisation: str | None):
        self.clear_tree()

        if (
            initialisation is None
            or initialisation == ""
            or initialisation == "default"
        ):
            self.default_init()
        elif initialisation == "one_branch":
            self.one_branch_init()

        self.branch_initialisation = initialisation

    def set_dspy_initialisation(self, initialisation: str | None):
        if initialisation is not None:
            mipro_suffix = "light_" if initialisation == "mipro" else ""
            self.model_filepath = f"elysia/training/dspy_models/decision/{initialisation}/{mipro_suffix}gemini-2.0-flash-001_gemini-2.0-flash-001.json"
        else:
            self.model_filepath = None

        for node in self.decision_nodes.values():
            node.model_filepath = self.model_filepath

        self.dspy_initialisation = initialisation

    def set_debug(self, debug: bool):
        self.debug = debug
        self.settings.LOCAL = debug

    def get_lm(self, model_type: str):
        if self.debug:  # if in debug mode, model is already loaded in the settings
            return (
                self.settings.BASE_MODEL_LM
                if model_type == "base"
                else self.settings.COMPLEX_MODEL_LM
            )
        else:  # if not in debug mode, model is hot-loaded
            return (
                load_base_lm(self.settings)
                if model_type == "base"
                else load_complex_lm(self.settings)
            )

    def _get_root(self):
        for decision_node in self.decision_nodes.values():
            if decision_node.root:
                if "root" in dir(self) and self.root != decision_node.id:
                    raise ValueError("Multiple root decision nodes found")

                self.root = decision_node.id

        if "root" not in dir(self):
            raise ValueError("No root decision node found")

    def _get_function_branches(self, func: Any) -> List[Dict]:
        """Analyzes a function for Branch objects without executing it."""
        source = inspect.getsource(func)
        tree = ast.parse(source.strip())
        visitor = BranchVisitor()
        visitor.visit(tree)
        return visitor.branches

    def _construct_tree(self, node_id: str, tree: dict):
        decision_node = self.decision_nodes[node_id]

        # Define desired key order
        key_order = ["name", "id", "description", "instruction", "reasoning", "options"]

        # Set the base node information
        tree["name"] = node_id.capitalize().replace("_", " ")
        tree["id"] = node_id
        if node_id == self.root:
            tree["description"] = ""
        tree["instruction"] = remove_whitespace(
            decision_node.instruction.replace("\n", "")
        )
        tree["reasoning"] = ""
        tree["options"] = {}

        # Order the top-level dictionary
        tree = {key: tree[key] for key in key_order if key in tree}

        # Initialize all options first with ordered dictionaries
        for option in decision_node.options:
            tree["options"][option] = {
                "description": remove_whitespace(
                    decision_node.options[option]["description"].replace("\n", "")
                )
            }

        # Then handle the recursive cases
        for option in decision_node.options:
            if decision_node.options[option]["action"] is not None:
                func = decision_node.options[option]["action"].__call__
                branches = self._get_function_branches(func)
                if branches:
                    tree["options"][option]["name"] = option.capitalize().replace(
                        "_", " "
                    )
                    tree["options"][option]["id"] = option
                    tree["options"][option]["instruction"] = ""
                    sub_tree = tree["options"][option]
                    for branch in branches:
                        branch_name = branch["name"].lower().replace(" ", "_")
                        if "options" not in sub_tree:
                            sub_tree["options"] = {}
                        sub_tree["options"][branch_name] = branch
                        sub_tree["options"][branch_name]["name"] = branch["name"]
                        sub_tree["options"][branch_name]["id"] = branch_name
                        sub_tree["options"][branch_name]["description"] = branch[
                            "description"
                        ]
                        sub_tree["options"][branch_name]["instruction"] = ""
                        sub_tree["options"][branch_name]["reasoning"] = ""
                        sub_tree = sub_tree["options"][branch_name]
                    sub_tree["options"] = {}

                else:
                    tree["options"][option]["name"] = option.capitalize().replace(
                        "_", " "
                    )
                    tree["options"][option]["id"] = option
                    tree["options"][option]["instruction"] = ""
                    tree["options"][option]["reasoning"] = ""
                    tree["options"][option]["options"] = {}

            elif decision_node.options[option]["next"] is not None:
                tree["options"][option] = self._construct_tree(
                    decision_node.options[option]["next"], tree["options"][option]
                )
            else:
                tree["options"][option]["name"] = option.capitalize().replace("_", " ")
                tree["options"][option]["id"] = option
                tree["options"][option]["instruction"] = ""
                tree["options"][option]["reasoning"] = ""
                tree["options"][option]["options"] = {}

            # Order each option's dictionary
            tree["options"][option] = {
                key: tree["options"][option][key]
                for key in key_order
                if key in tree["options"][option]
            }

        return tree

    async def set_collection_names(
        self,
        collection_names: list[str],
        client_manager: ClientManager,
    ):
        self.settings.logger.info(
            f"Using the following collection names: {collection_names}"
        )

        collection_names = await self.tree_data.set_collection_names(
            collection_names, client_manager
        )

    def _remove_empty_branches(self):
        empty_branches = []
        for branch_id, branch in self.decision_nodes.items():
            if len(branch.options) == 0:
                empty_branches.append(branch_id)

        for branch_id in self.decision_nodes:
            for empty_branch in empty_branches:
                self.decision_nodes[branch_id].remove_option(empty_branch)

        for empty_branch in empty_branches:
            self.settings.logger.warning(
                f"Removing empty branch: {empty_branch}",
                "No tools are attached to this branch, so it has been removed.",
                f"To add a tool to this branch, use .add_tool(tool_name, branch_id={empty_branch})",
            )
            del self.decision_nodes[empty_branch]

        return empty_branches

    def _get_function_inputs(self, tool_name: str, inputs: dict):
        if tool_name in self.tools:
            # any non-provided inputs are set to the default
            default_inputs = self.tools[tool_name].get_default_inputs()
            for default_input_name in default_inputs:
                if default_input_name not in inputs:
                    inputs[default_input_name] = default_inputs[default_input_name]

            # if the inputs match the 'schema' of keys: description, type, default, value, then take the value
            for input_name in inputs:
                if (
                    isinstance(inputs[input_name], dict)
                    and "value" in inputs[input_name]
                ):
                    inputs[input_name] = inputs[input_name]["value"]

            return inputs
        else:
            return {}

    def _check_rules(
        self, prev_rules_met: list[str], branch_id: str, client_manager: ClientManager
    ):
        base_lm = self.get_lm("base")
        complex_lm = self.get_lm("complex")

        branch = self.decision_nodes[branch_id]
        rules_met = []
        for function_name, option in branch.options.items():
            if option["rule"] and "rule_call" in dir(self.tools[function_name]):
                rule_met = self.tools[function_name].rule_call(
                    tree_data=self.tree_data,
                    client_manager=client_manager,
                    base_lm=base_lm,
                    complex_lm=complex_lm,
                )
                if rule_met and function_name not in prev_rules_met:
                    rules_met.append(function_name)

        return rules_met

    def set_conversation_id(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.returner.conversation_id = conversation_id

    def set_user_id(self, user_id: str):
        self.user_id = user_id
        self.returner.user_id = user_id

    def soft_reset(self):
        # conversation history is not reset
        # environment is not reset
        self.recursion_counter = 0
        self.tree_data.num_trees_completed = 0
        self.decision_history = []
        self.num_trees_completed = 0
        self.training_updates = []
        self.tree_data.soft_reset()
        self.action_information = []
        self.tree_index += 1
        self.returner.set_tree_index(self.tree_index)

    def save_history(self, query_id: str, time_taken_seconds: float):
        """
        What the tree did, results for saving feedback.
        """
        training_update = deepcopy(
            [update.to_json() for update in self.training_updates]
        )
        training_update = json.dumps(training_update)

        self.history[query_id] = {
            "num_trees_completed": self.num_trees_completed,
            "tree_data": deepcopy(self.tree_data),
            "action_information": deepcopy(self.action_information),
            "decision_history": deepcopy(self.decision_history),
            "base_lm_used": self.settings.BASE_MODEL,
            "complex_lm_used": self.settings.COMPLEX_MODEL,
            "time_taken_seconds": time_taken_seconds,
            "training_updates": training_update,
            "initialisation": f"{self.branch_initialisation}/{self.dspy_initialisation}",
        }
        # can remove training updates now
        self.training_updates = []

    def set_start_time(self):
        self.start_time = time.time()

    def add_tool(self, branch_id: str, tool: type, **kwargs):
        tool_instance = tool(
            logger=self.settings.logger,
            load_dspy_model=self.load_dspy_model,
            **kwargs,
        )

        if not isinstance(tool_instance, Tool):
            raise ValueError("The tool must be an instance of the Tool class")

        self.tools[tool_instance.name] = tool_instance

        if branch_id not in self.decision_nodes:
            raise ValueError(
                f"Branch {branch_id} not found. Use .add_branch() to add a branch before adding a tool."
            )

        self.decision_nodes[branch_id].add_option(
            id=tool_instance.name,
            description=tool_instance.description,
            inputs=tool_instance.inputs,
            action=self.tools[tool_instance.name],
            end=tool_instance.end,
            status=tool_instance.status,
            rule=tool_instance.rule,
        )

        # reconstruct tree
        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

    def remove_tool(self, branch_id: str, tool_name: str):
        self.decision_nodes[branch_id].remove_option(tool_name)
        del self.tools[tool_name]

        # reconstruct tree
        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

    def add_branch(
        self,
        branch_id: str,
        instruction: str,
        description: str = "",
        root: bool = False,
        from_branch_id: str = "",
        status: str = "",
    ):
        """
        Add a branch to the tree.
        args:
            branch_id: str - the id of the branch being added
            instruction: str - the general instruction for the branch, what is this branch containing?
                What kind of tools or actions are being decided on this branch?
            description: str - a description of the branch, if it is chosen from a previous branch
            root: bool - whether this is the root branch, i.e. the beginning of the tree
            from_branch_id: str - the id of the branch that this branch is stemming from
            status: str - the status message to be displayed when this branch is chosen
        """
        if not root and description == "":
            raise ValueError("Description is required for non-root branches.")
        if not root and from_branch_id == "":
            raise ValueError(
                "`from_branch_id` is required for non-root branches. "
                "Set `root=True` to create a root branch or choose where this branch stems from."
            )
        if root and description != "":
            self.settings.logger.warning(f"Description is not used for root branches. ")
            description = ""

        if root and from_branch_id != "":
            self.settings.logger.warning(
                "`from_branch_id` is not used for root branches. "
                "(As this is the root branch, it does not stem from any other branch.)"
                "If you wish this to be stemming from a previous branch, set `root=False`."
            )
            from_branch_id = ""

        if status == "":
            status = f"Running {branch_id}"

        if not root:
            self.decision_nodes[from_branch_id].add_option(
                id=branch_id,
                description=description,
                inputs={},
                action=None,
                end=False,
                status=status,
                next=branch_id,
            )

        decision_node = DecisionNode(
            id=branch_id,
            instruction=instruction,
            options={},
            root=root,
            load_dspy_model=self.load_dspy_model,
            model_filepath=self.model_filepath,
            logger=self.settings.logger,
        )
        self.decision_nodes[branch_id] = decision_node

        # reconstruct tree
        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

    def remove_branch(self, branch_id: str):
        for branch_id in self.decision_nodes:
            self.decision_nodes[branch_id].remove_option(branch_id)
        del self.decision_nodes[branch_id]

        # reconstruct tree
        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

    def _update_conversation_history(self, role: str, message: str):
        if message != "":
            # If the first message, create a new message
            if len(self.tree_data.conversation_history) == 0:
                self.tree_data.update_list(
                    "conversation_history", {"role": role, "content": message}
                )
            # If the last message is from the same role, append to the content
            elif self.tree_data.conversation_history[-1]["role"] == role:
                if self.tree_data.conversation_history[-1]["content"].endswith(" "):
                    self.tree_data.conversation_history[-1]["content"] += message
                else:
                    self.tree_data.conversation_history[-1]["content"] += " " + message
            # Otherwise, create a new message
            else:
                self.tree_data.update_list(
                    "conversation_history", {"role": role, "content": message}
                )

    def _update_actions_called(self, result: Result, decision: Decision):
        if self.user_prompt not in self.actions_called:
            self.actions_called[self.user_prompt] = []
            self.actions_called[self.user_prompt].append(
                {
                    "name": decision.function_name,
                    "inputs": decision.function_inputs,
                    "reasoning": decision.reasoning,
                    "output": None,
                }
            )
        if self.debug:
            self.actions_called[self.user_prompt][-1]["output"] = result.objects
        else:
            self.actions_called[self.user_prompt][-1]["output"] = []

    def _update_environment(self, result: Result, decision: Decision):
        """
        Given a yielded result from an action or otherwise, update the environment.
        I.e. the items within the LLM knowledge base/prompt for future decisions/actions
        All Result subclasses have their .to_json() method added to the environment.
        As well, all Result subclasses have their llm_parse() method added to the tasks_completed.
        """

        # add to environment (store of retrieved/called objects)
        self.tree_data.environment.add(result, decision.function_name)

        # add to log of actions called
        self.action_information.append(
            {
                "action_name": decision.function_name,
                **{key: value for key, value in result.metadata.items()},
            }
        )

        # add to tasks completed (parsed info / train of thought for LLM)
        self.tree_data.update_tasks_completed(
            prompt=self.user_prompt,
            task=decision.function_name,
            num_trees_completed=self.tree_data.num_trees_completed,
            reasoning=decision.reasoning,
            # retrieved_objects=result.to_json(),
            parsed_info=result.llm_parse(),
        )

        # add to log of actions called
        # TODO: is this the same thing as action_information? but better?
        self._update_actions_called(result, decision)

    async def _evaluate_action(self, result: Return | Decision | TreeUpdate):
        if isinstance(result, Decision):
            self.current_decision = result

        if isinstance(result, Result):
            self._update_environment(result, self.current_decision)

        if isinstance(result, TrainingUpdate):
            self.training_updates.append(result)

        if isinstance(result, Text):
            self._update_conversation_history("assistant", result.text)
            if self.settings.LOGGING_LEVEL_INT <= 20:
                print(
                    Panel.fit(
                        result.text,
                        title="Assistant response",
                        border_style="cyan",
                        padding=(1, 1),
                    )
                )

        return await self.returner(
            result,
            self.prompt_to_query_id[self.user_prompt],
            last_in_tree=self.current_decision.last_in_tree,
        )

    async def process(
        self,
        user_prompt: str,
        collection_names: list[str] = [],
        client_manager: ClientManager | None = None,
        training_route: str = "",
        query_id: str = "1",
        close_clients_after_completion: bool = True,
        _first_run: bool = True,
        **kwargs,
    ):

        if client_manager is None:
            client_manager = ClientManager(logger=self.settings.logger)

        # Some initial steps if this is the first run (no recursion yet)
        if _first_run:
            # Reset the tree (clear temporary data specific to the last user prompt)
            self.soft_reset()

            check_base_lm_settings(self.settings)
            check_complex_lm_settings(self.settings)

            # Initialise some objects
            self.set_start_time()
            self.num_trees_completed = 0
            self.query_id_to_prompt[query_id] = user_prompt
            self.prompt_to_query_id[user_prompt] = query_id
            self.tree_data.set_property("user_prompt", user_prompt)
            self._update_conversation_history("user", user_prompt)
            self.user_prompt = user_prompt

            # check and start clients if not already started
            await client_manager.start_clients()

            # Initialise the collections
            # TODO: with client manager error handling, another bool check to see if client manager is active
            # if active, keep collection names empty
            if collection_names == []:
                async with client_manager.connect_to_async_client() as client:
                    collection_names = await retrieve_all_collection_names(client)

            collection_names = [c.lower() for c in collection_names]
            await self.set_collection_names(
                collection_names,
                client_manager,
            )

            # If there are any empty branches, remove them (no tools attached to them)
            self._remove_empty_branches()

            # If training route is provided, split it into a list and set route_active to True
            if training_route != "":
                training_route = training_route.split("/")

            if self.settings.LOGGING_LEVEL_INT <= 20:
                print(
                    Panel.fit(
                        user_prompt,
                        title="User prompt",
                        border_style="yellow",
                        padding=(1, 1),
                    )
                )

        base_lm = self.get_lm("base")

        # Start the tree at the root node
        current_decision_node: DecisionNode = self.decision_nodes[self.root]
        rules_met = []

        # Loop through the tree until the end is reached
        while True:
            # Evaluate any rules from tools
            # hardcoded logic that is evalutaed at the start of each branch
            # actions (call()) are performed if rule_call() returns True
            rules_met = self._check_rules(
                rules_met, current_decision_node.id, client_manager
            )

            # TODO: reimplement this, currently doesn't exist
            # if len(rules_met) > 0:
            #     async for result in self._evaluate_rules(
            #         rules_met, current_decision_node.id, client_manager
            #     ):
            #         yield result

            # else:
            # If training, decide from the training route
            if len(training_route) > 0:
                self.settings.logger.info(f"Route that will be used: {training_route}")

                (
                    self.current_decision,
                    training_route,
                ) = current_decision_node.decide_from_route(training_route)
                force_text_response = (
                    self.current_decision.function_name == "text_response"
                )

            # Under normal circumstances decide from the decision node
            else:
                t = Timer("base_lm")
                t.start()
                async for result in current_decision_node(
                    tree_data=self.tree_data,
                    lm=base_lm,
                ):
                    yield await self._evaluate_action(
                        result
                    )  # this assigns self.current_decision

                t.end()
                self.base_lm_timer.update_avg_time(t.time_taken)

                # Force text response (later) if model chooses end actions
                # but no response will be generated from the node
                force_text_response = (
                    not current_decision_node.options[
                        self.current_decision.function_name
                    ]["end"]
                    and self.current_decision.end_actions
                )

            # Set default values for the function inputs for current call
            self.current_decision.function_inputs = self._get_function_inputs(
                self.current_decision.function_name,
                self.current_decision.function_inputs,
            )

            # end criteria, task picked is "text_response" or model chooses to end conversation
            completed = (
                self.current_decision.function_name == "text_response"
                or self.current_decision.end_actions
            )

            # additional end criteria, recursion limit reached
            if self.tree_data.num_trees_completed > self.tree_data.recursion_limit:
                self.settings.logger.warning(
                    f"Recursion limit reached! ({self.tree_data.num_trees_completed})"
                )
                yield await self.returner(
                    Warning("Decision tree reached recursion limit!"),
                    query_id=self.prompt_to_query_id[user_prompt],
                )
                completed = True

            # If the task is impossible, break the loop
            if self.current_decision.impossible:
                break

            # assign action function
            action_fn = current_decision_node.options[
                self.current_decision.function_name
            ]["action"]

            # update the decision history
            self.decision_history.append(self.current_decision.function_name)

            # print the current node information
            if self.settings.LOGGING_LEVEL_INT <= 20:
                print(
                    Panel.fit(
                        f"[bold]Node:[/bold] [magenta]{current_decision_node.id}[/magenta]\n"
                        f"[bold]Decision:[/bold] [green]{self.current_decision.function_name}[/green]\n"
                        f"[bold]Reasoning:[/bold] {self.current_decision.reasoning}\n",
                        title="Current Decision",
                        border_style="magenta",
                        padding=(1, 1),
                    )
                )

            # end of current tree
            self.current_decision.last_in_tree = (
                current_decision_node.options[self.current_decision.function_name][
                    "next"
                ]
                is None
            )

            # evaluate the action
            if action_fn is not None:
                async for result in action_fn(
                    tree_data=self.tree_data,
                    inputs=self.current_decision.function_inputs,
                    base_lm=self.get_lm("base"),
                    complex_lm=self.get_lm("complex"),
                    client_manager=client_manager,
                ):
                    action_result = await self._evaluate_action(result)
                    if action_result is not None:
                        yield action_result

            # check if the current node is the end of the tree
            if (
                current_decision_node.options[self.current_decision.function_name][
                    "next"
                ]
                is None
                or completed
            ):
                break
            else:
                current_decision_node = self.decision_nodes[
                    current_decision_node.options[self.current_decision.function_name][
                        "next"
                    ]
                ]

        # end of all trees
        if completed:
            self.tree_data.num_trees_completed += 1

            # firstly, if we reached the end of a tree at a node that shouldn't be the end, call text response tool here to respond
            if (
                not current_decision_node.options[self.current_decision.function_name][
                    "end"
                ]
                or force_text_response
            ):
                async for result in self.tools["final_text_response"](
                    tree_data=self.tree_data,
                    inputs={},
                    base_lm=self.get_lm("base"),
                    complex_lm=self.get_lm("complex"),
                ):
                    action_result = await self._evaluate_action(result)
                    if action_result is not None:
                        yield action_result

            self.save_history(
                query_id=self.prompt_to_query_id[user_prompt],
                time_taken_seconds=time.time() - self.start_time,
            )

            yield await self.returner(
                Completed(), query_id=self.prompt_to_query_id[user_prompt]
            )

            self.settings.logger.info(
                f"[bold green]Model identified overall goal as completed![/bold green]"
            )
            self.settings.logger.debug(
                f"Time taken: {time.time() - self.start_time:.2f} seconds"
            )
            self.settings.logger.info(f"Tasks completed:\n")
            for task in self.decision_history:
                self.settings.logger.info(f"     - {task}")

            if close_clients_after_completion:
                await client_manager.close_clients()

        # otherwise, end of the tree for this iteration, and recursively call process() to restart the tree
        else:
            self.settings.logger.info(
                f"Model did [bold red]not[/bold red] yet complete overall goal! "
            )
            self.settings.logger.debug(
                f"Restarting tree (Recursion: {self.tree_data.num_trees_completed+1}/{self.tree_data.recursion_limit})..."
            )

            # recursive call to restart the tree since the goal was not completed
            self.tree_data.num_trees_completed += 1
            async for result in self.process(
                user_prompt,
                collection_names,
                client_manager,
                training_route=training_route,
                query_id=query_id,
                _first_run=False,
                **kwargs,
            ):
                yield result

    def process_sync(self, *args, **kwargs):
        """Synchronous version of process() for testing purposes"""

        nest_asyncio.apply()

        async def run_process():
            results = []
            async for result in self.process(*args, **kwargs):
                if (
                    result is not None
                    and "type" in result
                    and result["type"] == "result"
                ):
                    results.append(result["payload"]["objects"])
            return results

        async def run_with_live():
            console = Console()

            with console.status("[bold indigo]Thinking...") as status:
                results = []
                async for result in self.process(*args, **kwargs):
                    if (
                        result is not None
                        and "type" in result
                        and result["type"] == "result"
                    ):
                        results.append(result["payload"]["objects"])

                    if (
                        result is not None
                        and "type" in result
                        and result["type"] == "status"
                    ):
                        status.update(f"[bold indigo]{result['payload']['text']}")

        loop = asyncio.get_event_loop()
        if self.settings.LOGGING_LEVEL_INT <= 10:
            yielded_results = loop.run_until_complete(run_with_live())
        else:
            yielded_results = loop.run_until_complete(run_process())

        text = self.tree_data.conversation_history[-1]["content"]

        return text, yielded_results

    def detailed_memory_usage(self) -> dict:
        """
        Returns a detailed breakdown of memory usage for all major objects in the Tree class.

        Returns:
            dict: Dictionary containing memory sizes (in bytes) for each major component
        """
        memory_usage = {}

        # Core data structures
        memory_usage["tree_data"] = asizeof.asizeof(self.tree_data)
        memory_usage["tree_data"] = asizeof.asizeof(self.tree_data)
        memory_usage["tree_data"] = asizeof.asizeof(self.tree_data)
        memory_usage["collection_data"] = asizeof.asizeof(
            self.tree_data.collection_data
        )

        # Decision nodes and tools
        memory_usage["decision_nodes"] = {
            node_id: asizeof.asizeof(node)
            for node_id, node in self.decision_nodes.items()
        }
        memory_usage["tools"] = {
            tool_name: asizeof.asizeof(tool) for tool_name, tool in self.tools.items()
        }

        # History and tracking
        memory_usage["decision_history"] = asizeof.asizeof(self.decision_history)
        memory_usage["history"] = asizeof.asizeof(self.history)
        memory_usage["training_updates"] = asizeof.asizeof(self.training_updates)
        memory_usage["action_information"] = asizeof.asizeof(self.action_information)

        # Mapping dictionaries
        memory_usage["query_mappings"] = {
            "query_id_to_prompt": asizeof.asizeof(self.query_id_to_prompt),
            "prompt_to_query_id": asizeof.asizeof(self.prompt_to_query_id),
        }

        # Calculate total
        memory_usage["total"] = sum(
            (v if isinstance(v, int) else sum(v.values()) if isinstance(v, dict) else 0)
            for v in memory_usage.values()
        )

        return memory_usage

    def __call__(self, *args, **kwargs):
        return self.process_sync(*args, **kwargs)
