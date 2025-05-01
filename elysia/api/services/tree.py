import asyncio
import datetime
import os
import random

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from copy import deepcopy
from functools import lru_cache

from elysia.config import Settings
from elysia.tree.tree import Tree
from elysia.util.client import ClientManager
from elysia.util.objects import Timer


from elysia import Tool
from elysia.tools import Ecommerce

# from elysia.tools.fun.dnd import SetTheScene, ActionConsequence, DiceRoll


class TreeManager:
    """
    Manages trees (different conversations) for a single user.
    """

    def __init__(self, user_id: str):
        self.trees = {}
        self.user_id = user_id

        self.timer = Timer("TreeManager")
        self.tree_timeout = datetime.timedelta(
            minutes=int(os.environ.get("USER_TREE_TIMEOUT", 10))
        )

    def add_tree(
        self, base_tree: Tree, conversation_id: str, debug: bool, config: Settings
    ):

        if not self.tree_exists(conversation_id):

            self.timer.start()
            branch_initialisation = "one_branch"
            # branch_initialisation = "empty"

            self.trees[conversation_id] = {}
            self.trees[conversation_id]["tree"] = deepcopy(base_tree)
            self.trees[conversation_id]["tree"].settings = config
            self.trees[conversation_id]["tree"].set_debug(debug)
            self.trees[conversation_id]["tree"].set_conversation_id(conversation_id)
            self.trees[conversation_id]["tree"].set_user_id(self.user_id)
            self.trees[conversation_id]["last_request"] = datetime.datetime.now()
            self.trees[conversation_id]["tree"].set_branch_initialisation(
                branch_initialisation
            )
            self.trees[conversation_id]["tree"].get_tree_lms()
            self.trees[conversation_id]["event"] = asyncio.Event()
            self.trees[conversation_id]["event"].set()

            # self.trees[conversation_id]["tree"].change_style(
            #     "Poetic, flamboyant, engaging, you are a dungeon master! Crafting worlds."
            # )
            # self.trees[conversation_id]["tree"].change_agent_description(
            #     """
            #     You are a dungeon master, crafting a story for the player.
            #     Your job is to describe what happens in this fantasy world, and interact with the player.
            #     They may choose actions, or they may require you to prompt them for more information.
            #     Sometimes, their action may not be descriptive enough, so before choosing any actions, you should ask them to be more specific.
            #     Then combine these responses to call on the appropriate tools and use them to create stories.
            #     Do not create anything yourself, you just call the correct tools with the correct arguments.
            #     """,
            # )
            # self.trees[conversation_id]["tree"].change_end_goal(
            #     "The player/the user's answer has been asked to be more specific, "
            #     "or all the actions outlined in the `user_prompt` have been completed. "
            #     "Either the scene has been set, or the story has evolved. "
            # )

            # self.trees[conversation_id]["tree"].add_tool(SetTheScene)
            # self.trees[conversation_id]["tree"].add_tool(ActionConsequence)
            # self.trees[conversation_id]["tree"].add_tool(DiceRoll)
            self.trees[conversation_id]["tree"].change_agent_description(
                "You are a travel agent. You are given a user prompt and you need to help the user plan their trip. "
                "You should always start by finding the destination before doing anything else. "
                "Once you have the destination, you can find the best hotels and attractions for the user. "
                "When searching for hotels or attractions, you should always use the `destination_id` which matches the original destination you retrieved, "
                "instead of the location itself."
            )
            self.trees[conversation_id]["tree"].change_end_goal(
                "You have a complete set of hotels and attractions for the user's destination. "
                "You should finish the conversation with a full summary and itenary of the trip. "
            )

            self.update_tree_last_request(conversation_id)

        self.timer.end(print_time=True, print_name="initialise_tree")

    def tree_exists(self, conversation_id: str):
        return conversation_id in self.trees

    def get_tree(self, conversation_id: str):
        return self.trees[conversation_id]["tree"]

    def get_event(self, conversation_id: str):
        return self.trees[conversation_id]["event"]

    async def process_tree(
        self,
        conversation_id: str,
        query: str,
        query_id: str,
        training_route: str,
        training_mimick_model: str,
        collection_names: list[str],
        client_manager: ClientManager,
    ):
        self.update_tree_last_request(conversation_id)
        tree: Tree = self.get_tree(conversation_id)

        self.trees[conversation_id]["event"].clear()
        async for yielded_result in tree.async_run(
            query,
            collection_names=collection_names,
            client_manager=client_manager,
            query_id=query_id,
            training_route=training_route,
            training_mimick_model=training_mimick_model,
            close_clients_after_completion=False,
        ):
            yield yielded_result
            self.update_tree_last_request(conversation_id)
        self.trees[conversation_id]["event"].set()

    def check_tree_timeout(self, conversation_id: str):
        # if tree not found, return True
        if conversation_id not in self.trees:
            return True

        # Remove any trees that have not been active in the last USER_TREE_TIMEOUT minutes
        if (
            "last_request" in self.trees[conversation_id]
            and datetime.datetime.now() - self.trees[conversation_id]["last_request"]
            > self.tree_timeout
        ):
            return True

        return False

    def update_tree_last_request(self, conversation_id: str):
        self.trees[conversation_id]["last_request"] = datetime.datetime.now()

    def check_all_trees_timeout(self):
        convs_to_remove = []
        for i, conversation_id in enumerate(self.trees):
            if self.check_tree_timeout(conversation_id):
                convs_to_remove.append(conversation_id)

        for conversation_id in convs_to_remove:
            del self.trees[conversation_id]
