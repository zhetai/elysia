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
from elysia.util.logging import backend_print
from elysia.util.objects import Timer


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
        self.timer.start()
        branch_initialisation = "one_branch"
        dspy_initialisation = None

        if conversation_id not in self.trees:
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
            self.trees[conversation_id]["tree"].set_dspy_initialisation(
                dspy_initialisation
            )
            self.trees[conversation_id]["event"] = asyncio.Event()
            self.trees[conversation_id]["event"].set()

            self.update_tree_last_request(conversation_id)

        self.timer.end(print_time=True, print_name="initialise_tree")

    def get_tree(self, base_tree: Tree, conversation_id: str):
        if conversation_id not in self.trees:
            self.add_tree(base_tree, conversation_id)
        return self.trees[conversation_id]["tree"]

    def get_event(self, conversation_id: str):
        return self.trees[conversation_id]["event"]

    async def process_tree(
        self,
        base_tree: Tree,
        conversation_id: str,
        query: str,
        query_id: str,
        training_route: str,
        training_mimick_model: str,
        collection_names: list[str],
        client_manager: ClientManager,
    ):
        self.update_tree_last_request(conversation_id)
        tree: Tree = self.get_tree(base_tree, conversation_id)

        self.trees[conversation_id]["event"].clear()
        async for yielded_result in tree.process(
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
            datetime.datetime.now() - self.trees[conversation_id]["last_request"]
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

        if len(convs_to_remove) > 0:
            backend_print(f"Removing {len(convs_to_remove)} trees that have timed out.")

        for conversation_id in convs_to_remove:
            del self.trees[conversation_id]
