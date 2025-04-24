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


class ShuffleAndDealCards(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="deal_cards",
            description="""
            This tool should always be run at the start of any interaction with the user.
            It defines any modifiers that get added to the conversation, from a random pool.
            It does so by 'dealing cards' to the user as if they were a player of a card game.
            These then will be displayed.
            Call this tool when specifically asked for, or at the start of any conversation.
            """,
            status="Dealing cards...",
            inputs={
                "num_cards": {
                    "description": "The number of cards to deal",
                    "type": int,
                    "default": 3,
                    "required": False,
                }
            },
            end=False,
        )
        self.logger = kwargs.get("logger", None)

    def select_random_cards(self, num_cards=3):
        possible_cards = [
            {
                "title": "The Jumbled",
                "effect": "Sometimes, the Elysia agent will say words in the wrong order.",
                "rarity": 3,
                "image": "https://i.imgur.com/KdGeZTp.png",
            },
            {
                "title": "The Comedian",
                "effect": "At the end of every sentence, the Elysia agent will tell a joke.",
                "rarity": 2,
                "image": "https://i.imgur.com/I8yVXHa.png",
            },
            {
                "title": "The Sarcastic",
                "effect": "Most interactions end with the Elysia agent making a sarcastic remark.",
                "rarity": 1,
                "image": "https://i.imgur.com/oFkwt1M.png",
            },
            {
                "title": "The Bro",
                "effect": "Elysia must now use the word 'bro' a lot more, and apply similar slang everywhere.",
                "rarity": 1,
                "image": "https://i.imgur.com/J6dLbTZ.png",
            },
            {
                "title": "The Philosopher",
                "effect": "The Elysia agent will now try to philosophise at every opportunity.",
                "rarity": 3,
                "image": "https://i.imgur.com/D6VSitF.png",
            },
        ]
        return random.choices(
            possible_cards, weights=[1 / card["rarity"] for card in possible_cards], k=3
        )

    async def __call__(self, tree_data, inputs, base_lm, complex_lm, client_manager):
        self.logger.info(f"Dealing {inputs['num_cards']} cards")
        cards = self.select_random_cards(inputs["num_cards"])

        yield Ecommerce(
            objects=cards,
            mapping={
                "description": "effect",
                "name": "title",
                "price": "rarity",
                "image": "image",
            },
            metadata={
                "num_cards": inputs["num_cards"],
            },
            llm_message="""
            Cards have been successfully dealt for this prompt! 
            Dealt {num_cards} out of a possible 5.
            Look at them in the environment to apply their modifiers to the conversation.
            Pay attention to what these cards do, and how they affect the conversation.
            You should apply the modifiers together, in a combination, not only one at a time.
            """,
        )


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

            self.trees[conversation_id]["tree"].add_tool(ShuffleAndDealCards)

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

        for conversation_id in convs_to_remove:
            del self.trees[conversation_id]
