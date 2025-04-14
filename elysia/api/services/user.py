import asyncio
import datetime
import os
import random
from asyncio import Lock

import weaviate
from dotenv import load_dotenv
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, Metrics
from weaviate.util import generate_uuid5

# Load environment variables from .env file
load_dotenv()

from elysia.api.services.tree import TreeManager
from elysia.config import Settings
from elysia.objects import Error, Update
from elysia.tree.tree import Tree
from elysia.util.client import ClientManager
from elysia.util.logging import backend_print
from elysia.util.objects import Timer
from elysia.util.parsing import format_datetime

ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "0") == "1"
FORCE_RATE_LIMITING = os.getenv("FORCE_RATE_LIMITING", "0") == "1"


class TreeTimeoutError(Update):
    def __init__(self):
        super().__init__(
            "tree_timeout_error",
            {
                "text": "This conversation has been timed out due to inactivity. Please start a new conversation."
            },
        )


class UserTimeoutError(Update):
    def __init__(self):
        super().__init__(
            "user_timeout_error",
            {
                "text": "You have been timed out due to inactivity. Please start a new conversation."
            },
        )


class RateLimitError(Update):
    def __init__(self, reset_time: datetime.datetime):
        self.reset_time = reset_time
        self.current_time = datetime.datetime.now(tz=datetime.timezone.utc)

        self.hours_left = int(
            round((self.reset_time - self.current_time).total_seconds() // 3600, 0)
        )
        self.minutes_left = int(
            round((self.reset_time - self.current_time).total_seconds() // 60 % 60, 0)
        )
        self.seconds_left = int(
            round((self.reset_time - self.current_time).total_seconds() % 60, 0)
        )

        super().__init__(
            "rate_limit_error",
            {
                "text": f"You have exceeded the maximum number of requests per day. Please try again tomorrow!",
                "reset_time": format_datetime(self.reset_time),
                "time_left": {
                    "hours": self.hours_left,
                    "minutes": self.minutes_left,
                    "seconds": self.seconds_left,
                },
            },
        )


class UserManager:
    def __init__(self):
        self.max_requests_per_day = int(os.environ.get("MAX_USER_REQUESTS_PER_DAY", 5))

        self.user_timeout = datetime.timedelta(
            minutes=int(os.environ.get("USER_TIMEOUT", 20))
        )
        self.tree_timeout = datetime.timedelta(
            minutes=int(os.environ.get("USER_TREE_TIMEOUT", 10))
        )

        self.timer = Timer("UserManager")
        self.manager_id = random.randint(0, 1000000)

        # this is used to wait for the tree to be completed
        self.event = asyncio.Event()
        self.event.set()

        self.date_of_reset = None
        self.users = {}
        self.base_tree = Tree(
            verbosity=2
        )  # TODO: is this fine to be here? should it be user-specific?

    async def get_user_local(self, user_id: str):
        # add user if it doesn't exist
        if user_id not in self.users:
            self.users[user_id] = {}

            self.users[user_id]["tree_manager"] = TreeManager(
                user_id=user_id,
            )

            # client manager starts with env variables, when config is updated, api keys are updated
            self.users[user_id]["client_manager"] = ClientManager()

            # config starts as empty dict
            self.users[user_id]["config"] = Settings()

        # update last request (adds last_request to user)
        await self.update_user_last_request(user_id)

        return self.users[user_id]

    async def get_tree(self, user_id: str, conversation_id: str):
        local_user = await self.get_user_local(user_id)
        return local_user["tree_manager"].get_tree(self.base_tree, conversation_id)

    async def check_all_trees_timeout(self):
        for user_id in self.users:
            self.users[user_id]["tree_manager"].check_all_trees_timeout()

    async def check_restart_clients(self):
        # these check within these methods if the client has been used in the last X minutes
        for user_id in self.users:
            self.users[user_id]["client_manager"].restart_client()
            await self.users[user_id]["client_manager"].restart_async_client()

    async def close_all_clients(self):
        for user_id in self.users:
            await self.users[user_id]["client_manager"].close_clients()

    async def initialise_tree(
        self, user_id: str, conversation_id: str, debug: bool = False
    ):
        local_user = await self.get_user_local(user_id)
        tree_manager: TreeManager = local_user["tree_manager"]
        tree_manager.add_tree(
            self.base_tree, conversation_id, debug, local_user["config"]
        )
        return tree_manager.get_tree(self.base_tree, conversation_id)

    async def update_user_last_request(self, user_id: str):
        self.users[user_id]["last_request"] = datetime.datetime.now()

    def check_tree_timeout(self, user_id: str, conversation_id: str):
        if user_id not in self.users:
            return True
        elif conversation_id not in self.users[user_id]["tree_manager"].trees:
            return True
        return False

    async def process_tree(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
        query_id: str,
        training_route: str,
        training_mimick_model: str,
        collection_names: list[str],
    ):
        local_user = await self.get_user_local(user_id)

        if user_id not in self.users:
            yield Error(
                "User not found. The app may have been restarted whilst you were using it. Please refresh the page."
            ).to_frontend(user_id, conversation_id, query_id)
            backend_print(
                f"\n\n[bold red]User not found (this error should not occur!)[/bold red]\n\n"
            )
            return
        elif self.check_tree_timeout(user_id, conversation_id):
            tree_timeout_error = TreeTimeoutError()
            error_payload = await tree_timeout_error.to_frontend(
                conversation_id, query_id
            )
            yield error_payload
            return
        await self.update_user_last_request(user_id)

        tree_manager: TreeManager = local_user["tree_manager"]

        self.event.clear()
        async for yielded_result in tree_manager.process_tree(
            self.base_tree,
            conversation_id,
            query,
            query_id,
            training_route,
            training_mimick_model,
            collection_names,
            local_user["client_manager"],
        ):
            yield yielded_result
            await self.update_user_last_request(user_id)

        self.event.set()
