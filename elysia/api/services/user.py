import datetime
import os
import random

from dotenv import load_dotenv

load_dotenv()

from elysia.api.services.tree import TreeManager
from elysia.config import Settings
from elysia.objects import Error, Update
from elysia.tree.tree import Tree
from elysia.util.client import ClientManager
from elysia.util.objects import Timer
from elysia.api.core.log import logger


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

        self.date_of_reset = None
        self.users = {}
        # TODO: is this fine to be here? should it be user-specific?
        self.base_tree = Tree(debug=False)

    async def get_user_local(self, user_id: str):
        # add user if it doesn't exist
        if user_id not in self.users:
            self.users[user_id] = {}

            self.users[user_id]["tree_manager"] = TreeManager(
                user_id=user_id,
            )

            # client manager starts with env variables, when config is updated, api keys are updated
            self.users[user_id]["client_manager"] = ClientManager(logger=logger)

            # config starts as empty dict
            self.users[user_id]["config"] = Settings()
            self.users[user_id]["config"].set_config_from_env()
            self.users[user_id]["config"].setup_app_logger(logger)

        # update last request (adds last_request to user)
        await self.update_user_last_request(user_id)

        return self.users[user_id]

    def get_user_config(self, user_id: str):
        return self.users[user_id]["config"]

    async def get_tree(self, user_id: str, conversation_id: str):
        local_user = await self.get_user_local(user_id)
        return local_user["tree_manager"].get_tree(conversation_id)

    async def check_all_trees_timeout(self):
        for user_id in self.users:
            self.users[user_id]["tree_manager"].check_all_trees_timeout()

    async def check_restart_clients(self):
        # these check within these methods if the client has been used in the last X minutes
        for user_id in self.users:
            if "client_manager" in self.users[user_id]:
                await self.users[user_id]["client_manager"].restart_client()
                await self.users[user_id]["client_manager"].restart_async_client()

    async def close_all_clients(self):
        for user_id in self.users:
            if "client_manager" in self.users[user_id]:
                await self.users[user_id]["client_manager"].close_clients()

    async def initialise_tree(
        self, user_id: str, conversation_id: str, debug: bool = False
    ):
        local_user = await self.get_user_local(user_id)
        tree_manager: TreeManager = local_user["tree_manager"]
        tree_manager.add_tree(
            self.base_tree, conversation_id, debug, local_user["config"]
        )
        return tree_manager.get_tree(conversation_id)

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
            logger.error(
                f"\n\n[bold red]User not found [italic](this error should not occur!)[/italic][/bold red]\n\n"
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

        async for yielded_result in tree_manager.process_tree(
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
