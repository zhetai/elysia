import datetime
import os
import random

from dotenv import load_dotenv
from typing import Any

load_dotenv(override=True)

from elysia.api.services.tree import TreeManager
from elysia.config import Settings
from elysia.objects import Update
from elysia.util.client import ClientManager
from elysia.api.core.log import logger
from elysia.api.api_types import Config
from elysia.api.utils.frontend_config import FrontendConfig


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
    """
    The UserManager is designed to manage, for each user:

    - TreeManager
    - ClientManager

    It contains methods for creating and updating these objects across a range of users, stored in a dictionary.

    It can be used as a dependency injection container for FastAPI.
    """

    def __init__(
        self,
        user_timeout: datetime.timedelta | int | None = None,
        tree_timeout: datetime.timedelta | int | None = None,
        client_timeout: datetime.timedelta | int | None = None,
    ):
        """
        Args:
            user_timeout (datetime.timedelta | int | None): Optional.
                The length of time a user can be idle before being timed out.
                Defaults to 20 minutes or the value of the USER_TIMEOUT environment variable.
                If an integer is provided, it is interpreted as the number of minutes.
            tree_timeout (datetime.timedelta | int | None): Optional.
                The length of time a tree can be idle before being timed out.
                Defaults to 10 minutes or the value of the USER_TREE_TIMEOUT environment variable.
                If an integer is provided, it is interpreted as the number of minutes.
            client_timeout (datetime.timedelta | int | None): Optional.
                The length of time a client can be idle before being timed out.
                Defaults to 10 minutes or the value of the CLIENT_TIMEOUT environment variable.
                If an integer is provided, it is interpreted as the number of minutes.
        """
        if user_timeout is None:
            self.user_timeout = datetime.timedelta(
                minutes=int(os.environ.get("USER_TIMEOUT", 20))
            )
        elif isinstance(user_timeout, int):
            self.user_timeout = datetime.timedelta(minutes=user_timeout)
        else:
            self.user_timeout = user_timeout

        if tree_timeout is None:
            self.tree_timeout = datetime.timedelta(
                minutes=int(os.environ.get("USER_TREE_TIMEOUT", 10))
            )
        elif isinstance(tree_timeout, int):
            self.tree_timeout = datetime.timedelta(minutes=tree_timeout)
        else:
            self.tree_timeout = tree_timeout

        if client_timeout is None:
            self.client_timeout = datetime.timedelta(
                minutes=int(os.environ.get("CLIENT_TIMEOUT", 10))
            )
        elif isinstance(client_timeout, int):
            self.client_timeout = datetime.timedelta(minutes=client_timeout)
        else:
            self.client_timeout = client_timeout

        self.manager_id = random.randint(0, 1000000)
        self.date_of_reset = None
        self.users = {}

    def user_exists(self, user_id: str):
        return user_id in self.users

    def update_config(self, user_id: str, config: Config):
        pass

    async def update_frontend_config(
        self,
        user_id: str,
        config: dict[str, Any],
    ):
        local_user = await self.get_user_local(user_id)
        frontend_config: FrontendConfig = local_user["frontend_config"]
        frontend_config.configure(**config)

    def add_user_local(
        self,
        user_id: str,
        style: str = "",
        agent_description: str = "",
        end_goal: str = "",
        branch_initialisation: str = "one_branch",
        settings: Settings | None = None,
    ):
        """
        Add a user to the UserManager.

        Args:
            user_id (str): Required. The unique identifier for the user.
            style (str | None): Optional. A writing style for all trees created for this user.
            agent_description (str | None): Optional. A description of the agent for all trees created for this user.
            end_goal (str | None): Optional. A goal for all trees created for this user.
            branch_initialisation (str | None): Optional. A branch initialisation for all trees created for this user.
                Defaults to "one_branch", which contains 'Query', 'Aggregate' and 'Summarize' tools.
                Other options are "multi_branch", which contains 'Query' and 'Aggregate' in a separate branch,
                and "empty", which contains no tools (only a base branch), designed to add more tools to.
            settings (Settings | None): Optional. The settings for the user.
                This setting config will be set for all trees associated with the user (in the TreeManager).
                Defaults to None, which will use the environment variables.
        """

        # add user if it doesn't exist
        if user_id not in self.users:
            self.users[user_id] = {}
            self.users[user_id]["tree_manager"] = TreeManager(
                user_id=user_id,
                style=style,
                agent_description=agent_description,
                end_goal=end_goal,
                branch_initialisation=branch_initialisation,
                settings=settings,
                tree_timeout=self.tree_timeout,
            )

            # client manager starts with env variables, when config is updated, api keys are updated
            self.users[user_id]["client_manager"] = ClientManager(
                logger=logger, client_timeout=self.client_timeout
            )

            self.users[user_id]["frontend_config"] = FrontendConfig(
                logger=logger, client_timeout=self.client_timeout
            )

    async def get_user_local(self, user_id: str):
        """
        Return a local user object.
        Will raise a ValueError if the user is not found.

        Args:
            user_id (str): Required. The unique identifier for the user.

        Returns:
            dict: A local user object, containing a TreeManager ("tree_manager"),
                User Config ("config") and ClientManager ("client_manager").
        """

        if user_id not in self.users:
            raise ValueError(
                f"User {user_id} not found. Please initialise a user first (by calling `initialise_tree` or `add_user_local`)."
            )

        # update last request (adds last_request to user)
        await self.update_user_last_request(user_id)

        return self.users[user_id]

    async def get_tree(self, user_id: str, conversation_id: str):
        """
        Get a tree for a user.

        Args:
            user_id (str): Required. The unique identifier for the user.
            conversation_id (str): Required. The unique identifier for the conversation.

        Returns:
            Tree: The tree.
        """
        local_user = await self.get_user_local(user_id)
        return local_user["tree_manager"].get_tree(conversation_id)

    async def check_all_trees_timeout(self):
        """
        Check all trees in all TreeManagers across all users and remove any that have not been active in the last tree_timeout.
        """
        for user_id in self.users:
            self.users[user_id]["tree_manager"].check_all_trees_timeout()

    def check_user_timeout(self, user_id: str):
        """
        Check if a user has been idle for the last user_timeout.

        Args:
            user_id (str): The user ID which contains the user.

        Returns:
            (bool): True if the user has been idle for the last user_timeout, False otherwise.
        """
        # if user not found, return True
        if user_id not in self.users:
            return True

        # Remove any trees that have not been active in the last USER_TREE_TIMEOUT minutes
        if (
            "last_request" in self.users[user_id]
            and datetime.datetime.now() - self.users[user_id]["last_request"]
            > self.user_timeout
        ):
            return True

        return False

    async def check_all_users_timeout(self):
        """
        Check all users in the UserManager and remove any that have not been active in the last user_timeout.
        """
        if self.user_timeout == datetime.timedelta(minutes=0):
            return

        for user_id in self.users:
            if self.check_user_timeout(user_id):
                del self.users[user_id]

    async def check_restart_clients(self):
        """
        Check all clients in all ClientManagers across all users and run the restart_client() method (for sync and async clients).
        The restart_client() methods will check if the client has been inactive for the last client_timeout minutes (set in init).
        """
        for user_id in self.users:
            if "client_manager" in self.users[user_id]:
                await self.users[user_id]["client_manager"].restart_client()
                await self.users[user_id]["client_manager"].restart_async_client()

    async def close_all_clients(self):
        """
        Close all clients in all ClientManagers across all users.
        """
        for user_id in self.users:
            if "client_manager" in self.users[user_id]:
                await self.users[user_id]["client_manager"].close_clients()

    async def initialise_tree(
        self,
        user_id: str,
        conversation_id: str,
        settings: Settings | None = None,
        style: str | None = None,
        agent_description: str | None = None,
        end_goal: str | None = None,
        branch_initialisation: str | None = None,
        low_memory: bool = False,
    ):
        """
        Initialises a tree for a user for an existing user at user_id.
        Requires a user to already exist in the UserManager, via `add_user_local`.
        This is a wrapper for the TreeManager.add_tree() method.

        Args:
            user_id (str): Required. The unique identifier for the user.
            conversation_id (str): Required. The unique identifier for a new conversation within the tree manager.
            settings (Settings | None): Optional. The settings for the tree.
                Defaults to the settings used to initialise the TreeManager for that specific user.
                Which itself defaults to the environment variables if not specified.
            style (str | None): Optional. The style for the tree.
                Defaults to the style used to initialise the TreeManager for that specific user.
            agent_description (str | None): Optional. The description of the agent for the tree.
                Defaults to the agent description used to initialise the TreeManager for that specific user.
            end_goal (str | None): Optional. The end goal for the tree.
                Defaults to the end goal used to initialise the TreeManager for that specific user.
            branch_initialisation (str | None): Optional. The initialisation for the branches of the tree.
                Defaults to the branch initialisation used to initialise the TreeManager for that specific user.
            low_memory (bool): Optional. Whether to use low memory mode for the tree.
                Controls the LM history being saved in the tree, and some other variables.
                Defaults to False.
        """
        # self.add_user_local(user_id)
        local_user = await self.get_user_local(user_id)
        tree_manager: TreeManager = local_user["tree_manager"]
        tree_manager.add_tree(
            conversation_id,
            settings,
            style,
            agent_description,
            end_goal,
            branch_initialisation,
            low_memory,
        )
        return tree_manager.get_tree(conversation_id)

    async def save_tree(self, user_id: str, conversation_id: str):
        """
        Save a tree to a Weaviate instance (set in the frontend config).
        This is a wrapper for the TreeManager.save_tree_weaviate() method.

        Args:
            user_id (str): Required. The unique identifier for the user stored in the UserManager.
            conversation_id (str): Required. The unique identifier for the conversation for the user.
        """
        local_user = await self.get_user_local(user_id)
        tree_manager: TreeManager = local_user["tree_manager"]
        await tree_manager.save_tree_weaviate(
            conversation_id, local_user["frontend_config"].save_location_client_manager
        )

    async def load_tree(self, user_id: str, conversation_id: str):
        """
        Load a tree from a Weaviate instance (set in the frontend config).
        This is a wrapper for the TreeManager.load_tree_weaviate() method.

        Args:
            user_id (str): Required. The unique identifier for the user stored in the UserManager.
            conversation_id (str): Required. The unique identifier for the conversation for the user.

        Returns:
            (list): A list of dictionaries, each containing a frontend payload that was used to generate the tree.
            The list is ordered by the time the payload was originally sent to the frontend (at the time it was saved).
        """
        local_user = await self.get_user_local(user_id)
        tree_manager: TreeManager = local_user["tree_manager"]
        return await tree_manager.load_tree_weaviate(
            conversation_id, local_user["frontend_config"].save_location_client_manager
        )

    async def delete_tree(self, user_id: str, conversation_id: str):
        """
        Delete a saved tree from a Weaviate instance (set in the frontend config).
        Also delete the tree from the local tree manager.
        This is a wrapper for the TreeManager.delete_tree_weaviate() method and the TreeManager.delete_tree_local() method.

        Args:
            user_id (str): Required. The unique identifier for the user stored in the UserManager.
            conversation_id (str): Required. The unique identifier for the conversation for the user.
        """
        local_user = await self.get_user_local(user_id)
        tree_manager: TreeManager = local_user["tree_manager"]
        await tree_manager.delete_tree_weaviate(
            conversation_id, local_user["frontend_config"].save_location_client_manager
        )
        tree_manager.delete_tree_local(conversation_id)

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
        query: str,
        user_id: str,
        conversation_id: str,
        query_id: str,
        training_route: str = "",
        collection_names: list[str] = [],
    ):
        """
        Wrapper for the TreeManager.process_tree() method.
        Which itself is a wrapper for the Tree.async_run() method.
        This is an async generator which yields results from the tree.async_run() method.
        Automatically sends error payloads if the user or tree has been timed out.

        Args:
            query (str): Required. The user input/prompt to process in the decision tree.
            user_id (str): Required. The unique identifier for the user.
            conversation_id (str): Required. The conversation ID which contains the tree.
                This should be the same conversation ID as the one used to initialise the tree (see `initialise_tree`).
            query_id (str): Required. A unique identifier for the query.
            training_route (str): Optional. The training route, a string of the form "tool1/tool2/tool1" etc.
                See the `tree.async_run()` method for more details.
            collection_names (list[str]): Optional. A list of collection names to use in the query.
                If not supplied, all collections will be used.
        """

        if self.check_user_timeout(user_id):
            user_timeout_error = UserTimeoutError()
            error_payload = await user_timeout_error.to_frontend(
                user_id, conversation_id, query_id
            )
            yield error_payload
            return

        if self.check_tree_timeout(user_id, conversation_id):
            tree_timeout_error = TreeTimeoutError()
            error_payload = await tree_timeout_error.to_frontend(
                user_id, conversation_id, query_id
            )
            yield error_payload
            return

        local_user = await self.get_user_local(user_id)
        await self.update_user_last_request(user_id)

        tree_manager: TreeManager = local_user["tree_manager"]

        async for yielded_result in tree_manager.process_tree(
            query,
            conversation_id,
            query_id,
            training_route,
            collection_names,
            local_user["client_manager"],
        ):
            yield yielded_result
            await self.update_user_last_request(user_id)

        frontend_config: FrontendConfig = local_user["frontend_config"]
        if frontend_config.config["save_trees_to_weaviate"]:
            await self.save_tree(user_id, conversation_id)
