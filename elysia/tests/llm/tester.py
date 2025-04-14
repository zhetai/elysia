import os
import unittest
from copy import deepcopy

from deepeval.test_case import ToolCall

from elysia.config import Settings
from elysia.tree.tree import Tree
from elysia.util.client import ClientManager


class TestTree(unittest.TestCase):
    base_tree = Tree(
        debug=True,
        branch_initialisation="one_branch",
        settings=Settings.from_default(),
    )

    def get_tools_called(self, tree: Tree, user_prompt: str):
        tools_called = []
        if user_prompt in tree.actions_called:
            for action in tree.actions_called[user_prompt]:
                if action["name"] in tree.tools:
                    tools_called.append(
                        ToolCall(
                            name=action["name"],
                            description=tree.tools[action["name"]].description,
                            input_parameters=action["inputs"],
                            output=action["output"] if "output" in action else [],
                        )
                    )
        return tools_called

    async def start_client_manager(self):
        client_manager = ClientManager(
            wcd_url=os.getenv("WCD_URL"),
            wcd_api_key=os.getenv("WCD_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        await client_manager.start_clients()
        return client_manager

    async def run_tree(self, user_prompt: str, collection_names: list[str]):
        client_manager = await self.start_client_manager()

        tree = deepcopy(self.base_tree)

        async for result in tree.process(
            user_prompt,
            client_manager=client_manager,
            collection_names=collection_names,
        ):
            pass
        return tree
