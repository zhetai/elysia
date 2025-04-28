import unittest
import asyncio
from dspy import LM, configure, ChatAdapter
from dspy.utils import DummyLM
from dspy.adapters import ChatAdapter
from pydantic_core import PydanticUndefined

from fastapi.responses import JSONResponse

import json

from elysia.api.core.log import logger, set_log_level
from elysia.api.dependencies.common import get_user_manager

set_log_level("CRITICAL")

from elysia.api.routes.tree import initialise_tree
from elysia.api.routes.config import default_config
from elysia.api.api_types import InitialiseTreeData, DefaultConfigData


def read_response(response: JSONResponse):
    return json.loads(response.body)


class TestTree(unittest.TestCase):

    def get_event_loop(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop

    def test_tree(self):

        loop = self.get_event_loop()

        try:
            user_manager = get_user_manager()
            user_id = "test_user"
            conversation_id = "test_conversation"

            out = loop.run_until_complete(
                default_config(
                    DefaultConfigData(user_id=user_id),
                    user_manager,
                )
            )
            response = read_response(out)
            self.assertIs(response["error"], "")

            out = loop.run_until_complete(
                initialise_tree(
                    InitialiseTreeData(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        debug=True,
                    ),
                    user_manager,
                )
            )
            response = read_response(out)
            self.assertIs(response["error"], "")

            out = loop.run_until_complete(
                initialise_tree(
                    InitialiseTreeData(
                        user_id=user_id + "2",
                        conversation_id=conversation_id + "2",
                        debug=False,
                    ),
                    user_manager,
                )
            )
            response = read_response(out)
            self.assertIs(response["error"], "")
        finally:
            loop.run_until_complete(get_user_manager().close_all_clients())


if __name__ == "__main__":
    unittest.main()
