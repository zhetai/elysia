import unittest
import asyncio

from fastapi.responses import JSONResponse

import json

from elysia.api.core.log import logger, set_log_level
from elysia.api.dependencies.common import get_user_manager

set_log_level("CRITICAL")

from elysia.api.routes.config import default_config
from elysia.api.routes.query import process
from elysia.api.api_types import QueryData, DefaultConfigData
from elysia.tests.dummy_adapter import dummy_adapter


def read_response(response: JSONResponse):
    return json.loads(response.body)


class fake_websocket:
    results = []

    async def send_json(self, data: dict):
        self.results.append(data)


with dummy_adapter():

    class TestQuery(unittest.TestCase):

        def get_event_loop(self):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop

        def test_query(self):

            loop = self.get_event_loop()

            try:

                user_id = "test_user"
                conversation_id = "test_conversation"
                query_id = "test_query"

                out = loop.run_until_complete(
                    default_config(
                        DefaultConfigData(user_id=user_id), get_user_manager()
                    )
                )

                websocket = fake_websocket()

                out = loop.run_until_complete(
                    process(
                        QueryData(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            query="test query!",
                            query_id=query_id,
                            collection_names=["test_collection_1", "test_collection_2"],
                        ).model_dump(),
                        websocket,
                        get_user_manager(),
                    )
                )

                # check all payloads are valid
                for result in websocket.results:
                    self.assertIsInstance(result, dict)
                    self.assertIn("type", result)
                    self.assertIn("id", result)
                    self.assertIn("conversation_id", result)
                    self.assertIn("query_id", result)
                    self.assertIn("payload", result)
                    self.assertIsInstance(result["payload"], dict)
                    if "objects" in result["payload"]:
                        for obj in result["payload"]["objects"]:
                            self.assertIsInstance(obj, dict)
                    if "metadata" in result["payload"]:
                        self.assertIsInstance(result["payload"]["metadata"], dict)
                    if "text" in result["payload"]:
                        self.assertIsInstance(result["payload"]["text"], str)
            finally:
                loop.run_until_complete(get_user_manager().close_all_clients())
