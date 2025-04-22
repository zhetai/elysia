import unittest
import asyncio
from dspy import LM, configure
from dspy.utils import DummyLM
from dspy.adapters import ChatAdapter
from pydantic_core import PydanticUndefined

from fastapi.responses import JSONResponse

import json

from elysia.api.core.log import logger, set_log_level
from elysia.api.dependencies.common import get_user_manager

set_log_level("CRITICAL")

from elysia.api.routes.config import default_config
from elysia.api.routes.query import process
from elysia.api.api_types import QueryData, DefaultConfigData


class DummyAdapter(ChatAdapter):

    def __call__(
        self,
        lm: LM,
        lm_kwargs,
        signature,
        demos,
        inputs,
    ):

        categories = []
        defaults = {}
        for field_name, field in signature.model_fields.items():
            if field.json_schema_extra.get("__dspy_field_type") == "output":
                categories.append(field_name)
                if field.default is not PydanticUndefined:
                    defaults[field_name] = field.default
                else:
                    try:
                        defaults[field_name] = list(field.annotation.__args__)[0]
                    except Exception as e:
                        try:
                            defaults[field_name] = field.annotation()
                        except Exception as e:
                            try:
                                defaults[field_name] = field.json_schema_extra[
                                    "format"
                                ]()
                            except Exception as e:
                                defaults[field_name] = "Test!"

        return super().__call__(
            DummyLM([{cat: defaults[cat] for cat in categories}]),
            lm_kwargs,
            signature,
            demos,
            inputs,
        )


configure(adapter=DummyAdapter())


def read_response(response: JSONResponse):
    return json.loads(response.body)


class fake_websocket:
    results = []

    async def send_json(self, data: dict):
        self.results.append(data)


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

        user_id = "test_user"
        conversation_id = "test_conversation"
        query_id = "test_query"

        out = loop.run_until_complete(
            default_config(DefaultConfigData(user_id=user_id), get_user_manager())
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

        loop.run_until_complete(get_user_manager().close_all_clients())


if __name__ == "__main__":
    unittest.main()
