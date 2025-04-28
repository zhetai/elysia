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
from elysia.api.routes.utils import (
    named_entity_recognition,
    title,
    follow_up_suggestions,
    debug,
)
from elysia.api.routes.config import default_config
from elysia.api.api_types import (
    NERData,
    TitleData,
    FollowUpSuggestionsData,
    DebugData,
    InitialiseTreeData,
    DefaultConfigData,
)


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


def read_response(response: JSONResponse):
    return json.loads(response.body)


class fake_websocket:
    results = []

    async def send_json(self, data: dict):
        self.results.append(data)


class TestUtils(unittest.TestCase):

    def get_event_loop(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop

    def test_ner(self):
        loop = self.get_event_loop()
        try:
            out = loop.run_until_complete(
                named_entity_recognition(NERData(text="Hello, world!"))
            )
            response = read_response(out)
            self.assertIs(response["error"], "")
        finally:
            loop.run_until_complete(get_user_manager().close_all_clients())

    def test_title(self):
        loop = self.get_event_loop()
        try:
            user_manager = get_user_manager()
            # local_user = user_manager.get_user_local(user_id="test_user")

            out = loop.run_until_complete(
                default_config(
                    DefaultConfigData(user_id="test_user"),
                    user_manager,
                ),
            )
            response = read_response(out)
            self.assertIs(response["error"], "")

            out = loop.run_until_complete(
                initialise_tree(
                    InitialiseTreeData(
                        user_id="test_user",
                        conversation_id="test_conversation",
                    ),
                    user_manager,
                )
            )
            response = read_response(out)
            self.assertIs(response["error"], "")

            out = loop.run_until_complete(
                title(
                    TitleData(
                        user_id="test_user",
                        conversation_id="test_conversation",
                        text="Hello, world!",
                    ),
                    user_manager,
                )
            )
            response = read_response(out)
            self.assertIs(response["error"], "")
        finally:
            loop.run_until_complete(get_user_manager().close_all_clients())

    def test_follow_up_suggestions(self):
        loop = self.get_event_loop()
        try:
            user_manager = get_user_manager()
            # local_user = awaituser_manager.get_user_local(user_id="test_user")

            out = loop.run_until_complete(
                default_config(
                    DefaultConfigData(user_id="test_user"),
                    user_manager,
                ),
            )
            response = read_response(out)
            self.assertIs(response["error"], "")

            out = loop.run_until_complete(
                initialise_tree(
                    InitialiseTreeData(
                        user_id="test_user",
                        conversation_id="test_conversation",
                    ),
                    user_manager,
                )
            )
            response = read_response(out)
            self.assertIs(response["error"], "")

            out = loop.run_until_complete(
                follow_up_suggestions(
                    FollowUpSuggestionsData(
                        user_id="test_user",
                        conversation_id="test_conversation",
                    ),
                    user_manager,
                )
            )
            response = read_response(out)
            self.assertIs(response["error"], "")
        finally:
            loop.run_until_complete(get_user_manager().close_all_clients())

    def test_debug(self):
        pass


if __name__ == "__main__":
    configure(adapter=DummyAdapter())
    unittest.main()
    configure(adapter=ChatAdapter())
