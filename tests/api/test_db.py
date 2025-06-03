import pytest

from elysia.api.api_types import (
    InitialiseUserData,
    InitialiseTreeData,
    QueryData,
    UpdateFrontendConfigData,
)
from elysia.api.routes.query import process
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.init import initialise_user, initialise_tree
from elysia.api.routes.user_config import update_frontend_config
from elysia.api.routes.db import (
    save_tree,
    load_tree,
    get_saved_trees,
    delete_tree,
)

from fastapi.responses import JSONResponse
import json


def read_response(response: JSONResponse):
    return json.loads(response.body)


from elysia.api.core.log import logger, set_log_level

set_log_level("CRITICAL")


def read_response(response: JSONResponse):
    return json.loads(response.body)


class fake_websocket:
    results = []

    async def send_json(self, data: dict):
        self.results.append(data)


class TestSaveLoad:

    @pytest.mark.asyncio
    async def test_manual_save_load_cycle(self):

        user_id = "test_user_full_save_load_cycle"
        conversation_id = "test_conversation_full_save_load_cycle"
        query_id = "test_query_full_save_load_cycle"

        websocket = fake_websocket()
        user_manager = get_user_manager()

        response = await initialise_user(
            InitialiseUserData(
                user_id=user_id,
                conversation_id=conversation_id,
                default_models=True,
            ),
            user_manager,
        )
        assert read_response(response)["error"] == ""

        response = await initialise_tree(
            InitialiseTreeData(
                user_id=user_id,
                conversation_id=conversation_id,
            ),
            user_manager,
        )
        assert read_response(response)["error"] == ""

        await update_frontend_config(
            user_id=user_id,
            data=UpdateFrontendConfigData(
                config={
                    "save_trees_to_weaviate": False,
                },
            ),
            user_manager=user_manager,
        )

        await process(
            QueryData(
                user_id=user_id,
                conversation_id=conversation_id,
                query="hi",
                query_id=query_id,
                collection_names=["Example_verba_slack_conversations"],
            ).model_dump(),
            websocket,
            user_manager,
        )

        # check that the tree is not saved to weaviate (automatic saving turned off)
        response = await get_saved_trees(
            user_id=user_id,
            user_manager=user_manager,
        )
        assert read_response(response)["error"] == ""
        assert conversation_id not in list(read_response(response)["trees"].keys())

        response = await save_tree(
            user_id=user_id,
            conversation_id=conversation_id,
            user_manager=user_manager,
        )
        assert read_response(response)["error"] == ""

        response = await load_tree(
            user_id=user_id,
            conversation_id=conversation_id,
            user_manager=user_manager,
        )
        assert read_response(response)["error"] == ""
        assert read_response(response)["rebuild"] != []
        for rebuilt in read_response(response)["rebuild"]:
            assert "type" in rebuilt
            assert "conversation_id" in rebuilt
            assert rebuilt["conversation_id"] == conversation_id
            assert "id" in rebuilt
            if "user_id" in rebuilt:
                assert rebuilt["user_id"] == user_id

        response = await get_saved_trees(
            user_id=user_id,
            user_manager=user_manager,
        )
        assert read_response(response)["error"] == ""
        assert len(list(read_response(response)["trees"].keys())) >= 1
        old_len = len(list(read_response(response)["trees"].keys()))
        assert "test_conversation_full_save_load_cycle" in list(
            read_response(response)["trees"].keys()
        )

        response = await delete_tree(
            user_id=user_id,
            conversation_id=conversation_id,
            user_manager=user_manager,
        )
        assert read_response(response)["error"] == ""

        response = await get_saved_trees(
            user_id=user_id,
            user_manager=user_manager,
        )
        assert read_response(response)["error"] == ""
        assert len(list(read_response(response)["trees"].keys())) < old_len
        assert "test_conversation_full_save_load_cycle" not in list(
            read_response(response)["trees"].keys()
        )

    @pytest.mark.asyncio
    async def test_automatic_save_load_cycle(self):

        user_id = "test_user_automatic_save_load_cycle"
        conversation_id = "test_conversation_automatic_save_load_cycle"
        query_id = "test_query_automatic_save_load_cycle"

        websocket = fake_websocket()
        user_manager = get_user_manager()

        response = await initialise_user(
            InitialiseUserData(
                user_id=user_id,
                conversation_id=conversation_id,
                default_models=True,
            ),
            user_manager,
        )
        assert read_response(response)["error"] == ""

        response = await initialise_tree(
            InitialiseTreeData(
                user_id=user_id,
                conversation_id=conversation_id,
            ),
            user_manager,
        )
        assert read_response(response)["error"] == ""

        await update_frontend_config(
            user_id=user_id,
            data=UpdateFrontendConfigData(
                config={
                    "save_trees_to_weaviate": True,
                },
            ),
            user_manager=user_manager,
        )

        await process(
            QueryData(
                user_id=user_id,
                conversation_id=conversation_id,
                query="hi",
                query_id=query_id,
                collection_names=["Example_verba_slack_conversations"],
            ).model_dump(),
            websocket,
            user_manager,
        )

        response = await load_tree(
            user_id=user_id,
            conversation_id=conversation_id,
            user_manager=user_manager,
        )
        assert read_response(response)["error"] == ""
        assert read_response(response)["rebuild"] != []
        for rebuilt in read_response(response)["rebuild"]:
            assert "type" in rebuilt
            assert "conversation_id" in rebuilt
            assert rebuilt["conversation_id"] == conversation_id
            assert "id" in rebuilt
            if "user_id" in rebuilt:
                assert rebuilt["user_id"] == user_id

        response = await get_saved_trees(
            user_id=user_id,
            user_manager=user_manager,
        )
        assert read_response(response)["error"] == ""
        assert len(list(read_response(response)["trees"].keys())) >= 1
        old_len = len(list(read_response(response)["trees"].keys()))
        assert "test_conversation_automatic_save_load_cycle" in list(
            read_response(response)["trees"].keys()
        )

        response = await delete_tree(
            user_id=user_id,
            conversation_id=conversation_id,
            user_manager=user_manager,
        )
        assert read_response(response)["error"] == ""

        response = await get_saved_trees(
            user_id=user_id,
            user_manager=user_manager,
        )
        assert read_response(response)["error"] == ""
        assert len(list(read_response(response)["trees"].keys())) < old_len
        assert "test_conversation_automatic_save_load_cycle" not in list(
            read_response(response)["trees"].keys()
        )
