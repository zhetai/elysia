import pytest
from fastapi.responses import JSONResponse
import os
import json

from elysia.api.core.log import set_log_level
from elysia.api.dependencies.common import get_user_manager

set_log_level("CRITICAL")

from elysia.api.routes.init import initialise_tree, initialise_user
from elysia.api.api_types import InitialiseTreeData, InitialiseUserData


def read_response(response: JSONResponse):
    return json.loads(response.body)


class TestTree:

    @pytest.mark.asyncio
    async def test_initialise_user(self):
        user_manager = get_user_manager()
        user_id = "test_new_user"
        conversation_id = "test_new_conversation"

        # Initialise when a user does not exist
        out = await initialise_user(
            InitialiseUserData(
                user_id=user_id,
            ),
            user_manager,
        )

        response = read_response(out)
        assert response["error"] == ""
        assert response["user_exists"] is False

        # check config options are environment variables
        if "BASE_MODEL" in os.environ:
            assert response["config"]["settings"]["BASE_MODEL"] == os.getenv(
                "BASE_MODEL"
            )
        if "OPENAI_API_KEY" in os.environ:
            assert response["config"]["settings"]["API_KEYS"][
                "openai_api_key"
            ] == os.getenv("OPENAI_API_KEY")

        # Initialise when a user exists
        out = await initialise_user(
            InitialiseUserData(
                user_id=user_id,
            ),
            user_manager,
        )

        response = read_response(out)
        assert response["error"] == ""
        assert response["user_exists"] is True

        # Initialise a user with config options
        user_id_2 = "test_user_2"
        out = await initialise_user(
            InitialiseUserData(
                user_id=user_id_2,
                style="test_style",
                agent_description="test_agent_description",
                end_goal="test_end_goal",
                branch_initialisation="empty",
            ),
            user_manager,
        )

        response = read_response(out)
        assert response["error"] == ""
        assert response["config"]["style"] == "test_style"
        assert response["config"]["agent_description"] == "test_agent_description"
        assert response["config"]["end_goal"] == "test_end_goal"
        assert response["config"]["branch_initialisation"] == "empty"

    @pytest.mark.asyncio
    async def test_tree(self):

        try:
            user_manager = get_user_manager()
            user_id = "test_user"
            conversation_id = "test_conversation"

            out = await initialise_user(
                InitialiseUserData(
                    user_id=user_id,
                    default_models=True,
                ),
                user_manager,
            )
            response = read_response(out)
            assert response["error"] == ""

            out = await initialise_tree(
                InitialiseTreeData(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    low_memory=True,
                ),
                user_manager,
            )

            response = read_response(out)
            assert response["error"] == ""

            out = await initialise_tree(
                InitialiseTreeData(
                    user_id=user_id + "2",
                    conversation_id=conversation_id + "2",
                    low_memory=True,
                ),
                user_manager,
            )

            response = read_response(out)
            assert response["error"] != ""  # should error as user doesn't exist
        finally:
            await get_user_manager().close_all_clients()
