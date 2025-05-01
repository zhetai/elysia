import pytest
from fastapi.responses import JSONResponse

import json

from elysia.api.core.log import set_log_level
from elysia.api.dependencies.common import get_user_manager

set_log_level("CRITICAL")

from elysia.api.routes.tree import initialise_tree
from elysia.api.routes.config import default_config
from elysia.api.api_types import InitialiseTreeData, DefaultConfigData


def read_response(response: JSONResponse):
    return json.loads(response.body)


class TestTree:

    @pytest.mark.asyncio
    async def test_tree(self):

        try:
            user_manager = get_user_manager()
            user_id = "test_user"
            conversation_id = "test_conversation"

            out = await default_config(
                DefaultConfigData(user_id=user_id),
                user_manager,
            )
            response = read_response(out)
            assert response["error"] == ""

            out = await initialise_tree(
                InitialiseTreeData(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    debug=True,
                ),
                user_manager,
            )

            response = read_response(out)
            assert response["error"] == ""

            out = await initialise_tree(
                InitialiseTreeData(
                    user_id=user_id + "2",
                    conversation_id=conversation_id + "2",
                    debug=False,
                ),
                user_manager,
            )

            response = read_response(out)
            assert response["error"] == ""
        finally:
            await get_user_manager().close_all_clients()
