import os
from fastapi.responses import JSONResponse

import json
import random
import pytest
from elysia.api.api_types import (
    DefaultConfigData,
    ChangeConfigData,
    ListConfigsData,
    LoadConfigData,
    SaveConfigData,
)
from elysia.util.async_util import asyncio_run
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.config import (
    save_config,
    load_config,
    change_config,
    default_config,
    list_configs,
)
from elysia.tree.tree import Tree
from elysia.api.core.log import logger, set_log_level

set_log_level("CRITICAL")


def read_response(response: JSONResponse):
    return json.loads(response.body)


class TestConfig:
    user_manager = get_user_manager()

    @pytest.mark.asyncio
    async def test_default_config(self):
        try:
            response = await default_config(
                data=DefaultConfigData(
                    user_id="test_user",
                ),
                user_manager=self.user_manager,
            )

            response = read_response(response)
            assert response["error"] == ""

            # just test a couple of the default settings
            local_user = await self.user_manager.get_user_local("test_user")
            assert local_user["config"].BASE_MODEL == os.getenv(
                "BASE_MODEL", "gemini-2.0-flash-001"
            )
            assert local_user["config"].COMPLEX_MODEL == os.getenv(
                "COMPLEX_MODEL", "gemini-2.0-flash-001"
            )

        finally:
            await self.user_manager.close_all_clients()

    @pytest.mark.asyncio
    async def test_save_config(self):
        try:
            response = await save_config(
                data=SaveConfigData(
                    user_id="test_user", config_id="test_config_save_config"
                ),
                user_manager=self.user_manager,
            )
            response = read_response(response)
            assert response["error"] == ""
        finally:
            await self.user_manager.close_all_clients()

    @pytest.mark.asyncio
    async def test_load_config(self):
        try:
            response = await save_config(
                data=SaveConfigData(
                    user_id="test_user", config_id="test_config_load_config"
                ),
                user_manager=self.user_manager,
            )

            response = read_response(response)
            assert response["error"] == ""

            response = await load_config(
                data=LoadConfigData(
                    user_id="test_user", config_id="test_config_load_config"
                ),
                user_manager=self.user_manager,
            )

            response = read_response(response)
            assert response["error"] == ""

        finally:
            await self.user_manager.close_all_clients()

    @pytest.mark.asyncio
    async def test_change_config(self):
        try:
            config_to_change = {
                "BASE_MODEL": "gpt-4o-mini",
                "COMPLEX_MODEL": "gpt-4o",
                "BASE_PROVIDER": "openai",
                "COMPLEX_PROVIDER": "openai",
                "MODEL_API_BASE": "test_api_base",
                "LOCAL": False,
                "WCD_URL": "test_url",
                "WCD_API_KEY": "test_key",
            }
            response = await change_config(
                data=ChangeConfigData(
                    user_id="test_user",
                    config=config_to_change,
                ),
                user_manager=self.user_manager,
            )
            response = read_response(response)
            assert response["error"] == ""

            local_user = await self.user_manager.get_user_local("test_user")
            assert local_user["config"].BASE_MODEL == "gpt-4o-mini"
            assert local_user["config"].COMPLEX_MODEL == "gpt-4o"
            assert local_user["config"].BASE_PROVIDER == "openai"
            assert local_user["config"].COMPLEX_PROVIDER == "openai"
            assert local_user["config"].MODEL_API_BASE == "test_api_base"
            assert local_user["config"].LOCAL == False
            assert local_user["config"].WCD_URL == "test_url"
            assert local_user["config"].WCD_API_KEY == "test_key"
        finally:
            await self.user_manager.close_all_clients()

    @pytest.mark.asyncio
    async def test_list_configs(self):
        try:
            response = await list_configs(
                data=ListConfigsData(user_id="test_user"),
                user_manager=self.user_manager,
            )
            response = read_response(response)
            assert response["error"] == ""
            assert isinstance(response["configs"], list)

        finally:
            await self.user_manager.close_all_clients()

    @pytest.mark.asyncio
    async def test_combination(self):

        try:
            # Create a config via creating a user (env settings)
            local_user = await self.user_manager.get_user_local("test_user")
            config = local_user["config"]

            # Create a tree
            tree: Tree = await self.user_manager.initialise_tree(
                user_id="test_user",
                conversation_id="test_conversation",
                debug=True,
            )

            # Change the config
            response = await change_config(
                data=ChangeConfigData(
                    user_id="test_user",
                    config={
                        "BASE_MODEL": "gpt-4o-mini",
                        "COMPLEX_MODEL": "gpt-4o",
                        "BASE_PROVIDER": "openai",
                        "COMPLEX_PROVIDER": "openai",
                    },
                ),
                user_manager=self.user_manager,
            )
            response = read_response(response)
            assert response["error"] == ""

            # Check the config has been changed
            assert local_user["config"].BASE_MODEL == "gpt-4o-mini"
            assert local_user["config"].COMPLEX_MODEL == "gpt-4o"
            assert local_user["config"].BASE_PROVIDER == "openai"
            assert local_user["config"].COMPLEX_PROVIDER == "openai"

            # Check the config has been changed in the tree
            assert tree.settings.BASE_MODEL == "gpt-4o-mini"
            assert tree.settings.COMPLEX_MODEL == "gpt-4o"
            assert tree.settings.BASE_PROVIDER == "openai"
            assert tree.settings.COMPLEX_PROVIDER == "openai"

            # Save the config
            r = random.randint(0, 1000000)
            config_id = f"test_config_{r}"
            response = await save_config(
                data=SaveConfigData(user_id="test_user", config_id=config_id),
                user_manager=self.user_manager,
            )
            response = read_response(response)
            assert response["error"] == ""

            # Load the config
            response = await load_config(
                data=LoadConfigData(user_id="test_user", config_id=config_id),
                user_manager=self.user_manager,
            )
            response = read_response(response)
            assert response["error"] == ""

            # Check the loaded config is the same as what we saved
            assert response["config"]["BASE_MODEL"] == "gpt-4o-mini"
            assert response["config"]["COMPLEX_MODEL"] == "gpt-4o"

            # Check the returned config from load_config is the same as the config we have now
            assert local_user["config"].BASE_MODEL == response["config"]["BASE_MODEL"]
            assert (
                local_user["config"].COMPLEX_MODEL
                == response["config"]["COMPLEX_MODEL"]
            )

            # Now the config_id should be in the list of configs
            response = await list_configs(
                data=ListConfigsData(user_id="test_user"),
                user_manager=self.user_manager,
            )
            response = read_response(response)
            assert response["error"] == ""
            assert config_id in response["configs"]

        finally:
            await self.user_manager.close_all_clients()
