import unittest
import os
from fastapi.responses import JSONResponse

import json
import random
import asyncio
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

from elysia.api.core.log import logger, set_log_level

set_log_level("CRITICAL")


def read_response(response: JSONResponse):
    return json.loads(response.body)


class TestConfig(unittest.TestCase):
    user_manager = get_user_manager()

    def get_event_loop(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop

    def test_default_config(self):
        loop = self.get_event_loop()
        try:
            response = loop.run_until_complete(
                default_config(
                    data=DefaultConfigData(
                        user_id="test_user",
                    ),
                    user_manager=self.user_manager,
                )
            )

            response = read_response(response)
            self.assertEqual(response["error"], "")

            # just test a couple of the default settings
            local_user = asyncio_run(self.user_manager.get_user_local("test_user"))
            self.assertEqual(
                local_user["config"].BASE_MODEL,
                os.getenv("BASE_MODEL", "gemini-2.0-flash-001"),
            )
            self.assertEqual(
                local_user["config"].COMPLEX_MODEL,
                os.getenv("COMPLEX_MODEL", "gemini-2.0-flash-001"),
            )

        finally:
            loop.run_until_complete(self.user_manager.close_all_clients())

    def test_save_config(self):
        loop = self.get_event_loop()
        try:
            response = loop.run_until_complete(
                save_config(
                    data=SaveConfigData(
                        user_id="test_user", config_id="test_config_save_config"
                    ),
                    user_manager=self.user_manager,
                )
            )
            response = read_response(response)
            self.assertEqual(response["error"], "")
        finally:
            loop.run_until_complete(self.user_manager.close_all_clients())

    def test_load_config(self):
        loop = self.get_event_loop()
        try:
            response = loop.run_until_complete(
                save_config(
                    data=SaveConfigData(
                        user_id="test_user", config_id="test_config_load_config"
                    ),
                    user_manager=self.user_manager,
                )
            )
            response = read_response(response)
            self.assertEqual(response["error"], "")

            response = loop.run_until_complete(
                load_config(
                    data=LoadConfigData(
                        user_id="test_user", config_id="test_config_load_config"
                    ),
                    user_manager=self.user_manager,
                )
            )
            response = read_response(response)
            self.assertEqual(response["error"], "")

        finally:
            loop.run_until_complete(self.user_manager.close_all_clients())

    def test_change_config(self):
        loop = self.get_event_loop()
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
            response = loop.run_until_complete(
                change_config(
                    data=ChangeConfigData(
                        user_id="test_user",
                        config=config_to_change,
                    ),
                    user_manager=self.user_manager,
                )
            )
            response = read_response(response)
            self.assertEqual(response["error"], "")

            local_user = asyncio_run(self.user_manager.get_user_local("test_user"))
            self.assertEqual(local_user["config"].BASE_MODEL, "gpt-4o-mini")
            self.assertEqual(local_user["config"].COMPLEX_MODEL, "gpt-4o")
            self.assertEqual(local_user["config"].BASE_PROVIDER, "openai")
            self.assertEqual(local_user["config"].COMPLEX_PROVIDER, "openai")
            self.assertEqual(local_user["config"].MODEL_API_BASE, "test_api_base")
            self.assertEqual(local_user["config"].LOCAL, False)
            self.assertEqual(local_user["config"].WCD_URL, "test_url")
            self.assertEqual(local_user["config"].WCD_API_KEY, "test_key")
        finally:
            loop.run_until_complete(self.user_manager.close_all_clients())

    def test_list_configs(self):
        loop = self.get_event_loop()
        try:
            response = loop.run_until_complete(
                list_configs(
                    data=ListConfigsData(user_id="test_user"),
                    user_manager=self.user_manager,
                )
            )
            response = read_response(response)
            self.assertEqual(response["error"], "")
            self.assertIsInstance(response["configs"], list)

        finally:
            loop.run_until_complete(self.user_manager.close_all_clients())

    def test_combination(self):
        loop = self.get_event_loop()

        try:
            # Create a config
            local_user = asyncio_run(self.user_manager.get_user_local("test_user"))
            config = local_user["config"]

            # Change the config
            response = loop.run_until_complete(
                change_config(
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
            )
            response = read_response(response)
            print("\nResponse error: ", response["error"])
            self.assertEqual(response["error"], "")

            # Save the config
            r = random.randint(0, 1000000)
            config_id = f"test_config_{r}"
            loop = self.get_event_loop()
            response = loop.run_until_complete(
                save_config(
                    data=SaveConfigData(user_id="test_user", config_id=config_id),
                    user_manager=self.user_manager,
                )
            )
            response = read_response(response)
            print("\nResponse error: ", response["error"])
            self.assertEqual(response["error"], "")

            # Load the config
            response = loop.run_until_complete(
                load_config(
                    data=LoadConfigData(user_id="test_user", config_id=config_id),
                    user_manager=self.user_manager,
                )
            )
            response = read_response(response)
            print("\nResponse error: ", response["error"])
            self.assertEqual(response["error"], "")

            # Check the loaded config is the same as what we saved
            self.assertEqual(response["config"]["BASE_MODEL"], "gpt-4o-mini")
            self.assertEqual(response["config"]["COMPLEX_MODEL"], "gpt-4o")

            # Check the returned config from load_config is the same as the config we have now
            self.assertEqual(
                local_user["config"].BASE_MODEL, response["config"]["BASE_MODEL"]
            )
            self.assertEqual(
                local_user["config"].COMPLEX_MODEL, response["config"]["COMPLEX_MODEL"]
            )

            # Now the config_id should be in the list of configs
            loop = self.get_event_loop()
            response = loop.run_until_complete(
                list_configs(
                    data=ListConfigsData(user_id="test_user"),
                    user_manager=self.user_manager,
                )
            )
            response = read_response(response)
            print("\nResponse error: ", response["error"])
            self.assertEqual(response["error"], "")
            self.assertIn(config_id, response["configs"])

        finally:
            loop.run_until_complete(self.user_manager.close_all_clients())


if __name__ == "__main__":
    unittest.main()
    # TestConfig().test_load_config()
