import asyncio
import os
import pytest

from elysia.util.client import ClientManager


def get_event_loop(self):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop


def test_client_manager_starts(self):
    try:
        client_manager = ClientManager()
    finally:
        asyncio.run(client_manager.close_clients())


def test_client_manager_starts_with_env_vars(self):
    try:
        client_manager = ClientManager()
        self.assertTrue(client_manager.wcd_url == os.getenv("WCD_URL"))
        self.assertTrue(client_manager.wcd_api_key == os.getenv("WCD_API_KEY"))
    finally:
        asyncio.run(client_manager.close_clients())


def test_client_manager_starts_with_api_keys(self):
    try:
        client_manager = ClientManager()

        if "OPENAI_API_KEY" in os.environ:
            self.assertIn("X-OpenAI-Api-Key", client_manager.headers)
            self.assertTrue(
                client_manager.headers["X-OpenAI-Api-Key"]
                == os.getenv("OPENAI_API_KEY")
            )
    finally:
        asyncio.run(client_manager.close_clients())


def test_clients_are_connected(self):
    loop = self.get_event_loop()
    try:
        client_manager = ClientManager()
        loop.run_until_complete(client_manager.start_clients())
        self.assertTrue(client_manager.client.is_ready())
    finally:
        loop.run_until_complete(client_manager.close_clients())


def test_sync_client_connects(self):
    try:
        client_manager = ClientManager()
        with client_manager.connect_to_client() as client:
            self.assertTrue(client_manager.sync_in_use_counter == 1)
        self.assertTrue(client_manager.sync_in_use_counter == 0)
    finally:
        asyncio.run(client_manager.close_clients())


async def async_connect(self, client_manager: ClientManager):
    async with client_manager.connect_to_async_client() as client:
        self.assertTrue(client_manager.async_in_use_counter == 1)
        pass
    self.assertTrue(client_manager.async_in_use_counter == 0)


def test_async_client_connects(self):
    loop = self.get_event_loop()
    try:
        client_manager = ClientManager()
        loop.run_until_complete(self.async_connect(client_manager))
    finally:
        loop.run_until_complete(client_manager.close_clients())


if __name__ == "__main__":
    unittest.main()
