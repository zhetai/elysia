import unittest

from fastapi.responses import JSONResponse

import asyncio
import json
import random
from elysia.api.api_types import (
    ProcessCollectionData,
)
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.processor import process_collection
from elysia.util.client import ClientManager
from elysia.api.core.log import logger, set_log_level

from weaviate.classes.config import Configure

set_log_level("CRITICAL")

user_manager = get_user_manager()


def read_response(response: JSONResponse):
    return json.loads(response.body)


class fake_websocket:
    results = []

    async def send_json(self, data: dict):
        self.results.append(data)


def create_collection(client_manager: ClientManager):
    """
    Create a new collection for testing.
    """

    r = random.randint(0, 1000)
    collection_name = f"test_elysia_collection_{r}"

    # create the collection
    vectorizer = Configure.Vectorizer.text2vec_openai(model="text-embedding-3-small")

    with client_manager.connect_to_client() as client:
        client.collections.create(
            collection_name,
            vectorizer_config=vectorizer,
        )

    lorem_ipsum = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    ]

    # upload some random data
    with client_manager.connect_to_client() as client:
        collection = client.collections.get(collection_name)
        for i in range(4):
            collection.data.insert(
                {"name": f"test_object_{i}", "description": lorem_ipsum[i]}
            )

    return collection_name


class TestProcessor(unittest.TestCase):

    def get_event_loop(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop

    async def run_processor(self, data: ProcessCollectionData):
        websocket = fake_websocket()
        await process_collection(data.model_dump(), websocket, user_manager)
        return websocket.results

    def test_process_collection(self):

        loop = self.get_event_loop()

        try:
            local_user = loop.run_until_complete(
                user_manager.get_user_local("test_user")
            )
            client_manager = local_user["client_manager"]
            local_user["config"].default_config()
            local_user["config"].configure(
                base_model="gpt-4o-mini", base_provider="openai"
            )

            collection_name = create_collection(client_manager)

            response = loop.run_until_complete(
                self.run_processor(
                    ProcessCollectionData(
                        user_id="test_user", collection_name=collection_name
                    ),
                )
            )
            for result in response:
                self.assertTrue(result["error"] == "")

            for result in response[:-1]:
                self.assertTrue(result["type"] == "update")
                self.assertTrue(result["progress"] < 1)

            self.assertTrue(response[-1]["type"] == "completed")
            self.assertTrue(response[-1]["progress"] == 1)

        finally:

            # delete the collection
            try:
                with client_manager.connect_to_client() as client:
                    if client.collections.exists(collection_name):
                        client.collections.delete(collection_name)
            except Exception as e:
                print(f"Can't delete collection: {e}")
                pass

            loop.run_until_complete(user_manager.close_all_clients())


if __name__ == "__main__":
    # unittest.main()
    TestProcessor().test_process_collection()
