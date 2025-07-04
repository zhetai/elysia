import pytest

from fastapi.responses import JSONResponse

import json
import random
from elysia.api.api_types import (
    ProcessCollectionData,
)
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.init import initialise_user
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


class TestProcessor:

    async def run_processor(self, data: ProcessCollectionData):
        websocket = fake_websocket()
        await process_collection(data.model_dump(), websocket, user_manager)
        return websocket.results

    @pytest.mark.asyncio
    async def test_process_collection(self):

        user_id = "test_user_Process_collection"
        try:
            await initialise_user(
                user_id,
                user_manager,
            )
            local_user = await user_manager.get_user_local(user_id)
            client_manager = local_user["client_manager"]

            collection_name = create_collection(client_manager)

            response = await self.run_processor(
                ProcessCollectionData(user_id=user_id, collection_name=collection_name),
            )
            for result in response:
                assert result["error"] == ""

            for result in response[:-1]:
                assert result["type"] == "update"
                assert result["progress"] < 1

            assert response[-1]["type"] == "completed"
            assert response[-1]["progress"] == 1

            with client_manager.connect_to_client() as client:
                assert client.collections.exists(collection_name)
                assert client.collections.exists(
                    f"ELYSIA_METADATA_{collection_name.lower()}__"
                )

        finally:

            # delete the collections
            try:
                with client_manager.connect_to_client() as client:

                    if client.collections.exists(collection_name):
                        client.collections.delete(collection_name)

                    if client.collections.exists(
                        f"ELYSIA_METADATA_{collection_name.lower()}"
                    ):
                        client.collections.delete(
                            f"ELYSIA_METADATA_{collection_name.lower()}__"
                        )
            except Exception as e:
                print(f"Can't delete collection: {str(e)}")
                pass

            await user_manager.close_all_clients()
