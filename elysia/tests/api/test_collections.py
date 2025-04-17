import os
import random
import sys
import unittest

from fastapi.responses import JSONResponse

sys.path.append(os.getcwd())

import elysia
from elysia.tree.tree import Tree

from elysia.preprocess.collection import CollectionPreprocessor

from elysia.util.client import ClientManager

# api imports
import asyncio
import os
import sys

sys.path.append(os.getcwd())
# os.chdir("../..")

import json

from rich import print
from weaviate.util import generate_uuid5

from elysia.api.api_types import (
    GetCollectionsData,
    GetObjectData,
    ViewPaginatedCollectionData,
    UserCollectionsData,
    CollectionMetadataData,
)
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.collections import (
    collections,
    view_paginated_collection,
    collection_metadata,
    get_object,
)

from elysia.api.core.logging import logger

logger.setLevel("CRITICAL")


def read_response(response: JSONResponse):
    return json.loads(response.body)


class TestEndpoints(unittest.TestCase):

    def get_event_loop(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop

    def test_collections(self):

        loop = self.get_event_loop()
        try:
            user_manager = get_user_manager()
            basic = loop.run_until_complete(
                collections(UserCollectionsData(user_id="test_user"), user_manager)
            )
            basic = read_response(basic)
            self.assertTrue(basic["error"] == "")
        finally:
            loop.run_until_complete(user_manager.close_all_clients())

    def test_collection_metadata(self):
        loop = self.get_event_loop()
        try:
            user_manager = get_user_manager()
            basic = loop.run_until_complete(
                collection_metadata(
                    CollectionMetadataData(
                        user_id="test_user",
                        collection_name="example_verba_github_issues",
                    ),
                    user_manager,
                )
            )
            basic = read_response(basic)
            self.assertTrue(basic["error"] == "")
        finally:
            loop.run_until_complete(user_manager.close_all_clients())

    def test_get_object(self):
        loop = self.get_event_loop()
        try:
            user_manager = get_user_manager()

            # first manually get a UUID
            user_local = loop.run_until_complete(
                user_manager.get_user_local("test_user")
            )
            client_manager = user_local["client_manager"]
            with client_manager.connect_to_client() as client:
                collection = client.collections.get("example_verba_github_issues")
                uuid = str(collection.query.fetch_objects(limit=1).objects[0].uuid)

            basic = loop.run_until_complete(
                get_object(
                    GetObjectData(
                        user_id="test_user",
                        collection_name="example_verba_github_issues",
                        uuid=uuid,
                    ),
                    user_manager,
                )
            )
            basic = read_response(basic)
            self.assertTrue(basic["error"] == "")

            # test with invalid UUID
            invalid = loop.run_until_complete(
                get_object(
                    GetObjectData(
                        user_id="test_user",
                        collection_name="example_verba_github_issues",
                        uuid="invalid",
                    ),
                    user_manager,
                )
            )
            invalid = read_response(invalid)
            self.assertTrue(invalid["error"] != "")

        finally:
            loop.run_until_complete(user_manager.close_all_clients())

    def test_view_paginated_collection(self):
        loop = self.get_event_loop()
        try:

            user_manager = get_user_manager()

            basic = loop.run_until_complete(
                view_paginated_collection(
                    ViewPaginatedCollectionData(
                        user_id="test_user",
                        collection_name="example_verba_github_issues",
                        page_size=50,
                        page_number=1,
                        sort_on="issue_created_at",
                        ascending=True,
                        filter_config={},
                    ),
                    user_manager,
                )
            )
            basic = read_response(basic)
            self.assertTrue(basic["error"] == "")

            # test with filter
            filter = loop.run_until_complete(
                view_paginated_collection(
                    ViewPaginatedCollectionData(
                        user_id="test_user",
                        collection_name="example_verba_github_issues",
                        page_size=50,
                        page_number=1,
                        sort_on="issue_created_at",
                        ascending=True,
                        filter_config={
                            "type": "all",
                            "filters": [
                                {
                                    "field": "issue_state",
                                    "operator": "equal",
                                    "value": "open",
                                }
                            ],
                        },
                    ),
                    user_manager,
                )
            )
            filter = read_response(filter)
            self.assertTrue(filter["error"] == "")

            out_of_bounds = loop.run_until_complete(
                view_paginated_collection(
                    ViewPaginatedCollectionData(
                        user_id="test_user",
                        collection_name="example_verba_github_issues",
                        page_size=50,
                        page_number=100,
                        sort_on="issue_created_at",
                        ascending=True,
                        filter_config={
                            "type": "all",
                            "filters": [
                                {
                                    "field": "issue_state",
                                    "operator": "equal",
                                    "value": "open",
                                }
                            ],
                        },
                    ),
                    user_manager,
                )
            )
            out_of_bounds = read_response(out_of_bounds)
            if out_of_bounds["error"] != "":
                print(out_of_bounds["error"])
            self.assertTrue(out_of_bounds["error"] == "")

            self.assertTrue(len(out_of_bounds["items"]) == 0)

        finally:
            loop.run_until_complete(user_manager.close_all_clients())


if __name__ == "__main__":
    unittest.main()
