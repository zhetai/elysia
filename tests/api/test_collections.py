import pytest

from fastapi.responses import JSONResponse

import asyncio
import json

from elysia.api.api_types import (
    ViewPaginatedCollectionData,
    UpdateCollectionMetadataData,
    MetadataNamedVectorData,
    MetadataFieldData,
)
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.collections import (
    collections_list,
    view_paginated_collection,
    collection_metadata,
    get_object,
    mapping_types,
    update_metadata,
)
from elysia.api.routes.init import initialise_user
from elysia.api.api_types import InitialiseUserData

from elysia.api.core.log import logger, set_log_level

set_log_level("CRITICAL")


def read_response(response: JSONResponse):
    return json.loads(response.body)


class TestEndpoints:

    @pytest.mark.asyncio
    async def test_collections(self):

        try:
            user_manager = get_user_manager()
            await initialise_user(
                InitialiseUserData(user_id="test_user", default_models=True),
                user_manager,
            )
            basic = await collections_list(
                user_id="test_user", user_manager=user_manager
            )
            basic = read_response(basic)
            assert basic["error"] == ""
        finally:
            await user_manager.close_all_clients()

    @pytest.mark.asyncio
    async def test_collection_metadata(self):
        try:
            user_manager = get_user_manager()
            await initialise_user(
                InitialiseUserData(user_id="test_user", default_models=True),
                user_manager,
            )
            basic = await collection_metadata(
                user_id="test_user",
                collection_name="example_verba_github_issues",
                user_manager=user_manager,
            )
            basic = read_response(basic)
            assert basic["error"] == ""
        finally:
            await user_manager.close_all_clients()

    @pytest.mark.asyncio
    async def test_get_object(self):
        try:
            user_manager = get_user_manager()
            await initialise_user(
                InitialiseUserData(user_id="test_user", default_models=True),
                user_manager,
            )

            # first manually get a UUID
            user_local = await user_manager.get_user_local("test_user")
            client_manager = user_local["client_manager"]
            async with client_manager.connect_to_async_client() as client:
                collection = client.collections.get("example_verba_github_issues")
                response = await collection.query.fetch_objects(limit=1)
                uuid = str(response.objects[0].uuid)

            # test with valid UUID
            basic = await get_object(
                user_id="test_user",
                collection_name="example_verba_github_issues",
                uuid=uuid,
                user_manager=user_manager,
            )

            basic = read_response(basic)
            assert basic["error"] == ""

            # test with invalid UUID
            invalid = await get_object(
                user_id="test_user",
                collection_name="example_verba_github_issues",
                uuid="invalid",
                user_manager=user_manager,
            )

            invalid = read_response(invalid)
            assert invalid["error"] != ""

        finally:
            await user_manager.close_all_clients()

    @pytest.mark.asyncio
    async def test_view_paginated_collection(self):
        try:

            user_manager = get_user_manager()
            await initialise_user(
                InitialiseUserData(user_id="test_user", default_models=True),
                user_manager,
            )

            basic = await view_paginated_collection(
                user_id="test_user",
                collection_name="example_verba_github_issues",
                data=ViewPaginatedCollectionData(
                    page_size=50,
                    page_number=1,
                    sort_on="issue_created_at",
                    ascending=True,
                    filter_config={},
                ),
                user_manager=user_manager,
            )
            basic = read_response(basic)
            assert basic["error"] == ""

            # test with filter
            filter = await view_paginated_collection(
                user_id="test_user",
                collection_name="example_verba_github_issues",
                data=ViewPaginatedCollectionData(
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
                user_manager=user_manager,
            )
            filter = read_response(filter)
            assert filter["error"] == ""

            # test with out of bounds page number, should return empty list
            out_of_bounds = await view_paginated_collection(
                user_id="test_user",
                collection_name="example_verba_github_issues",
                data=ViewPaginatedCollectionData(
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
                user_manager=user_manager,
            )
            out_of_bounds = read_response(out_of_bounds)
            assert out_of_bounds["error"] == ""
            assert len(out_of_bounds["items"]) == 0

        finally:
            await user_manager.close_all_clients()

    @pytest.mark.asyncio
    async def test_mapping_types(self):
        mapping_types()

    @pytest.mark.asyncio
    async def test_update_metadata(self):
        try:
            user_id = "test_user_update_metadata"
            user_manager = get_user_manager()
            await initialise_user(
                InitialiseUserData(user_id=user_id, default_models=True),
                user_manager,
            )

            # retrieve the metadata
            metadata = await collection_metadata(
                user_id=user_id,
                collection_name="example_verba_github_issues",
                user_manager=user_manager,
            )
            metadata = read_response(metadata)
            assert metadata["error"] == ""
            metadata = metadata["metadata"]

            # record values
            old_named_vectors = metadata["named_vectors"]
            old_summary = metadata["summary"]
            old_mappings = metadata["mappings"]
            old_fields = metadata["fields"]

            # what to update
            new_named_vectors = [
                MetadataNamedVectorData(
                    name="issue_content",
                    enabled=True,
                    description="Directly related to the issue content.",
                )
            ]

            new_summary = "This is a test summary."
            new_mappings = ["ecommerce"]
            new_fields = [
                MetadataFieldData(
                    name="issue_title",
                    description="A test description for issue title.",
                )
            ]

            # update the metadata
            update_metadata_response = await update_metadata(
                user_id=user_id,
                collection_name="example_verba_github_issues",
                data=UpdateCollectionMetadataData(
                    named_vectors=new_named_vectors,
                    summary=new_summary,
                    mappings=new_mappings,
                    fields=new_fields,
                ),
                user_manager=user_manager,
            )
            update_metadata_response = read_response(update_metadata_response)
            assert update_metadata_response["error"] == ""

            # retrieve the metadata again
            metadata = await collection_metadata(
                user_id=user_id,
                collection_name="example_verba_github_issues",
                user_manager=user_manager,
            )
            metadata = read_response(metadata)
            assert metadata["error"] == ""
            metadata = metadata["metadata"]

            # check the values have been updated
            assert metadata["named_vectors"]["issue_content"]["enabled"] == True
            assert (
                metadata["named_vectors"]["issue_content"]["description"]
                == "Directly related to the issue content."
            )
            assert metadata["summary"] == new_summary
            assert "ecommerce" in metadata["mappings"]
            assert (
                metadata["fields"]["issue_title"]["description"]
                == new_fields[0].description
            )

            # save current values
            midway_named_vectors = metadata["named_vectors"]
            midway_summary = metadata["summary"]
            midway_mappings = metadata["mappings"]
            midway_fields = metadata["fields"]

            # only update a portion of the metadata
            new_mappings2 = ["conversation", "message", "ticket"]
            new_fields2 = [
                MetadataFieldData(
                    name="issue_updated_at",
                    description="A test description for issue title.",
                ),
                MetadataFieldData(
                    name="issue_created_at",
                    description="A test description for issue created at.",
                ),
            ]

            # update the metadata
            update_metadata2_response = await update_metadata(
                user_id=user_id,
                collection_name="example_verba_github_issues",
                data=UpdateCollectionMetadataData(
                    mappings=new_mappings2,
                    fields=new_fields2,
                ),
                user_manager=user_manager,
            )
            update_metadata2_response = read_response(update_metadata2_response)
            assert update_metadata2_response["error"] == ""

            # retrieve the metadata again
            metadata = await collection_metadata(
                user_id=user_id,
                collection_name="example_verba_github_issues",
                user_manager=user_manager,
            )
            metadata = read_response(metadata)
            assert metadata["error"] == ""
            metadata = metadata["metadata"]

            # check the values have been updated
            assert "conversation" in metadata["mappings"]
            assert "message" in metadata["mappings"]
            assert "ticket" in metadata["mappings"]
            assert (
                metadata["fields"]["issue_updated_at"]["description"]
                == new_fields2[0].description
            )
            assert (
                metadata["fields"]["issue_created_at"]["description"]
                == new_fields2[1].description
            )

            # check that the other values are unchanged
            assert metadata["named_vectors"] == midway_named_vectors
            assert metadata["summary"] == midway_summary

            old_named_vectors_data = [
                MetadataNamedVectorData(
                    name=name,
                    enabled=old_named_vectors[name]["enabled"],
                    description=old_named_vectors[name]["description"],
                )
                for name in old_named_vectors
            ]
            old_fields_data = [
                MetadataFieldData(
                    name=field,
                    description=old_fields[field]["description"],
                )
                for field in old_fields
            ]
            old_mappings_data = list(old_mappings.keys())

            # change back to the original values
            update_metadata3_response = await update_metadata(
                user_id=user_id,
                collection_name="example_verba_github_issues",
                data=UpdateCollectionMetadataData(
                    named_vectors=old_named_vectors_data,
                    summary=old_summary,
                    mappings=old_mappings_data,
                    fields=old_fields_data,
                ),
                user_manager=user_manager,
            )
            update_metadata3_response = read_response(update_metadata3_response)
            assert update_metadata3_response["error"] == ""

        finally:
            from rich import print

            print(
                "\n\n[bold red]AN ERROR OCCURED DURING TESTING test_update_metadata[/bold red]\n"
                "This means that the example_verba_github_issues collection has been left in an unknown state.\n"
                "The metadata was being edited in this test, but the test did not finish successfully and it was not reset.\n"
                "The collection needs reprocessing to return to a working state.\n"
            )
            await user_manager.close_all_clients()
