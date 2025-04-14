# TODO: with user management: most of these functions depend on the user (as they will have their own collections/API/client)
# TODO: and a lot of these use the 'global' user client which needs to be phased out

import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

# API Types
from elysia.api.api_types import (
    CollectionMetadataData,
    GetObjectData,
    SetCollectionsData,
    UserCollectionsData,
    ViewPaginatedCollectionData,
)

# Logging
from elysia.api.core.logging import logger

# user manager
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager

# Tree
from elysia.tree.tree import Tree

# Util
from elysia.util.collection_metadata import (
    async_get_collection_data_types,
    paginated_collection,
)

router = APIRouter()


@router.post("/collections")
async def collections(user_manager: UserManager = Depends(get_user_manager)):
    client_manager = user_manager.user_client_manager

    async with client_manager.connect_to_async_client() as client:
        collections = [
            c
            for c in await client.collections.list_all()
            if not c.startswith("ELYSIA_")
        ]

    async with client_manager.connect_to_async_client() as client:
        # get collection metadata
        metadata = []
        for collection_name in collections:
            # for vectoriser
            collection = client.collections.get(collection_name)
            config = await collection.config.get()

            # for total count
            all_aggregate = await collection.aggregate.over_all(total_count=True)
            len = all_aggregate.total_count

            # for processed
            processed = await client.collections.exists(
                f"ELYSIA_METADATA_{collection_name}__"
            )

            vector_config = {}
            if config.vector_config:
                for field_name, vectorised_field in config.vector_config.items():
                    model = vectorised_field.vectorizer.model
                    if isinstance(model, dict) and "model" in model:
                        model = model["model"]

                    vector_config[field_name] = {
                        "vectorizer": vectorised_field.vectorizer.vectorizer.name,
                        "model": model,
                    }

            if config.vectorizer_config:
                model = config.vectorizer_config.model
                if isinstance(model, dict) and "model" in model:
                    model = model["model"]

                vector_config["global"] = {
                    "vectorizer": config.vectorizer_config.vectorizer.name,
                    "model": model,
                }
            else:
                vector_config["global"] = {
                    "vectorizer": None,
                    "model": None,
                }

            metadata.append(
                {
                    "name": collection_name,
                    "total": len,
                    "vectorizer": vector_config,
                    "processed": processed,
                }
            )

        return JSONResponse(
            content={"collections": metadata, "error": ""}, status_code=200
        )


# @router.post("/user_collections")
# async def user_collections(
#     data: UserCollectionsData,
#     user_manager: UserManager = Depends(get_user_manager)
# ):

#     tree_manager: TreeManager = user_manager.get_tree_manager(data.user_id)
#     client_manager = user_manager.user_client_manager
#     with client_manager.connect_to_client() as client:

#         # get collection metadata
#         t0 = time.time()
#         metadata = []
#         for collection_name in tree_manager.collection_names:
#             collection = client.collections.get(collection_name)
#             metadata.append(
#                 {
#                     "name": collection_name,
#                     "total": len(collection),
#                     "vectorizer": collection.config.get().vectorizer,  # None when using namedvectors, TODO: implement for named vectors
#                     "processed": client.collections.exists(f"ELYSIA_METADATA_{collection_name}__")
#                 }
#             )

#         logger.info(f"Returning collections: {metadata}")
#         backend_print(f"Collections got in {time.time() - t0} seconds")
#         return JSONResponse(content={"collections": metadata, "error": ""}, status_code=200)


@router.post("/view_paginated_collection")
async def view_paginated_collection(
    data: ViewPaginatedCollectionData,
    user_manager: UserManager = Depends(get_user_manager),
):
    client_manager = user_manager.user_client_manager
    async with client_manager.connect_to_async_client() as client:
        # get collection properties
        data_types = await async_get_collection_data_types(client, data.collection_name)

        # obtain paginated results from collection
        items = await paginated_collection(
            client=client,
            collection_name=data.collection_name,
            page_size=data.page_size,
            page_number=data.page_number,
            sort_on=data.sort_on,
            ascending=data.ascending,
            filter_config=data.filter_config,
        )

        logger.info(f"Returning collection info for {data.collection_name}")
        return JSONResponse(
            content={"properties": data_types, "items": items, "error": ""},
            status_code=200,
        )


@router.post("/view_paginated_collection_admin")
async def view_paginated_collection_admin(
    data: ViewPaginatedCollectionData,
    user_manager: UserManager = Depends(get_user_manager),
):
    client_manager = user_manager.auth_client_manager
    async with client_manager.connect_to_async_client() as client:
        # get collection properties
        data_types = await async_get_collection_data_types(client, data.collection_name)

        # obtain paginated results from collection
        items = await paginated_collection(
            client=client,
            collection_name=data.collection_name,
            page_size=data.page_size,
            page_number=data.page_number,
            sort_on=data.sort_on,
            ascending=data.ascending,
            filter_config=data.filter_config,
        )

        logger.info(f"Returning collection info for {data.collection_name}")
        return JSONResponse(
            content={"properties": data_types, "items": items, "error": ""},
            status_code=200,
        )


@router.post("/set_collections")
async def set_collections(
    data: SetCollectionsData, user_manager: UserManager = Depends(get_user_manager)
):
    client_manager = user_manager.user_client_manager
    tree: Tree = await user_manager.get_tree(data.user_id, data.conversation_id)
    await tree.set_collection_names(
        data.collection_names,
        client_manager=client_manager,
    )
    return JSONResponse(content={"error": ""}, status_code=200)


@router.post("/get_object")
async def get_object(
    data: GetObjectData, user_manager: UserManager = Depends(get_user_manager)
):
    client_manager = user_manager.user_client_manager
    async with client_manager.connect_to_async_client() as client:
        error = ""
        collection = client.collections.get(data.collection_name)

        try:
            object = await collection.query.fetch_object_by_id(data.uuid)
            object = object.properties
        except Exception as e:
            error = "No object found with this UUID."

        data_types = await async_get_collection_data_types(client, data.collection_name)

    return JSONResponse(
        content={"properties": data_types, "items": [object], "error": error},
        status_code=200,
    )


@router.post("/collection_metadata")
async def collection_metadata(
    data: CollectionMetadataData, user_manager: UserManager = Depends(get_user_manager)
):
    error_message = ""
    try:
        tree = user_manager.base_tree
    except Exception as e:
        error_message = str(e)
    return JSONResponse(
        content={
            "metadata": tree.tree_data.output_collection_metadata(with_mappings=True),
            "error": error_message,
        },
        status_code=200,
    )
