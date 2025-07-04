from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

# API Types
from elysia.api.api_types import (
    ViewPaginatedCollectionData,
    UpdateCollectionMetadataData,
)

# Logging
from elysia.api.core.log import logger

# user manager
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager

# Utils
from elysia.util.parsing import format_dict_to_serialisable

# Util
from elysia.util.collection import (
    retrieve_all_collection_names,
    async_get_collection_data_types,
    paginated_collection,
)

from elysia.preprocess.collection import edit_preprocessed_collection

from elysia.util.return_types import specific_return_types, types_dict

router = APIRouter()


@router.get("/mapping_types")
async def mapping_types():
    return JSONResponse(
        content={"mapping_types": types_dict},
        status_code=200,
    )


@router.get("/{user_id}/list")
async def collections_list(
    user_id: str, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/collections API request received")

    headers = {"Cache-Control": "no-cache"}

    try:

        user_local = await user_manager.get_user_local(user_id)
        client_manager = user_local["client_manager"]

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
                logger.debug(
                    f"Gathering information for collection_name: {collection_name}"
                )

                try:
                    # for vectoriser
                    collection = client.collections.get(collection_name)
                    config = await collection.config.get()

                    # for total count
                    all_aggregate = await collection.aggregate.over_all(
                        total_count=True
                    )
                    len = all_aggregate.total_count

                    # for processed
                    processed = await client.collections.exists(
                        f"ELYSIA_METADATA_{collection_name.lower()}__"
                    )

                    vector_config = {"fields": {}, "global": {}}
                    if config.vector_config:
                        for (
                            named_vector_name,
                            named_vector_config,
                        ) in config.vector_config.items():

                            # for this vectoriser, what model is being used?
                            model = named_vector_config.vectorizer.model
                            if isinstance(model, dict) and "model" in model:
                                model = model["model"]
                            else:
                                model = "Unknown"

                            # check what fields are being vectorised
                            fields = named_vector_config.vectorizer.source_properties
                            if fields is None:
                                fields = [
                                    c.name
                                    for c in config.properties
                                    if c.data_type[:].startswith("text")
                                ]

                            # for each field, add the vectoriser and model to the vector config
                            for field_name in fields:
                                if field_name not in vector_config["fields"]:
                                    vector_config["fields"][field_name] = [
                                        {
                                            "named_vector": named_vector_name,
                                            "vectorizer": named_vector_config.vectorizer.vectorizer.name,
                                            "model": model,
                                        }
                                    ]
                                else:
                                    vector_config["fields"][field_name].append(
                                        {
                                            "named_vector": named_vector_name,
                                            "vectorizer": named_vector_config.vectorizer.vectorizer.name,
                                            "model": model,
                                        }
                                    )

                            # add the vectoriser and model to the global vector config
                            # vector_config["fields"][field_name] = {
                            #     "vectorizer": vectorised_field.vectorizer.vectorizer.name,
                            #     "model": model,
                            # }

                    if config.vectorizer_config:
                        model = config.vectorizer_config.model
                        if isinstance(model, dict) and "model" in model:
                            model = model["model"]

                        vector_config["global"] = {
                            "vectorizer": config.vectorizer_config.vectorizer.name,
                            "model": model,
                        }
                    else:
                        vector_config["global"] = {}

                    metadata.append(
                        {
                            "name": collection_name,
                            "total": len,
                            "vectorizer": vector_config,
                            "processed": processed,
                            "error": False,
                        }
                    )
                except Exception as e:
                    metadata.append(
                        {
                            "name": collection_name,
                            "total": 0,
                            "vectorizer": {},
                            "processed": False,
                            "error": True,
                        }
                    )

            return JSONResponse(
                content={"collections": metadata, "error": ""},
                status_code=200,
                headers=headers,
            )

    except Exception as e:
        logger.exception(f"Error in /collections API")
        return JSONResponse(
            content={"collections": [], "error": str(e)},
            status_code=500,
            headers=headers,
        )


@router.post("/{user_id}/view/{collection_name}")
async def view_paginated_collection(
    user_id: str,
    collection_name: str,
    data: ViewPaginatedCollectionData,
    user_manager: UserManager = Depends(get_user_manager),
):

    logger.debug(f"/view_paginated_collection API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Query: {data.query}")
    logger.debug(f"Collection name: {collection_name}")
    logger.debug(f"Page size: {data.page_size}")
    logger.debug(f"Page number: {data.page_number}")
    logger.debug(f"Sort on: {data.sort_on}")
    logger.debug(f"Ascending: {data.ascending}")
    logger.debug(f"Filter config: {data.filter_config}")

    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager = user_local["client_manager"]
        async with client_manager.connect_to_async_client() as client:
            # get collection properties
            data_types = await async_get_collection_data_types(client, collection_name)

            # obtain paginated results from collection
            items = await paginated_collection(
                client=client,
                collection_name=collection_name,
                query=data.query,
                page_size=data.page_size,
                page_number=data.page_number,
                sort_on=data.sort_on,
                ascending=data.ascending,
                filter_config=data.filter_config,
            )

            logger.info(f"Returning collection info for {collection_name}")
            return JSONResponse(
                content={"properties": data_types, "items": items, "error": ""},
                status_code=200,
            )
    except Exception as e:
        logger.exception(f"Error in /view_paginated_collection API")
        return JSONResponse(
            content={"properties": [], "items": [], "error": str(e)}, status_code=500
        )


@router.get("/{user_id}/get_object/{collection_name}/{uuid}")
async def get_object(
    user_id: str,
    collection_name: str,
    uuid: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    logger.debug(f"/get_object API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Collection name: {collection_name}")
    logger.debug(f"UUID: {uuid}")

    headers = {"Cache-Control": "max-age=300"}  # cache for 5 minutes

    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager = user_local["client_manager"]
        async with client_manager.connect_to_async_client() as client:
            collection = client.collections.get(collection_name)

            try:
                object = await collection.query.fetch_object_by_id(uuid)
                object = object.properties
            except Exception as e:
                logger.error(
                    f"Error in /get_object API: No object found with UUID {uuid} in collection {collection_name}"
                )
                return JSONResponse(
                    content={
                        "properties": [],
                        "items": [],
                        "error": f"No object found with UUID {uuid} in collection {collection_name}.",
                    },
                    status_code=500,
                    headers=headers,
                )

            data_types = await async_get_collection_data_types(client, collection_name)
            format_dict_to_serialisable(object)

    except Exception as e:
        logger.exception(f"Error in /get_object API")
        return JSONResponse(
            content={"properties": [], "items": [], "error": str(e)},
            status_code=500,
            headers=headers,
        )

    return JSONResponse(
        content={"properties": data_types, "items": [object], "error": ""},
        status_code=200,
        headers=headers,
    )


@router.get("/{user_id}/metadata/{collection_name}")
async def collection_metadata(
    user_id: str,
    collection_name: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    logger.debug(f"/collection_metadata API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Collection name: {collection_name}")

    headers = {"Cache-Control": "no-cache"}

    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager = user_local["client_manager"]

        async with client_manager.connect_to_async_client() as client:
            metadata_name = f"ELYSIA_METADATA_{collection_name.lower()}__"

            # check if the collection itself exists
            if not await client.collections.exists(collection_name):
                raise Exception(f"Collection {collection_name} does not exist")

            # check if the metadata collection exists
            if not await client.collections.exists(metadata_name):
                raise Exception(
                    f"Metadata collection for {collection_name} does not exist"
                )
            else:
                metadata_collection = client.collections.get(metadata_name)
                metadata = await metadata_collection.query.fetch_objects(limit=1)
                properties = metadata.objects[0].properties
                format_dict_to_serialisable(properties)

    except Exception as e:
        logger.exception(f"Error in /collection_metadata API")
        return JSONResponse(
            content={
                "metadata": {},
                "error": str(e),
            },
            status_code=200,
            headers=headers,
        )

    return JSONResponse(
        content={
            "metadata": properties,
            "error": "",
        },
        status_code=200,
        headers=headers,
    )


@router.patch("/{user_id}/metadata/{collection_name}")
async def update_metadata(
    user_id: str,
    collection_name: str,
    data: UpdateCollectionMetadataData,
    user_manager: UserManager = Depends(get_user_manager),
):
    logger.debug(f"/update_metadata API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Collection name: {collection_name}")
    logger.debug(f"Data: {data}")

    # retrieve the current metadata
    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager = user_local["client_manager"]

        properties = await edit_preprocessed_collection(
            collection_name=collection_name,
            client_manager=client_manager,
            named_vectors=(
                [
                    named_vector.model_dump()
                    for named_vector in data.named_vectors
                    if named_vector
                ]
                if data.named_vectors
                else None
            ),
            summary=data.summary,
            mappings=data.mappings,
            fields=(
                [field.model_dump() if field else None for field in data.fields]
                if data.fields
                else None
            ),
        )
    except Exception as e:
        logger.exception(f"Error in /update_metadata API:")
        return JSONResponse(
            content={"metadata": {}, "error": str(e)},
            status_code=200,
        )

    return JSONResponse(
        content={"metadata": properties, "error": ""},
        status_code=200,
    )


@router.delete("/{user_id}/metadata/{collection_name}")
async def delete_metadata(
    user_id: str,
    collection_name: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    logger.debug(f"/delete_metadata API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Collection name: {collection_name}")

    # retrieve the current metadata
    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager = user_local["client_manager"]

        async with client_manager.connect_to_async_client() as client:
            metadata_name = f"ELYSIA_METADATA_{collection_name.lower()}__"

            # check if the collection itself exists
            if not await client.collections.exists(collection_name.lower()):
                raise Exception(f"Collection {collection_name} does not exist")

            # check if the metadata collection exists
            if not await client.collections.exists(metadata_name):
                raise Exception(
                    f"Metadata collection for {collection_name} does not exist"
                )

            else:
                await client.collections.delete(metadata_name)

                return JSONResponse(content={"error": ""}, status_code=200)

    except Exception as e:
        logger.exception(f"Error in /delete_metadata API")
        return JSONResponse(content={"error": str(e)}, status_code=200)
