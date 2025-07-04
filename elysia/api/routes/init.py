from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.api_types import InitialiseTreeData
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.api.core.log import logger
from elysia.config import Settings

from uuid import uuid4

router = APIRouter()


@router.post("/user/{user_id}")
async def initialise_user(
    user_id: str, user_manager: UserManager = Depends(get_user_manager)
):
    """
    Create or retrieve a user.

    Returns:
        (JSONResponse): A JSON response with the following fields:
            - error (str): Any error message (empty string if no error).
            - user_exists (bool): True if the user exists, False otherwise.
            - config (dict): The user's config.
            - frontend_config (dict): The user's frontend config.
    """
    logger.debug(f"/initialise_user API request received")

    try:

        user_exists = user_manager.user_exists(user_id)

        # if a user does not exist, create a user and set up the configs
        if not user_exists:
            user_manager.add_user_local(
                user_id,
            )  # leave config empty to create defaults for a new user

        # if a user exists, get the existing configs
        user = await user_manager.get_user_local(user_id)
        config = user["tree_manager"].config.to_json()
        frontend_config = user["frontend_config"].config

    except Exception as e:
        logger.exception(f"Error in /initialise_user API")
        return JSONResponse(
            content={
                "error": str(e),
                "user_exists": None,
                "config": {},
                "frontend_config": {},
            },
            status_code=500,
        )
    return JSONResponse(
        content={
            "error": "",
            "user_exists": user_exists,
            "config": config,
            "frontend_config": frontend_config,
        },
        status_code=200,
    )


@router.post("/tree/{user_id}/{conversation_id}")
async def initialise_tree(
    user_id: str,
    conversation_id: str,
    data: InitialiseTreeData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Create or retrieve a tree.

    Returns:
        (JSONResponse): A JSON response with the following fields:
            - conversation_id (str): The conversation ID.
            - tree (dict): The tree.
            - error (str): Any error message (empty string if no error).
    """
    logger.debug(f"/initialise_tree API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Conversation ID: {conversation_id}")
    logger.debug(f"Low Memory: {data.low_memory}")

    try:
        tree = await user_manager.initialise_tree(
            user_id,
            conversation_id,
            low_memory=data.low_memory,
        )
    except Exception as e:
        logger.exception(f"Error in /initialise_tree API")
        return JSONResponse(
            content={
                "conversation_id": conversation_id,
                "tree": "",
                "error": str(e),
            },
            status_code=500,
        )

    return JSONResponse(
        content={
            "conversation_id": conversation_id,
            "tree": tree.tree,
            "error": "",
        },
        status_code=200,
    )
