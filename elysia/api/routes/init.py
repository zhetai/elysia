from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.api_types import InitialiseTreeData, Config
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
    Create a user with a specified config.
    """
    logger.debug(f"/initialise_user API request received")

    try:

        settings = Settings()
        settings.smart_setup()
        settings.setup_app_logger(logger)

        user_exists = user_manager.user_exists(user_id)

        # if a user does not exist, create a user and set up the configs
        if not user_exists:

            config = Config(
                id=str(uuid4()),
                name="New Config",
                settings=settings.to_json(),
                style="Informative, polite and friendly.",
                agent_description="You search and query Weaviate to satisfy the user's query, providing a concise summary of the results.",
                end_goal=(
                    "You have satisfied the user's query, and provided a concise summary of the results. "
                    "Or, you have exhausted all options available, or asked the user for clarification."
                ),
                branch_initialisation="one_branch",
            )

            user_manager.add_user_local(
                user_id,
                config,
            )

            frontend_config = (await user_manager.get_user_local(user_id))[
                "frontend_config"
            ].config

        # if a user exists, get the existing configs
        else:
            user = await user_manager.get_user_local(user_id)  #
            config = user["tree_manager"].config
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
            "config": config.model_dump(),
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
