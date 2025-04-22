from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.api_types import InitialiseTreeData
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.api.core.log import logger

router = APIRouter()


@router.post("/initialise_tree")
async def initialise_tree(
    data: InitialiseTreeData, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/initialise_tree API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Conversation ID: {data.conversation_id}")
    logger.debug(f"Debug: {data.debug}")

    try:
        tree = await user_manager.initialise_tree(
            data.user_id, data.conversation_id, data.debug
        )
    except Exception as e:
        logger.exception(f"Error in /initialise_tree API")
        return JSONResponse(
            content={
                "conversation_id": data.conversation_id,
                "tree": "",
                "error": str(e),
            },
            status_code=500,
        )

    return JSONResponse(
        content={
            "conversation_id": data.conversation_id,
            "tree": tree.tree,
            "error": "",
        },
        status_code=200,
    )
