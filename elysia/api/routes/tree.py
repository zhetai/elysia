from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from elysia.api.api_types import InitialiseTreeData, ChangeAtlasData
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.api.core.log import logger
from elysia.tree.tree import Tree


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


@router.post("/change_atlas")
async def change_atlas(
    data: ChangeAtlasData, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/change_atlas API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Conversation ID: {data.conversation_id}")
    logger.debug(f"Style: {data.style}")
    logger.debug(f"Agent Description: {data.agent_description}")
    logger.debug(f"End Goal: {data.end_goal}")

    try:

        local_user = await user_manager.get_user_local(data.user_id)
        tree: Tree = local_user["tree_manager"].get_tree(data.conversation_id)

        tree.change_style(data.style)
        tree.change_agent_description(data.agent_description)
        tree.change_end_goal(data.end_goal)

        return JSONResponse(content={"error": ""}, status_code=200)
    except Exception as e:
        logger.exception(f"Error in /change_atlas API")
        return JSONResponse(content={"error": str(e)}, status_code=500)
