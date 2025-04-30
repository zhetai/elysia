from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from elysia.api.api_types import InitialiseTreeData
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.api.core.log import logger
from elysia.tree.tree import Tree


class ChangeGuideData(BaseModel):
    user_id: str
    conversation_id: str
    style: str
    agent_description: str
    end_goal: str


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


@router.post("/change_guide")
async def change_guide(
    data: ChangeGuideData, user_manager: UserManager = Depends(get_user_manager)
):
    try:

        local_user = await user_manager.get_user_local(data.user_id)
        tree: Tree = local_user["tree_manager"].get_tree(data.conversation_id)

        tree.change_style(data.style)
        tree.change_agent_description(data.agent_description)
        tree.change_end_goal(data.end_goal)

        return JSONResponse(content={"success": True}, status_code=200)
    except Exception as e:
        logger.exception(f"Error in /change_guide API")
        return JSONResponse(content={"error": str(e)}, status_code=500)
