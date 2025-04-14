from uuid import UUID, uuid5

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from elysia.api.api_types import InitialiseTreeData
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.tree.tree import Tree

router = APIRouter()


@router.post("/initialise_tree")
async def initialise_tree(
    data: InitialiseTreeData, user_manager: UserManager = Depends(get_user_manager)
):
    try:
        tree = await user_manager.initialise_tree(
            data.user_id, data.conversation_id, data.debug
        )
    except Exception as e:
        print(f"\n\nError: {e}")
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
