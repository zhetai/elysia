from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.api_types import (
    AddFeedbackData,
    FeedbackMetadataData,
    RemoveFeedbackData,
)
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.api.utils.feedback import create_feedback, feedback_metadata, remove_feedback

router = APIRouter()


@router.post("/add_feedback")
async def run_add_feedback(
    data: AddFeedbackData, user_manager: UserManager = Depends(get_user_manager)
):
    tree = await user_manager.get_tree(data.user_id, data.conversation_id)

    async with user_manager.auth_client_manager.connect_to_async_client() as client:
        try:
            await create_feedback(
                data.user_id,
                data.conversation_id,
                data.query_id,
                data.feedback,
                tree,
                client,
            )
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    return JSONResponse(content={"error": ""}, status_code=200)


@router.post("/remove_feedback")
async def run_remove_feedback(
    data: RemoveFeedbackData, user_manager: UserManager = Depends(get_user_manager)
):
    async with user_manager.auth_client_manager.connect_to_async_client() as client:
        try:
            await remove_feedback(
                data.user_id, data.conversation_id, data.query_id, client
            )
            return JSONResponse(content={"error": ""}, status_code=200)
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/feedback_metadata")
async def run_feedback_metadata(
    data: FeedbackMetadataData, user_manager: UserManager = Depends(get_user_manager)
):
    async with user_manager.auth_client_manager.connect_to_async_client() as client:
        return JSONResponse(
            content=await feedback_metadata(client, data.user_id), status_code=200
        )
