import asyncio

from fastapi import APIRouter, Depends, WebSocket
from starlette.websockets import WebSocketDisconnect

# API Types
from elysia.api.api_types import ProcessCollectionData

# Logging
from elysia.api.core.logging import logger

# User manager
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager

# Websocket
from elysia.api.utils.websocket import help_websocket

# Preprocessing
from elysia.preprocess.collection import CollectionPreprocessor

# LM
from elysia.config import load_lm

router = APIRouter()


async def process_collection(
    data: ProcessCollectionData, websocket: WebSocket, user_manager: UserManager
):
    user = await user_manager.get_user_local(data["user_id"])
    config = user["config"]

    # try:
    preprocessor = CollectionPreprocessor(
        lm=load_lm(
            provider=config.BASE_PROVIDER,
            lm_name=config.BASE_MODEL,
            model_api_base=config.MODEL_API_BASE,
        )
    )
    async for result in preprocessor(
        collection_name=data["collection_name"],
        client_manager=user["client_manager"],
        force=True,
    ):
        try:
            await websocket.send_json(result)
        except WebSocketDisconnect:
            logger.info("Client disconnected during process_collection")
            break
        await asyncio.sleep(0.001)


@router.websocket("/process_collection")
async def process_collection_websocket(
    websocket: WebSocket, user_manager: UserManager = Depends(get_user_manager)
):
    await help_websocket(
        websocket, lambda data, ws: process_collection(data, ws, user_manager)
    )
