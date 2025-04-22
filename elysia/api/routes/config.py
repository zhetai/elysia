# TODO: with user management: per-user config that gets saved to the user's db i.e. ELYSIA_CONFIG__ as a collection
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

# API Types
from elysia.api.api_types import (
    ChangeConfigData,
    DefaultConfigData,
    LoadConfigData,
    SaveConfigData,
)

from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.util.client import ClientManager
from elysia.config import Settings
from elysia.api.core.log import logger

router = APIRouter()


@router.post("/default_config")
async def default_config(
    data: DefaultConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/default_config API request received")
    logger.debug(f"User ID: {data.user_id}")

    try:
        user = await user_manager.get_user_local(data.user_id)
        user["config"].default_config()
        client_manager: ClientManager = user["client_manager"]
        await client_manager.set_keys_from_settings(user["config"])

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

    return JSONResponse(content={"error": ""})


@router.post("/change_config")
async def change_config(
    data: ChangeConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/change_config API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Config: {data.config}")

    try:
        user = await user_manager.get_user_local(data.user_id)
        config: Settings = user["config"]
        config.configure(**data.config)
        client_manager: ClientManager = user["client_manager"]
        await client_manager.set_keys_from_settings(config)

    except Exception as e:
        logger.exception(f"Error in /change_config API")
        return JSONResponse(content={"error": str(e)})

    return JSONResponse(content={"error": ""})


@router.post("/save_config")
async def save_config(
    data: SaveConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/save_config API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Filepath: {data.filepath}")

    try:
        user = await user_manager.get_user_local(data.user_id)
        config: Settings = user["config"]
        config.export_config(data.filepath)
    except Exception as e:
        logger.exception(f"Error in /save_config API")
        return JSONResponse(content={"error": str(e)})

    return JSONResponse(content={"error": ""})


@router.post("/load_config")
async def load_config(
    data: LoadConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/load_config API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Filepath: {data.filepath}")

    try:
        user = await user_manager.get_user_local(data.user_id)
        config: Settings = user["config"]
        config.load_config(data.filepath)
        client_manager: ClientManager = user["client_manager"]
        await client_manager.set_keys_from_settings(config)

    except Exception as e:
        logger.exception(f"Error in /load_config API")
        return JSONResponse(content={"error": str(e)})

    return JSONResponse(content={"error": ""})
