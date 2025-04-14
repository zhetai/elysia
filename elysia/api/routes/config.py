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

# user manager
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager

router = APIRouter()


@router.post("/default_config")
async def default_config(
    data: DefaultConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    try:
        user = await user_manager.get_user_local(data.user_id)
        user["config"].default_config()
        await user["client_manager"].set_keys_from_settings(user["config"])
        user["tree_manager"].update_settings(user["config"])
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

    return JSONResponse(content={"error": ""})


@router.post("/change_config")
async def change_config(
    data: ChangeConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    try:
        user = await user_manager.get_user_local(data.user_id)
        user["config"].configure(**data.config)
        await user["client_manager"].set_keys_from_settings(user["config"])
        user["tree_manager"].update_settings(user["config"])
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

    return JSONResponse(content={"error": ""})


@router.post("/save_config")
async def save_config(
    data: SaveConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    try:
        user = await user_manager.get_user_local(data.user_id)
        user["config"].export_config(data.filepath)
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

    return JSONResponse(content={"error": ""})


@router.post("/load_config")
async def load_config(
    data: LoadConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    try:
        user = await user_manager.get_user_local(data.user_id)
        user["config"].load_config(data.filepath)
        await user["client_manager"].set_keys_from_settings(user["config"])
        user["tree_manager"].update_settings(user["config"])
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

    return JSONResponse(content={"error": ""})
