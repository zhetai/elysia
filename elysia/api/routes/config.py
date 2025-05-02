# TODO: with user management: per-user config that gets saved to the user's db i.e. ELYSIA_CONFIG__ as a collection
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

# API Types
from elysia.api.api_types import (
    ChangeConfigData,
    DefaultConfigData,
    LoadConfigData,
    SaveConfigData,
    ListConfigsData,
)

from elysia.api.core.log import logger
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.util.client import ClientManager
from elysia.util.parsing import format_dict_to_serialisable
from elysia.config import Settings


from weaviate.util import generate_uuid5
import weaviate.classes.config as wc

router = APIRouter()


@router.post("/default_config")
async def default_config(
    data: DefaultConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    """
    Set the config to the default values (gemini 2.0 flash or environment variables).
    Any changes to the config will propagate to all trees associated with the user, as well as update the ClientManager to use the new API keys.

    Args:
        data (DefaultConfigData): A class with the following attributes:
            user_id (str): The user ID.
        user_manager (UserManager): The user manager instance.
    """
    logger.debug(f"/default_config API request received")
    logger.debug(f"User ID: {data.user_id}")

    try:
        user = await user_manager.get_user_local(data.user_id)
        user["config"].default_config()
        client_manager: ClientManager = user["client_manager"]
        await client_manager.set_keys_from_settings(user["config"])

    except Exception as e:
        logger.exception(f"Error in /default_config API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": user["config"].to_json()})


@router.post("/change_config")
async def change_config(
    data: ChangeConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    """
    Change the config for a user.
    The `config` dictionary can contain any of the settings, and will override the current config for the user.
    You do not need to specify all the settings, only the ones you wish to change.
    Any changes to the config will propagate to all trees associated with the user, as well as update the ClientManager to use the new API keys.

    Args:
        data (ChangeConfigData): A class with the following attributes:
            user_id (str): The user ID.
            config (dict): A dictionary of config values to set, follows the same format as the Settings.configure() method.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """
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
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": config.to_json()})


@router.post("/save_config")
async def save_config(
    data: SaveConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    """
    Save a config to the weaviate database.
    If the collection ELYSIA_CONFIG__ does not exist, it will be created.

    Args:
        data (SaveConfigData): A class with the following attributes:
            user_id (str): The user ID.
            config_id (str): The ID of the config to save.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """

    logger.debug(f"/save_config API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Config ID: {data.config_id}")

    try:
        # Retrieve the user info
        user = await user_manager.get_user_local(data.user_id)
        config: Settings = user["config"]
        client_manager: ClientManager = user["client_manager"]

        # Get the dict representation of the config
        config_item = config.to_json()
        config_item["config_id"] = data.config_id

        uuid = generate_uuid5(data.config_id)
        async with client_manager.connect_to_async_client() as client:

            # Create a collection if it doesn't exist
            if await client.collections.exists("ELYSIA_CONFIG__"):
                collection = client.collections.get("ELYSIA_CONFIG__")
            else:
                collection = await client.collections.create(
                    "ELYSIA_CONFIG__",
                    vectorizer_config=wc.Configure.Vectorizer.none(),
                )

            if await collection.data.exists(uuid=uuid):
                await collection.data.update(properties=config_item, uuid=uuid)
            else:
                await collection.data.insert(config_item, uuid=uuid)

    except Exception as e:
        logger.exception(f"Error in /save_config API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": config.to_json()})


@router.post("/load_config")
async def load_config(
    data: LoadConfigData, user_manager: UserManager = Depends(get_user_manager)
):
    """
    Load a config from the weaviate database.

    Args:
        data (LoadConfigData): A class with the following attributes:
            user_id (str): The user ID.
            config_id (str): The ID of the config to load. Can be found in the /list_configs API.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """
    logger.debug(f"/load_config API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Config ID: {data.config_id}")

    try:
        # Retrieve the user info
        user = await user_manager.get_user_local(data.user_id)
        config: Settings = user["config"]
        client_manager: ClientManager = user["client_manager"]

        # Retrieve the config from the weaviate database
        uuid = generate_uuid5(data.config_id)
        async with client_manager.connect_to_async_client() as client:
            collection = client.collections.get("ELYSIA_CONFIG__")
            config_item = await collection.query.fetch_object_by_id(uuid=uuid)

        # Load the configs to the current user
        format_dict_to_serialisable(config_item.properties)
        config.load_config(config_item.properties)
        await client_manager.set_keys_from_settings(config)

    except Exception as e:
        logger.exception(f"Error in /load_config API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": config.to_json()})


@router.post("/list_configs")
async def list_configs(
    data: ListConfigsData, user_manager: UserManager = Depends(get_user_manager)
):
    """
    List all the configs for a user.

    Args:
        data (ListConfigsData): A class with the following attributes:
            user_id (str): The user ID.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            configs (list): A list of config IDs.
    """
    logger.debug(f"/list_configs API request received")
    logger.debug(f"User ID: {data.user_id}")

    try:
        user = await user_manager.get_user_local(data.user_id)
        client_manager: ClientManager = user["client_manager"]

        async with client_manager.connect_to_async_client() as client:

            collection = client.collections.get("ELYSIA_CONFIG__")

            len_collection = await collection.aggregate.over_all(total_count=True)
            len_collection = len_collection.total_count

            response = await collection.query.fetch_objects(limit=len_collection)
            configs = [obj.properties["config_id"] for obj in response.objects]

    except Exception as e:
        logger.exception(f"Error in /list_configs API")
        return JSONResponse(content={"error": str(e), "configs": []})

    return JSONResponse(content={"error": "", "configs": configs})
