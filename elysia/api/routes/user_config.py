import os
import jwt
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, set_key

from pydantic import BaseModel
from typing import Optional

load_dotenv(override=True)

# API Types
from elysia.api.api_types import (
    SaveConfigData,
    LoadConfigData,
    Config,
    UpdateSaveLocationData,
)

from elysia.api.core.log import logger
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.util.client import ClientManager
from elysia.util.parsing import format_dict_to_serialisable
from elysia.config import Settings
from elysia.api.services.tree import TreeManager

import weaviate
import weaviate.classes.config as wc
from weaviate.util import generate_uuid5
from weaviate.auth import Auth


def rename_keys(config_item: dict):

    renamed_settings = {
        key.upper(): config_item["settings"][key] for key in config_item["settings"]
    }

    if "API_KEYS" in config_item["settings"]:
        renamed_api_keys = {
            key.lower(): config_item["settings"]["API_KEYS"][key]
            for key in config_item["settings"]["API_KEYS"]
        }
        renamed_settings["API_KEYS"] = renamed_api_keys

    renamed_config = {key.lower(): config_item[key] for key in config_item}
    renamed_config["settings"] = renamed_settings

    return renamed_config


router = APIRouter()


@router.patch("/save_location/{user_id}")
async def update_save_location(
    user_id: str,
    data: UpdateSaveLocationData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Update a user's save location for the configs.
    """
    logger.debug(f"/update_save_location API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"WCD URL: {data.wcd_url}")
    logger.debug(f"WCD API Key: {data.wcd_api_key}")

    try:
        await user_manager.update_save_location(user_id, data.wcd_url, data.wcd_api_key)
        return JSONResponse(content={"error": ""})
    except Exception as e:
        logger.exception(f"Error in /update_save_location API")
        return JSONResponse(content={"error": str(e)})


@router.get("/{user_id}")
async def get_user_config(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Get the config for a single user.
    """
    logger.debug(f"/get_user_config API request received")
    logger.debug(f"User ID: {user_id}")

    headers = {"Cache-Control": "no-cache"}

    try:
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]

        config = Config(
            settings=tree_manager.settings.to_json(),
            style=tree_manager.style,
            agent_description=tree_manager.agent_description,
            end_goal=tree_manager.end_goal,
            branch_initialisation=tree_manager.branch_initialisation,
        )

    except Exception as e:
        logger.exception(f"Error in /get_user_config API")
        return JSONResponse(content={"error": str(e), "config": {}}, headers=headers)

    return JSONResponse(
        content={"error": "", "config": config.model_dump()}, headers=headers
    )


@router.post("/{user_id}/default_models")
async def default_models_user(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Set the base_model and complex_model to the default values (gemini 2.0 flash or environment variables), for an entire user.
    Any changes will propagate to all trees associated with the user that were not initialised with their own settings.

    Args:
        user_id (str): The user ID.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """
    logger.debug(f"/default_models_user API request received")
    logger.debug(f"User ID: {user_id}")

    try:
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]
        tree_manager.settings.default_models()

        config = Config(
            settings=tree_manager.settings.to_json(),
            style=tree_manager.style,
            agent_description=tree_manager.agent_description,
            end_goal=tree_manager.end_goal,
            branch_initialisation=tree_manager.branch_initialisation,
        )

    except Exception as e:
        logger.exception(f"Error in /default_models_user API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": config.model_dump()})


@router.post("/{user_id}/from_env")
async def environment_settings_user(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Set the settings for a user, to be used as default settings for all trees associated with that user.
    These will have been automatically set if none were provided during the initialisation of the user (initialise_user).
    Any changes to the api keys will be automatically propagated to the ClientManager for that user.

    Args:
        user_id (str): The user ID.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """

    try:
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]
        tree_manager.settings.set_api_keys_from_env()

        client_manager: ClientManager = user["client_manager"]
        await client_manager.set_keys_from_settings(tree_manager.settings)

        config = Config(
            settings=tree_manager.settings.to_json(),
            style=tree_manager.style,
            agent_description=tree_manager.agent_description,
            end_goal=tree_manager.end_goal,
            branch_initialisation=tree_manager.branch_initialisation,
        )

    except Exception as e:
        logger.exception(f"Error in /environment_settings_user API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": config.model_dump()})


@router.patch("/{user_id}")
async def change_config_user(
    user_id: str,
    data: Config,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Change the global config for a user.
    The `config` dictionary can contain any of the settings (dict), style, agent_description, end_goal and branch_initialisation.
    You do not need to specify all the settings, only the ones you wish to change.

    Any changes to the _settings_ will propagate to all trees associated with the user that were not initialised with their own settings,
    as well as update the ClientManager to use the new API keys.
    However, changes to the _style_, _agent_description_, _end_goal_ and _branch_initialisation_ will not be changed for existing trees.
    I.e., only changes to API keys and model choices will be changed for existing trees (Settings).

    Args:
        user_id (str): Required. The user ID.
        data (Config): Required. Containing the following attributes:
            settings(dict): Optional. A dictionary of config values to set, follows the same format as the Settings.configure() method.
            style (str): Optional. The writing style of the agent.
            agent_description (str): Optional. The description of the agent.
            end_goal (str): Optional. The end goal of the agent.
            branch_initialisation (str): Optional. The branch initialisation of the agent.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """
    logger.debug(f"/change_config_user API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Settings: {data.settings}")
    logger.debug(f"Style: {data.style}")
    logger.debug(f"Agent description: {data.agent_description}")
    logger.debug(f"End goal: {data.end_goal}")
    logger.debug(f"Branch initialisation: {data.branch_initialisation}")

    try:
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]

        if data.settings is not None:
            tree_manager.settings.configure(**data.settings)

        if data.style is not None:
            tree_manager.change_style(data.style)

        if data.agent_description is not None:
            tree_manager.change_agent_description(data.agent_description)

        if data.end_goal is not None:
            tree_manager.change_end_goal(data.end_goal)

        if data.branch_initialisation is not None:
            tree_manager.change_branch_initialisation(data.branch_initialisation)

        client_manager: ClientManager = user["client_manager"]
        await client_manager.set_keys_from_settings(tree_manager.settings)

        config = Config(
            settings=tree_manager.settings.to_json(),
            style=tree_manager.style,
            agent_description=tree_manager.agent_description,
            end_goal=tree_manager.end_goal,
            branch_initialisation=tree_manager.branch_initialisation,
        )

    except Exception as e:
        logger.exception(f"Error in /change_config_user API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": config.model_dump()})


@router.post("/{user_id}/save")
async def save_config(
    user_id: str,
    data: SaveConfigData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Save a config to the weaviate database.
    If the collection ELYSIA_CONFIG__ does not exist, it will be created.
    Uses the WCD_URL and WCD_API_KEY from /update_save_location to connect to the weaviate database.
    (These are initialised to the environment variables).

    Args:
        user_id (str): The user ID.
        data (SaveConfigData): A class with the following attributes:
            config_id (str): Required. The ID of the config to save. Can be a user submitted name.
            config (Config): Required. The config to save.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """

    logger.debug(f"/save_config API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Config ID: {data.config_id}")
    logger.debug(f"Config: {data.config}")

    try:
        # Retrieve the user info
        user = await user_manager.get_user_local(user_id)

        if "wcd_url" not in user or "wcd_api_key" not in user:
            raise Exception("WCD URL or API key not found.")

        if user["wcd_url"] is None or user["wcd_api_key"] is None:
            raise Exception(
                "No valid destination for config save location found. "
                "Please update the save location using the /update_save_location API."
            )

        # check if an auth key is set in the environment variables
        if "API_KEY_AUTH_KEY" in os.environ:
            auth_key = os.environ["API_KEY_AUTH_KEY"]
        else:  # create one
            auth_key = str(uuid4())
            set_key(".env", "API_KEY_AUTH_KEY", auth_key)

        # encode all api keys
        if "API_KEYS" in data.config.settings:
            for key in data.config.settings["API_KEYS"]:
                data.config.settings["API_KEYS"][key] = jwt.encode(
                    {"key": data.config.settings["API_KEYS"][key]},
                    auth_key,
                    algorithm="HS256",
                )
        if "WCD_API_KEY" in data.config.settings:
            data.config.settings["WCD_API_KEY"] = jwt.encode(
                {"key": data.config.settings["WCD_API_KEY"]},
                auth_key,
                algorithm="HS256",
            )

        async with weaviate.use_async_with_weaviate_cloud(
            cluster_url=user["wcd_url"],
            auth_credentials=Auth.api_key(user["wcd_api_key"]),
        ) as client:

            uuid = generate_uuid5(data.config_id)

            # Create a collection if it doesn't exist
            if await client.collections.exists("ELYSIA_CONFIG__"):
                collection = client.collections.get("ELYSIA_CONFIG__")
            else:
                collection = await client.collections.create(
                    "ELYSIA_CONFIG__",
                    vectorizer_config=wc.Configure.Vectorizer.none(),
                )

            config_item = data.config.model_dump()
            format_dict_to_serialisable(config_item)

            if await collection.data.exists(uuid=uuid):
                await collection.data.update(properties=config_item, uuid=uuid)
            else:
                await collection.data.insert(config_item, uuid=uuid)

    except Exception as e:
        logger.exception(f"Error in /save_config API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": config_item})


@router.post("/{user_id}/load")
async def load_config_user(
    user_id: str,
    data: LoadConfigData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Load a config from the weaviate database for a user.
    This will set the default config for all new trees associated with the user.
    It will also propagate the config to any trees set with default settings.
    It will not affect trees that have been set up with their own config.

    It will also update the ClientManager to use the new API keys.

    Args:
        user_id (str): The user ID.
        data (LoadConfigData): A class with the following attributes:
            config_id (str): The ID of the config to load. Can be found in the /config/list API.
            include_atlas (bool): Whether to include the atlas (style, agent_description, end_goal) in the config.
            include_branch_initialisation (bool): Whether to include the branch initialisation in the config.
            include_settings (bool): Whether to include the settings (LM choice, etc.) in the config.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """
    logger.debug(f"/load_config_user API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Config ID: {data.config_id}")

    try:
        # Retrieve the user info
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]
        settings: Settings = tree_manager.settings
        client_manager: ClientManager = user["client_manager"]

        if "wcd_url" not in user or "wcd_api_key" not in user:
            raise Exception("WCD URL or API key not found.")

        if user["wcd_url"] is None or user["wcd_api_key"] is None:
            raise Exception(
                "No valid destination for config load location found. "
                "Please update the save location using the /update_save_location API."
            )

        if "API_KEY_AUTH_KEY" in os.environ:
            auth_key = os.environ["API_KEY_AUTH_KEY"]
        else:
            raise Exception(
                "API key auth key not found, cannot decode API keys in config."
            )

        # Retrieve the config from the weaviate database
        async with weaviate.use_async_with_weaviate_cloud(
            cluster_url=user["wcd_url"],
            auth_credentials=Auth.api_key(user["wcd_api_key"]),
        ) as client:
            uuid = generate_uuid5(data.config_id)
            collection = client.collections.get("ELYSIA_CONFIG__")
            config_item = await collection.query.fetch_object_by_id(uuid=uuid)

        # Load the configs to the current user
        format_dict_to_serialisable(config_item.properties)

        # decode all api keys
        if (
            "settings" in config_item.properties
            and "API_KEYS" in config_item.properties["settings"]
        ):
            for key in config_item.properties["settings"]["API_KEYS"]:
                config_item.properties["settings"]["API_KEYS"][key] = jwt.decode(
                    config_item.properties["settings"]["API_KEYS"][key],
                    auth_key,
                    algorithms=["HS256"],
                )["key"]

        if (
            "settings" in config_item.properties
            and "WCD_API_KEY" in config_item.properties["settings"]
        ):
            config_item.properties["settings"]["WCD_API_KEY"] = jwt.decode(
                config_item.properties["settings"]["WCD_API_KEY"],
                auth_key,
                algorithms=["HS256"],
            )["key"]

        renamed_config = rename_keys(config_item.properties)

        if data.include_settings:
            settings.load_settings(renamed_config["settings"])
            await client_manager.set_keys_from_settings(settings)

        if data.include_atlas:
            tree_manager.change_style(renamed_config["style"])
            tree_manager.change_agent_description(renamed_config["agent_description"])
            tree_manager.change_end_goal(renamed_config["end_goal"])

        if data.include_branch_initialisation:
            tree_manager.change_branch_initialisation(
                renamed_config["branch_initialisation"]
            )

    except Exception as e:
        logger.exception(f"Error in /load_config API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": renamed_config})


@router.get("/{user_id}/list")
async def list_configs(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
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
    logger.debug(f"User ID: {user_id}")

    headers = {"Cache-Control": "no-cache"}

    try:
        user = await user_manager.get_user_local(user_id)

        if (
            "wcd_url" not in user
            or "wcd_api_key" not in user
            or user["wcd_url"] is None
            or user["wcd_api_key"] is None
        ):
            logger.warning(
                "In /list_configs API, "
                "no valid destination for config location found. "
                "Returning no error but an empty list of configs."
            )
            return JSONResponse(content={"error": "", "configs": []}, headers=headers)

        async with weaviate.use_async_with_weaviate_cloud(
            cluster_url=user["wcd_url"],
            auth_credentials=Auth.api_key(user["wcd_api_key"]),
        ) as client:

            collection = client.collections.get("ELYSIA_CONFIG__")

            len_collection = await collection.aggregate.over_all(total_count=True)
            len_collection = len_collection.total_count

            response = await collection.query.fetch_objects(limit=len_collection)
            configs = [obj.properties["config_id"] for obj in response.objects]

    except Exception as e:
        logger.exception(f"Error in /list_configs API")
        return JSONResponse(content={"error": str(e), "configs": []}, headers=headers)

    return JSONResponse(content={"error": "", "configs": configs}, headers=headers)
