import os
from cryptography.fernet import Fernet

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, set_key

load_dotenv(override=True)

from elysia.api.api_types import (
    SaveConfigUserData,
    UpdateFrontendConfigData,
)
from elysia.api.utils.config import Config
from elysia.api.core.log import logger
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.util.client import ClientManager
from elysia.util.parsing import format_dict_to_serialisable, format_datetime
from elysia.config import Settings
from elysia.api.services.tree import TreeManager
from elysia.api.utils.config import FrontendConfig

import weaviate.classes.config as wc
from weaviate.util import generate_uuid5
from weaviate.classes.query import MetadataQuery, Sort, Filter
from uuid import uuid4


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


@router.patch("/frontend_config/{user_id}")
async def update_frontend_config(
    user_id: str,
    data: UpdateFrontendConfigData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Update a user's frontend config, e.g. whether to save trees to weaviate or not, and the save location.
    """
    logger.debug(f"/update_frontend_config API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Config: {data.config}")

    try:
        await user_manager.update_frontend_config(user_id, data.config)
        return JSONResponse(content={"error": ""})
    except Exception as e:

        if "Could not connect to Weaviate" in str(e):
            return JSONResponse(
                content={
                    "error": "Could not connect to Weaviate. Double check that the WCD_URL and WCD_API_KEY are correct."
                }
            )
        else:
            logger.exception(f"Error in /update_frontend_config API")
            return JSONResponse(content={"error": str(e)})


@router.get("/{user_id}")
async def get_current_user_config(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Get the currnet config for a single user (in memory).
    """
    logger.debug(f"/get_current_user_config API request received")
    logger.debug(f"User ID: {user_id}")

    headers = {"Cache-Control": "no-cache"}

    try:
        user = await user_manager.get_user_local(user_id)
        config = user["tree_manager"].config.to_json()
        frontend_config = user["frontend_config"].config

    except Exception as e:
        logger.exception(f"Error in /get_current_user_config API")
        return JSONResponse(
            content={"error": str(e), "config": {}, "frontend_config": {}},
            headers=headers,
        )

    return JSONResponse(
        content={
            "error": "",
            "config": config,
            "frontend_config": frontend_config,
        },
        headers=headers,
    )


@router.post("/{user_id}/new")
async def new_user_config(
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
            config (dict): A dictionary of the new config values.
    """
    logger.debug(f"/new_user_config API request received")
    logger.debug(f"User ID: {user_id}")

    try:
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]

        settings = Settings()
        settings.smart_setup()

        tree_manager.settings.load_settings(settings.to_json())
        tree_manager.config.settings = settings.to_json()
        tree_manager.config.style = "Informative, polite and friendly."
        tree_manager.config.agent_description = "You search and query Weaviate to satisfy the user's query, providing a concise summary of the results."
        tree_manager.config.end_goal = (
            "You have satisfied the user's query, and provided a concise summary of the results. "
            "Or, you have exhausted all options available, or asked the user for clarification."
        )
        tree_manager.config.branch_initialisation = "one_branch"
        tree_manager.config.name = "New Config"
        tree_manager.config.id = str(uuid4())

    except Exception as e:
        logger.exception(f"Error in /new_user_config API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": tree_manager.config.to_json()})


@router.post("/{user_id}/{config_id}")
async def save_config_user(
    user_id: str,
    config_id: str,
    data: SaveConfigUserData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Change the global config for a user.
    The `config` dictionary can contain any of the `settings` (dict), style, agent_description, end_goal and branch_initialisation.
    You do not need to specify all the `settings`, only the ones you wish to change.

    Any changes to the _settings_ will propagate to all trees associated with the user that were not initialised with their own settings,
    as well as update the ClientManager to use the new API keys.
    However, changes to the _style_, _agent_description_, _end_goal_ and _branch_initialisation_ will not be changed for existing trees.
    I.e., only changes to API keys and model choices will be changed for existing trees (Settings).

    Additionally saves a config to the weaviate database.
    If the collection ELYSIA_CONFIG__ does not exist, it will be created.
    Uses the WCD_URL and WCD_API_KEY from /update_save_location to connect to the weaviate database.
    (These are initialised to the environment variables).

    Args:
        user_id (str): Required. The user ID.
        data (SaveConfigUserData): Required. Containing the following attributes:
            name (str): Optional. The name of the config.
            settings (dict): Optional. A dictionary of config values to set, follows the same format as the Settings.configure() method.
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
    logger.debug(f"Config ID: {config_id}")
    logger.debug(f"Name: {data.name}")
    logger.debug(f"Settings: {data.settings}")
    logger.debug(f"Style: {data.style}")
    logger.debug(f"Agent description: {data.agent_description}")
    logger.debug(f"End goal: {data.end_goal}")
    logger.debug(f"Branch initialisation: {data.branch_initialisation}")

    # Save the config in memory for this user (Apply step)
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
        await client_manager.reset_keys(
            wcd_url=tree_manager.settings.WCD_URL,
            wcd_api_key=tree_manager.settings.WCD_API_KEY,
            api_keys=tree_manager.settings.API_KEYS,
        )

    except Exception as e:
        logger.exception(f"Error in /save_config_user API during in-memory update")
        return JSONResponse(content={"error": str(e), "config": {}})

    if not user["frontend_config"].config["save_configs_to_weaviate"]:
        return JSONResponse(
            content={
                "error": "",
                "config": tree_manager.config.to_json(),
                "frontend_config": user["frontend_config"].config,
            }
        )

    # Save the config to the weaviate database
    try:

        # Retrieve the user info
        user = await user_manager.get_user_local(user_id)
        settings: Settings = user["tree_manager"].settings
        settings_dict = settings.to_json()

        # extract style, agent_description, end_goal, branch_initialisation from user
        style = user["tree_manager"].config.style
        agent_description = user["tree_manager"].config.agent_description
        end_goal = user["tree_manager"].config.end_goal
        branch_initialisation = user["tree_manager"].config.branch_initialisation

        # Check if the user has a valid save location
        if (
            "frontend_config" not in user
            or "save_location_wcd_url" not in user["frontend_config"].__dict__
            or "save_location_wcd_api_key" not in user["frontend_config"].__dict__
        ):
            raise Exception("WCD URL or API key not found.")

        if (
            user["frontend_config"].save_location_wcd_url is None
            or user["frontend_config"].save_location_wcd_api_key is None
        ):
            raise Exception(
                "No valid destination for config save location found. "
                "Please update the save location using the /update_save_location API."
            )

        # Check if an auth key is set in the environment variables
        if "FERNET_KEY" in os.environ:
            auth_key = os.environ["FERNET_KEY"]
        else:  # create one
            auth_key = Fernet.generate_key().decode("utf-8")
            set_key(".env", "FERNET_KEY", auth_key)

        # Encode all api keys
        f = Fernet(auth_key.encode("utf-8"))
        if "API_KEYS" in settings_dict:
            for key in settings_dict["API_KEYS"]:
                settings_dict["API_KEYS"][key] = f.encrypt(
                    settings_dict["API_KEYS"][key].encode("utf-8")
                ).decode("utf-8")

        if "WCD_API_KEY" in settings_dict:
            settings_dict["WCD_API_KEY"] = f.encrypt(
                settings_dict["WCD_API_KEY"].encode("utf-8")
            ).decode("utf-8")

        # Connect to the weaviate database
        async with user[
            "frontend_config"
        ].save_location_client_manager.connect_to_async_client() as client:

            uuid = generate_uuid5(config_id)

            # Create a collection if it doesn't exist
            if await client.collections.exists("ELYSIA_CONFIG__"):
                collection = client.collections.get("ELYSIA_CONFIG__")
            else:
                collection = await client.collections.create(
                    "ELYSIA_CONFIG__",
                    vectorizer_config=wc.Configure.Vectorizer.none(),
                    inverted_index_config=wc.Configure.inverted_index(
                        index_timestamps=True
                    ),
                )

            format_dict_to_serialisable(settings_dict)

            config_item = {
                "name": data.name,
                "settings": settings_dict,
                "style": style,
                "agent_description": agent_description,
                "end_goal": end_goal,
                "branch_initialisation": branch_initialisation,
                "frontend_config": user["frontend_config"].config,
                "config_id": config_id,
                "user_id": user_id,
            }

            if await collection.data.exists(uuid=uuid):
                await collection.data.update(properties=config_item, uuid=uuid)
            else:
                await collection.data.insert(config_item, uuid=uuid)

    except Exception as e:
        logger.exception(f"Error in /save_config_user API during weaviate save")
        return JSONResponse(
            content={"error": str(e), "config": {}, "frontend_config": {}}
        )

    return JSONResponse(
        content={
            "error": "",
            "config": tree_manager.config.to_json(),
            "frontend_config": user["frontend_config"].config,
        }
    )


@router.post("/{user_id}/{config_id}/load")
async def load_config_user(
    user_id: str,
    config_id: str,
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
        config_id (str): The config ID you want to load.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """
    logger.debug(f"/load_config_user API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Config ID: {config_id}")

    try:
        # Retrieve the user info
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]
        settings: Settings = tree_manager.settings
        client_manager: ClientManager = user["client_manager"]
        frontend_config: FrontendConfig = user["frontend_config"]

        # check if the user has a valid save location
        if (
            "frontend_config" not in user
            or "save_location_wcd_url" not in user["frontend_config"].__dict__
            or "save_location_wcd_api_key" not in user["frontend_config"].__dict__
        ):
            raise Exception("WCD URL or API key not found.")

        if (
            user["frontend_config"].save_location_wcd_url is None
            or user["frontend_config"].save_location_wcd_api_key is None
        ):
            raise Exception(
                "No valid destination for config load location found. "
                "Please update the save location using the /update_save_location API."
            )

        # check if the auth key is set in the environment variables
        if "FERNET_KEY" in os.environ:
            auth_key = os.environ["FERNET_KEY"]
        else:
            raise Exception(
                "API key auth key not found, cannot decode API keys in config."
            )

        # Retrieve the config from the weaviate database
        async with user[
            "frontend_config"
        ].save_location_client_manager.connect_to_async_client() as client:
            uuid = generate_uuid5(config_id)
            collection = client.collections.get("ELYSIA_CONFIG__")
            config_item = await collection.query.fetch_object_by_id(uuid=uuid)

        # Load the configs to the current user
        format_dict_to_serialisable(config_item.properties)

        # decode all api keys
        f = Fernet(auth_key.encode("utf-8"))

        if (
            "settings" in config_item.properties
            and "API_KEYS" in config_item.properties["settings"]
        ):
            for key in config_item.properties["settings"]["API_KEYS"]:
                config_item.properties["settings"]["API_KEYS"][key] = f.decrypt(
                    config_item.properties["settings"]["API_KEYS"][key].encode("utf-8")
                ).decode("utf-8")

        if (
            "settings" in config_item.properties
            and "WCD_API_KEY" in config_item.properties["settings"]
        ):
            config_item.properties["settings"]["WCD_API_KEY"] = f.decrypt(
                config_item.properties["settings"]["WCD_API_KEY"].encode("utf-8")
            ).decode("utf-8")

        # Rename the keys to the correct format
        renamed_config = rename_keys(config_item.properties)

        # Load the settings
        settings.load_settings(renamed_config["settings"])

        # Restart the client manager with new WCD_URL and WCD_API_KEY etc.
        await client_manager.reset_keys(
            wcd_url=settings.WCD_URL,
            wcd_api_key=settings.WCD_API_KEY,
            api_keys=settings.API_KEYS,
        )

        # Update the style, agent_description, end_goal, branch_initialisation in the tree manager
        tree_manager.change_style(renamed_config["style"])
        tree_manager.change_agent_description(renamed_config["agent_description"])
        tree_manager.change_end_goal(renamed_config["end_goal"])

        # tree_manager.change_branch_initialisation(
        #     renamed_config["branch_initialisation"]
        # )

        # Update the frontend config
        frontend_config.config = renamed_config["frontend_config"]

    except Exception as e:
        logger.exception(f"Error in /load_config API")
        return JSONResponse(
            content={"error": str(e), "config": {}, "frontend_config": {}}
        )

    return JSONResponse(
        content={
            "error": "",
            "config": tree_manager.config.to_json(),
            "frontend_config": frontend_config.config,
        }
    )


@router.delete("/{user_id}/{config_id}")
async def delete_config(
    user_id: str,
    config_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Delete a config from the weaviate database.
    """
    logger.debug(f"/delete_config API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Config ID: {config_id}")

    try:
        user = await user_manager.get_user_local(user_id)
        client_manager: ClientManager = user["client_manager"]
        async with client_manager.connect_to_async_client() as client:
            collection = client.collections.get("ELYSIA_CONFIG__")
            uuid = generate_uuid5(config_id)
            await collection.data.delete_by_id(uuid)

    except Exception as e:
        logger.exception(f"Error in /delete_config API")
        return JSONResponse(content={"error": str(e)})

    return JSONResponse(content={"error": ""})


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
            "frontend_config" not in user
            or "save_location_wcd_url" not in user["frontend_config"].__dict__
            or "save_location_wcd_api_key" not in user["frontend_config"].__dict__
            or user["frontend_config"].save_location_wcd_url is None
            or user["frontend_config"].save_location_wcd_api_key is None
        ):
            logger.warning(
                "In /list_configs API, "
                "no valid destination for config location found. "
                "Returning no error but an empty list of configs."
            )
            return JSONResponse(content={"error": "", "configs": []}, headers=headers)

        if user_id is not None:
            user_id_filter = Filter.by_property("user_id").equal(user_id)
        else:
            user_id_filter = None

        async with user[
            "frontend_config"
        ].save_location_client_manager.connect_to_async_client() as client:

            collection = client.collections.get("ELYSIA_CONFIG__")

            len_collection = await collection.aggregate.over_all(total_count=True)
            len_collection = len_collection.total_count

            response = await collection.query.fetch_objects(
                limit=len_collection,
                sort=Sort.by_update_time(ascending=False),
                return_metadata=MetadataQuery(last_update_time=True),
                filters=user_id_filter,
            )

            configs = [
                {
                    "config_id": obj.properties["config_id"],
                    "name": obj.properties["name"],
                    "last_update_time": format_datetime(obj.metadata.last_update_time),
                }
                for obj in response.objects
            ]

    except Exception as e:
        logger.exception(f"Error in /list_configs API")
        return JSONResponse(content={"error": str(e), "configs": []}, headers=headers)

    return JSONResponse(content={"error": "", "configs": configs}, headers=headers)
