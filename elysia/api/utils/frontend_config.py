from elysia.util.client import ClientManager
from logging import Logger
import datetime
import os


class FrontendConfig:
    """
    Handles frontend-specific config options.
    Anything that does not fit into the Elysia backend/python package, but relevant to the app is here.
    """

    def __init__(self, logger: Logger):
        """
        Args:
            logger (Logger): The logger to use for logging.

        Config options:
            - save_trees_to_weaviate (bool): Whether to save trees to Weaviate.
            - save_config_to_weaviate (bool): Whether to save config to Weaviate.
            - tree_timeout (int): Optional.
                The length of time in minutes a tree can be idle before being timed out.
                Defaults to 10 minutes or the value of the TREE_TIMEOUT environment variable (integer, minutes).
                If an integer is provided, it is interpreted as the number of minutes.
            - client_timeout (int): Optional.
                The length of time in minutes a client can be idle before being timed out.
                Defaults to 3 minutes or the value of the CLIENT_TIMEOUT environment variable (integer, minutes).
                If an integer is provided, it is interpreted as the number of minutes.
        """

        self.logger = logger

        client_timeout = int(os.getenv("CLIENT_TIMEOUT", 3))
        tree_timeout = int(os.getenv("TREE_TIMEOUT", 10))

        self.config: dict = {
            "save_trees_to_weaviate": True,
            "save_configs_to_weaviate": True,
            "client_timeout": client_timeout,
            "tree_timeout": tree_timeout,
        }
        self.save_location_wcd_url: str | None = os.getenv("WCD_URL", None)
        self.save_location_wcd_api_key: str | None = os.getenv("WCD_API_KEY", None)
        self.save_location_client_manager: ClientManager = ClientManager(
            wcd_url=self.save_location_wcd_url,
            wcd_api_key=self.save_location_wcd_api_key,
            logger=logger,
            client_timeout=self.config["client_timeout"],
        )

    def update_save_location(self, wcd_url: str, wcd_api_key: str):
        self.save_location_wcd_url = wcd_url
        self.save_location_wcd_api_key = wcd_api_key

    def get_save_location(self):
        return self.save_location_wcd_url, self.save_location_wcd_api_key

    async def configure(self, **kwargs):

        reload_client_manager = False
        if "save_location_wcd_url" in kwargs:
            self.save_location_wcd_url = kwargs["save_location_wcd_url"]
            reload_client_manager = True
        if "save_location_wcd_api_key" in kwargs:
            self.save_location_wcd_api_key = kwargs["save_location_wcd_api_key"]
            reload_client_manager = True
        if "save_trees_to_weaviate" in kwargs:
            self.config["save_trees_to_weaviate"] = kwargs["save_trees_to_weaviate"]
        if "save_configs_to_weaviate" in kwargs:
            self.config["save_configs_to_weaviate"] = kwargs["save_configs_to_weaviate"]

        if reload_client_manager:
            await self.save_location_client_manager.reset_keys(
                wcd_url=self.save_location_wcd_url,
                wcd_api_key=self.save_location_wcd_api_key,
            )
