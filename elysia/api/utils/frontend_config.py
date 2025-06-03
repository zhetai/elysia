from elysia.util.client import ClientManager
from logging import Logger
import datetime
import os


class FrontendConfig:

    def __init__(self, logger: Logger, client_timeout: datetime.timedelta):

        self.logger = logger
        self.client_timeout = client_timeout

        self.save_trees_to_weaviate: bool = True
        self.save_location_wcd_url: str | None = os.getenv("WCD_URL", None)
        self.save_location_wcd_api_key: str | None = os.getenv("WCD_API_KEY", None)
        self.save_location_client_manager: ClientManager = ClientManager(
            wcd_url=self.save_location_wcd_url,
            wcd_api_key=self.save_location_wcd_api_key,
            logger=logger,
            client_timeout=client_timeout,
        )

    def update_save_location(self, wcd_url: str, wcd_api_key: str):
        self.save_location_wcd_url = wcd_url
        self.save_location_wcd_api_key = wcd_api_key

    def update_save_trees_to_weaviate(self, save_trees_to_weaviate: bool):
        self.save_trees_to_weaviate = save_trees_to_weaviate

    def get_save_location(self):
        return self.save_location_wcd_url, self.save_location_wcd_api_key

    def configure(self, **kwargs):

        if "save_location_wcd_url" in kwargs:
            self.save_location_wcd_url = kwargs["save_location_wcd_url"]
        if "save_location_wcd_api_key" in kwargs:
            self.save_location_wcd_api_key = kwargs["save_location_wcd_api_key"]
        if "save_trees_to_weaviate" in kwargs:
            self.save_trees_to_weaviate = kwargs["save_trees_to_weaviate"]

        self.save_location_client_manager = ClientManager(
            wcd_url=self.save_location_wcd_url,
            wcd_api_key=self.save_location_wcd_api_key,
            logger=self.logger,
            client_timeout=self.client_timeout,
        )
