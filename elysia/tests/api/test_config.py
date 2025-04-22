import unittest

from fastapi.responses import JSONResponse

import asyncio
import json

from elysia.api.api_types import (
    GetObjectData,
    ViewPaginatedCollectionData,
    UserCollectionsData,
    CollectionMetadataData,
)
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.config import (
    save_config,
    load_config,
    change_config,
    default_config,
)

from elysia.api.core.log import logger, set_log_level

set_log_level("CRITICAL")

# TODO: tests once load/save config works for the weaviate collection
