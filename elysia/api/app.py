import gc
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from elysia.api.core.logging import logger
from elysia.api.dependencies.common import get_user_manager

# Middleware
from elysia.api.middleware.error_handlers import register_error_handlers

# Routes
from elysia.api.routes import (
    collections,
    config,
    feedback,
    processor,
    query,
    tree,
    utils,
)

# Services
from elysia.api.services.user import UserManager
from elysia.api.utils.resources import print_resources


async def perform_garbage_collection():
    gc.collect(generation=2)


async def check_timeouts():
    user_manager = get_user_manager()
    await user_manager.check_all_trees_timeout()


async def output_resources():
    user_manager = get_user_manager()
    await print_resources(user_manager, save_to_file=True)


async def check_restart_clients():
    user_manager = get_user_manager()
    await user_manager.check_restart_clients()


@asynccontextmanager
async def lifespan(app: FastAPI):
    user_manager = get_user_manager()

    scheduler = AsyncIOScheduler()

    # use prime numbers for intervals so they don't overlap
    scheduler.add_job(perform_garbage_collection, "interval", seconds=23)
    scheduler.add_job(check_timeouts, "interval", seconds=29)
    scheduler.add_job(check_restart_clients, "interval", seconds=31)
    scheduler.add_job(output_resources, "interval", seconds=1103)

    scheduler.start()
    yield
    scheduler.shutdown()

    await user_manager.close_all_clients()


# Create FastAPI app instance
app = FastAPI(title="Elysia API", version="0.3.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register error handlers
register_error_handlers(app)

# Include routers
app.include_router(utils.router, prefix="/api", tags=["utilities"])
app.include_router(collections.router, prefix="/api", tags=["collections"])
app.include_router(tree.router, prefix="/api", tags=["tree"])
app.include_router(query.router, prefix="/ws", tags=["websockets"])
app.include_router(processor.router, prefix="/ws", tags=["websockets"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])
app.include_router(config.router, prefix="/api", tags=["config"])


# Health check endpoint (kept in main app.py due to its simplicity)
@app.get("/api/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {"status": "healthy"}
