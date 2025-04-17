import logging
from rich.logging import RichHandler
import litellm
import dspy

FORMAT = "%(message)s"
logging.basicConfig(
    level="DEBUG",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.INFO)
logging.getLogger("fastapi").setLevel(logging.INFO)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("starlette").setLevel(logging.INFO)
logging.getLogger("grpc").setLevel(logging.INFO)
logging.getLogger("grpc._channel").setLevel(logging.WARNING)
logging.getLogger("dspy").setLevel(logging.WARNING)
logging.getLogger("dspy").propagate = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").propagate = False
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").propagate = False
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").propagate = False


def get_loggers():
    """Get a list of all configured loggers and their levels."""
    loggers = []
    for name in logging.root.manager.loggerDict:
        logger_obj = logging.getLogger(name)
        if isinstance(logger_obj, logging.Logger):
            loggers.append(
                {
                    "name": name,
                    "level": logging.getLevelName(logger_obj.level),
                    "propagate": logger_obj.propagate,
                }
            )
    return loggers


dspy.disable_litellm_logging()
dspy.disable_logging()
litellm.suppress_debug_info = True

# Your application logger
logger = logging.getLogger("rich")
