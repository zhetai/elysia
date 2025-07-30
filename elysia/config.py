import os
import logging
import litellm
from litellm import check_valid_key
from rich.logging import RichHandler
from typing import Callable, Literal

import spacy
import random

from dotenv import load_dotenv
from dspy import LM

load_dotenv(override=True)

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    spacy.cli.download("en_core_web_sm")  # type: ignore
    nlp = spacy.load("en_core_web_sm")


class Settings:
    """
    Settings for Elysia.
    This class handles the configuration of various settings within Elysia.
    This includes:
    - The base and complex models to use.
    - The providers for the base and complex models.
    - The Weaviate cloud URL and API key.
    - The API keys for the providers.
    - The logger and logging level.

    The Settings object is set as at a default `settings` object if running Elysia as a package.
    You can import this via `elysia.config.settings`.
    This is initialised to default values from the environment variables.

    Or, you can create your own Settings object, and configure it as you wish.
    The Settings object can be passed to different Elysia classes and functions, such as `Tree` and `preprocess`.
    These will not use the global `settings` object, but instead use the Settings object you passed to them.
    """

    def __init__(self):
        """
        Initialize the settings for Elysia.
        These are all settings initialised to None, and should be set using the `configure` method.
        """
        # Default settings
        self.SETTINGS_ID = str(random.randint(100000000000000, 999999999999999))

        self.base_init()

    def base_init(self):
        self.BASE_MODEL: str | None = None
        self.BASE_PROVIDER: str | None = None
        self.COMPLEX_MODEL: str | None = None
        self.COMPLEX_PROVIDER: str | None = None

        self.WCD_URL: str = ""
        self.WCD_API_KEY: str = ""

        self.MODEL_API_BASE: str | None = None

        self.API_KEYS: dict[str, str] = {}

        self.logger = logging.getLogger("rich")
        self.logger.setLevel(logging.INFO)

        # Remove any existing handlers before adding a new one
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))

        self.logger.propagate = False
        self.LOGGING_LEVEL = "INFO"
        self.LOGGING_LEVEL_INT = 20

        # Experimental features
        self.USE_FEEDBACK = False
        self.BASE_USE_REASONING = True
        self.COMPLEX_USE_REASONING = True

    def setup_app_logger(self, logger: logging.Logger):
        """
        Override existing logger with the app-level logger.

        Args:
            logger (Logger): The logger to use.
        """
        self.logger = logger
        self.LOGGING_LEVEL_INT = logger.level
        inverted_logging_mapping = {
            v: k for k, v in logging.getLevelNamesMapping().items()
        }
        self.LOGGING_LEVEL = inverted_logging_mapping[self.LOGGING_LEVEL_INT]

    def configure_logger(
        self,
        level: Literal[
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"
        ] = "NOTSET",
    ):
        """
        Configure the logger with a RichHandler.

        Args:
            level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"]): The logging level to use.
        """
        self.logger.setLevel(level)
        self.LOGGING_LEVEL = level
        self.LOGGING_LEVEL_INT = logging.getLevelNamesMapping()[level]

    def set_api_key(self, api_key: str, api_key_name: str) -> None:
        self.API_KEYS[api_key_name] = api_key

    def get_api_key(self, api_key_name: str) -> str:
        return self.API_KEYS[api_key_name]

    def load_settings(self, settings: dict):
        for item in settings:
            setattr(self, item, settings[item])

        # self.logger = logging.getLogger("rich")
        # self.logger.setLevel(self.LOGGING_LEVEL)
        # for handler in self.logger.handlers[:]:
        #     self.logger.removeHandler(handler)
        # self.logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))

    @classmethod
    def from_smart_setup(cls):
        settings = cls()
        settings.set_from_env()
        settings.smart_setup()
        return settings

    @classmethod
    def from_env_vars(cls):
        settings = cls()
        settings.set_from_env()
        return settings

    def set_from_env(self):
        self.BASE_MODEL = os.getenv("BASE_MODEL", None)
        self.COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", None)
        self.BASE_PROVIDER = os.getenv("BASE_PROVIDER", None)
        self.COMPLEX_PROVIDER = os.getenv("COMPLEX_PROVIDER", None)
        self.MODEL_API_BASE = os.getenv("MODEL_API_BASE", None)
        self.LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "NOTSET")
        self.set_api_keys_from_env()

    def set_api_keys_from_env(self):
        self.WCD_URL = os.getenv("WCD_URL", "")
        self.WCD_API_KEY = os.getenv("WCD_API_KEY", "")

        self.API_KEYS = {
            env_var.lower(): os.getenv(env_var, "")
            for env_var in os.environ
            if (
                (
                    env_var.lower().endswith("api_key")
                    or env_var.lower().endswith("apikey")
                )
                and env_var.lower() != "wcd_api_key"
            )
        }
        for api_key in self.API_KEYS:
            self.set_api_key(self.API_KEYS[api_key], api_key)

    def smart_setup(self):

        # Check if the user has set the base model etc from the environment variables
        if os.getenv("BASE_MODEL", None):
            self.BASE_MODEL = os.getenv("BASE_MODEL")
        if os.getenv("COMPLEX_MODEL", None):
            self.COMPLEX_MODEL = os.getenv("COMPLEX_MODEL")
        if os.getenv("BASE_PROVIDER", None):
            self.BASE_PROVIDER = os.getenv("BASE_PROVIDER")
        if os.getenv("COMPLEX_PROVIDER", None):
            self.COMPLEX_PROVIDER = os.getenv("COMPLEX_PROVIDER")

        self.MODEL_API_BASE = os.getenv("MODEL_API_BASE", None)
        self.LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "NOTSET")

        self.set_api_keys_from_env()

        # check what API keys are available
        if (
            self.BASE_MODEL is None
            or self.COMPLEX_MODEL is None
            or self.BASE_PROVIDER is None
            or self.COMPLEX_PROVIDER is None
        ):
            if os.getenv("OPENROUTER_API_KEY", None):
                # use gemini 2.0 flash
                self.BASE_PROVIDER = "openrouter/google"
                self.COMPLEX_PROVIDER = "openrouter/google"
                self.BASE_MODEL = "gemini-2.0-flash-001"
                self.COMPLEX_MODEL = "gemini-2.5-flash"
            elif os.getenv("GEMINI_API_KEY", None):
                # use gemini 2.0 flash
                self.BASE_PROVIDER = "gemini"
                self.COMPLEX_PROVIDER = "gemini"
                self.BASE_MODEL = "gemini-2.0-flash-001"
                self.COMPLEX_MODEL = "gemini-2.5-flash"
            elif os.getenv("OPENAI_API_KEY", None):
                # use gpt family
                self.BASE_PROVIDER = "openai"
                self.COMPLEX_PROVIDER = "openai"
                self.BASE_MODEL = "gpt-4.1-mini"
                self.COMPLEX_MODEL = "gpt-4.1"
            elif os.getenv("ANTHROPIC_API_KEY", None):
                # use claude family
                self.BASE_PROVIDER = "anthropic"
                self.COMPLEX_PROVIDER = "anthropic"
                self.BASE_MODEL = "claude-3-5-haiku-latest"
                self.COMPLEX_MODEL = "claude-sonnet-4-0"

    def configure(
        self,
        replace: bool = False,
        **kwargs,
    ):
        """
        Configure the settings for Elysia for the current Settings object.

        Args:
            replace (bool): Whether to override the current settings with the new settings.
                When this is True, all existing settings are removed, and only the new settings are used.
                Defaults to False.
            **kwargs (str): One or more of the following:
                - base_model (str): The base model to use. e.g. "gpt-4o-mini"
                - complex_model (str): The complex model to use. e.g. "gpt-4o"
                - base_provider (str): The provider to use for base_model. E.g. "openai" or "openrouter/openai"
                - complex_provider (str): The provider to use for complex_model. E.g. "openai" or "openrouter/openai"
                - model_api_base (str): The API base to use.
                - wcd_url (str): The Weaviate cloud URL to use.
                - wcd_api_key (str): The Weaviate cloud API key to use.
                - logging_level (str): The logging level to use. e.g. "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
                - use_feedback (bool): EXPERIMENTAL. Whether to use feedback from previous runs of the tree.
                    If True, the tree will use TrainingUpdate objects that have been saved in previous runs of the decision tree.
                    These are implemented via few-shot examples for the decision node.
                    They are collected in the 'feedback' collection (ELYSIA_FEEDBACK__).
                    Relevant examples are retrieved from the collection based on searching the collection via the user's prompt.
                - base_use_reasoning (bool): Whether to use reasoning output for the base model.
                    If True, the model will generate reasoning before coming to its solution.
                - complex_use_reasoning (bool): Whether to use reasoning output for the complex model.
                    If True, the model will generate reasoning before coming to its solution.
                - Additional API keys to set. E.g. `openai_apikey="..."`, if this argument ends with `apikey` or `api_key`,
                    it will be added to the `API_KEYS` dictionary.

        """
        if replace:
            self.base_init()

        # convert all kwargs to lowercase for consistency
        kwargs = {kwarg.lower(): kwargs[kwarg] for kwarg in kwargs}

        if "base_model" in kwargs:
            if "base_provider" not in kwargs:
                raise ValueError(
                    "Provider must be specified if base_model is set. "
                    "E.g. `elysia.config.configure(base_model='gpt-4o-mini', base_provider='openai')`"
                )

            if kwargs["base_provider"] == "ollama" and (
                not kwargs["model_api_base"] or "MODEL_API_BASE" not in dir(self)
            ):
                raise ValueError(
                    "Using local models via ollama requires MODEL_API_BASE to be set. "
                    "This is likely to be http://localhost:11434. "
                    "e.g. `elysia.config.configure(model_api_base='http://localhost:11434')`"
                )

            self.BASE_MODEL = kwargs["base_model"]
            self.BASE_PROVIDER = kwargs["base_provider"]

            kwargs.pop("base_model")
            kwargs.pop("base_provider")

            # self.load_base_dspy_model()

        if "complex_model" in kwargs:
            if "complex_provider" not in kwargs:
                raise ValueError(
                    "Provider must be specified if complex_model is set. "
                    "E.g. `elysia.config.configure(complex_model='gpt-4o', complex_provider='openai')`"
                )

            if kwargs["complex_provider"] == "ollama" and (
                not kwargs["model_api_base"] or "MODEL_API_BASE" not in dir(self)
            ):
                raise ValueError(
                    "Using local models via ollama requires MODEL_API_BASE to be set. "
                    "This is likely to be http://localhost:11434. "
                    "e.g. `elysia.config.configure(model_api_base='http://localhost:11434')`"
                )

            self.COMPLEX_MODEL = kwargs["complex_model"]
            self.COMPLEX_PROVIDER = kwargs["complex_provider"]

            kwargs.pop("complex_model")
            kwargs.pop("complex_provider")

            # self.load_complex_dspy_model()

        if "model_api_base" in kwargs:
            self.MODEL_API_BASE = kwargs["model_api_base"]
            kwargs.pop("model_api_base")

        if "wcd_url" in kwargs:
            self.WCD_URL = kwargs["wcd_url"]
            kwargs.pop("wcd_url")

        if "wcd_api_key" in kwargs:
            self.WCD_API_KEY = kwargs["wcd_api_key"]
            kwargs.pop("wcd_api_key")

        if "logging_level" in kwargs or "logger_level" in kwargs:

            self.LOGGING_LEVEL = (
                kwargs["logging_level"]
                if "logging_level" in kwargs
                else kwargs["logger_level"]
            )
            self.LOGGING_LEVEL_INT = logging.getLevelNamesMapping()[self.LOGGING_LEVEL]
            self.logger.setLevel(self.LOGGING_LEVEL)
            if "logging_level" in kwargs:
                kwargs.pop("logging_level")
            if "logger_level" in kwargs:
                kwargs.pop("logger_level")
            if "logging_level_int" in kwargs:
                kwargs.pop("logging_level_int")
            if "logger_level_int" in kwargs:
                kwargs.pop("logger_level_int")

        if "logging_level_int" in kwargs or "logger_level_int" in kwargs:

            self.LOGGING_LEVEL_INT = (
                kwargs["logging_level_int"]
                if "logging_level_int" in kwargs
                else kwargs["logger_level_int"]
            )
            self.LOGGING_LEVEL = {
                v: k for k, v in logging.getLevelNamesMapping().items()
            }[self.LOGGING_LEVEL_INT]
            self.logger.setLevel(self.LOGGING_LEVEL)

        if "settings_id" in kwargs:
            self.SETTINGS_ID = kwargs["settings_id"]
            kwargs.pop("settings_id")

        if "use_feedback" in kwargs:
            self.USE_FEEDBACK = kwargs["use_feedback"]
            kwargs.pop("use_feedback")

        if "base_use_reasoning" in kwargs:
            self.BASE_USE_REASONING = kwargs["base_use_reasoning"]
            kwargs.pop("base_use_reasoning")

        if "complex_use_reasoning" in kwargs:
            self.COMPLEX_USE_REASONING = kwargs["complex_use_reasoning"]
            kwargs.pop("complex_use_reasoning")

        if "api_keys" in kwargs and isinstance(kwargs["api_keys"], dict):
            for key, value in kwargs["api_keys"].items():
                self.set_api_key(value, key)
            kwargs.pop("api_keys")

        # remainder of kwargs are API keys or saved there
        removed_kwargs = []
        for key, value in kwargs.items():
            if key.endswith("api_key"):
                self.set_api_key(value, key)
                removed_kwargs.append(key)

        for key in removed_kwargs:
            kwargs.pop(key)

        # remaining kwargs are not API keys
        if len(kwargs) > 0:
            self.logger.warning(
                "Unknown arguments to configure: " + ", ".join(kwargs.keys())
            )

    def __repr__(self) -> str:
        out = ""
        if "BASE_MODEL" in dir(self) and self.BASE_MODEL is not None:
            out += f"Base model: {self.BASE_MODEL}\n"
        else:
            out += "Base model: not set\n"
        if "COMPLEX_MODEL" in dir(self) and self.COMPLEX_MODEL is not None:
            out += f"Complex model: {self.COMPLEX_MODEL}\n"
        else:
            out += "Complex model: not set\n"
        if "BASE_PROVIDER" in dir(self) and self.BASE_PROVIDER is not None:
            out += f"Base provider: {self.BASE_PROVIDER}\n"
        else:
            out += "Base provider: not set\n"
        if "COMPLEX_PROVIDER" in dir(self) and self.COMPLEX_PROVIDER is not None:
            out += f"Complex provider: {self.COMPLEX_PROVIDER}\n"
        else:
            out += "Complex provider: not set\n"
        if "MODEL_API_BASE" in dir(self):
            out += f"Model API base: {self.MODEL_API_BASE}\n"
        else:
            out += "Model API base: not set\n"
        return out

    def to_json(self):
        return {
            item: getattr(self, item)
            for item in dir(self)
            if not item.startswith("_")
            and not isinstance(getattr(self, item), Callable)
            and item not in ["BASE_MODEL_LM", "COMPLEX_MODEL_LM", "logger"]
        }

    @classmethod
    def from_json(cls, json_data: dict):
        settings = cls()
        for item in json_data:
            if item not in ["logger"]:
                setattr(settings, item, json_data[item])

        settings.logger.setLevel(settings.LOGGING_LEVEL)

        return settings

    def check(self):
        return {
            "base_model": self.BASE_MODEL is not None and self.BASE_MODEL != "",
            "base_provider": self.BASE_PROVIDER is not None
            and self.BASE_PROVIDER != "",
            "complex_model": self.COMPLEX_MODEL is not None
            and self.COMPLEX_MODEL != "",
            "complex_provider": self.COMPLEX_PROVIDER is not None
            and self.COMPLEX_PROVIDER != "",
            "wcd_url": self.WCD_URL != "",
            "wcd_api_key": self.WCD_API_KEY != "",
        }


class ElysiaKeyManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def __enter__(self):

        # save existing keys
        self.existing_openrouter_key = litellm.openrouter_key
        self.existing_openai_key = litellm.openai_key
        self.existing_anthropic_key = litellm.anthropic_key
        self.existing_cohere_key = litellm.cohere_key
        self.existing_groq_key = litellm.groq_key
        self.existing_llama_api_key = litellm.llama_api_key
        self.existing_ollama_key = litellm.ollama_key
        self.existing_azure_key = litellm.azure_key

        # save existing env
        self.existing_openrouter_key_env = (
            os.environ["OPENROUTER_API_KEY"]
            if "OPENROUTER_API_KEY" in os.environ
            else None
        )
        self.existing_openai_key_env = (
            os.environ["OPENAI_API_KEY"] if "OPENAI_API_KEY" in os.environ else None
        )
        self.existing_anthropic_key_env = (
            os.environ["ANTHROPIC_API_KEY"]
            if "ANTHROPIC_API_KEY" in os.environ
            else None
        )
        self.existing_cohere_key_env = (
            os.environ["COHERE_API_KEY"] if "COHERE_API_KEY" in os.environ else None
        )
        self.existing_groq_key_env = (
            os.environ["GROQ_API_KEY"] if "GROQ_API_KEY" in os.environ else None
        )
        self.existing_llama_api_key_env = (
            os.environ["LLAMA_API_KEY"] if "LLAMA_API_KEY" in os.environ else None
        )
        self.existing_ollama_key_env = (
            os.environ["OLLAMA_API_KEY"] if "OLLAMA_API_KEY" in os.environ else None
        )
        self.existing_azure_key_env = (
            os.environ["AZURE_API_KEY"] if "AZURE_API_KEY" in os.environ else None
        )

        # api keys in lower case
        api_keys = {k.lower(): v for k, v in self.settings.API_KEYS.items()}

        # set new keys in litellm
        litellm.openrouter_key = (
            api_keys["openrouter_api_key"] if "openrouter_api_key" in api_keys else None
        )
        litellm.openai_key = (
            api_keys["openai_api_key"] if "openai_api_key" in api_keys else None
        )
        litellm.anthropic_key = (
            api_keys["anthropic_api_key"] if "anthropic_api_key" in api_keys else None
        )
        litellm.cohere_key = (
            api_keys["cohere_api_key"] if "cohere_api_key" in api_keys else None
        )
        litellm.groq_key = (
            api_keys["groq_api_key"] if "groq_api_key" in api_keys else None
        )
        litellm.llama_api_key = (
            api_keys["llama_api_key"] if "llama_api_key" in api_keys else None
        )
        litellm.ollama_key = (
            api_keys["ollama_api_key"] if "ollama_api_key" in api_keys else None
        )
        litellm.azure_key = (
            api_keys["azure_api_key"] if "azure_api_key" in api_keys else None
        )

        # set in env
        if self.existing_openrouter_key_env is not None:
            os.environ["OPENROUTER_API_KEY"] = ""
        if self.existing_openai_key_env is not None:
            os.environ["OPENAI_API_KEY"] = ""
        if self.existing_anthropic_key_env is not None:
            os.environ["ANTHROPIC_API_KEY"] = ""
        if self.existing_cohere_key_env is not None:
            os.environ["COHERE_API_KEY"] = ""
        if self.existing_groq_key_env is not None:
            os.environ["GROQ_API_KEY"] = ""
        if self.existing_llama_api_key_env is not None:
            os.environ["LLAMA_API_KEY"] = ""
        if self.existing_ollama_key_env is not None:
            os.environ["OLLAMA_API_KEY"] = ""
        if self.existing_azure_key_env is not None:
            os.environ["AZURE_API_KEY"] = ""

        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        litellm.openrouter_key = self.existing_openrouter_key
        litellm.openai_key = self.existing_openai_key
        litellm.anthropic_key = self.existing_anthropic_key
        litellm.cohere_key = self.existing_cohere_key
        litellm.groq_key = self.existing_groq_key
        litellm.llama_api_key = self.existing_llama_api_key
        litellm.ollama_key = self.existing_ollama_key
        litellm.azure_key = self.existing_azure_key

        # reset env
        if self.existing_openrouter_key_env is not None:
            os.environ["OPENROUTER_API_KEY"] = self.existing_openrouter_key_env
        if self.existing_openai_key_env is not None:
            os.environ["OPENAI_API_KEY"] = self.existing_openai_key_env
        if self.existing_anthropic_key_env is not None:
            os.environ["ANTHROPIC_API_KEY"] = self.existing_anthropic_key_env
        if self.existing_cohere_key_env is not None:
            os.environ["COHERE_API_KEY"] = self.existing_cohere_key_env
        if self.existing_groq_key_env is not None:
            os.environ["GROQ_API_KEY"] = self.existing_groq_key_env
        if self.existing_llama_api_key_env is not None:
            os.environ["LLAMA_API_KEY"] = self.existing_llama_api_key_env
        if self.existing_ollama_key_env is not None:
            os.environ["OLLAMA_API_KEY"] = self.existing_ollama_key_env
        if self.existing_azure_key_env is not None:
            os.environ["AZURE_API_KEY"] = self.existing_azure_key_env
        pass


def check_base_lm_settings(settings: Settings):
    if "BASE_MODEL" not in dir(settings) or settings.BASE_MODEL is None:
        raise ValueError(
            "No base model specified. "
            "Use `elysia.config.configure(base_model=..., base_provider=...)` to set a base model. "
            "E.g. `elysia.config.configure(base_model='gpt-4o-mini', base_provider='openai')`"
        )

    if "BASE_PROVIDER" not in dir(settings) or settings.BASE_PROVIDER is None:
        raise ValueError(
            "No base provider specified. "
            "Use `elysia.config.configure(base_model=..., base_provider=...)` to set a base model. "
            "E.g. `elysia.config.configure(base_model='gpt-4o-mini', base_provider='openai')`"
        )


def check_complex_lm_settings(settings: Settings):
    if "COMPLEX_MODEL" not in dir(settings) or settings.COMPLEX_MODEL is None:
        raise ValueError(
            "No complex model specified. "
            "Use `elysia.config.configure(complex_model=..., complex_provider=...)` to set a complex model. "
            "E.g. `elysia.config.configure(complex_model='gpt-4o', complex_provider='openai')`"
        )

    if "COMPLEX_PROVIDER" not in dir(settings) or settings.COMPLEX_PROVIDER is None:
        raise ValueError(
            "No complex provider specified. "
            "Use `elysia.config.configure(complex_model=..., complex_provider=...)` to set a complex model. "
            "E.g. `elysia.config.configure(complex_model='gpt-4o', complex_provider='openai')`"
        )


def load_base_lm(settings: Settings) -> LM:
    check_base_lm_settings(settings)

    return load_lm(
        settings.BASE_PROVIDER,
        settings.BASE_MODEL,
        settings.MODEL_API_BASE if "MODEL_API_BASE" in dir(settings) else None,
    )


def load_complex_lm(settings: Settings) -> LM:
    check_complex_lm_settings(settings)

    return load_lm(
        settings.COMPLEX_PROVIDER,
        settings.COMPLEX_MODEL,
        settings.MODEL_API_BASE if "MODEL_API_BASE" in dir(settings) else None,
    )


def load_lm(
    provider: str | None,
    lm_name: str | None,
    model_api_base: str | None = None,
) -> LM:

    if provider is None or lm_name is None:
        raise ValueError("Provider and LM name must be set")

    api_base = model_api_base if provider == "ollama" else None
    full_lm_name = f"{provider}/{lm_name}"

    if lm_name.startswith("o1") or lm_name.startswith("o3"):
        return LM(
            model=full_lm_name, api_base=api_base, max_tokens=8000, temperature=1.0
        )

    return LM(model=full_lm_name, api_base=api_base, max_tokens=8000)


# global settings that should never be used by the frontend
# but used when using Elysia as a package
settings = Settings()
settings.smart_setup()

DEFAULT_SETTINGS = Settings()
DEFAULT_SETTINGS.smart_setup()


def reset_settings() -> None:
    settings.base_init()
    settings.smart_setup()


def configure(**kwargs) -> None:
    """
    Configure the settings for Elysia for the global settings object.

    Args:
        **kwargs (str): One or more of the following:
            - base_model (str): The base model to use. e.g. "gpt-4o-mini"
            - complex_model (str): The complex model to use. e.g. "gpt-4o"
            - base_provider (str): The provider to use for base_model. E.g. "openai" or "openrouter/openai"
            - complex_provider (str): The provider to use for complex_model. E.g. "openai" or "openrouter/openai"
            - model_api_base (str): The API base to use.
            - wcd_url (str): The Weaviate cloud URL to use.
            - wcd_api_key (str): The Weaviate cloud API key to use.
            - logging_level (str): The logging level to use. e.g. "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            - use_feedback (bool): EXPERIMENTAL. Whether to use feedback from previous runs of the tree.
                If True, the tree will use TrainingUpdate objects that have been saved in previous runs of the decision tree.
                These are implemented via few-shot examples for the decision node.
                They are collected in the 'feedback' collection (ELYSIA_FEEDBACK__).
                Relevant examples are retrieved from the collection based on searching the collection via the user's prompt.
            - Additional API keys to set. E.g. `openai_apikey="..."`, if this argument ends with `apikey` or `api_key`,
                it will be added to the `API_KEYS` dictionary.

    """
    settings.configure(**kwargs)
