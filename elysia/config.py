import json
import os
from typing import Callable

import spacy
from dotenv import load_dotenv

load_dotenv()

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class Settings:
    CLIENT_TIMEOUT: int = os.getenv("CLIENT_TIMEOUT", 3)
    VERSION = "0.2.0"

    # Other API keys (for vectorisers etc.)
    API_KEYS = {}

    def set_api_key(self, api_key: str, api_key_name: str):
        os.environ[api_key_name] = api_key
        self.API_KEYS[api_key_name] = api_key

    def get_api_key(self, api_key_name: str):
        return self.API_KEYS[api_key_name]

    def export_config(self, filepath: str = "elysia_config.json"):
        with open(filepath, "w") as f:
            json.dump(
                {
                    item: getattr(self, item)
                    for item in dir(self)
                    if not item.startswith("_")
                    and not isinstance(getattr(self, item), Callable)
                },
                f,
                indent=4,
            )

    def load_config(self, filepath: str = "elysia_config.json"):
        with open(filepath, "r") as f:
            config = json.load(f)
        for item in config:
            setattr(self, item, config[item])

    @classmethod
    def from_config(cls, config: dict):
        settings = cls()
        for item in config:
            setattr(settings, item, config[item])
        return settings

    @classmethod
    def from_default(cls):
        settings = cls()
        settings.default_config()
        return settings

    def set_api_keys_from_env(self):
        self.WCD_URL = os.getenv("WCD_URL", "")
        self.WCD_API_KEY = os.getenv("WCD_API_KEY", "")

        self.API_KEYS = {
            env_var.lower(): os.getenv(env_var)
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

        if (
            not any(api_key.startswith("openrouter") for api_key in self.API_KEYS)
            and "OPENROUTER" in dir(self)
            and self.OPENROUTER
        ):
            raise Exception(
                "Either `openrouter` is True, the default model is set, but no openrouter API keys are set. "
                "Set `OPENROUTER_API_KEY` in your environment variables, or choose a different default model."
            )

    def default_config(self):
        self.BASE_MODEL = os.getenv("BASE_MODEL", "gemini-2.0-flash-001")
        self.COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", "gemini-2.0-flash-001")
        self.BASE_PROVIDER = os.getenv("BASE_PROVIDER", "google")
        self.COMPLEX_PROVIDER = os.getenv("COMPLEX_PROVIDER", "google")
        self.MODEL_API_BASE = os.getenv("MODEL_API_BASE", None)
        self.OPENROUTER = os.getenv("OPENROUTER", "1") == "1"
        self.LOCAL = os.getenv("LOCAL", "0") == "1"
        self.WCD_URL = os.getenv("WCD_URL")
        self.WCD_API_KEY = os.getenv("WCD_API_KEY")
        self.set_api_keys_from_env()

    def configure(
        self,
        **kwargs,
    ):
        """
        Configure the settings for the Elysia API.

        Args:
            base_model: The base model to use (str). e.g. "gpt-4o-mini"
            complex_model: The complex model to use (str). e.g. "gpt-4o"
            base_provider: The provider to use for base_model (str). E.g. "openai"
            complex_provider: The provider to use for complex_model (str). E.g. "openai"
            openrouter: Whether to use openrouter (True/False).
            model_api_base: The API base to use (str).
            wcd_url: The Weaviate cloud URL to use (str).
            wcd_api_key: The Weaviate cloud API key to use (str).
            **kwargs: Additional API keys to set. E.g. `openai_apikey="..."`
        """

        if "base_model" in kwargs:
            if "base_provider" not in kwargs:
                raise ValueError(
                    "Provider must be specified if base_model is set. "
                    "E.g. `elysia.config.configure(base_model='gpt-4o-mini', base_provider='openai')`"
                )
            if "openrouter" not in kwargs:
                kwargs["openrouter"] = False

            if kwargs["base_provider"] == "ollama" and (
                not kwargs["model_api_base"] or "MODEL_API_BASE" not in self
            ):
                raise ValueError(
                    "Using local models via ollama requires MODEL_API_BASE to be set. "
                    "This is likely to be http://localhost:11434. "
                    "e.g. `elysia.config.configure(model_api_base='http://localhost:11434')`"
                )

            self.BASE_MODEL = kwargs["base_model"]
            self.BASE_PROVIDER = kwargs["base_provider"]
            self.OPENROUTER = kwargs["openrouter"]

            kwargs.pop("base_model")
            kwargs.pop("base_provider")
            kwargs.pop("openrouter")

        if "complex_model" in kwargs:
            if "complex_provider" not in kwargs:
                raise ValueError(
                    "Provider must be specified if complex_model is set. "
                    "E.g. `elysia.config.configure(complex_model='gpt-4o', complex_provider='openai')`"
                )
            if "openrouter" not in kwargs:
                kwargs["openrouter"] = False

            if kwargs["complex_provider"] == "ollama" and (
                not kwargs["model_api_base"] or "MODEL_API_BASE" not in self
            ):
                raise ValueError(
                    "Using local models via ollama requires MODEL_API_BASE to be set. "
                    "This is likely to be http://localhost:11434. "
                    "e.g. `elysia.config.configure(model_api_base='http://localhost:11434')`"
                )

            self.COMPLEX_MODEL = kwargs["complex_model"]
            self.COMPLEX_PROVIDER = kwargs["complex_provider"]
            self.OPENROUTER = kwargs["openrouter"]

            kwargs.pop("complex_model")
            kwargs.pop("complex_provider")
            kwargs.pop("openrouter")

        if "model_api_base" in kwargs:
            self.MODEL_API_BASE = kwargs["model_api_base"]
            kwargs.pop("model_api_base")

        if "wcd_url" in kwargs:
            self.WCD_URL = kwargs["wcd_url"]
            kwargs.pop("wcd_url")

        if "wcd_api_key" in kwargs:
            self.WCD_API_KEY = kwargs["wcd_api_key"]
            kwargs.pop("wcd_api_key")

        # remainder of kwargs are API keys or saved there
        for key, value in kwargs.items():
            self.set_api_key(value, key)

    def __repr__(self) -> str:
        if "BASE_MODEL" in self and self.BASE_MODEL is not None:
            print("Base model: ", self.BASE_MODEL)
        else:
            print("Base model: not set")
        if "COMPLEX_MODEL" in self and self.COMPLEX_MODEL is not None:
            print("Complex model: ", self.COMPLEX_MODEL)
        else:
            print("Complex model: not set")
        if "BASE_PROVIDER" in self and self.BASE_PROVIDER is not None:
            print("Base provider: ", self.BASE_PROVIDER)
        else:
            print("Base provider: not set")
        if "COMPLEX_PROVIDER" in self and self.COMPLEX_PROVIDER is not None:
            print("Complex provider: ", self.COMPLEX_PROVIDER)
        else:
            print("Complex provider: not set")
        if "OPENROUTER" in self:
            print("Openrouter: ", self.OPENROUTER)
        if "MODEL_API_BASE" in self:
            print("Model API base: ", self.MODEL_API_BASE)


# TODO: move this to a frontend input
# Frontend should read this file and call the corresponding endpoint to activate it
# Otherwise this will conflict with Elysia as a python package
# if os.path.exists("elysia_config.json"):
#     with open("elysia_config.json", "r") as f:
#         config = json.load(f)
#     settings = Settings.from_config(config)
# else:

# global settings that should never be used by the frontend
# but used when using Elysia as a package
settings = Settings()
settings.set_api_keys_from_env()


def configure(**kwargs):
    settings.configure(**kwargs)
