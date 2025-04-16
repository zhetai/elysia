import json
import os
from typing import Callable

import spacy
from dotenv import load_dotenv

from dspy import LM
import uuid

load_dotenv()

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class Settings:
    CLIENT_TIMEOUT: int = os.getenv("CLIENT_TIMEOUT", 3)
    VERSION = "0.2.0"
    SETTINGS_ID = str(uuid.uuid4())

    BASE_MODEL: str | None = None
    BASE_PROVIDER: str | None = None
    COMPLEX_MODEL: str | None = None
    COMPLEX_PROVIDER: str | None = None

    BASE_MODEL_LM: LM | None = None
    COMPLEX_MODEL_LM: LM | None = None

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
            if item not in ["BASE_MODEL_LM", "COMPLEX_MODEL_LM"]:
                setattr(settings, item, config[item])
        settings.load_base_dspy_model()
        settings.load_complex_dspy_model()
        return settings

    @classmethod
    def from_default(cls):
        settings = cls()
        settings.default_config()
        return settings

    @classmethod
    def from_env_vars(cls):
        settings = cls()
        settings.set_config_from_env()
        return settings

    def set_config_from_env(self):
        self.BASE_MODEL = os.getenv("BASE_MODEL", None)
        self.COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", None)
        self.BASE_PROVIDER = os.getenv("BASE_PROVIDER", None)
        self.COMPLEX_PROVIDER = os.getenv("COMPLEX_PROVIDER", None)
        self.MODEL_API_BASE = os.getenv("MODEL_API_BASE", None)
        self.LOCAL = os.getenv("LOCAL", "1") == "1"
        self.set_api_keys_from_env()
        self.load_base_dspy_model()
        self.load_complex_dspy_model()

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

    def default_config(self):
        self.BASE_MODEL = os.getenv("BASE_MODEL", "gemini-2.0-flash-001")
        self.COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", "gemini-2.0-flash-001")
        self.BASE_PROVIDER = os.getenv("BASE_PROVIDER", "openrouter/google")
        self.COMPLEX_PROVIDER = os.getenv("COMPLEX_PROVIDER", "openrouter/google")
        self.MODEL_API_BASE = os.getenv("MODEL_API_BASE", None)
        self.LOCAL = os.getenv("LOCAL", "1") == "1"
        self.WCD_URL = os.getenv("WCD_URL")
        self.WCD_API_KEY = os.getenv("WCD_API_KEY")
        self.set_api_keys_from_env()
        self.load_base_dspy_model()
        self.load_complex_dspy_model()

    def load_base_dspy_model(self):
        if (
            "LOCAL" in dir(self)
            and self.LOCAL
            and "BASE_MODEL" in dir(self)
            and self.BASE_MODEL
        ):
            self.BASE_MODEL_LM = load_base_lm(self)
        else:
            self.BASE_MODEL_LM = None

    def load_complex_dspy_model(self):
        if (
            "LOCAL" in dir(self)
            and self.LOCAL
            and "COMPLEX_MODEL" in dir(self)
            and self.COMPLEX_MODEL
        ):
            self.COMPLEX_MODEL_LM = load_complex_lm(self)
        else:
            self.COMPLEX_MODEL_LM = None

    def configure(
        self,
        **kwargs,
    ):
        """
        Configure the settings for the Elysia API.

        Args:
            base_model: The base model to use (str). e.g. "gpt-4o-mini"
            complex_model: The complex model to use (str). e.g. "gpt-4o"
            base_provider: The provider to use for base_model (str). E.g. "openai" or "openrouter/openai"
            complex_provider: The provider to use for complex_model (str). E.g. "openai" or "openrouter/openai"
            model_api_base: The API base to use (str).
            wcd_url: The Weaviate cloud URL to use (str).
            wcd_api_key: The Weaviate cloud API key to use (str).
            local: Whether Elysia is running locally (bool).
            **kwargs: Additional API keys to set. E.g. `openai_apikey="..."`
        """

        if "local" in kwargs:
            self.LOCAL = kwargs["local"]
            kwargs.pop("local")

        if "base_model" in kwargs:
            if "base_provider" not in kwargs:
                raise ValueError(
                    "Provider must be specified if base_model is set. "
                    "E.g. `elysia.config.configure(base_model='gpt-4o-mini', base_provider='openai')`"
                )

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

            kwargs.pop("base_model")
            kwargs.pop("base_provider")

            self.load_base_dspy_model()

        if "complex_model" in kwargs:
            if "complex_provider" not in kwargs:
                raise ValueError(
                    "Provider must be specified if complex_model is set. "
                    "E.g. `elysia.config.configure(complex_model='gpt-4o', complex_provider='openai')`"
                )

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

            kwargs.pop("complex_model")
            kwargs.pop("complex_provider")

            self.load_complex_dspy_model()

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


def load_base_lm(settings: Settings):
    check_base_lm_settings(settings)

    return load_lm(
        settings.BASE_PROVIDER,
        settings.BASE_MODEL,
        settings.MODEL_API_BASE if "MODEL_API_BASE" in dir(settings) else None,
    )


def load_complex_lm(settings: Settings):
    check_complex_lm_settings(settings)

    return load_lm(
        settings.COMPLEX_PROVIDER,
        settings.COMPLEX_MODEL,
        settings.MODEL_API_BASE if "MODEL_API_BASE" in dir(settings) else None,
    )


def load_lm(
    provider: str,
    lm_name: str,
    model_api_base: str | None = None,
):
    api_base = model_api_base if provider == "ollama" else None
    full_lm_name = f"{provider}/{lm_name}"

    if lm_name.startswith("o1") or lm_name.startswith("o3"):
        lm = LM(model=full_lm_name, api_base=api_base, max_tokens=8000, temperature=1.0)
    else:
        lm = LM(model=full_lm_name, api_base=api_base, max_tokens=8000)

    return lm


# global settings that should never be used by the frontend
# but used when using Elysia as a package
settings = Settings()
settings.set_config_from_env()


def configure(**kwargs):
    settings.configure(**kwargs)
