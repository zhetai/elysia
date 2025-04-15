from dspy import LM

from elysia.config import Settings


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
        settings.MODEL_API_BASE,
    )


def load_complex_lm(settings: Settings):
    check_complex_lm_settings(settings)

    return load_lm(
        settings.COMPLEX_PROVIDER,
        settings.COMPLEX_MODEL,
        settings.MODEL_API_BASE,
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
