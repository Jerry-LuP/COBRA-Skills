from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    base_url: str
    api_key_env: str
    model: str
    auth_mode: str = "openai_compatible"


DEFAULT_MODELS = {
    "openrouter": "openai/gpt-5.4-nano",
    "yunwu": "openai/gpt-5.4-nano",
    "custom": "openai/gpt-5.4-nano",
    "local": "local-model",
}

DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "yunwu": "https://yunwu.ai/v1",
    "custom": "",
    "local": "http://127.0.0.1:8000/v1",
}

DEFAULT_KEY_ENVS = {
    "openrouter": "TEACHER_API_KEY",
    "yunwu": "TEACHER_API_KEY",
    "custom": "TEACHER_API_KEY",
    "local": "TARGET_API_KEY",
}


def resolve_provider(
    *,
    provider: str,
    model: str = "",
    base_url: str = "",
    api_key_env: str = "",
) -> ProviderConfig:
    key = (provider or "openrouter").strip().lower()
    if key not in DEFAULT_BASE_URLS:
        raise SystemExit(
            f"Unsupported provider={provider!r}. Use openrouter, yunwu, custom, or local."
        )
    env_base_name = "TARGET_BASE_URL" if key == "local" else "TEACHER_BASE_URL"
    resolved_base_url = base_url or os.environ.get(env_base_name) or DEFAULT_BASE_URLS[key]
    return ProviderConfig(
        provider=key,
        base_url=resolved_base_url,
        api_key_env=api_key_env or DEFAULT_KEY_ENVS[key],
        model=model or DEFAULT_MODELS[key],
    )


def child_env_for_provider(
    cfg: ProviderConfig,
    *,
    api_key: str = "",
    role: str = "target",
    openai_base_url_env: str = "OPENAI_BASE_URL",
) -> dict[str, str]:
    env: dict[str, str] = {}
    key_value = api_key or os.environ.get(cfg.api_key_env, "")
    role_prefix = role.upper()
    if cfg.base_url:
        env["BASE_URL"] = cfg.base_url
        env[openai_base_url_env] = cfg.base_url
        if role_prefix in {"TARGET", "TEACHER", "EMBEDDING"}:
            env[f"{role_prefix}_BASE_URL"] = cfg.base_url
    if key_value:
        env["API_KEY"] = key_value
        env[cfg.api_key_env] = key_value
        if role_prefix in {"TARGET", "TEACHER", "EMBEDDING"}:
            env[f"{role_prefix}_API_KEY"] = key_value
    env["MODEL"] = cfg.model
    if role_prefix in {"TARGET", "TEACHER", "EMBEDDING"}:
        env[f"{role_prefix}_MODEL"] = cfg.model
    if role in {"teacher", "optimizer", "skill"}:
        env["SKILL_GEN_MODEL"] = cfg.model
        env["SKILL_MUTATE_MODEL"] = cfg.model
    return env
