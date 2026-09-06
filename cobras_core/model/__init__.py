"""COBRAS model API with runtime backend selection."""

from __future__ import annotations

from typing import Any

from cobras.cobras_core.model import openai as _openai
from cobras.cobras_core.model.backend_config import (  # noqa: F401
    configure_codex_exec,
    configure_claude_code_exec,
    get_codex_exec_config,
    get_claude_code_exec_config,
    get_target_backend,
    get_optimizer_backend,
    is_target_chat_backend,
    is_target_exec_backend,
    is_optimizer_chat_backend,
    set_target_backend,
    set_optimizer_backend,
)


def set_backend(name: str | None) -> str:
    """Backward-compatible global backend setter.

    Historically the codebase used one shared backend for both optimizer and
    target. Keep that entry point so older scripts continue to work, while
    mapping it onto the split optimizer/target backend model.
    """
    normalized = str(name or "openai_chat").strip().lower()
    if normalized in {"azure_openai", "openai_chat", "azure", "azure-openai"}:
        set_optimizer_backend("openai_chat")
        set_target_backend("openai_chat")
        return "openai_chat"
    if normalized in {"codex", "codex_exec", "claude_code_exec"}:
        set_optimizer_backend("openai_chat")
        if normalized == "codex":
            target_backend = "codex_exec"
        else:
            target_backend = normalized
        set_target_backend(target_backend)
        return target_backend
    raise ValueError(f"Unsupported legacy backend: {name!r}")


def get_backend_name() -> str:
    """Best-effort backward-compatible backend summary."""
    optimizer = get_optimizer_backend()
    target = get_target_backend()
    if optimizer == "openai_chat" and target == "openai_chat":
        return "openai_chat"
    return f"{optimizer}+{target}"


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    return _openai.chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    if not is_target_chat_backend():
        raise NotImplementedError(
            "chat_target is only supported with target_backend=openai_chat. "
            "Exec backends are handled in environment-specific rollout code."
        )
    return _openai.chat_target(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict]:
    return _openai.chat_optimizer_messages(
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def chat_target_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict]:
    if not is_target_chat_backend():
        raise NotImplementedError(
            "chat_target_messages is only supported with target_backend=openai_chat. "
            "Exec backends are handled in environment-specific rollout code."
        )
    return _openai.chat_target_messages(
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def chat_messages_with_deployment(
    deployment: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict]:
    return _openai.chat_messages_with_deployment(
        deployment=deployment,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def chat_with_deployment(
    deployment: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    return _openai.chat_with_deployment(
        deployment=deployment,
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def get_token_summary() -> dict:
    summary = _openai.get_token_summary()
    total = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for stage, values in summary.items():
        if stage == "_total":
            continue
        total["calls"] += values["calls"]
        total["prompt_tokens"] += values["prompt_tokens"]
        total["completion_tokens"] += values["completion_tokens"]
        total["total_tokens"] += values["total_tokens"]
    summary["_total"] = total
    return summary


def reset_token_tracker() -> None:
    _openai.reset_token_tracker()


def configure_openai(
    *,
    endpoint: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    auth_mode: str | None = None,
    ad_scope: str | None = None,
    managed_identity_client_id: str | None = None,
    optimizer_endpoint: str | None = None,
    optimizer_api_version: str | None = None,
    optimizer_api_key: str | None = None,
    optimizer_auth_mode: str | None = None,
    optimizer_ad_scope: str | None = None,
    optimizer_managed_identity_client_id: str | None = None,
    target_endpoint: str | None = None,
    target_api_version: str | None = None,
    target_api_key: str | None = None,
    target_auth_mode: str | None = None,
    target_ad_scope: str | None = None,
    target_managed_identity_client_id: str | None = None,
) -> None:
    _openai.configure_openai(
        endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,
        auth_mode=auth_mode,
        ad_scope=ad_scope,
        managed_identity_client_id=managed_identity_client_id,
        optimizer_endpoint=optimizer_endpoint,
        optimizer_api_version=optimizer_api_version,
        optimizer_api_key=optimizer_api_key,
        optimizer_auth_mode=optimizer_auth_mode,
        optimizer_ad_scope=optimizer_ad_scope,
        optimizer_managed_identity_client_id=optimizer_managed_identity_client_id,
        target_endpoint=target_endpoint,
        target_api_version=target_api_version,
        target_api_key=target_api_key,
        target_auth_mode=target_auth_mode,
        target_ad_scope=target_ad_scope,
        target_managed_identity_client_id=target_managed_identity_client_id,
    )


configure_azure_openai = configure_openai




def set_reasoning_effort(effort: str | None) -> None:
    _openai.set_reasoning_effort(effort)


def set_target_deployment(deployment: str) -> None:
    _openai.set_target_deployment(deployment)


def get_target_deployment() -> str:
    """Return the target model name independently of the active transport."""
    return _openai.TARGET_DEPLOYMENT


def set_optimizer_deployment(deployment: str) -> None:
    _openai.set_optimizer_deployment(deployment)
