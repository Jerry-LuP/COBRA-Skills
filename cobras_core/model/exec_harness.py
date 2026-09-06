"""Helpers for running exec backends as the target harness."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import threading
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

from cobras.cobras_core.model.backend_config import (
    get_claude_code_exec_config,
    get_codex_exec_config,
    get_target_backend,
)
from cobras.cobras_core.model.common import tracker
from cobras.cobras_core.token_usage import normalize_usage, record_token_usage


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "final_response": {
            "type": "string",
            "description": "The exact final answer text to return, preserving required <answer>...</answer> tags.",
        },
        "final_answer": {
            "type": "string",
            "description": "The concise answer value without explanation, if separable.",
        },
    },
    "required": ["final_response", "final_answer"],
    "additionalProperties": False,
}


def render_skill_md(
    skill_content: str,
    *,
    name: str = "cobras-target",
    description: str = "Dynamic ReflACT skill for the current benchmark task.",
    preamble: str = "",
) -> str:
    body = skill_content.strip() or "No additional dynamic guidance was provided for this task."
    chunks = [
        "---",
        f'name: "{name}"',
        f'description: "{description}"',
        "---",
        "",
        "# ReflACT Target Skill",
        "",
    ]
    if preamble.strip():
        chunks.append(preamble.strip())
        chunks.append("")
    chunks.extend([
        "## Dynamic Guidance",
        "",
        body,
        "",
    ])
    return "\n".join(chunks)


def _safe_relative_path(value: str, *, label: str) -> Path:
    raw = str(value or "").replace("\\", "/").strip()
    relative = PurePosixPath(raw)
    if (
        not raw
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} must be a safe relative path: {value!r}")
    return Path(*relative.parts)


def _regular_source(path: str, *, label: str) -> Path:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {source}")
    return source.resolve()


def _write_private_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def _workspace_root(work_dir: str) -> Path:
    root = Path(work_dir).expanduser().absolute()
    allowed = os.environ.get("COBRAS_EXEC_ALLOWED_ROOT", "").strip()
    if allowed:
        allowed_root = Path(allowed).expanduser().resolve()
        try:
            root.resolve(strict=False).relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(
                f"Exec workspace is outside COBRAS_EXEC_ALLOWED_ROOT={allowed_root}: {root}"
            ) from exc
    if root.is_symlink():
        raise ValueError(f"Exec workspace cannot be a symlink: {root}")
    return root


def prepare_workspace(
    *,
    work_dir: str,
    skill_md: str,
    task_text: str = "",
    task_filename: str = "task.md",
    images: list[str] | None = None,
    extra_files: dict[str, str] | None = None,
    copy_files: list[tuple[str, str]] | None = None,
    link_dirs: list[tuple[str, str]] | None = None,
) -> tuple[str, str]:
    root = _workspace_root(work_dir)
    if link_dirs:
        raise ValueError("Symlinked directories are not allowed in exec workspaces.")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)

    skill_path = root / ".agents" / "skills" / "cobras-target" / "SKILL.md"
    _write_private_text(skill_path, skill_md)

    task_relative = _safe_relative_path(task_filename, label="task_filename")
    task_path = root / task_relative
    if task_text:
        _write_private_text(task_path, task_text)

    for rel_path, content in (extra_files or {}).items():
        relative = _safe_relative_path(rel_path, label="extra file destination")
        _write_private_text(root / relative, content)

    for src, rel_dst in copy_files or []:
        source = _regular_source(src, label="copied input")
        relative = _safe_relative_path(rel_dst, label="copied input destination")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source, destination)
        destination.chmod(0o600)

    attachment_lines: list[str] = []
    attachments_dir = root / "attachments"
    for index, image in enumerate(images or [], 1):
        source = _regular_source(image, label="image attachment")
        attachments_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        safe_base = Path(source.name).name or f"image_{index}"
        destination = attachments_dir / f"{index:02d}_{safe_base}"
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
        attachment_lines.append(f"- `{destination.relative_to(root).as_posix()}`")

    if attachment_lines:
        _write_private_text(
            root / "ATTACHMENTS.md",
            "# Attachments\n\n"
            "Use only these workspace-local files when the task refers to attached images or documents.\n\n"
            + "\n".join(attachment_lines)
            + "\n",
        )

    return str(skill_path), str(task_path)


def workspace_attachment_paths(work_dir: str) -> list[str]:
    root = _workspace_root(work_dir)
    attachments = root / "attachments"
    if not attachments.exists():
        return []
    paths: list[str] = []
    for path in sorted(attachments.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Invalid workspace attachment: {path}")
        paths.append(str(path.resolve()))
    return paths


def _build_codex_trace_summary(raw: str, response: str) -> str:
    usage = _codex_usage_from_jsonl(raw)
    errors = []
    for line in (raw or "").splitlines():
        lowered = line.lower()
        if "error" in lowered or "traceback" in lowered:
            errors.append(line.strip())
        if len(errors) >= 3:
            break
    answer_format = "missing"
    if "<answer>" in (response or "").lower():
        answer_format = "tagged"
    elif (response or "").strip():
        answer_format = "plain_text"
    return "\n".join(
        [
            "Codex Trace Summary",
            f"- final answer format: {answer_format}",
            f"- final response chars: {len(response or '')}",
            f"- input tokens: {usage['input_tokens']}",
            f"- output tokens: {usage['output_tokens']}",
            f"- errors: {' | '.join(errors) if errors else 'none'}",
        ]
    )


def _build_claude_trace_summary(raw: str, response: str) -> str:
    usage = _claude_code_usage_from_jsonl(raw)
    answer_format = "missing"
    if "<answer>" in (response or "").lower():
        answer_format = "tagged"
    elif (response or "").strip():
        answer_format = "plain_text"
    errors: list[str] = []
    for ln in (raw or "").splitlines():
        if "error" in ln.lower() or "traceback" in ln.lower():
            errors.append(ln.strip())
        if len(errors) >= 3:
            break
    parts = ["Claude Code Trace Summary", f"- final answer format: {answer_format}"]
    parts.extend(
        [
            f"- final response chars: {len(response or '')}",
            f"- input tokens: {usage['input_tokens']}",
            f"- output tokens: {usage['output_tokens']}",
            f"- errors: {' | '.join(errors) if errors else 'none'}",
        ]
    )
    return "\n".join(parts)


def _persist_artifacts(
    *,
    work_dir: str,
    raw: str,
    response: str,
    prefix: str,
    summary_builder,
) -> None:
    pred_dir = os.path.dirname(work_dir.rstrip(os.sep))
    raw_path = os.path.join(pred_dir, f"{prefix}_raw.txt")
    summary_path = os.path.join(pred_dir, f"{prefix}_trace_summary.txt")

    combined_raw = raw
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            prev = f.read()
        combined_raw = f"{prev}\n\n===== TURN BREAK =====\n\n{raw}" if prev.strip() else raw

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(combined_raw)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_builder(combined_raw, response))


def _persist_codex_artifacts(work_dir: str, raw: str, response: str) -> None:
    _persist_artifacts(
        work_dir=work_dir,
        raw=raw,
        response=response,
        prefix="codex",
        summary_builder=_build_codex_trace_summary,
    )


def _persist_claude_artifacts(work_dir: str, raw: str, response: str) -> None:
    _persist_artifacts(
        work_dir=work_dir,
        raw=raw,
        response=response,
        prefix="claude",
        summary_builder=_build_claude_trace_summary,
    )


_DENIED_DATA_DIR_NAMES = {"legacy_doc_split", "sealqa_split"}


def _normalize_tools(allowed_tools: list[str] | str | None) -> str:
    if allowed_tools is None:
        return ""
    if isinstance(allowed_tools, str):
        return ",".join(part.strip() for part in allowed_tools.split(",") if part.strip())
    return ",".join(str(tool).strip() for tool in allowed_tools if str(tool).strip())


def _tools_list(allowed_tools: list[str] | str | None) -> list[str]:
    tools = _normalize_tools(allowed_tools)
    return [part.strip() for part in tools.split(",") if part.strip()]


def _default_claude_tools(allow_file_edits: bool) -> list[str]:
    return ["Read", "Bash", "Write", "Edit"] if allow_file_edits else ["Read", "Bash"]


def _validate_exec_path(path: str) -> str:
    resolved = os.path.realpath(os.path.abspath(path))
    parts = set(resolved.split(os.sep))
    denied = parts & _DENIED_DATA_DIR_NAMES
    if denied:
        raise ValueError(f"Refusing to expose denied data directory to exec backend: {', '.join(sorted(denied))}")
    return resolved


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validated_add_dirs(work_dir: str, data_dirs: list[str] | None, images: list[str] | None) -> list[str]:
    root = Path(_validate_exec_path(work_dir))
    for value in data_dirs or []:
        resolved = Path(_validate_exec_path(value))
        if not _is_inside(resolved, root):
            raise ValueError(f"Exec data directory must be inside its workspace: {resolved}")
    for value in images or []:
        resolved = Path(_validate_exec_path(value))
        if not _is_inside(resolved, root):
            raise ValueError(f"Exec image must be copied inside its workspace: {resolved}")
    return [str(root)]


def _anthropic_base_url(value: str) -> str:
    """Return the API root expected by Claude Code.

    Claude Code appends /v1/messages itself. OpenAI-compatible target
    endpoints in COBRA-Skills conventionally end in /v1, so passing the
    value through unchanged would make Claude Code request /v1/v1/messages.
    """
    base_url = str(value or "").strip().rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    return base_url


def _target_subprocess_env(work_dir: str) -> dict[str, str]:
    home = Path(work_dir) / ".exec_home"
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    home.chmod(0o700)
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    codex_home.chmod(0o700)
    passthrough = {
        "PATH",
        "LD_LIBRARY_PATH",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "no_proxy",
        "COBRAS_CODEX_ALLOWED_ROOT",
        "COBRAS_CODEX_ROOTFS",
        "COBRAS_CODEX_NATIVE",
        "COBRAS_CODEX_PYTHON_ENV",
        "COBRAS_CLAUDE_CODE_ALLOWED_ROOT",
        "COBRAS_CLAUDE_CODE_ROOTFS",
        "COBRAS_CLAUDE_CODE_NATIVE",
        "COBRAS_CLAUDE_CODE_NODE",
        "COBRAS_CLAUDE_CODE_RUNTIME",
        "COBRAS_CLAUDE_CODE_PYTHON_ENV",
        "COBRAS_CLAUDE_CODE_BASE_URL",
    }
    env = {key: os.environ[key] for key in passthrough if os.environ.get(key)}
    env.setdefault("PATH", os.defpath)
    env.setdefault("LANG", "C")
    env.setdefault("LC_ALL", "C")
    env["HOME"] = str(home)
    env["CODEX_HOME"] = str(home / ".codex")
    env["COBRAS_TARGET_API_KEY"] = os.environ.get("TARGET_API_KEY", "dummy") or "dummy"
    env["COBRAS_TARGET_BASE_URL"] = os.environ.get("TARGET_BASE_URL", "")
    if get_target_backend() == "claude_code_exec":
        claude_config = home / ".claude"
        claude_config.mkdir(parents=True, exist_ok=True, mode=0o700)
        claude_config.chmod(0o700)
        configured_base = os.environ.get(
            "COBRAS_CLAUDE_CODE_BASE_URL",
            os.environ.get("TARGET_BASE_URL", ""),
        )
        base_url = _anthropic_base_url(configured_base)
        env["ANTHROPIC_API_KEY"] = os.environ.get("TARGET_API_KEY", "")
        env["ANTHROPIC_AUTH_TOKEN"] = os.environ.get("TARGET_API_KEY", "")
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url
        env.update(
            {
                "CLAUDE_CONFIG_DIR": str(claude_config),
                "CLAUDE_CODE_SIMPLE": "1",
                "CLAUDE_CODE_SAFE_MODE": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
                "DISABLE_TELEMETRY": "1",
                "TERM": "dumb",
                "SHELL": "/bin/bash",
            }
        )
    return env


def _run_process(
    cmd: list[str],
    *,
    cwd: str,
    timeout: int,
    env: dict[str, str],
) -> tuple[int, str, str]:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        exc.stdout = stdout
        exc.stderr = stderr
        raise
    return process.returncode, stdout or "", stderr or ""


def _record_exec_usage(*, model: str, stage: str, usage: Any, provider: str) -> None:
    normalized = normalize_usage(usage)
    if normalized["input_tokens"] or normalized["output_tokens"]:
        tracker.record(stage, normalized["input_tokens"], normalized["output_tokens"])
        record_token_usage(
            role="student",
            model=model,
            stage=stage,
            usage=normalized,
            provider=provider,
        )


def _codex_usage_from_jsonl(raw: str) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for line in (raw or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "turn.completed":
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            turn = event.get("turn")
            usage = turn.get("usage") if isinstance(turn, dict) else {}
        normalized = normalize_usage(usage)
        for key in totals:
            totals[key] += normalized[key]
    return totals


def _claude_code_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    text = (raw or "").strip()
    if not text:
        return events
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return [payload]
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _claude_code_usage_from_jsonl(raw: str) -> dict[str, int]:
    result_usage: dict[str, int] | None = None
    assistant_usage: dict[str, dict[str, int]] = {}
    for event in _claude_code_events(raw):
        if event.get("type") == "result" and isinstance(event.get("usage"), dict):
            result_usage = normalize_usage(event["usage"])
            continue
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("usage"), dict):
            continue
        identity = str(
            message.get("id")
            or event.get("uuid")
            or f"assistant-{len(assistant_usage)}"
        )
        assistant_usage[identity] = normalize_usage(message["usage"])
    if result_usage is not None:
        return result_usage
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in assistant_usage.values():
        for key in totals:
            totals[key] += usage[key]
    return totals


def _claude_code_response_from_jsonl(raw: str) -> str:
    response = ""
    for event in _claude_code_events(raw):
        if event.get("type") == "result" and event.get("result") is not None:
            response = str(event.get("result") or "").strip()
            continue
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        text_parts = [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        if text_parts:
            response = "".join(text_parts).strip()
    return response


def _claude_code_error_from_jsonl(raw: str) -> str:
    """Extract a structurally reported Claude Code API/harness error."""
    for event in _claude_code_events(raw):
        if event.get("type") == "result" and (
            bool(event.get("is_error"))
            or str(event.get("terminal_reason") or "").lower() == "api_error"
        ):
            status = event.get("api_error_status")
            reason = event.get("terminal_reason") or event.get("subtype") or "error"
            detail = str(event.get("result") or event.get("error") or "").strip()
            parts = [str(reason)]
            if status is not None:
                parts.append(f"HTTP {status}")
            if detail:
                parts.append(detail)
            return ": ".join(parts)

        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        if not message.get("is_api_error_message") and not message.get("error"):
            continue
        content = message.get("content")
        detail = ""
        if isinstance(content, list):
            detail = "".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
        reason = str(message.get("error") or "api_error")
        return f"{reason}: {detail}" if detail else reason
    return ""


def _sdk_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in {"1", "true", "yes", "on", "sdk"}:
        return "sdk"
    if mode in {"0", "false", "no", "off", "cli"}:
        return "cli"
    return "auto"


def _claude_effort(value: Any) -> str:
    effort = str(value or "medium").strip().lower()
    if effort in {"", "none", "off"}:
        return ""
    if effort == "xhigh":
        return "max"
    if effort not in {"low", "medium", "high", "max"}:
        return "medium"
    return effort


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            box["exception"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()
    if "exception" in box:
        raise box["exception"]
    return box.get("result")


def _exec_prompt(prompt: str, *, allow_file_edits: bool = False) -> str:
    edit_instruction = (
        "You may modify files in the workspace when the task asks you to create an artifact. "
        if allow_file_edits
        else "Do not modify files. "
    )
    return (
        "Use the workspace files to solve the task. Read task.md and the skill at "
        ".agents/skills/cobras-target/SKILL.md before answering. "
        "If ATTACHMENTS.md exists, read it and inspect the listed local files. "
        "Do not call a Skill tool; the ReflACT guidance is a local markdown file. "
        f"Do not ask for permission. {edit_instruction}"
        "Return only the final answer text, keeping any required <answer>...</answer> tags exactly.\n\n"
        f"{_normalize_target_exec_prompt(prompt)}"
    )


def _retry_prompt(prompt: str, attempt: int) -> str:
    if attempt <= 0:
        return prompt
    return (
        f"{prompt}\n\n"
        "Previous execution returned an empty final response. Re-read task.md and "
        ".agents/skills/cobras-target/SKILL.md. If ATTACHMENTS.md exists, use the listed files. "
        "Then produce the final answer inside <answer>...</answer>."
    )


def _normalize_target_exec_prompt(prompt: str) -> str:
    """Avoid wording that makes Claude Code call an unregistered Skill tool."""
    text = prompt or ""
    replacements = {
        "Use the `cobras-target` skill available in this workspace.": (
            "Read `.agents/skills/cobras-target/SKILL.md` directly; do not call a Skill tool."
        ),
        "- Use the local `cobras-target` skill before writing code.": (
            "- Read `.agents/skills/cobras-target/SKILL.md` before writing code; do not call a Skill tool."
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict = json.loads(json.dumps(schema))
    strict["additionalProperties"] = False
    properties = strict.get("properties") or {}
    strict["required"] = list(properties.keys())
    return strict


def _structured_response(data: Any) -> tuple[str, str]:
    if not isinstance(data, dict):
        return "", f"Structured output was not an object: {type(data).__name__}"
    final_response = str(data.get("final_response") or "").strip()
    final_answer = str(data.get("final_answer") or "").strip()
    if final_response:
        return final_response, ""
    if final_answer:
        if "<answer>" in final_answer.lower():
            return final_answer, ""
        return f"<answer>{final_answer}</answer>", ""
    return "", "Structured output did not contain a final response."


def _extract_claude_structured_output(messages: list[Any]) -> Any:
    """Claude Code SDK can finish with error_during_execution after StructuredOutput."""
    for msg in reversed(messages):
        structured = getattr(msg, "structured_output", None)
        if isinstance(structured, dict):
            return structured

        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in reversed(content):
            name = getattr(item, "name", None)
            payload = getattr(item, "input", None)
            if isinstance(item, dict):
                name = item.get("name", name)
                payload = item.get("input", payload)
            if name == "StructuredOutput" and isinstance(payload, dict):
                return payload
    return None


def _raw_exception(label: str, exc: BaseException) -> str:
    return _json_dumps({
        "backend": label,
        "is_error": True,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    })


def _run_claude_code_sdk_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    allowed_tools: list[str] | str | None = None,
    permission_mode: str | None = None,
    allow_file_edits: bool = False,
    stage: str = "rollout",
) -> tuple[str, str]:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    async def _query() -> tuple[str, str]:
        system_prompt: dict[str, Any] = {
            "type": "preset",
            "preset": "claude_code",
            "append": (
                "Use the workspace files to solve the task. Read task.md and the skill at "
                ".agents/skills/cobras-target/SKILL.md before answering. "
                "If ATTACHMENTS.md exists, read it and inspect the listed local files. "
                "Do not call a Skill tool; the ReflACT guidance is a local markdown file. "
                + (
                    "You may modify files in the workspace when the task asks you to create an artifact. "
                    if allow_file_edits
                    else "Do not modify files. "
                )
                + "Return structured output whose final_response preserves required <answer>...</answer> tags."
            ),
        }
        kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "output_format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            "allowed_tools": _tools_list(allowed_tools) or _default_claude_tools(allow_file_edits),
            "cwd": str(work_dir),
            "permission_mode": permission_mode or "bypassPermissions",
            "add_dirs": _validated_add_dirs(work_dir, data_dirs, images),
            "max_buffer_size": 8 * 1024 * 1024,
        }
        config = get_claude_code_exec_config()
        effort = _claude_effort(config.get("effort"))
        if effort:
            kwargs["effort"] = effort
        max_thinking_tokens = int(config.get("max_thinking_tokens", 0) or 0)
        if max_thinking_tokens > 0:
            kwargs["max_thinking_tokens"] = max_thinking_tokens
        options = ClaudeAgentOptions(**kwargs)
        if model:
            options.model = model.split("/", 1)[1] if model.startswith("anthropic/") else model

        messages = []
        async with ClaudeSDKClient(options) as client:
            await client.query(_normalize_target_exec_prompt(prompt))
            messages = [msg async for msg in client.receive_response()]
        last = messages[-1] if messages else None
        raw_structured_output = _extract_claude_structured_output(messages)
        response, parse_error = _structured_response(raw_structured_output)
        first = messages[0] if messages else None
        first_data = getattr(first, "data", {}) if first is not None else {}
        terminal_is_error = bool(getattr(last, "is_error", False)) if last is not None else False
        usage = getattr(last, "usage", {}) if last is not None else {}
        _record_exec_usage(
            model=model,
            stage=stage,
            usage=usage,
            provider="claude_code_exec",
        )
        raw = _json_dumps({
            "backend": "claude_code_sdk",
            "uuid": first_data.get("uuid", "") if isinstance(first_data, dict) else "",
            "session_id": getattr(last, "session_id", "") if last is not None else "",
            "model": first_data.get("model", model) if isinstance(first_data, dict) else model,
            "tools": first_data.get("tools", _tools_list(allowed_tools)) if isinstance(first_data, dict) else _tools_list(allowed_tools),
            "duration_ms": getattr(last, "duration_ms", 0) if last is not None else 0,
            "total_cost_usd": getattr(last, "total_cost_usd", 0.0) if last is not None else 0.0,
            "num_turns": getattr(last, "num_turns", 0) if last is not None else 0,
            "usage": usage,
            "result": getattr(last, "result", "") if last is not None else "",
            "is_error": bool(parse_error) or (terminal_is_error and not response.strip()),
            "terminal_is_error": terminal_is_error,
            "parse_error": parse_error,
            "raw_structured_output": raw_structured_output,
            "messages": messages,
        })
        return response, raw

    return _run_async(asyncio.wait_for(_query(), timeout=timeout))


def _run_claude_code_cli_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    allowed_tools: list[str] | str | None = None,
    permission_mode: str | None = None,
    allow_file_edits: bool = False,
    stage: str = "rollout",
) -> tuple[str, str]:
    config = get_claude_code_exec_config()
    selected_tools = _tools_list(allowed_tools) or _default_claude_tools(allow_file_edits)
    tools = ",".join(selected_tools)
    _validated_add_dirs(work_dir, data_dirs, images)
    cmd = [
        str(config["path"]),
        "-p",
        "--verbose",
        "--output-format",
        "stream-json",
        "--no-session-persistence",
        "--permission-mode",
        permission_mode or "dontAsk",
        "--add-dir",
        work_dir,
        "--tools",
        tools,
        "--allowedTools",
        tools,
    ]
    if config.get("profile"):
        cmd.extend(["--settings", '{"env":{"CLAUDE_CODE_USE_BEDROCK":"0"}}'])
        cmd.extend(["--append-system-prompt", f"Profile: {config['profile']}"])
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["--", _exec_prompt(prompt, allow_file_edits=allow_file_edits)])

    timeout_override = os.environ.get("CLAUDE_CODE_EXEC_TIMEOUT_SECONDS", "").strip()
    if timeout_override:
        try:
            inner_timeout = max(1, int(float(timeout_override)))
        except ValueError as exc:
            raise ValueError(
                "CLAUDE_CODE_EXEC_TIMEOUT_SECONDS must be a positive number, "
                f"got {timeout_override!r}"
            ) from exc
        outer_timeout = max(timeout, inner_timeout + 15)
    else:
        inner_timeout = max(1, timeout - 15)
        outer_timeout = timeout
    subprocess_env = _target_subprocess_env(work_dir)
    subprocess_env.update(
        {
            "COBRAS_CLAUDE_CODE_INNER_TIMEOUT_SECONDS": str(inner_timeout),
            "COBRAS_CLAUDE_CODE_ALLOW_FILE_EDITS": "1" if allow_file_edits else "0",
            "COBRAS_CLAUDE_CODE_MODEL": model,
            "COBRAS_CLAUDE_CODE_CONTEXT_WINDOW": str(config["context_window"]),
            "COBRAS_CLAUDE_CODE_MAX_TOKENS": str(config["max_tokens"]),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(config["max_tokens"]),
        }
    )
    try:
        returncode, stdout, stderr = _run_process(
            cmd,
            cwd=work_dir,
            timeout=outer_timeout,
            env=subprocess_env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "")
        raw = stdout
        if stderr:
            raw = f"{raw}\n[stderr]\n{stderr}" if raw else stderr
        response = _claude_code_response_from_jsonl(stdout)
        _record_exec_usage(
            model=model,
            stage=stage,
            usage=_claude_code_usage_from_jsonl(stdout),
            provider="claude_code_exec",
        )
        _persist_claude_artifacts(work_dir, raw, response)
        return "", raw

    raw = stdout
    if stderr:
        raw = f"{raw}\n[stderr]\n{stderr}" if raw else stderr

    response = _claude_code_response_from_jsonl(stdout)
    if not response and returncode == 0:
        response = stdout.strip()
    _record_exec_usage(
        model=model,
        stage=stage,
        usage=_claude_code_usage_from_jsonl(stdout),
        provider="claude_code_exec",
    )
    terminal_error = _claude_code_error_from_jsonl(stdout)
    if terminal_error:
        _persist_claude_artifacts(work_dir, raw, response)
        raise RuntimeError(f"claude code exec API error: {terminal_error[:4000]}")
    if returncode != 0:
        _persist_claude_artifacts(work_dir, raw, response)
        detail = (stderr or stdout).strip()
        raise RuntimeError(
            f"claude code exec failed with exit code {returncode}: {detail[:4000]}"
        )
    return response, raw


def run_claude_code_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    allowed_tools: list[str] | str | None = None,
    permission_mode: str | None = None,
    allow_file_edits: bool = False,
    stage: str = "rollout",
) -> tuple[str, str]:
    config = get_claude_code_exec_config()
    mode = _sdk_mode(config.get("use_sdk"))
    retries = int(config.get("empty_response_retries", 0) or 0)
    last_response = ""
    all_raw: list[str] = []

    for attempt in range(retries + 1):
        attempt_prompt = _retry_prompt(prompt, attempt)
        if mode != "cli":
            try:
                response, raw = _run_claude_code_sdk_exec(
                    work_dir=work_dir,
                    prompt=attempt_prompt,
                    model=model,
                    timeout=timeout,
                    images=images,
                    data_dirs=data_dirs,
                    allowed_tools=allowed_tools,
                    permission_mode=permission_mode,
                    allow_file_edits=allow_file_edits,
                    stage=stage,
                )
                all_raw.append(f"===== CLAUDE SDK ATTEMPT {attempt + 1} =====\n{raw}")
                if response.strip():
                    combined = "\n\n".join(all_raw)
                    _persist_claude_artifacts(work_dir, combined, response)
                    return response, combined
            except (ImportError, ModuleNotFoundError) as exc:
                raw = _raw_exception("claude_code_sdk", exc)
                all_raw.append(f"===== CLAUDE SDK ATTEMPT {attempt + 1} =====\n{raw}")
                if mode == "sdk":
                    _persist_claude_artifacts(work_dir, "\n\n".join(all_raw), "")
                    raise
            except Exception as exc:  # noqa: BLE001
                raw = _raw_exception("claude_code_sdk", exc)
                all_raw.append(f"===== CLAUDE SDK ATTEMPT {attempt + 1} =====\n{raw}")
                if mode == "sdk" and attempt >= retries:
                    _persist_claude_artifacts(work_dir, "\n\n".join(all_raw), "")
                    raise
        if mode != "sdk":
            response, raw = _run_claude_code_cli_exec(
                work_dir=work_dir,
                prompt=attempt_prompt,
                model=model,
                timeout=timeout,
                images=images,
                data_dirs=data_dirs,
                allowed_tools=allowed_tools,
                permission_mode=permission_mode,
                allow_file_edits=allow_file_edits,
                stage=stage,
            )
            all_raw.append(f"===== CLAUDE CLI ATTEMPT {attempt + 1} =====\n{raw}")
            last_response = response
            if response.strip():
                combined = "\n\n".join(all_raw)
                _persist_claude_artifacts(work_dir, combined, response)
                return response, combined

    combined = "\n\n".join(all_raw)
    _persist_claude_artifacts(work_dir, combined, last_response)
    return last_response, combined


def _toml_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def _codex_provider_args() -> list[str]:
    config = get_codex_exec_config()
    base_url = os.environ.get("TARGET_BASE_URL", "").strip()
    if not base_url:
        raise RuntimeError("TARGET_BASE_URL is required for target_backend=codex_exec")
    return [
        "-c",
        'model_provider="cobras_target"',
        "-c",
        'model_providers.cobras_target.name="COBRAS target"',
        "-c",
        f"model_providers.cobras_target.base_url={_toml_string(base_url)}",
        "-c",
        'model_providers.cobras_target.env_key="COBRAS_TARGET_API_KEY"',
        "-c",
        'model_providers.cobras_target.wire_api="responses"',
        "-c",
        f"model_context_window={int(config['context_window'])}",
        "-c",
        f"model_providers.cobras_target.request_max_retries={int(config['request_max_retries'])}",
        "-c",
        f"model_providers.cobras_target.stream_max_retries={int(config['stream_max_retries'])}",
        "-c",
        f"model_providers.cobras_target.stream_idle_timeout_ms={int(config['stream_idle_timeout_ms'])}",
        "-c",
        'approval_policy="never"',
        "-c",
        f"web_search={_toml_string('live' if config.get('web_search') else 'disabled')}",
    ]


def _run_codex_cli_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    sandbox: str | None = None,
    allow_file_edits: bool = False,
    stage: str = "rollout",
) -> tuple[str, str]:
    config = get_codex_exec_config()
    _validated_add_dirs(work_dir, data_dirs, images)
    last_message_path = os.path.join(work_dir, "codex_last_message.txt")
    if sandbox:
        actual_sandbox = str(sandbox)
    elif allow_file_edits:
        actual_sandbox = "workspace-write"
    else:
        actual_sandbox = str(config.get("sandbox") or "read-only")
    cmd = [
        str(config["path"]),
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--json",
        "--color",
        "never",
        "-C",
        work_dir,
        "--sandbox",
        actual_sandbox,
        *_codex_provider_args(),
    ]
    reasoning_effort = str(config.get("reasoning_effort", "") or "").strip().lower()
    if reasoning_effort not in {"", "none", "off"}:
        cmd.extend(["-c", f"model_reasoning_effort={_toml_string(reasoning_effort)}"])
    if model:
        cmd.extend(["-m", model])
    for image in images or []:
        cmd.extend(["-i", _validate_exec_path(image)])
    cmd.extend(["--output-last-message", last_message_path, _exec_prompt(
        prompt,
        allow_file_edits=allow_file_edits,
    )])

    subprocess_env = _target_subprocess_env(work_dir)
    subprocess_env["COBRAS_CODEX_INNER_TIMEOUT_SECONDS"] = str(max(1, timeout - 15))
    try:
        returncode, stdout, stderr = _run_process(
            cmd,
            cwd=work_dir,
            timeout=timeout,
            env=subprocess_env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "")
        raw = stdout
        if stderr:
            raw = f"{raw}\n[stderr]\n{stderr}" if raw else stderr
        _record_exec_usage(
            model=model,
            stage=stage,
            usage=_codex_usage_from_jsonl(stdout),
            provider="codex_exec",
        )
        _persist_codex_artifacts(work_dir, raw, "")
        raise

    last_message = ""
    if os.path.exists(last_message_path):
        last_message = Path(last_message_path).read_text(encoding="utf-8", errors="replace").strip()
    raw = stdout
    if stderr:
        raw = f"{raw}\n[stderr]\n{stderr}" if raw else stderr
    usage = _codex_usage_from_jsonl(stdout)
    _record_exec_usage(
        model=model,
        stage=stage,
        usage=usage,
        provider="codex_exec",
    )
    if returncode != 0:
        _persist_codex_artifacts(work_dir, raw, last_message)
        detail = (stderr or stdout).strip()
        raise RuntimeError(
            f"codex exec failed with exit code {returncode}: {detail[:4000]}"
        )
    return last_message, raw


def run_codex_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    sandbox: str | None = None,
    allow_file_edits: bool = False,
    stage: str = "rollout",
) -> tuple[str, str]:
    config = get_codex_exec_config()
    mode = _sdk_mode(config.get("use_sdk"))
    if mode == "sdk":
        raise ValueError(
            "COBRAS Codex target isolation currently requires CODEX_EXEC_USE_SDK=cli"
        )
    retries = int(config.get("empty_response_retries", 0) or 0)
    last_response = ""
    all_raw: list[str] = []
    for attempt in range(retries + 1):
        response, raw = _run_codex_cli_exec(
            work_dir=work_dir,
            prompt=_retry_prompt(prompt, attempt),
            model=model,
            timeout=timeout,
            images=images,
            data_dirs=data_dirs,
            sandbox=sandbox,
            allow_file_edits=allow_file_edits,
            stage=stage,
        )
        all_raw.append(f"===== CODEX CLI ATTEMPT {attempt + 1} =====\n{raw}")
        last_response = response
        if response.strip():
            combined = "\n\n".join(all_raw)
            _persist_codex_artifacts(work_dir, combined, response)
            return response, combined
    combined = "\n\n".join(all_raw)
    _persist_codex_artifacts(work_dir, combined, last_response)
    return last_response, combined


def run_target_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    allowed_tools: list[str] | str | None = None,
    permission_mode: str | None = None,
    sandbox: str | None = None,
    allow_file_edits: bool = False,
    stage: str = "rollout",
) -> tuple[str, str]:
    backend = get_target_backend()
    if backend == "codex_exec":
        return run_codex_exec(
            work_dir=work_dir,
            prompt=prompt,
            model=model,
            timeout=timeout,
            images=images,
            data_dirs=data_dirs,
            sandbox=sandbox,
            allow_file_edits=allow_file_edits,
            stage=stage,
        )
    if backend == "claude_code_exec":
        return run_claude_code_exec(
            work_dir=work_dir,
            prompt=prompt,
            model=model,
            timeout=timeout,
            images=images,
            data_dirs=data_dirs,
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
            allow_file_edits=allow_file_edits,
            stage=stage,
        )
    raise ValueError(f"Unsupported exec backend: {backend}")
