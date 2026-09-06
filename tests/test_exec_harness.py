from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from cobras.cobras_core.model import (
    configure_claude_code_exec,
    configure_codex_exec,
    get_claude_code_exec_config,
    get_codex_exec_config,
    get_target_backend,
    set_target_backend,
)
from cobras.cobras_core.model.exec_harness import (
    _claude_code_error_from_jsonl,
    _claude_code_usage_from_jsonl,
    _codex_usage_from_jsonl,
    _target_subprocess_env,
    prepare_workspace,
    run_target_exec,
    workspace_attachment_paths,
)


def test_prepare_workspace_contains_inputs_and_rejects_unsafe_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COBRAS_EXEC_ALLOWED_ROOT", raising=False)
    source = tmp_path / "source.txt"
    image = tmp_path / "page.png"
    source.write_text("source data", encoding="utf-8")
    image.write_bytes(b"not-a-real-png")

    work_dir = tmp_path / "run"
    skill_path, task_path = prepare_workspace(
        work_dir=str(work_dir),
        skill_md="# Skill\nUse evidence.",
        task_text="Answer the task.",
        copy_files=[(str(source), "inputs/source.txt")],
        images=[str(image)],
    )

    assert Path(skill_path).is_file()
    assert Path(task_path).read_text(encoding="utf-8") == "Answer the task."
    assert (work_dir / "inputs" / "source.txt").read_text(encoding="utf-8") == "source data"
    attachments = workspace_attachment_paths(str(work_dir))
    assert len(attachments) == 1
    manifest = (work_dir / "ATTACHMENTS.md").read_text(encoding="utf-8")
    assert str(image.resolve()) not in manifest
    assert "attachments/01_page.png" in manifest

    with pytest.raises(ValueError, match="safe relative path"):
        prepare_workspace(
            work_dir=str(tmp_path / "unsafe"),
            skill_md="# Skill",
            task_text="task",
            extra_files={"../escape.txt": "no"},
        )

    symlink = tmp_path / "source-link"
    symlink.symlink_to(source)
    with pytest.raises(ValueError, match="non-symlink"):
        prepare_workspace(
            work_dir=str(tmp_path / "symlink"),
            skill_md="# Skill",
            task_text="task",
            copy_files=[(str(symlink), "source.txt")],
        )


def test_target_subprocess_environment_excludes_teacher_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TARGET_API_KEY", "student-secret")
    monkeypatch.setenv("TEACHER_API_KEY", "teacher-secret")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-secret")

    env = _target_subprocess_env(str(tmp_path / "workspace"))

    assert env["COBRAS_TARGET_API_KEY"] == "student-secret"
    assert "TARGET_API_KEY" not in env
    assert "TEACHER_API_KEY" not in env
    assert "EMBEDDING_API_KEY" not in env


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://127.0.0.1:8001/v1", "http://127.0.0.1:8001"),
        ("http://127.0.0.1:8001/v1/", "http://127.0.0.1:8001"),
        ("http://127.0.0.1:8001", "http://127.0.0.1:8001"),
    ],
)
def test_claude_subprocess_environment_normalizes_api_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: str,
) -> None:
    previous_backend = get_target_backend()
    monkeypatch.setenv("TARGET_BASE_URL", configured)
    monkeypatch.setenv("TARGET_API_KEY", "student-secret")
    try:
        set_target_backend("claude_code_exec")
        env = _target_subprocess_env(str(tmp_path / "workspace"))
    finally:
        set_target_backend(previous_backend)

    assert env["ANTHROPIC_BASE_URL"] == expected
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["DISABLE_TELEMETRY"] == "1"


def test_usage_parsers_support_current_cli_events() -> None:
    codex_raw = json.dumps(
        {
            "type": "turn.completed",
            "turn": {
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                }
            },
        }
    )
    claude_raw = json.dumps(
        {
            "type": "result",
            "result": "<answer>B</answer>",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 4,
                "total_tokens": 15,
            },
        }
    )

    assert _codex_usage_from_jsonl(codex_raw) == {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }
    assert _claude_code_usage_from_jsonl(claude_raw) == {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
    }


def test_claude_error_parser_detects_terminal_api_error() -> None:
    raw = json.dumps(
        {
            "type": "result",
            "is_error": True,
            "terminal_reason": "api_error",
            "api_error_status": 404,
            "result": "The selected model is unavailable.",
        }
    )

    assert _claude_code_error_from_jsonl(raw) == (
        "api_error: HTTP 404: The selected model is unavailable."
    )


def _restore_codex(config: dict[str, object]) -> None:
    configure_codex_exec(
        path=str(config["path"]),
        sandbox=str(config["sandbox"]),
        profile=str(config["profile"]),
        reasoning_effort=str(config["reasoning_effort"]),
        use_sdk=str(config["use_sdk"]),
        network_access=bool(config["network_access"]),
        web_search=bool(config["web_search"]),
        approval_policy=str(config["approval_policy"]),
        context_window=int(config["context_window"]),
    )


def test_codex_cli_exec_records_student_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output-last-message') + 1])\n"
        "out.write_text('<answer>A</answer>', encoding='utf-8')\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': "
        "{'input_tokens': 11, 'output_tokens': 7, 'total_tokens': 18}}))\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)

    previous_backend = get_target_backend()
    previous_config = get_codex_exec_config()
    token_log = tmp_path / "tokens.jsonl"
    work_dir = tmp_path / "workspace"
    prepare_workspace(
        work_dir=str(work_dir),
        skill_md="# Skill\nChoose carefully.",
        task_text="Question with choices.",
    )
    monkeypatch.delenv("COBRAS_EXEC_ALLOWED_ROOT", raising=False)
    monkeypatch.setenv("TARGET_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("TARGET_API_KEY", "dummy")
    monkeypatch.setenv("COBRAS_TOKEN_LOG_PATH", str(token_log))

    try:
        set_target_backend("codex_exec")
        configure_codex_exec(
            path=str(fake_codex),
            sandbox="read-only",
            reasoning_effort="none",
            use_sdk="cli",
            network_access=False,
            web_search=False,
        )
        response, raw = run_target_exec(
            work_dir=str(work_dir),
            prompt="Read task.md and answer.",
            model="target-test",
            timeout=10,
            stage="unit_rollout",
        )
    finally:
        set_target_backend(previous_backend)
        _restore_codex(previous_config)

    assert response == "<answer>A</answer>"
    assert '"turn.completed"' in raw
    event = json.loads(token_log.read_text(encoding="utf-8").strip())
    assert event["role"] == "student"
    assert event["provider"] == "codex_exec"
    assert event["input_tokens"] == 11
    assert event["output_tokens"] == 7


def test_codex_timeout_records_partial_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = json.dumps(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 101,
                "output_tokens": 23,
                "total_tokens": 124,
            },
        }
    )

    def fake_run_process(cmd, *, cwd, timeout, env):
        raise subprocess.TimeoutExpired(cmd, timeout, output=raw, stderr="timeout")

    monkeypatch.setattr(
        "cobras.cobras_core.model.exec_harness._run_process",
        fake_run_process,
    )
    previous_backend = get_target_backend()
    previous_config = get_codex_exec_config()
    token_log = tmp_path / "tokens.jsonl"
    work_dir = tmp_path / "workspace"
    prepare_workspace(
        work_dir=str(work_dir),
        skill_md="# Skill",
        task_text="Question.",
    )
    monkeypatch.delenv("COBRAS_EXEC_ALLOWED_ROOT", raising=False)
    monkeypatch.setenv("TARGET_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("TARGET_API_KEY", "dummy")
    monkeypatch.setenv("COBRAS_TOKEN_LOG_PATH", str(token_log))

    try:
        set_target_backend("codex_exec")
        configure_codex_exec(path="/unused/codex", use_sdk="cli")
        with pytest.raises(subprocess.TimeoutExpired):
            run_target_exec(
                work_dir=str(work_dir),
                prompt="Answer.",
                model="target-test",
                timeout=40,
                stage="unit_rollout",
            )
    finally:
        set_target_backend(previous_backend)
        _restore_codex(previous_config)

    event = json.loads(token_log.read_text(encoding="utf-8").strip())
    assert event["provider"] == "codex_exec"
    assert event["input_tokens"] == 101
    assert event["output_tokens"] == 23


def _restore_claude(config: dict[str, object]) -> None:
    configure_claude_code_exec(
        path=str(config["path"]),
        profile=str(config["profile"]),
        use_sdk=str(config["use_sdk"]),
        effort=str(config["effort"]),
        max_thinking_tokens=int(config["max_thinking_tokens"]),
        context_window=int(config["context_window"]),
        max_tokens=int(config["max_tokens"]),
    )


def test_claude_cli_uses_standard_flags_and_records_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_claude = tmp_path / "fake_claude.py"
    fake_claude.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "pathlib.Path('claude_args.json').write_text(json.dumps(sys.argv[1:]))\n"
        "print(json.dumps({'type': 'result', 'result': '<answer>C</answer>', "
        "'usage': {'input_tokens': 17, 'output_tokens': 5, 'total_tokens': 22}}))\n",
        encoding="utf-8",
    )
    fake_claude.chmod(0o755)

    previous_backend = get_target_backend()
    previous_config = get_claude_code_exec_config()
    token_log = tmp_path / "tokens.jsonl"
    work_dir = tmp_path / "workspace"
    prepare_workspace(
        work_dir=str(work_dir),
        skill_md="# Skill",
        task_text="Question.",
    )
    monkeypatch.delenv("COBRAS_EXEC_ALLOWED_ROOT", raising=False)
    monkeypatch.setenv("TARGET_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("TARGET_API_KEY", "dummy")
    monkeypatch.setenv("COBRAS_TOKEN_LOG_PATH", str(token_log))

    try:
        set_target_backend("claude_code_exec")
        configure_claude_code_exec(
            path=str(fake_claude),
            use_sdk="cli",
            effort="none",
            max_thinking_tokens=0,
        )
        response, _ = run_target_exec(
            work_dir=str(work_dir),
            prompt="Answer.",
            model="target-test",
            timeout=10,
            stage="unit_rollout",
        )
    finally:
        set_target_backend(previous_backend)
        _restore_claude(previous_config)

    args = json.loads((work_dir / "claude_args.json").read_text(encoding="utf-8"))
    assert response == "<answer>C</answer>"
    assert "--permission-mode" in args
    assert "dontAsk" in args
    assert "--bare" not in args
    assert "--safe-mode" not in args
    assert "--disable-slash-commands" not in args
    event = json.loads(token_log.read_text(encoding="utf-8").strip())
    assert event["provider"] == "claude_code_exec"
    assert event["input_tokens"] == 17
    assert event["output_tokens"] == 5
