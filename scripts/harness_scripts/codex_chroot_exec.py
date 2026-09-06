#!/usr/bin/env python3
"""Run Codex in a per-call unprivileged chroot and export approved files only."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import uuid
from pathlib import Path


COBRAS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTFS = COBRAS_ROOT / ".codex_chroot" / "rootfs"
DEFAULT_ALLOWED_ROOT = COBRAS_ROOT / "outputs"
MAX_EXPORT_BYTES = 8 * 1024 * 1024
TIMEOUT_EXIT_CODE = 124


def _flag_value(args: list[str], *flags: str) -> tuple[int, str]:
    for index, value in enumerate(args[:-1]):
        if value in flags:
            return index, args[index + 1]
    raise ValueError(f"Missing required Codex argument: {'/'.join(flags)}")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _copy_input_tree(src: Path, dst: Path) -> None:
    for current_root, dirs, files in os.walk(src, followlinks=False):
        current = Path(current_root)
        target_dir = dst / current.relative_to(src)
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in list(dirs):
            source = current / name
            if source.is_symlink():
                raise RuntimeError(f"Symlinks are not allowed in Codex input workspace: {source}")
        for name in files:
            source = current / name
            if source.is_symlink() or not source.is_file():
                raise RuntimeError(f"Only regular files are allowed in Codex input workspace: {source}")
            shutil.copy2(source, target_dir / name)


def _chown_tree(path: Path, uid: int, gid: int) -> None:
    os.chown(path, uid, gid)
    for current_root, dirs, files in os.walk(path, followlinks=False):
        current = Path(current_root)
        os.chown(current, uid, gid)
        for name in dirs:
            os.chown(current / name, uid, gid, follow_symlinks=False)
        for name in files:
            os.chown(current / name, uid, gid, follow_symlinks=False)


def _safe_export(src: Path, dst: Path) -> bool:
    try:
        metadata = src.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"Refusing to export non-regular sandbox output: {src}")
    if metadata.st_size > MAX_EXPORT_BYTES:
        raise RuntimeError(
            f"Refusing to export oversized sandbox output ({metadata.st_size} bytes): {src}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(f".{dst.name}.codex-export-{os.getpid()}")
    shutil.copyfile(src, temporary)
    os.replace(temporary, dst)
    return True


def _session_usage(codex_home: Path) -> dict[str, int]:
    """Return the latest cumulative usage from this isolated Codex session."""
    latest: dict[str, int] = {}
    for path in sorted(codex_home.glob("sessions/**/*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            payload = event.get("payload") if isinstance(event, dict) else None
            if not isinstance(payload, dict) or payload.get("type") != "token_count":
                continue
            info = payload.get("info")
            usage = info.get("total_token_usage") if isinstance(info, dict) else None
            if not isinstance(usage, dict):
                continue
            prompt = int(usage.get("input_tokens") or 0)
            completion = int(usage.get("output_tokens") or 0)
            latest = {
                "input_tokens": prompt,
                "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
                "cache_write_input_tokens": int(usage.get("cache_write_input_tokens") or 0),
                "output_tokens": completion,
                "reasoning_output_tokens": int(usage.get("reasoning_output_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or (prompt + completion)),
            }
    return latest


def _has_turn_completed(stdout: bytes) -> bool:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(event, dict) and event.get("type") == "turn.completed":
            return True
    return False


def _append_timeout_usage(stdout: bytes, usage: dict[str, int]) -> bytes:
    if not usage or _has_turn_completed(stdout):
        return stdout
    event = json.dumps(
        {
            "type": "turn.completed",
            "usage": usage,
            "timeout": True,
            "usage_source": "codex_session_token_count",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    separator = b"" if not stdout or stdout.endswith(b"\n") else b"\n"
    return stdout + separator + event + b"\n"


def _run_command(
    command: list[str],
    *,
    env: dict[str, str],
    timeout_seconds: int,
) -> tuple[bytes, bytes, int, bool]:
    """Run Codex and terminate its entire process group on task timeout."""
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return stdout, stderr, int(proc.returncode or 0), False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        return stdout, stderr, TIMEOUT_EXIT_CODE, True


def _rewrite_args(
    args: list[str],
    host_workspace: Path,
    sandbox_workspace: Path,
) -> tuple[list[str], Path, Path, bool]:
    rewritten = list(args)
    work_index, _ = _flag_value(rewritten, "-C", "--cd")
    rewritten[work_index + 1] = str(sandbox_workspace)

    output_index, output_value = _flag_value(rewritten, "-o", "--output-last-message")
    host_output = Path(output_value).resolve()
    if not _inside(host_output, host_workspace):
        raise ValueError(f"Codex output must stay inside its workspace: {host_output}")
    output_relative = host_output.relative_to(host_workspace)
    sandbox_output = sandbox_workspace / output_relative
    rewritten[output_index + 1] = str(sandbox_output)

    index = 0
    while index < len(rewritten):
        value = rewritten[index]
        if value == "--add-dir":
            raise ValueError("Additional host directories are not allowed by the COBRAS Codex chroot.")
        if value in {"-i", "--image"}:
            source = Path(rewritten[index + 1]).resolve()
            if not source.is_file() or source.is_symlink() or not _inside(source, host_workspace):
                raise ValueError(f"Codex image must be a regular workspace-local file: {source}")
            rewritten[index + 1] = str(sandbox_workspace / source.relative_to(host_workspace))
            index += 2
            continue
        index += 1

    allow_workspace_export = False
    cleaned: list[str] = []
    index = 0
    while index < len(rewritten):
        value = rewritten[index]
        if value == "--ephemeral":
            index += 1
            continue
        if value == "--full-auto":
            index += 1
            continue
        if value in {"-s", "--sandbox"}:
            allow_workspace_export = rewritten[index + 1] == "workspace-write"
            index += 2
            continue
        cleaned.append(value)
        index += 1

    insert_at = 1 if cleaned and cleaned[0] == "exec" else 0
    cleaned.insert(insert_at, "--dangerously-bypass-approvals-and-sandbox")
    return cleaned, host_output, sandbox_output, allow_workspace_export


def main() -> int:
    args = sys.argv[1:]
    _, workspace_value = _flag_value(args, "-C", "--cd")
    host_workspace = Path(workspace_value).resolve()
    allowed_root = Path(
        os.environ.get("COBRAS_CODEX_ALLOWED_ROOT", DEFAULT_ALLOWED_ROOT)
    ).resolve()
    if not host_workspace.is_dir() or not _inside(host_workspace, allowed_root):
        raise ValueError(f"Codex workspace is outside allowed root {allowed_root}: {host_workspace}")

    rootfs = Path(os.environ.get("COBRAS_CODEX_ROOTFS", DEFAULT_ROOTFS)).resolve()
    marker = rootfs / ".cobras-codex-rootfs.json"
    if not marker.is_file():
        raise RuntimeError(
            f"Codex chroot is not initialized: {rootfs}; run setup_codex_chroot.py"
        )

    digest = hashlib.sha256(
        f"{host_workspace}:{os.getpid()}:{uuid.uuid4()}".encode()
    ).hexdigest()[:16]
    call_relative = Path("sandbox_runs") / f"call-{digest}"
    call_root = rootfs / call_relative
    host_sandbox_workspace = call_root / "workspace"
    host_sandbox_home = call_root / "home"
    host_sandbox_tmp = call_root / "tmp"
    sandbox_workspace = Path("/") / call_relative / "workspace"
    sandbox_home = Path("/") / call_relative / "home"
    sandbox_tmp = Path("/") / call_relative / "tmp"
    uid = 200000 + (int(digest[:8], 16) % 1000000000)
    gid = uid
    exported: list[str] = []

    try:
        call_root.mkdir(mode=0o700)
        host_sandbox_workspace.mkdir(mode=0o700)
        host_sandbox_home.mkdir(mode=0o700)
        host_sandbox_tmp.mkdir(mode=0o700)
        _copy_input_tree(host_workspace, host_sandbox_workspace)

        codex_home = host_sandbox_home / ".codex"
        codex_home.mkdir(mode=0o700)

        rewritten, host_output, sandbox_output, allow_workspace_export = _rewrite_args(
            args,
            host_workspace,
            sandbox_workspace,
        )
        _chown_tree(call_root, uid, gid)

        clean_env = {
            "PATH": "/opt/cobras/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(sandbox_home),
            "CODEX_HOME": str(sandbox_home / ".codex"),
            "COBRAS_TARGET_API_KEY": os.environ.get("COBRAS_TARGET_API_KEY", "dummy"),
            "LANG": "C",
            "LC_ALL": "C",
            "SHELL": "/bin/bash",
            "TMPDIR": str(sandbox_tmp),
            "TMP": str(sandbox_tmp),
            "TEMP": str(sandbox_tmp),
            "PYTHONDONTWRITEBYTECODE": "1",
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
        command = [
            "/usr/sbin/chroot",
            f"--userspec={uid}:{gid}",
            f"--groups={gid}",
            str(rootfs),
            "/usr/local/bin/codex",
            *rewritten,
        ]
        print(
            f"[codex-chroot] uid={uid} workspace={host_workspace} exposed={sandbox_workspace}",
            file=sys.stderr,
            flush=True,
        )
        timeout_seconds = max(
            1,
            int(float(os.environ.get("COBRAS_CODEX_INNER_TIMEOUT_SECONDS", "600"))),
        )
        stdout, stderr, returncode, timed_out = _run_command(
            command,
            env=clean_env,
            timeout_seconds=timeout_seconds,
        )
        if timed_out:
            usage = _session_usage(codex_home)
            stdout = _append_timeout_usage(stdout, usage)
            recovered = int(usage.get("total_tokens") or 0)
            stderr += (
                f"\n[codex-chroot] timeout after {timeout_seconds}s; "
                f"recovered_tokens={recovered}\n"
            ).encode("utf-8")
        sys.stdout.buffer.write(stdout)
        sys.stderr.buffer.write(stderr)

        sandbox_output_host = rootfs / str(sandbox_output).lstrip("/")
        if _safe_export(sandbox_output_host, host_output):
            exported.append(host_output.name)
        if allow_workspace_export and _safe_export(
            host_sandbox_workspace / "solution.py",
            host_workspace / "solution.py",
        ):
            exported.append("solution.py")
        print(
            f"[codex-chroot] exported={','.join(exported) or 'none'}",
            file=sys.stderr,
            flush=True,
        )
        return returncode
    finally:
        shutil.rmtree(call_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[codex-chroot] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
