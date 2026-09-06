#!/usr/bin/env python3
"""Build the minimal immutable root filesystem used by COBRAS Codex target calls."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path


COBRAS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTFS = COBRAS_ROOT / ".codex_chroot" / "rootfs"
DEFAULT_PYTHON_ENV = Path(sys.prefix)

CORE_TOOLS = (
    "/bin/bash",
    "/usr/bin/basename",
    "/usr/bin/cat",
    "/usr/bin/chmod",
    "/usr/bin/cp",
    "/usr/bin/cut",
    "/usr/bin/dirname",
    "/usr/bin/env",
    "/usr/bin/find",
    "/usr/bin/grep",
    "/usr/bin/head",
    "/usr/bin/ls",
    "/usr/bin/mkdir",
    "/usr/bin/mv",
    "/usr/bin/printf",
    "/usr/bin/readlink",
    "/usr/bin/rm",
    "/usr/bin/sed",
    "/usr/bin/sort",
    "/usr/bin/stat",
    "/usr/bin/tail",
    "/usr/bin/tee",
    "/usr/bin/touch",
    "/usr/bin/tr",
    "/usr/bin/uniq",
    "/usr/bin/wc",
    "/usr/bin/xargs",
)


def _is_native_binary(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return path.is_file() and stream.read(4) == b"\x7fELF"
    except OSError:
        return False


def _codex_binary() -> Path:
    executable = shutil.which("codex")
    if not executable:
        raise FileNotFoundError("codex is not on PATH; install the Codex CLI first")
    resolved = Path(executable).resolve()
    if _is_native_binary(resolved):
        return resolved
    package_root = resolved.parent.parent
    candidates = sorted(
        package_root.glob("node_modules/@openai/codex-*/vendor/*/bin/codex")
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate the native Codex binary next to {resolved}; "
            "pass --codex-binary explicitly"
        )
    return candidates[0]


def _rg_binary(codex_binary: Path) -> Path:
    bundled = codex_binary.parent.parent / "codex-path" / "rg"
    if bundled.is_file():
        return bundled
    executable = shutil.which("rg")
    if not executable:
        raise FileNotFoundError("rg is not on PATH; pass --rg-binary explicitly")
    return Path(executable).resolve()


def _readonly_mode(mode: int) -> int:
    return stat.S_IMODE(mode) & ~0o022


def _copy_file(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst.chmod(_readonly_mode(src.stat().st_mode))


def _ldd_dependencies(binary: Path) -> set[Path]:
    completed = subprocess.run(
        ["ldd", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    dependencies: set[Path] = set()
    for line in completed.stdout.splitlines():
        match = re.search(r"=>\s+(/\S+)", line)
        if not match:
            match = re.match(r"\s*(/\S+)", line)
        if match:
            path = Path(match.group(1))
            if path.exists():
                dependencies.add(path)
    return dependencies


def _copy_binary(rootfs: Path, path: str | Path) -> None:
    requested = Path(path)
    if not requested.exists():
        raise FileNotFoundError(requested)
    _copy_file(requested, rootfs / requested.relative_to("/"))
    for dependency in _ldd_dependencies(requested.resolve()):
        _copy_file(dependency, rootfs / dependency.relative_to("/"))


def _copy_tree_readonly(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=False)
    for current_root, dirs, files in os.walk(dst):
        current = Path(current_root)
        current.chmod(_readonly_mode(current.stat().st_mode) or 0o755)
        for name in dirs:
            path = current / name
            if not path.is_symlink():
                path.chmod(_readonly_mode(path.stat().st_mode) or 0o755)
        for name in files:
            path = current / name
            if not path.is_symlink():
                path.chmod(_readonly_mode(path.stat().st_mode))


def _copy_python_runtime(rootfs: Path, python_env: Path) -> None:
    python = python_env / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(f"Python executable not found under {python_env}")
    version = subprocess.run(
        [str(python), "-c", "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    python_binary = python_env / "bin" / version
    if not python_binary.exists():
        python_binary = python.resolve()

    prefix = rootfs / "opt" / "cobras"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    _copy_file(python_binary, prefix / "bin" / version)
    for dependency in _ldd_dependencies(python_binary.resolve()):
        _copy_file(dependency, rootfs / dependency.relative_to("/"))
    (prefix / "bin" / "python").symlink_to(version)
    (prefix / "bin" / "python3").symlink_to(version)

    stdlib_src = python_env / "lib" / version
    stdlib_dst = prefix / "lib" / version
    shutil.copytree(
        stdlib_src,
        stdlib_dst,
        symlinks=False,
        ignore=shutil.ignore_patterns("site-packages"),
    )
    site_packages = stdlib_dst / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    for pattern in ("openpyxl", "openpyxl-*.dist-info", "et_xmlfile", "et_xmlfile-*.dist-info"):
        for src in (stdlib_src / "site-packages").glob(pattern):
            dst = site_packages / src.name
            if src.is_dir():
                _copy_tree_readonly(src, dst)
            else:
                _copy_file(src, dst)

    for src in (python_env / "lib").glob("*.so*"):
        dst = prefix / "lib" / src.name
        if src.is_symlink():
            dst.symlink_to(os.readlink(src))
        elif src.is_file():
            _copy_file(src, dst)

    for current_root, dirs, files in os.walk(prefix):
        current = Path(current_root)
        current.chmod(0o755)
        for name in dirs:
            path = current / name
            if not path.is_symlink():
                path.chmod(0o755)
        for name in files:
            path = current / name
            if not path.is_symlink():
                path.chmod(_readonly_mode(path.stat().st_mode))


def _make_devices(rootfs: Path) -> None:
    dev = rootfs / "dev"
    dev.mkdir(mode=0o755)
    for name, major, minor, mode in (
        ("null", 1, 3, 0o666),
        ("zero", 1, 5, 0o666),
        ("urandom", 1, 9, 0o444),
    ):
        path = dev / name
        os.mknod(path, stat.S_IFCHR | mode, os.makedev(major, minor))
        path.chmod(mode)


def build_rootfs(rootfs: Path, python_env: Path, codex_binary: Path, rg_binary: Path, force: bool) -> None:
    marker = rootfs / ".cobras-codex-rootfs.json"
    identity = {
        "version": 2,
        "python_env": str(python_env.resolve()),
        "python_mtime_ns": (python_env / "bin" / "python").resolve().stat().st_mtime_ns,
        "codex_binary": str(codex_binary.resolve()),
        "codex_mtime_ns": codex_binary.stat().st_mtime_ns,
    }
    if marker.is_file() and not force:
        if json.loads(marker.read_text(encoding="utf-8")) == identity:
            print(f"[codex-chroot-setup] ready rootfs={rootfs}")
            return
        raise RuntimeError(f"Existing rootfs does not match current runtime: {rootfs}; rerun with --force")

    build_dir = rootfs.with_name(f"{rootfs.name}.building-{os.getpid()}")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    try:
        for directory in (
            "bin",
            "etc",
            "home",
            "lib",
            "lib64",
            "opt",
            "sandbox_runs",
            "tmp",
            "usr/bin",
            "usr/local/bin",
        ):
            (build_dir / directory).mkdir(parents=True, exist_ok=True)
        (build_dir / "sandbox_runs").chmod(0o711)
        # Each Codex call receives its own writable TMPDIR under sandbox_runs.
        (build_dir / "tmp").chmod(0o555)

        _copy_file(codex_binary, build_dir / "usr" / "local" / "bin" / "codex")
        _copy_file(rg_binary, build_dir / "usr" / "bin" / "rg")
        for tool in CORE_TOOLS:
            _copy_binary(build_dir, tool)
        (build_dir / "bin" / "sh").symlink_to("bash")
        (build_dir / "proc" / "self").mkdir(parents=True)
        (build_dir / "proc" / "self" / "exe").symlink_to("/usr/local/bin/codex")
        _copy_python_runtime(build_dir, python_env)
        _make_devices(build_dir)

        (build_dir / "etc" / "passwd").write_text("root:x:0:0:root:/root:/bin/bash\n", encoding="utf-8")
        (build_dir / "etc" / "group").write_text("root:x:0:\n", encoding="utf-8")
        for name in ("hosts", "nsswitch.conf", "resolv.conf"):
            src = Path("/etc") / name
            if src.is_file():
                _copy_file(src, build_dir / "etc" / name)
        cert = Path("/etc/ssl/certs/ca-certificates.crt")
        if cert.is_file():
            _copy_file(cert, build_dir / cert.relative_to("/"))

        marker_path = build_dir / marker.name
        marker_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        marker_path.chmod(0o444)
        build_dir.chmod(0o755)

        if rootfs.exists():
            if not force:
                raise FileExistsError(rootfs)
            shutil.rmtree(rootfs)
        rootfs.parent.mkdir(parents=True, exist_ok=True)
        build_dir.rename(rootfs)
    except BaseException:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise

    print(f"[codex-chroot-setup] built rootfs={rootfs}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rootfs",
        type=Path,
        default=Path(os.environ.get("COBRAS_CODEX_ROOTFS", DEFAULT_ROOTFS)),
    )
    parser.add_argument(
        "--python-env",
        type=Path,
        default=Path(os.environ.get("COBRAS_CODEX_PYTHON_ENV", DEFAULT_PYTHON_ENV)),
    )
    parser.add_argument(
        "--codex-binary",
        type=Path,
        default=None,
    )
    parser.add_argument("--rg-binary", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    codex_binary = args.codex_binary
    if codex_binary is None:
        configured = os.environ.get("COBRAS_CODEX_NATIVE", "").strip()
        codex_binary = Path(configured) if configured else _codex_binary()
    rg_binary = args.rg_binary or _rg_binary(codex_binary.resolve())
    build_rootfs(
        args.rootfs.resolve(),
        args.python_env.resolve(),
        codex_binary.resolve(),
        rg_binary.resolve(),
        args.force,
    )


if __name__ == "__main__":
    main()
