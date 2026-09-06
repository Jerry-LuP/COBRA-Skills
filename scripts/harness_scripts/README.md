# Codex chroot harness

The default Codex harness runs each benchmark item in a minimal chroot. This
avoids nested bubblewrap failures on container hosts and prevents the target
agent from reading the repository, experiment outputs, or teacher credentials.

## Setup

Activate the environment used to run COBRA-Skills, ensure `codex` is on
`PATH`, then initialize the rootfs:

```bash
cobra-codex-setup
```

The setup script discovers the active Python environment, the native binary
bundled with the installed Codex CLI, and its bundled `rg`. Override discovery
only when needed:

```bash
cobra-codex-setup \
  --python-env /path/to/conda/env \
  --codex-binary /path/to/native/codex \
  --rg-binary /path/to/rg
```

Linux and root access are required because the harness uses `chroot` and
creates minimal device nodes. Run with `--force` after changing the Python
environment or Codex installation.

## Isolation

Each request receives a copied per-item workspace and a fresh unprivileged UID.
The subprocess gets only the target endpoint credential. Text-task workspace
changes are discarded; SpreadsheetBench exports only `solution.py`. The final
Codex response and token events are returned to the evaluator.
