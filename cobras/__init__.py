"""Source-checkout package shim.

Editable and wheel installs map the repository root to the ``cobras`` package.
This shim makes ``python -m cobras...`` work directly from an unpacked checkout
without requiring an installation step first.
"""
from __future__ import annotations

from pathlib import Path


__path__ = [str(Path(__file__).resolve().parent.parent)]
