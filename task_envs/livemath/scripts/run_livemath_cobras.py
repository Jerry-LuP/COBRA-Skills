#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cobras.scripts.cobras_dynamic import main


if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv.extend(["--dataset", "livemath"])
    if "--selected-eval-split" not in sys.argv:
        sys.argv.extend(["--selected-eval-split", "train"])
    main()
