#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cobras.scripts.cobras_dynamic import main


if __name__ == "__main__":
    sys.argv.extend(["--task-package", "cobras.task_envs.socialmaze_hard.scripts"])
    if "--selected-eval-split" not in sys.argv:
        sys.argv.extend(["--selected-eval-split", "train"])
    main()
