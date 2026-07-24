from __future__ import annotations

"""Shared progress-bar policy for CLI entry points.

Batch-level tqdm output is useful in an interactive terminal but becomes a very
large notebook cell output under Papermill/Kaggle.  ``BTXRD_DISABLE_TQDM`` can
force either behaviour; otherwise progress bars are shown only on a TTY.
"""

import os
import sys


def should_disable_tqdm() -> bool:
    value = os.environ.get("BTXRD_DISABLE_TQDM")
    if value is not None:
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(
            "BTXRD_DISABLE_TQDM must be one of: 1/0, true/false, yes/no, on/off"
        )
    return not sys.stderr.isatty()
