"""Make ``electropolybench`` (in src/) and ``data`` importable without an editable install.

This lets ``pytest`` and the experiment scripts run straight from a fresh checkout.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT / "src"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
