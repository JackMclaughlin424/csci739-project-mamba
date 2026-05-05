"""Pytest config — adds ``src/`` to ``sys.path`` so tests can ``import mamba``."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
