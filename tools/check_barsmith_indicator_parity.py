"""
Compatibility shim.

Historical code imported `scripts.check_barsmith_indicator_parity`. Keep that import path
working while the implementation lives under `algotrade.barsmith_parity`.
"""

from algotrade.barsmith_parity.check_barsmith_indicator_parity import *  # noqa: F403
