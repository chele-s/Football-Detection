"""
Football Tracker Pipeline
Sistema de seguimiento inteligente de balón en tiempo real.
"""

from importlib import import_module
import sys

__version__ = '1.0.0'


def _ensure_case_alias(primary: str, secondary: str) -> None:
    pkg_name = __name__
    primary_mod = f"{pkg_name}.{primary}"
    secondary_mod = f"{pkg_name}.{secondary}"

    def _import(submodule: str):
        try:
            return import_module(f".{submodule}", pkg_name)
        except ModuleNotFoundError:
            return None

    module = _import(primary)
    if module is not None:
        sys.modules.setdefault(secondary_mod, module)
        return

    module = _import(secondary)
    if module is not None:
        sys.modules.setdefault(primary_mod, module)


_ensure_case_alias('Inference', 'inference')
