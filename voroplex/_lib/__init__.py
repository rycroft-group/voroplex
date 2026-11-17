"""Native extension bindings used by VoroPlex."""

import sysconfig

try:
    from . import voroplusplus_
except ImportError as e:
    expected_suffix = sysconfig.get_config_var("EXT_SUFFIX") or "<unknown>"
    raise ImportError(
        f"Failed to import C++ extension: {e}\n"
        "The VoroPlex C++ extension may not be compiled.\n"
        f"Expected extension suffix for this Python is {expected_suffix}.\n"
        "ABI mismatches such as cpython-312 vs cpython-313 require rebuilding.\n"
        "Try running: pip install -e ."
    )

__all__ = ["voroplusplus_"]
