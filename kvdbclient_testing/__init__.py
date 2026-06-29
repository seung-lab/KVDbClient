"""Test doubles for the registered kvdbclient backends; not imported by the runtime library.

`backends()` discovers, for every backend registered in `kvdbclient`, the harness that starts a
local instance of it. Adding a backend (with its harness here) makes every consumer's suite run
against it with no change on their side.
"""

import importlib
import importlib.util

from kvdbclient import available_backends

from .base import BackendHarness


def backends():
    """The harness for every registered backend; a registered backend without one is an error."""
    out = []
    for name in available_backends():
        mod_name = f"kvdbclient_testing.{name}.harness"
        if importlib.util.find_spec(mod_name) is None:
            raise RuntimeError(
                f"backend {name!r} is registered but ships no test harness ({mod_name})"
            )
        harness = importlib.import_module(mod_name).harness
        if harness.name != name:
            raise RuntimeError(
                f"test harness for backend {name!r} reports name {harness.name!r}"
            )
        out.append(harness)
    return out
