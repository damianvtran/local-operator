"""Launch only the attested source runtime with a process-local endpoint shim.

WHY: a global PYTHONPATH shim would leak into task tools. This launcher imports
only in the harness process and never reads a host auth database or persists
credentials. Public transport uses a dummy key; the host meter owns real auth.
CONSTRAINT: source 07c324e9 is not the installed 0.94.22 release. The driver
requires a frozen runtime attestation before this code may be dispatched.
"""

import runpy
import sys


def main():
    import lop_shim  # noqa: F401 -- intentional process-local endpoint rebinding

    sys.argv = ["lop"] + sys.argv[1:]
    runpy.run_module("local_operator.cli", run_name="__main__")


if __name__ == "__main__":
    main()
