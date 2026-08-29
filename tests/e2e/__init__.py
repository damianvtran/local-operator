"""Headless end-to-end tests that drive the real TUI application.

Separate from ``tests/unit`` because these are a different KIND of test with a
different failure mode. A unit test that breaks prints an assertion; a test in
here breaks by HANGING, because the defects it exists to catch are event-loop
freezes. That difference is why this package carries its own watchdog
(:mod:`tests.e2e.watchdog`) and its own ``e2e`` marker, and why the default
``pytest`` run deselects it (see ``addopts`` in ``pyproject.toml``): the
bounding machinery is load-bearing rather than incidental, and it should not
be paid for on every unit run.
"""
