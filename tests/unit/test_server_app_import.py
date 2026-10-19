"""Import-time guard for the server application.

``lop serve`` imports every route module before it binds a port, and those
modules construct Pydantic models while importing. A model that cannot be built
therefore takes the whole server down before ``/health`` can answer, while any
test that never imports the app stays green.

v0.77.0 shipped exactly that: ``RadientTranscriptionResponseData`` annotated a
field with ``list[dict[str, Any]]`` in a class body that also defines a ``dict``
method. Python 3.14 evaluates annotations after the class body exists (PEP 649),
so Pydantic resolved ``dict`` to the method and raised
``TypeError: 'function' object is not subscriptable`` at import time. PR CI runs
Python 3.12, where annotations are eager and the same source imports cleanly;
``tests/unit/test_pydantic_annotation_shadowing.py`` is the version-independent
guard for that trap. This test covers every *other* import-time failure.
"""

import importlib


def test_server_app_imports_without_error() -> None:
    """Importing the server app must not raise at import time."""
    app_module = importlib.import_module("local_operator.server.app")

    assert app_module.app is not None
