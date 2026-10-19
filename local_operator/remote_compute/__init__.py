"""Remote compute: the preview switch and the tenant binding every remote run needs.

WHAT THIS PACKAGE IS FOR. A remote run leaves the operator's machine, so two
questions have to be answered before one can be built at all: is the preview
switched on, and WHICH mesh network does this workload belong to. The second is
the tenant segmentation guarantee on the client side — a request may only carry a
network THIS DEVICE IS A MEMBER OF, resolved from the local mesh store, never a
name a caller invented and never a guess between two networks. Everything here
serves those two answers; the transport, the wire fields besides ``network_id``
and every surface that offers the feature belong to later slices.

IMPORTS ARE DELIBERATELY LIGHT. :func:`is_enabled` is called from
``server.features.feature_flags()``, which runs on every capability negotiation,
so this package's module-scope imports are stdlib plus its own two submodules.
The config reader, :mod:`local_operator.paths` and the mesh store are imported
inside the functions that need them: a capability read must not pay for the
mesh's record/identity graph, and the import-graph guard in
``tests/unit/test_import_graph.py`` exists because that kind of eager import is
invisible in review.
"""

from __future__ import annotations

from pathlib import Path

from local_operator.remote_compute.request import RunCreateRequest
from local_operator.remote_compute.resolve import (
    CODE_AMBIGUOUS_NETWORK,
    CODE_DUPLICATE_NETWORK_ID,
    CODE_INVALID_NETWORK_ID,
    CODE_NO_NETWORK,
    CODE_NOT_A_MEMBER,
    CODE_UNKNOWN_NETWORK,
    NETWORK_ID_PATTERN,
    NetworkBinding,
    resolve_network,
)

#: What ``remote_compute.enabled`` falls back to when the key is absent.
#:
#: The REAL default, imported by ``_consumer_defaults()`` in
#: ``tests/unit/test_settings_io.py`` so the registry row and this reader cannot
#: drift. OFF because the feature sends work to Radient-hosted compute: a
#: preview that shipped switched on would move a user's first task off their
#: machine before they asked for it.
DEFAULT_REMOTE_COMPUTE_ENABLED = False

#: The path of the switch inside ``values`` — one spelling, shared by the reader
#: and the registry row's ``path`` (which is spelled in ``settings_io``).
SETTING_PATH: tuple[str, ...] = ("remote_compute", "enabled")


def is_enabled(config_dir: Path | None = None) -> bool:
    """Whether the remote-compute preview is on RIGHT NOW.

    Reads ``config.yml`` from disk on every call rather than through a
    long-lived :class:`~local_operator.config.ConfigManager`, because the
    capability bit this gates must follow a live toggle: a server holds the
    config it read at boot, so a write made by the TUI, the CLI or another
    process would otherwise never reach a running daemon. This is the reading
    ``web_fetch`` does per call (a fresh manager there) done with the
    side-effect-free reader instead, for the reason
    :func:`~local_operator.config.read_config_values` documents: a timer- or
    request-driven reader must never move an unparseable file aside.

    FAILS CLOSED. ``read_config_values`` answers ``None`` for a file that cannot
    be read or parsed right now (a torn editor save); that is not evidence the
    user turned the switch on, so the default applies. Only a real boolean
    ``true`` enables it — a hand-edited ``"yes"`` or ``1`` is not coerced,
    because the value decides whether work leaves the device.
    """
    # IMPORTED HERE, not at module scope: see the module docstring.
    from local_operator.config import read_config_values
    from local_operator.paths import config_dir as default_config_dir

    values = read_config_values(config_dir if config_dir is not None else default_config_dir())
    if values is None:
        return DEFAULT_REMOTE_COMPUTE_ENABLED
    section = values.get(SETTING_PATH[0])
    if not isinstance(section, dict):
        return DEFAULT_REMOTE_COMPUTE_ENABLED
    return section.get(SETTING_PATH[1], DEFAULT_REMOTE_COMPUTE_ENABLED) is True


__all__ = [
    "CODE_AMBIGUOUS_NETWORK",
    "CODE_DUPLICATE_NETWORK_ID",
    "CODE_INVALID_NETWORK_ID",
    "CODE_NOT_A_MEMBER",
    "CODE_NO_NETWORK",
    "CODE_UNKNOWN_NETWORK",
    "DEFAULT_REMOTE_COMPUTE_ENABLED",
    "NETWORK_ID_PATTERN",
    "NetworkBinding",
    "RunCreateRequest",
    "SETTING_PATH",
    "is_enabled",
    "resolve_network",
]
