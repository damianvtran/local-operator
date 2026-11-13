"""Remote files parity: a viewer's bounded cache of a peer session's bytes.

docs/design/remote-files-parity.md. ``cache`` is the evictable store the viewer
lands pulled bytes in (never the authoritative ``<config>/attachments``);
``fetch`` is the single-flight fetch-on-miss the desktop routes wait on.
"""
