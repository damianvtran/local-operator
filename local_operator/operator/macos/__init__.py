"""The macOS key agent: the presence-bearing process the operator key lives in.

One module, :mod:`local_operator.operator.macos.keyagent`, and one reason for the
package: the helper must sit at a PREDICTABLE path inside an installation
(``site-packages/local_operator/operator/macos/lop-keyagent.app``) so that a wheel
can carry it, and so that ``keyagent.helper_bundle_path()`` resolves it identically
for an editable install, a ``uv tool`` generation and a checkout. The bundle itself
is not in this repository — it is built and signed by
``packaging/macos/assemble_keyagent_bundle.sh`` and injected into the macOS wheel
by ``packaging/macos/make_macos_wheel.py``, because a signed artefact cannot be
built on a host without the signing identity, and because committing a signed
binary would put a new blob in git on every rebuild.
"""
