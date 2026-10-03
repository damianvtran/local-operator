# Vendored voicing map

`speech_voicing_map.v1.json` and `vectors.v1.json` are **byte-for-byte copies** of the
canonical artifacts that live in `radient-ml/agent-server` at
`internal/services/speech_voicing/`. The hub serves the map verbatim at
`GET /v1/tools/speech/voicing`, and both repositories run the same golden
vectors, which is how the Go executor and this Python one are kept from
drifting: a mapping change that lands in one repo and not the other turns a
test red instead of changing how a user's speech sounds.

Nothing here is edited by hand. The map is data, not code — the two rules it
exists to enforce are:

1. every descriptor field is either `applied` exactly or named in a `Note`,
   and
2. identical `(descriptor, map_version)` always produce identical params.

## Sync discipline

**A map update is vendored, never re-authored.** The procedure is:

1. Land the change in `agent-server` first. The hub is the source of truth and
   it is what a descriptor-bearing request actually talks to; a daemon that
   moved first would map to a version the hub has never heard of.
2. Copy both files across verbatim and record the source commit:
   ```sh
   cd ~/radient-ml/agent-server && git show origin/main:internal/services/speech_voicing/speech_voicing_map.v1.json \
     > ~/local-operator/local_operator/tts/voicing_map/speech_voicing_map.v1.json
   cd ~/radient-ml/agent-server && git show origin/main:internal/services/speech_voicing/vectors.v1.json \
     > ~/local-operator/local_operator/tts/voicing_map/vectors.v1.json
   ```
3. Update `VENDORED_VERSION` in `local_operator/tts/adapters.py` and the
   `VENDORED_FROM` commit line beside it.
4. Run `pytest tests/unit/tts -q`. The conformance test runs every vector
   against this repo's adapters; a MAJOR map bump that changes an existing
   vector's expectation fails until the Python executor is updated to match.

**A MINOR bump** (new rows, new tones, new phrases) needs no code change:
the adapters read the file. **A MAJOR bump** (a meaning change, or a changed
expectation for an existing descriptor) is a code change here as well, because
the Python executor has to reproduce the new output exactly.

## Version guard

`adapters.VENDORED_VERSION` must equal the file's own `map_version`. The
conformance test asserts it, so a half-vendored pair — new JSON, stale
version constant — is a red test rather than a silent skew. The daemon also
reports the version it mapped with (`X-Radient-Speech-Map` is the hub's; the
daemon's own report is in `/v1/tts/paths` and the speech route's headers), so
a skew between a pinned daemon and a moved hub is visible in the field rather
than only in the audio.

## Why not fetch the map at runtime?

The daemon must map descriptors **offline** — it maps for BYO rungs where the
user's own key is the only credential, and it can be run with no Radient
account at all. A runtime refresh from the hub is an optimisation, never a
requirement, and it must never trust a newer major (see the design note §3).
