# Output attachments — end-to-end evidence

The output-attachment contract's own proof on the harness side: a REAL session
whose transcript carries an `AttachmentContent` block (registered through the
real `cache_media`) plus both legacy image shapes — read back by the real
reader, mounted by the real TUI, degraded honestly when a store is gone, and
served to the phone. Desktop-UI frames live with the desktop scene in
`damianvtran/local-operator-ui:docs/evidence/output-artifact/`.

Every run below uses an ISOLATED config root (`env -i HOME=… LOCAL_OPERATOR_CONFIG_DIR=…`);
the operator's live sessions are only ever READ (`cp -c`, byte-identical and
cheap) and never written.

## 1. Seed the session (real code paths)

```sh
RIG=<scratch>/rig
mkdir -p "$RIG/home/.local-operator"
printf 'values:\n  hosting: test\n  model_name: mock-model\n' > "$RIG/home/.local-operator/config.yml"
env -i HOME="$RIG/home" LOCAL_OPERATOR_CONFIG_DIR="$RIG/home/.local-operator" \
  PATH="$PATH" TERM=xterm-256color \
  <worktree>/.venv/bin/python docs/evidence/output-attachments/seed_output_artifact.py "$RIG"
```

Reading: `block shapes on disk: [text, text, image, text, legacy-ref, text,
inline, text]` — one artifact (the new block), one externalised digest
reference and one sub-floor inline `data:` block (the two legacy shapes) — and
`artifact digest: 40d8683fdcafc5ef8d7a98169d9772a2`, `store file exists: True
9436`, `artifacts replayed: 1 {'kind': 'image', 'content_type': 'image/png',
'attachment': '40d8683f…', 'source_url': 'https://provider.example/…',
'size_bytes': 9436, 'width': 640, 'height': 360, 'name': 'output-artifact.png'}`.

## 2. Serving — desktop route and phone daemon

```sh
env -i HOME="$RIG/home" LOCAL_OPERATOR_CONFIG_DIR="$RIG/home/.local-operator" \
  PATH="$PATH" TERM=xterm-256color LOCAL_OPERATOR_DESKTOP_TOKEN=<token> \
  <worktree>/.venv/bin/local-operator serve --host 127.0.0.1 --port <free> \
  --hosting test --model mock-model
```

```
$ curl -H "Authorization: Bearer …" .../sessions/a77ac41f0001/history | head -c …
  → the seeded rows, artifact block intact
$ curl -H "Authorization: Bearer …" .../sessions/a77ac41f0001/attachments/40d8683f…
  200 image/png 9436            # viewed: the striped 640x360 artifact
```

The PHONE daemon (same store, the endpoint the phone's `<img>` fetches):

```sh
env -i HOME="$RIG/home" … LOP_MOBILE_PASSWORD=$(cat "$RIG/mobile.pass") \
  <worktree>/.venv/bin/local-operator mobile serve --port <free>
```

```
$ curl -sd "password=…" -c cookies /login                       → 303
$ curl -b cookies /api/sessions                                 → "a77ac41f0001","Artifact demo"
$ curl -b cookies "/api/sessions/a77ac41f0001/image?entry=cd8d6326…&i=0"
  200 image/png 9436            # the artifact, to the phone
$ curl -b cookies "…&i=9"                                        → 404
$ curl            "…&i=0"                                        → 401
```

The phone WEB currently paints images on USER rows only; a tool row's artifact
cell is the surfaces lane's follow-on (the daemon half — refs and bytes — is
what these readings prove). `LOP_MOBILE_PASSWORD` is the documented dev path,
so no Keychain is touched under the isolated HOME.

## 3. TUI — a real resume, read by a second process

The session was written by the seed process; the frames below are that saved
session re-read from disk twice — once by a real pty (`local-operator --resume
a77ac41f0001` in a console surface) and once by the compositor capture:

```sh
env -u NO_COLOR TERM=xterm-256color <worktree>/.venv/bin/python \
  docs/evidence/output-attachments/artifact_resume_shot.py \
  "$RIG/home/.local-operator/sessions/a77ac41f0001" out.svg
rsvg-convert out.svg -o out.png
```

Reading: `mounted blocks: 8; image blocks: 3` — the artifact plus both legacy
shapes mount on the real replay, and `tui-artifact-resume.png` shows them under
the settled `generate_image … 4.2s` card. `.geometry.json` beside the PNG
carries the capture helper's own screen/widget sizes.

## 4. An old real transcript keeps working (tolerant reader)

A 2026-09-16 user session with legacy image references, copied read-only with
`cp -c` and replayed under this build:

```
$ <the same shot script> <old-copy> tui-old-transcript.svg
store: 2 digest(s) referenced, 0 file(s) copied
transcript references missing attachment 780cab9c…   # pruned since September
transcript references missing attachment 5d474700…
mounted blocks: 68; image blocks: 0
```

The whole pre-feature transcript parses and paints (68 blocks), and the two
digests this store no longer holds degrade to the unavailable receipt — the
honest path — instead of crashing or vanishing. A second old session whose
digests DO survive (`9a3dd391bff1`) mounts its pictures:
`mounted blocks: 116; image blocks: 2` → `tui-legacy-images.png`.

## Frames

- `tui-artifact-resume.png` — the artifact + both legacy images on a real
  resumed session (the main TUI claim).
- `tui-legacy-images.png` — a pre-existing session's own images mounting.
- `tui-old-transcript.png` — the 09-16 transcript painting with pruned digests
  degraded.
- Scripts: `seed_output_artifact.py` (the fixture), `artifact_resume_shot.py`
  (capture; `isolate_capture()` before app imports, per the repo's Visual
  validation recipe).

## Additional states (design review round 1, D1 addendum)

Two further states were re-shot by the design round from the same rig (scripts
`shoot_receipt.py`, `shoot_narrow.py`; geometry in `receipt.geometry.json`):

- `receipt.png` — the MISSING-STORE state: the artifact and the externalised
  legacy image degrade to `image unavailable — no longer in the transcript`
  under an amber explanation, while the sub-floor inline image still paints
  from its own bytes. No crash, no vanishing.
- `narrow.png` — the same transcript at 80x24.
