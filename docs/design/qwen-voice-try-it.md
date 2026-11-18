# Trying session voice locally (backend + desktop app)

Status: a demo setup, not a release. Two branches have to be used together.

| Repo | Branch |
|---|---|
| `local-operator` (this one) | `feat/qwen-voice-replay-2026-11-17` |
| `local-operator-ui` | `feat/qwen-voice-ui-2026-11-17` |

Both sit on top of their repo's `main` as of 2026-11-17 (backend) and the
pre-existing `feat/qwen-voice` line (UI). Neither is a PR.

## What it is

Per-session voice mode. The backend adds `/v1/desktop/voice` (slot),
`/utterance`, `/text` and `/speech` (`features.voice: 2`) and uses the
harness's own speech cascades (Radient, ElevenLabs, OpenAI, and Groq as the
last rung). Design: `docs/design/qwen-voice.md` and
`docs/design/session-voice-groq.md`. Voice is OFF by default and off loads
nothing.

## Run it

```sh
# Backend: install this tree and make it the build the desktop app talks to
cd ~/path/to/local-operator
git switch feat/qwen-voice-replay-2026-11-17
uv pip install -e ".[all,dev]" --python .venv/bin/python
.venv/bin/python -c "import local_operator; print(local_operator.__file__)"   # must print THIS checkout

# Desktop app (macOS)
cd ~/path/to/local-operator-ui
git switch feat/qwen-voice-ui-2026-11-17
pnpm install --frozen-lockfile
npx electron-vite build
```

Launch the built app **through macOS** so it is its own app for permissions:

```sh
open -n -a node_modules/.pnpm/electron@*/node_modules/electron/dist/Electron.app \
  --args "$PWD" --user-data-dir="$HOME/qwen-voice-profile" --window-mode=normal
```

Do not start it from a terminal or from an agent session: macOS then attributes
the microphone request to that process, which has no usage description, and kills
the app (SIGABRT, `TCC` in the crash report).

In the app: Settings, Appearance, turn on **Session voice**; open a
conversation; use the voice button in the composer; click Allow on the
microphone prompt.

## What you need

- A speech provider on the backend's account, and ONE chosen: sign in to Radient
  or store an ElevenLabs key (`/login`), then set `speech.provider` to `radient`
  or `elevenlabs` (`lop config edit speech.provider elevenlabs`). Voice uses only
  that provider for hearing and speaking and refuses (`voice_provider_unset`)
  until one is chosen. Stored OpenAI / Groq keys are not used by voice.
- Node 24 for `scripts/i18n/generate.py` if you touch i18n catalogues.
- Speech spends real credits on the account it uses.

## Known gaps

- The UI at this commit still drives the local Qwen rig. The cloud voice page is
  built from pure parts only (`b853cc2246`: "nothing calls it yet"), so the app
  does not yet use the backend's Groq routes.
- Three CI guards fail on the backend branch and need a human decision:
  `test_language_manifest` (the `wire.voice` strings have no translations in the
  other 7 locales), `test_envelope_locale_ratchet` (one new unpinned
  `.envelope()["text"]` in `desktop_voice.py`) and `test_no_session_deletion`
  (four temp-file and cache removals in `desktop_voice.py`, none of which can
  reach `sessions/`).
- Live audio (a real spoken turn) has not been exercised end to end.
