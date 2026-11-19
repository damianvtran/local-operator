# H-B body rows: Mac-lane QA evidence for local-operator #4238 and local-operator-ui #1977

This branch is evidence only. It is never merged.

**Host.** The operator's Mac (Darwin arm64).

**Builds under test.**

| Repo | Commit | Notes |
|---|---|---|
| core #4238 | `1cdf69a0` | Its runtime product code is byte-identical to the current head `48512042`: `git diff --quiet 1cdf69a0 48512042 -- local_operator` returns 0, and the delta is one black-formatting test file. |
| UI #1977 | `5f56c659` | Merged as `7eb45184` while this lane was running. |

**Isolation.** Every process ran under `env -i` with:

- `HOME` set to a scratch directory, `LOCAL_OPERATOR_CONFIG_DIR=$HOME/.local-operator`, and the working directory inside that home;
- `GIT_CONFIG_SYSTEM=/dev/null`;
- no inherited `CMUX_*` or `LOP_*` variables.

The operator's real store was not touched. `~/.local-operator/team_bridge/display_bodies.sqlite3` does not exist after the run.

## What is real and what is a stub

**The hub is the only stub.** `stub_hub.py` is a real HTTP server on loopback. The product's own unpatched `TeamBridgeClient` reached it through `RADIENT_API_BASE_URL=http://127.0.0.1:<port>` and `RADIENT_ORG_ALLOW_NONCANONICAL_BASE=1`. That means these all ran as shipped:

- URL and query building;
- bearer attachment;
- the house `{msg, result}` envelope parse;
- HTTP status → reason mapping;
- the destination guard.

**Everything else is real:** `lop serve`, `lop mobile serve` (the relay), `lop logout radient`, `PATCH /v1/settings`, the stores, and the resolver.

**The stub refuses two kinds of request:**

- the classification fetch `/messages/{id}/body` answers 418 and logs `BOMB`;
- any write method answers 405 and logs `WRITE`.

Across every log: `BOMB` = 0 and `WRITE` = 0.

## Seeding

`seed.py` seeds the store through the PR's own APIs:

- `Store.commit_enrollment`, with the owner's `canonical_base_url` set to the stub origin;
- a Radient OAuth row in `auth.db` (credential id 7, `account_id` `acc_1`);
- `publish_screening_policy` and `record_body_reservation`;
- the **real single writer**: `PendingTurn.open` → `for_call` → `settle`, which returned `completed`;
- one unsettled reservation, for the pending state;
- `persist_hold` with `model_hold` and with `provider_error`.

Fourteen journal cards are written in the real `type:message` / `kind:custom` shape. All ids are synthetic: `acc_1`, `ep_1`, `bcv_abcdefgh`, `bcmsg_*`, and session `abcdef012345`.

## Scripts

| Script | What it does |
|---|---|
| `up.sh` / `down.sh` | Restore a pristine config, then start or stop the stub and `lop serve` as process-group leaders. Teardown uses `kill -- -pgid`, and `kill -0` is checked afterwards. |
| `q.sh` | One authenticated desktop-route read. |
| `rows.py` | Read-only dump of the display store. |
| `switch.py` | Re-enrolls the store under another account (`acc_2`/`ep_2`). |
| `model/run.mts` | Runs the UI's production `bridgeCardBodyOf`, copied from 5f56c659 with only the import extension changed, over the real JSON that `lop serve` returned (`real/*.json`). |

## Frames

`frames/sb-*.png` are Storybook frames from the harness browser, taken at UI 5f56c659 through `scripts/with-storybook.mjs`.
