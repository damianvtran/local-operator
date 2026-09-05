# Background subagent accounting evidence

## Acceptance contract

C2 standard/pre-authorized reliability fix. A background owner's known child
spend must reach a cold viewer without viewer-side provider discovery. Direct
rows remain self/current-attempt only. The session total includes descendants,
prior attempts and retention-swept work exactly once, including after restart.
Known zero, unknown money and a known lower bound remain different states.

No changes to delegation controls, provider billing rates, history navigation,
startup, model-default commands, or the adjacent runtime PRs. No migration of
historically unknown prices into invented dollar amounts.

## Reproduction and real execution

All execution used an isolated HOME and LOCAL_OPERATOR_CONFIG_DIR. The worktree
has its own Python 3.12 editable venv. Production Session, child runner, bash,
transcripts, OwnedSessionHandle, RuntimeServer TCP socket, RemoteSession and
OperatorApp were exercised. Only provider events and owner model-listing prices
were scripted; this proves application accounting, not vendor invoice accuracy.

The original model shape was a dynamically listed Anthropic model, 57 token
receipts and no provider USD receipts. The owner knew its price. The cold viewer
ran `job_stats` in a worker thread; paint-cache refresh refuses an off-loop
caller. Retrying therefore could never populate the cache. Explicit USD receipts
worked throughout, ruling out blanket usage loss in transport.

Independent QA ran separate owner and viewer processes. Before: owner and wire
cost `.006`, viewer row `$—`, context `1200`. After: owner and viewer `.006`.
The coder's real socket reproduction independently returned `.375` owner vs
`None` cold viewer before, and `.375` on both after. Its screen and virtual size
were both 118×38; no screen scrollbar appeared.

| Real path | Actual outcome after fix |
| --- | --- |
| Table estimate, cold viewer | `.006`, with context 1200; no viewer discovery required |
| Receipt control | `.125`, unchanged |
| Mixed `.125` receipt plus unpriced call | row `$0.125+`; unknown amount not erased |
| Estimated `.006` + mixed child | canonical and rendered footer `≥$0.131` |
| Free vs unknown | `$0.0000` vs `$—` |
| Offline reconstructed owner/viewer | estimate `.006`, mixed `.125+`, total `≥.131` retained |
| Live nested work | manager, owner and socket viewer include running grandchild |
| Sweep, restart, new billed work | lifetime survives instead of resetting to retained rows |
| Failed and cancelled billed tool loops | prior `.125` retained; real bash wrote side-effect files |
| Queued child | no provider request before promotion; promoted on capacity release |
| Invalid resume, wrong owner, missing/invalid socket auth | rejected, no unauthorized control |

The committed automated real-path guard is
`tests/e2e/test_subagent_accounting_e2e.py`: a child requests real bash writing
`billed`, reports `.125`, fails its next provider call, resumes to `.25`, is
retention-swept, is reconstructed from disk, and spends another `.125` for `.375`
lifetime. It asserts owner and socket-viewer totals and the actual sidecar.

```sh
env -u NO_COLOR HOME=/tmp/accounting-validation \
  LOCAL_OPERATOR_CONFIG_DIR=/tmp/accounting-validation/config TERM=xterm-256color \
  .venv/bin/python -m pytest tests/e2e -m e2e -n0 -q
# 10 passed, 2 existing Pydantic deprecation warnings
```

The existing cancellation-finalizer test was extended to inspect the durable
checkpoint: descendant cancellation changes 4 tokens to 6 during cleanup;
retention drops the descendant row, but parent and restored ledger keep 6 once.
Fixtures that previously rewrote already-settled display rows now mutate a live
ledger before ownership transfers, matching real runner ordering.

## Rendered frames

Real OperatorApp with its stylesheet, `run_test(130×35)` over RemoteSession;
SVG stills rendered to PNG and viewed. Independent QA captured consecutive
settled frames. No layout, control or navigation change is intended.

- [Before cold viewer](before-cold-viewer.png): row unknown despite priced footer.
- [After cold viewer](after-cold-viewer.png): both show the owner's `.006`.
- [Before mixed money](before-money.png): known mixed receipt omitted.
- [After offline restart](after-offline.png): estimate, genuine zero, unknown and
  partial rows; footer uses the same authoritative `.131` lower bound.

## Persistence and rollback

Optional `Usage.estimated_usd_cost` records a validated nonnegative finite table
estimate separately from authoritative `usd_cost` receipts. Price partitions
remain separate during bounded folding. Outer aggregates carry components, not
an incomplete aggregate receipt. Optional canonical cost fields preserve legacy
readers; absent knowledge uses legacy pricing, explicit unknown does not.
Daemonless cold viewers reconstruct sidecar row summaries only from persisted
receipts/estimates, retaining known lower bounds without provider discovery.
The strict AsyncJob sidecar shape is unchanged.

The standalone session-factory gate was independently run on unmodified main
619ec60a and the accounting head: both report 18 failures and 93 passes, with
identical failed node IDs ([comparison](baseline-factory-failures.txt)). These
existing background-refactor factory contract failures were not silently waived
or modified by the accounting patch. A narrow viewport landing test failed once
in an integrated run, then passed isolated on both branches; the complete view
file passed on main (130) and accounting (129, one existing skip). No viewport
code was changed to mask it.

The existing v1 roster sidecar gains an optional compact `accounting` checkpoint.
Restore replaces row-derived accounting with it; it is never added to retained
rows. Old sidecars still reconstruct from rows. No destructive migration occurs.
Downgrading remains readable but old binaries cannot preserve this new ledger
across future rewrites/sweeps and restore the original viewer limitation.

Observed pre-existing malformed-auth `null` logs AttributeError before closing
the socket. Authorization still denies access. It is outside this accounting
change and was not modified; no issue was created.
