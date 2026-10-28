# vi / wire.errors.notifications independent translation audit

Auditor: translation-auditor (audit-vi, lopdev team)
Session: cfdff5945838
Actual model: openai/gpt-6.1-sol
Host model-switch notice: openai/gpt-6.1-sol after anthropic/claude-opus-5-5 and anthropic/claude-sonnet-5-5 quota failures; substantive audit performed on this fallback.
Separate auditing agent/context from UI translator tr-vi and L1-core-notify coder; authored none of the translations; did not edit catalogues.

## Verdict
passed; 4 English and 4 Vietnamese keys read side by side with the context sidecar.
All four error keys are clean: lock contention preserves no-write/retry guidance; full disk preserves volume and {where}; unavailable store preserves non-retryability and {where}; unknown item names Thông báo, never Inbox. Tone/register and storage terminology are coherent. No plural/markup/date skeleton is present.

## Execution evidence
The adjacent audit.log is actual stdout from the repository runtime ICU parser and renderer, not reconstructed notes. Source hash assertions and key/argument parity passed. Synthetic /synthetic/config proves {where} survives interpolation. Reproduction script is retained as audit-script.py.txt.

Command executed (one namespace per invocation):
```sh
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/vi-isolated-home" LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/vi-isolated-home/config" PATH="$PATH" PYTHONPATH=~/workspace/repos/lo-notifications-l1 ~/local-operator/.venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/core-vi-check.py" wire.errors.notifications
```

Setup recovery: an initial import of optional wcwidth failed (ModuleNotFoundError); no dependency was installed. The final successful script derives cells from Unicode NFC/combining/East-Asian-width properties, asserting no combining/control codepoints. These short Vietnamese/ASCII messages contain no wide characters. This is a string-cell check, not native rendered-fit evidence.

## Limits
LLM review only; no human native-speaker review, browser/native rendering, release eligibility, or whole-locale audit is claimed. No catalogue, code, generated file, ledger or git state was changed.
