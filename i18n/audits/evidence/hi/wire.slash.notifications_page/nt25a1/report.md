# Independent Hindi audit — wire.slash.notifications_page

Auditor: translation-auditor / audit-hi-c9c5d8079e51, session c9c5d8079e51, model openai/gpt-6.1-sol (quota fallback; source: session model-switch notice). Independent of L1-core-notify coder/translator, who used anthropic/claude-opus-5-5.
Read all 1 English/Hindi key pairs with context, glossary and Hindi style guide. Verdict: **clean-with-findings**, no blocker/major.

**minor wire.slash.notifications_page.help**
Current: सवाल और एजेंट अपडेट; read सब पढ़ा हुआ करता है
Proposed: सवाल व एजेंट अपडेट; read सबको पढ़ा हुआ चिह्नित करे
Reason: सब पढ़ा हुआ करता है is an unnatural literal construction for marking read; the proposed description states the status operation idiomatically, preserves literal read, and measures 38 cells against maxCells 50 (current 36).

## Actual commands
`cd ~/workspace/repos/lo-notifications-l1 && PYTHONPATH="$PWD" .venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/audit-hi-core.py"`
Audit log is captured stdout/stderr from that execution, not a reconstructed prose checklist. This narrow probe imports the repo runtime parser/render_message and plural rules; it does not start the service or claim rendered UI validation.

Source pins verified against exact bytes. Key parity, parse, argument types/names and selectors passed. No native-speaker or rendered-fit review claimed.
