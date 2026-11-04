# ur / tui.sound — independent audit

Reviewer: C4-i18n-audit / openai/gpt-6.1-sol / 796c1db4d0d0 (no fallback).
Scope: 46053ab26a6395d3e1ea501efc3b7018f39ea1b7..27695a49855fc4ddae0450af5e215346ce5b66b5

Verdict: passed; terminal for these corrected bytes. All 27 values independently compared with English/source context. No unresolved blocker or major.

آپ respectful imperative; اطلاع کی آواز, ترتیب محفوظ; Latin digits; logical-order storage without bidi controls; TUI remains best-effort/degraded.

Corrected keys: tui.sound.save_failed, tui.sound.slash. Full before/after dispositions in record.json.

Key/argument/technical literal/NFC/storage checks and real runtime render passed; no ICU plural/select branches. Screenshot viewed; only transcript copy and 120×40 wrapping are exercised here. Native terminal shaping, full bidi, real audio, and settings UX are not certified.
