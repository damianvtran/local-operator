# Caveat: the state of the cmux host during this capture

Recorded because it bounds what the cmux evidence in this directory can honestly
be claimed to show. It is a fact about the machine, not about this change.

## What was wrong

During validation the installed cmux was mid-update and damaged:

- `/Applications/cmux.app/Contents/MacOS/` was **empty** (mtime 13:55). The main
  application executable was gone from disk.
- The running app (pid 73906, started 13:45) was still executing from memory out
  of that deleted binary, so `cmux version` kept reporting
  `0.64.22 (102) [ddd4a01bc]` and the control socket kept answering.
- The bundled CLI at `Contents/Resources/bin/cmux` was intact, which is why
  every `cmux rpc` call in these captures worked normally.

## What it means for the measurements

Observations taken before and after that ~13:55 boundary are not from the same
build, and behaviour was not self-consistent across it. Specifically:

- Early runs showed a published binding holding `auto_resume: true` for minutes.
- Later runs showed the same binding retiring to `manual` within 2–8 seconds.
- A control run then showed a binding **staying** `auto_resume: true` for 75s
  after its agent process was killed — which a working
  `isStaleAgentHookBinding` should have retired.

The last of those is the informative one: it says cmux's liveness-based
retirement was **not running** at that point. That also means this host could not
be used to demonstrate the retirement rule the vault registration defends
against, and equally could not be used to prove the registration is what defeats
it. On this host, a settled process measured a 100% duty cycle
(45/45 samples over 3 minutes at 4s resolution) both **with** and **without** the
registration installed — so the registration was not the discriminator here.

## What is therefore claimed, and what is not

**Claimed** (independent of the retirement machinery, and directly observed):

- `lop` publishes a correct `agent-hook` binding with `auto_resume: true`.
- That binding reaches the on-disk autosave a crash restores from, against the
  correct workspace and surface ids.
- It is withdrawn on a clean exit, and is never published for a subagent
  session, under the kill switch, without a multiplexer, or without a cmux CLI.

**Not claimed:**

- Any steady-state duty cycle figure as characterising a *healthy* cmux. The
  100% measured here was taken on a host whose retirement path was demonstrably
  inactive, so it is not evidence that retirement has been defeated.
- That the vault registration is what keeps the binding alive. cmux's own
  `docs/vault.md` documents it as the supported mechanism, and it is documented
  as required for that reason — not because this capture isolated its effect.

The re-assert timer in `multiplexer/cmux.py` is retained on the strength of
cmux's source (`SharedLiveAgentIndex.cacheTTL` is 60s and
`retireAgentHookResumeBinding` latches `autoResume = false`), not on the strength
of these measurements. It is cheap, silent, and stops on exit, so it costs
little if this host's behaviour turns out to be the normal one.

A clean reinstall of cmux is worth doing before treating any cmux-side timing
number from this machine as authoritative. It was deliberately not attempted
here: the operator had ~15 live sessions in that process, and restarting cmux to
tidy up an evidence run would have destroyed them.
