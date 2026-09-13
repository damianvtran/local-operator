# DeepSeek thinking-mode reasoning echo: live check

`reasoning_echo_probe.py` is the manual acceptance check for
`ModelSpec.requires_reasoning_echo` — the wire rule that every assistant turn in
a DeepSeek thinking-mode request must carry back `reasoning_content`, or the API
answers HTTP 400 ("The `reasoning_content` in the thinking mode must be passed
back to the API") for the whole request.

It needs a live DeepSeek credential in the local auth store and spends real
tokens (the session replay below is a ~100k-token request), so it is **not** a
CI test. The unit half of the evidence is
`tests/unit/providers/test_deepseek.py::test_thinking_route_*` and
`tests/unit/harness/test_loop.py::test_reasoning_echo_*`, which pin the body
shape and the bounded recovery with no network at all.

## Run it

```sh
cd <checkout-of-this-branch>
.venv/bin/python docs/evidence/deepseek-reasoning-echo/reasoning_echo_probe.py
.venv/bin/python docs/evidence/deepseek-reasoning-echo/reasoning_echo_probe.py \
    --session ~/.local-operator/sessions/9daa47ece7ad
```

Both shapes go through `OpenAICompatClient._build_body`, so the tool schemas,
system blocks, native replay and every body extra are the runtime's own. Each is
posted twice: once with the capability turned OFF (the body the pre-fix builder
produced) and once with it ON (this branch). Exit status is 0 only when the OFF
body is refused and the ON body is accepted with nothing left blank.

## Observed, 2026-09-12, head `3d765516a`

```
--- minimal synthetic tool loop ---
400  echo OFF (pre-fix body)  messages=6 assistant=2 blank=2
       The `reasoning_content` in the thinking mode must be passed back to the API.
200  echo ON (this fix)  messages=6 assistant=2 blank=0 placeholders=2
recorded credential scope present: True
--- real session 9daa47ece7ad ---
400  echo OFF (pre-fix body)  messages=491 assistant=230 blank=66
       The `reasoning_content` in the thinking mode must be passed back to the API.
200  echo ON (this fix)  messages=491 assistant=230 blank=0 placeholders=66
PASS
```

`9daa47ece7ad` is a session that recorded this refusal; `--session` replays it
with the credential scope its own transcript recorded, which matters — a
mismatched scope invalidates every stored native payload and is itself one of
the ways users meet this 400.

## The pre-fix body is what `main` builds

The capability-off path is byte-identical to the unfixed builder, proven against
a clean `origin/main` worktree rather than argued:

```sh
git worktree add --detach /tmp/lo-before origin/main
PYTHONPATH=/tmp/lo-before .venv/bin/python \
    docs/evidence/deepseek-reasoning-echo/body_digest.py --session ~/.local-operator/sessions/9daa47ece7ad
PYTHONPATH=$PWD .venv/bin/python \
    docs/evidence/deepseek-reasoning-echo/body_digest.py --session ~/.local-operator/sessions/9daa47ece7ad
```

```
tree: /tmp/lo-before/local_operator/__init__.py
sha256: 8e11b60e70c43bb27efcfc0cb35fd1dc0e0fb695db1fe9d84b1fe654d9d041b4
messages: 491 assistant: 230 blank_reasoning: 66
tree: /Users/damian/local-operator-worktrees/deepseek-reasoning-echo/local_operator/__init__.py
sha256: 8e11b60e70c43bb27efcfc0cb35fd1dc0e0fb695db1fe9d84b1fe654d9d041b4
messages: 491 assistant: 230 blank_reasoning: 66
```

The digest prints the file it imported, so a run cannot accidentally report one
tree while reading another. `body_digest.py` needs no credential and makes no
network call.

## What this does NOT show

The runtime's own fresh requests answered 200 on `main` too, in sandboxed
`local-operator exec --resume 9daa47ece7ad` runs with this branch and with
`main`'s tree (`HOME`/`LOCAL_OPERATOR_CONFIG_DIR` redirected, credential store
and session copied; a two-round bash tool loop completed in both, four requests
each, all 200). Bodies logged from those runs carried 66-68 assistant turns with
no `reasoning_content` key and were still accepted, so the refusal depends on a
shape those runs did not hit. It is reproduced at the request level above, on
the request built from the transcript at the point of failure — which is the
request the probe rebuilds, and the shape the operator's own sessions recorded
as `[session incident (deepseek/deepseek-flash)]`.
