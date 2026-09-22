"""Refuse a tool call in which a secret-bearing value reaches a printing sink.

**Why this exists.** Until this module, the only thing standing between a
``lop secret get`` value and the model's context was an OUTPUT FILTER:
``local_operator/secrets/runtime.py`` registers each value it hands out and
``str.replace``-es it out of every result afterwards. A filter decides after the
decision to print has already been made, and it only recognises the spellings it
holds. A real incident in a Minerva QA session is the demonstration: an agent ran

    for k in A B C D; do v=$(lop secret get "$k") && echo "$k = $v" || echo "$k = MISSING"; done

and the values were in the transcript; then, to read a hostname the mask kept
replacing, the agent printed it reversed. The first half is the accident this
scan refuses before a child process exists. The second half is why the rule keys
on the DATA FLOW — a secret-bearing source reaching a printing sink — and never
on the value's text: a re-spelling (``rev``, ``base64``, a substring, a
different quoting) is still the same flow, so it is still refused.

**Where it runs.** ``local_operator/tools/builtin.execute_bash`` and
``local_operator/tools/eval.execute_eval`` call this beside their argument
validation, BEFORE the spawn. ``execute_bash`` deliberately has no second
approval gate (write/exec approval is the LOOP's; see its comment), so this
REFUSES: it returns an error tool result naming the rule, the span and the
rewrite, and never prompts.

**What it is not.** It is not a sandbox and it does not look for values — it
looks for the FLOW. So it cannot see a value the model already holds (only the
pre-execution refusal prevents that in the first place), it does not model a
script invoked by name that an earlier call wrote, and it does not follow a
value out of the tool in a later call (that is the session-scoped tainted-path
ledger in the design doc, deliberately not part of this module: this module is a
pure function with no I/O and no state, which is what makes it testable one rule
at a time).

**Cost.** Stdlib only (``ast``, ``re``, ``dataclasses``, ``typing``) and cheap to
import, because it sits on the bash and eval tool path — the same constraint
``secrets/cli.py`` documents for the CLI half. Every scan begins with a
prefilter that returns ``none`` without tokenizing unless the text can carry a
source at all (see :func:`may_carry_a_shell_source`).

Design of record: ``docs/design/secret-get-hardening.md`` §3, on branch
``design/secret-get-hardening``.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

#: What the scan concluded about the text as a whole.
#:
#: ``none``        nothing in the text can produce a stored value;
#: ``consumer``    a value is produced and USED (a request header, a client
#:                 argv, a length) — the sanctioned form, allowed;
#: ``printing``    a value reaches a sink that this result carries back to the
#:                 model — refused;
#: ``unresolved``  a source is present and a region of the text could not be
#:                 classified — refused (see :func:`scan_command` for why the
#:                 asymmetry justifies that).
Verdict = Literal["consumer", "printing", "none", "unresolved"]

Lang = Literal["shell", "python"]


@dataclass(frozen=True)
class Rule:
    """One refusal (or one allowance) as a readable, testable unit.

    The table is the inspection surface: a reviewer reads one ``Rule`` and sees
    the case it catches, the constraint behind it, and — the half that keeps a
    scanner from becoming a menace — the counterexamples it must NOT fire on.
    ``examples``/``counterexamples`` are not prose: ``tests/unit/harness/
    test_secret_sinks.py`` iterates this tuple and runs every entry through the
    scanner, so a rule that stops firing its own examples fails the suite.
    """

    label: str
    verdict: Verdict
    lang: Lang
    question: str
    why: str
    rewrite: str
    examples: tuple[str, ...]
    counterexamples: tuple[str, ...]


#: The rules, in one literal table. Labels are stable ids: they appear in the
#: refusal the model reads, in the tests, and in the design doc.
RULES: tuple[Rule, ...] = (
    Rule(
        label="shell.print-of-source",
        verdict="printing",
        lang="shell",
        question="Does a command that writes its argument to stdout hold a secret?",
        why=(
            "The incident's Case A. `echo`, `printf`, `cat`, `tee`, `sed`, "
            "`base64`, `rev` and their class write their operands to stdout, and "
            "stdout of this command IS the tool result — so the value lands in "
            "the transcript, which is the model's context, and the only cure is "
            "a rotation. The rule keys on the flow (a source or a variable "
            "holding one, as an operand), not on the value, so re-spelling the "
            "value to get past the output filter (`| rev`, `| base64`, a "
            "substring) changes nothing: the `echo` is still there."
        ),
        rewrite=(
            "Bind the value to a variable only to interpolate it into the "
            "consuming command that needs it, e.g. "
            '`curl -H "Authorization: Bearer $(lop secret get NAME)" https://…`; '
            "or hand it over without stdout at all: "
            "`lop secret run --secret NAME -- …`, or `lop secret file NAME -- …` "
            "for a file-shaped secret. To identify a secret without its value, "
            "`lop secret describe NAME`."
        ),
        examples=(
            'echo "$(lop secret get GITHUB_TOKEN)"',
            'v=$(lop secret get GITHUB_TOKEN); echo "$v"',
            'for k in A B; do v=$(lop secret get "$k") && echo "$k = $v"; done',
            "printf '%s\\n' \"$(lop secret get GITHUB_TOKEN)\"",
            'v=$(lop secret get GITHUB_TOKEN); base64 <<< "$v"',
            'v=$(lop secret get GITHUB_TOKEN); echo "${v:0:8}"',
            'cat "$(lop secret get GITHUB_TOKEN)"',
        ),
        counterexamples=(
            # (i) the sanctioned form: bound to a variable, then consumed.
            'v=$(lop secret get GITHUB_TOKEN); curl -H "Authorization: Bearer $v" https://x',
            # (ii) length only.
            "lop secret get GITHUB_TOKEN | wc -c",
            # (iii) no value exists.
            "lop secret list",
            # (iv) a HEREDOC whose pattern lives in a literal region, writing a file.
            "cat > /tmp/probe.sh <<'EOF'\n"
            'ADMIN_API_KEY="$(lop secret get MINERVA_API_KEY_DEV)"\nEOF',
            # ... and the same body, unquoted, whose target is a file: the
            # expansion runs, but the bytes go to the file, not to this result.
            'cat > /tmp/probe.sh <<EOF\nADMIN_API_KEY="$(lop secret get MINERVA_API_KEY_DEV)"\nEOF',
            # A search for the pattern is not a use of a secret.
            "grep -rn 'lop secret get' docs/",
        ),
    ),
    Rule(
        label="shell.bare-source-in-command-position",
        verdict="printing",
        lang="shell",
        question="Is `lop secret get NAME` run on its own, with nothing to consume it?",
        why=(
            "`lop secret get` writes the exact stored bytes to stdout and exits "
            "non-zero with EMPTY stdout on failure — which is what makes `$( )` "
            "safe and what makes the bare form the leak: with nothing consuming "
            "its stdout, the value IS the tool result. This is the incident's "
            "shape with no `echo` to blame, and it is the reason the guide's "
            "prohibition is about the value reaching the transcript rather than "
            "about any particular command."
        ),
        rewrite=(
            "Interpolate it into the command that needs it "
            '(`curl -H "Authorization: Bearer $(lop secret get NAME)" …`), hand '
            "it over without stdout (`lop secret run --secret NAME -- …`), or — "
            "if you only need to know it exists — `lop secret describe NAME`."
        ),
        examples=(
            "lop secret get GITHUB_TOKEN",
            "cd /tmp && lop secret get GITHUB_TOKEN",
            "lop secret get GITHUB_TOKEN; echo done",
            'if [ -n "$X" ]; then lop secret get GITHUB_TOKEN; fi',
        ),
        counterexamples=(
            "v=$(lop secret get GITHUB_TOKEN)",
            "lop secret get GITHUB_TOKEN | wc -c",
            "lop secret get GITHUB_TOKEN > /tmp/token",
            'curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" https://x',
            "lop secret get --help",
        ),
    ),
    Rule(
        label="shell.pipe-of-source",
        verdict="printing",
        lang="shell",
        question="Does the source's stdout end in a stage that writes it back out?",
        why=(
            "`lop secret get NAME | base64` and `… | cat` are the same leak one "
            "pipe later, and the filtered spelling is the shape the incident's "
            "second half used. A pipeline's last stage writes to this result, so "
            "the value is refused whether it reached that stage verbatim or "
            "transformed; the transform only decides whether the OUTPUT filter "
            "would have recognised it, and the output filter is not the control."
        ),
        rewrite=(
            "Pipe it into the consumer that needs it, not into a printer: "
            "`lop secret get NAME | docker login --username u --password-stdin`, "
            "or `lop secret run --secret NAME -- …`. If you want a digest or a "
            "length, `lop secret get NAME | shasum -a 256` and "
            "`lop secret get NAME | wc -c` are allowed."
        ),
        examples=(
            # The re-spelling the incident measured: `rev` does not launder the
            # flow, and the pipeline's last stage is what this result holds.
            'v=$(lop secret get GITHUB_TOKEN); echo "$v" | rev',
            "lop secret get GITHUB_TOKEN | base64",
            "lop secret get GITHUB_TOKEN | rev",
            "lop secret get GITHUB_TOKEN | cat",
            "v=$(lop secret get GITHUB_TOKEN); printf '%s' \"$v\" | sed 's/./& /g'",
            "lop secret get GITHUB_TOKEN | xxd",
        ),
        counterexamples=(
            "lop secret get GITHUB_TOKEN | wc -c",
            "lop secret get GITHUB_TOKEN | shasum -a 256",
            "lop secret get GITHUB_TOKEN | docker login --username u --password-stdin",
            'v=$(lop secret get GITHUB_TOKEN); curl -H "Authorization: Bearer $v" https://x',
        ),
    ),
    Rule(
        label="shell.length-only-sink",
        verdict="consumer",
        lang="shell",
        question="Does the value stop at a length or digest sink?",
        why=(
            "A length reveals nothing — `handlers._set` prints `len(value)` "
            "deliberately — and a digest is one-way, so a pipeline that ENDS in "
            "`wc` or the shasum family is a use, not a print. The list is kept "
            "tight on purpose: a hash is fine, but `cmp -l` and `diff` are not, "
            "because their output contains bytes of the subject."
        ),
        rewrite=(
            "Nothing to rewrite: this form is allowed. Bind it or pipe it "
            "straight into `wc`/`shasum` if that is all you need."
        ),
        examples=(
            "lop secret get GITHUB_TOKEN | wc -c",
            "lop secret get GITHUB_TOKEN | wc -m",
            "lop secret get GITHUB_TOKEN | shasum -a 256",
            "lop secret get GITHUB_TOKEN | sha256sum",
            "v=$(lop secret get GITHUB_TOKEN); printf '%s' \"$v\" | wc -c",
            'v=$(lop secret get GITHUB_TOKEN); echo "${#v}"',
        ),
        counterexamples=(
            'echo "$(lop secret get GITHUB_TOKEN)"',
            "lop secret get GITHUB_TOKEN | base64",
            'v=$(lop secret get GITHUB_TOKEN); echo "${v:0:8}"',
        ),
    ),
    Rule(
        label="shell.read-of-secret-file-path",
        verdict="printing",
        lang="shell",
        question="Does anything read back a file a secret was written into?",
        why=(
            "The guide sanctions materialising a value to disk (`lop secret get "
            "NAME > /tmp/token`) as contained-but-debt, and it is contained only "
            "for as long as nothing reads it: the read is what turns the file "
            "into a transcript entry. Both halves are refused here — reading a "
            "file this command wrote from a value, and reading the path "
            "`lop secret file NAME` returns. The cross-CALL half (a later tool "
            "call reading it) is not detectable from one command's text and "
            "belongs to the session-scoped ledger in the design doc."
        ),
        rewrite=(
            "Give the command the path instead of reading it: "
            "`lop secret file GCP_SA_JSON -- gcloud auth activate-service-account "
            '--key-file "$GOOGLE_APPLICATION_CREDENTIALS"` — '
            "or delete the copy without reading it (`rm -f`)."
        ),
        examples=(
            "lop secret get GITHUB_TOKEN > /tmp/token; cat /tmp/token",
            "lop secret get GITHUB_TOKEN > /tmp/token; head -c 8 /tmp/token",
            'p=$(lop secret file GCP_SA_JSON); cat "$p"',
            "cat <<EOF > /tmp/creds\n$(lop secret get GITHUB_TOKEN)\nEOF\nsed -n 1p /tmp/creds",
        ),
        counterexamples=(
            'p=$(lop secret file GCP_SA_JSON); gcloud --key-file "$p" auth …',
            "lop secret get GITHUB_TOKEN > /tmp/token",
            "lop secret file GCP_SA_JSON -- gcloud auth activate-service-account "
            '--key-file "$GOOGLE_APPLICATION_CREDENTIALS"',
        ),
    ),
    Rule(
        label="shell.secret-verb-emitting-consumer",
        verdict="printing",
        lang="shell",
        question="Does the consumer handed to the secret verb print the value?",
        why=(
            "`lop secret file NAME -- CMD` materialises the value for CMD and "
            "removes it afterwards — which is exactly right when CMD reads it, "
            "and exactly the leak when CMD is `cat`/`echo`: the verb's stdout is "
            "this result, so the value is printed by a command the harness ran "
            "on the model's behalf."
        ),
        rewrite=(
            "Hand the path to the program that needs it "
            'lop secret file GCP_SA_JSON -- gcloud … --key-file "$GOOGLE_APPLICATION_CREDENTIALS"` '
            "or use `lop secret run --secret NAME -- …` for an environment "
            "variable. Do not make the consumer a printer."
        ),
        examples=(
            "lop secret file GCP_SA_JSON -- cat",
            "lop secret file GCP_SA_JSON -- tee /tmp/out.json",
            "lop secret file GCP_SA_JSON -- sh -c 'cat'",
        ),
        counterexamples=(
            "lop secret file GCP_SA_JSON -- gcloud auth activate-service-account "
            '--key-file "$GOOGLE_APPLICATION_CREDENTIALS"',
            "lop secret file GCP_SA_JSON -- /bin/true",
        ),
    ),
    Rule(
        label="shell.xtrace-of-source",
        verdict="printing",
        lang="shell",
        question="Is the shell tracing its own expansions while a secret is in the text?",
        why=(
            "Under `set -x` (equivalently `bash -x`, `sh -x`, `set -o xtrace`, a "
            "`PS4` assignment) the shell writes every expansion to stderr, and "
            "stderr is captured into this result — so the sink is the shell "
            "itself, and the value is printed whether or not any command in the "
            "text prints anything. Debug traces are a routine thing to add to a "
            "command that fetches a credential, which is why this is a rule "
            "rather than a footnote."
        ),
        rewrite=(
            "Drop the trace (`set -x`/`-x`), or run the traced part where its "
            "stderr is not this command's stderr, and never trace a command that "
            "carries a secret — the trace prints the value."
        ),
        examples=(
            "set -x; TOKEN=$(lop secret get GITHUB_TOKEN); " 'curl -H "Bearer $TOKEN" https://x',
            "set -eux -o pipefail; v=$(lop secret get GITHUB_TOKEN) "
            '&& curl -H "Bearer $v" https://x',
            "bash -x -c 'true'; v=$(lop secret get GITHUB_TOKEN) "
            '&& curl -H "Bearer $v" https://x',
            "PS4='+ '; v=$(lop secret get GITHUB_TOKEN) && curl -H 'Bearer $v' https://x",
        ),
        counterexamples=(
            'v=$(lop secret get GITHUB_TOKEN); curl -H "Authorization: Bearer $v" https://x',
            "set -x; echo hello",
            "set -e; v=$(lop secret get GITHUB_TOKEN); curl -H 'Bearer $v' https://x",
        ),
    ),
    Rule(
        label="shell.argv-visible-source",
        verdict="printing",
        lang="shell",
        question="Can another process read this argv while the secret is in it?",
        why=(
            "A value passed in argv is readable by any same-uid process — the "
            "secret store's own design rejects argv as a value channel for "
            "exactly that reason. So a command that both puts a secret in argv "
            "and shows the process table (or reads `/proc/*/cmdline`) is refused: "
            "the request to look at the process table is what makes the argv a "
            "sink, not the argv itself (otherwise the sanctioned "
            '`curl -H "…$(lop secret get NAME)"` form would be refused, and it '
            "is the form the guide recommends)."
        ),
        rewrite=(
            "Pass the value in the child's environment or over stdin instead of "
            "argv: `lop secret run --secret NAME -- …`, or "
            "`lop secret get NAME | CMD --password-stdin`. If you are looking "
            "for a process, `pgrep`/`ps` without the value alongside is fine."
        ),
        examples=(
            'curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" https://x & ps -ef',
            'v=$(lop secret get GITHUB_TOKEN); curl -H "Bearer $v" https://x '
            "& cat /proc/$!/cmdline",
            'v=$(lop secret get GITHUB_TOKEN); curl -H "Bearer $v" https://x; ps -ww',
        ),
        counterexamples=(
            # The value is in a SHELL VARIABLE here, so the process table shows
            # nothing of it: refusing this would be the nuisance refusal that
            # gets a guard worked around.
            "v=$(lop secret get GITHUB_TOKEN); ps -ww -p $$",
            "ps -ef",
            "ps aux | grep curl",
            'v=$(lop secret get GITHUB_TOKEN); curl -H "Authorization: Bearer $v" https://x',
        ),
    ),
    Rule(
        label="shell.source-to-stderr",
        verdict="printing",
        lang="shell",
        question="Is a printing stage's stdout redirected into the captured stderr?",
        why=(
            "`execute_bash` reads stderr into the same tool result as stdout, so "
            '`echo "$v" 1>&2` is not a way out of the refusal — the `2>&1` '
            "split is an output-channel detail, not a containment boundary. "
            "Refused for the same flow as `shell.print-of-source`."
        ),
        rewrite=(
            "Send diagnostics to stderr only when they carry no secret; for a "
            "value, use `lop secret run --secret NAME -- …` or "
            "`lop secret file NAME -- …`."
        ),
        examples=(
            'v=$(lop secret get GITHUB_TOKEN); echo "$v" 1>&2',
            "v=$(lop secret get GITHUB_TOKEN); printf '%s' \"$v\" >&2",
            "lop secret get GITHUB_TOKEN >&2",
        ),
        counterexamples=(
            'v=$(lop secret get GITHUB_TOKEN); curl -H "Authorization: Bearer $v" https://x 2>&1',
            "lop secret get GITHUB_TOKEN > /tmp/token",
        ),
    ),
    Rule(
        label="shell.interpreter-inline-source",
        verdict="printing",
        lang="shell",
        question="Does an inline `-c` program (or `xargs`) print the value?",
        why=(
            "`sh -c 'echo $(lop secret get NAME)'`, `bash -c \"echo $v\"` and "
            "`xargs echo` are printing constructs that the outer text does not "
            "show as one: the program that prints is inside an argument. The "
            "scan follows the inline program one level and applies the same "
            "rules there, so the refusal is the same refusal rather than a "
            "special case."
        ),
        rewrite=(
            "Keep the inline program on the consuming side "
            "(`sh -c 'curl -H \"Authorization: Bearer $TOKEN\" …'` with `TOKEN` "
            "supplied by `lop secret run --secret GITHUB_TOKEN=TOKEN -- sh -c …`), "
            "or use `lop secret run --secret NAME -- …` directly."
        ),
        examples=(
            "sh -c 'echo $(lop secret get GITHUB_TOKEN)'",
            'v=$(lop secret get GITHUB_TOKEN); bash -c "echo $v"',
            "v=$(lop secret get GITHUB_TOKEN); python3 -c \"print('$v')\"",
            "lop secret get GITHUB_TOKEN | xargs echo",
            "xargs -I{} echo {} < <(lop secret get GITHUB_TOKEN)",
        ),
        counterexamples=(
            "sh -c 'echo hello'",
            "v=$(lop secret get GITHUB_TOKEN); sh -c 'curl -H \"Bearer $TOKEN\" https://x',"
            "find . -name '*.py' | xargs grep -l TODO",
        ),
    ),
    Rule(
        label="shell.consumer-of-source",
        verdict="consumer",
        lang="shell",
        question="Is the value used by the command that needs it?",
        why=(
            "Non-finding (i): the sanctioned form. `v=$(lop secret get NAME)` "
            "then a curl header, or the substitution inline in the consumer's "
            "argv, crosses a pipe into the child and never enters the "
            "transcript. It is the form the guide recommends and Case B's "
            "intent, so refusing it would make the guard a nuisance that gets "
            "worked around."
        ),
        rewrite="Nothing to rewrite: this form is allowed.",
        examples=(
            'v=$(lop secret get GITHUB_TOKEN); curl -H "Authorization: Bearer $v" https://x',
            'curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" https://x',
            'v=$(lop secret get GITHUB_TOKEN); docker login --username u --password-stdin <<< "$v"',
            "lop secret get GITHUB_TOKEN | wc -c",
        ),
        counterexamples=(
            'echo "$(lop secret get GITHUB_TOKEN)"',
            "lop secret get GITHUB_TOKEN",
            "lop secret get GITHUB_TOKEN | base64",
        ),
    ),
    Rule(
        label="shell.unresolved-source-region",
        verdict="unresolved",
        lang="shell",
        question="Is a source present in text the scanner could not classify?",
        why=(
            "Fail closed, and only here: an unterminated quote, an unbalanced "
            "`$(`, an unterminated heredoc leaves the region unknown, so whether "
            "the value is printed is unknown too. The asymmetry is the whole "
            "justification — a false refusal costs one re-spelling (the rewrite "
            "in the message carries the same information in a form the scanner "
            "accepts), while a false allow is a credential in the transcript, "
            "and nothing undoes that. A command with NO source is never refused "
            "for a lexing failure, so a build, a test suite or a train loop is "
            "untouched by this rule."
        ),
        rewrite=(
            "Re-spell the call so the secret's use is a single command "
            "substitution inside the command that consumes it "
            '(`curl -H "Authorization: Bearer $(lop secret get NAME)" …`), or use '
            "`lop secret run --secret NAME -- …`. If you only need to identify "
            "the secret, `lop secret describe NAME`."
        ),
        examples=(
            'echo "$(lop secret get GITHUB_TOKEN)\n',
            "cat <<EOF\n$(lop secret get GITHUB_TOKEN)\n",
            'echo $(lop secret get "GITHUB_TOKEN',
        ),
        counterexamples=(
            "echo 'unterminated-looking but no source here",
            "cat <<'EOF'\n$(lop secret get X)\nEOF",
            "ls -la /tmp",
        ),
    ),
    Rule(
        label="python.print-of-source",
        verdict="printing",
        lang="python",
        question="Does an eval cell print or log a retrieved value?",
        why=(
            "Same sink as the shell half, on the surface where the value is a "
            "real `str`: `print`, `sys.stdout.write`, `sys.stderr.write`, "
            "`logging.*` (and any `logger.info`-shaped call) all reach stdout, "
            "stderr or the log, and the eval tool returns those to the model. "
            "The store's own `SecretValue.__repr__` shows `[redacted]`, and the "
            "cell's output is scrubbed — but the guide says plainly that the "
            "scrub is a safety net for accidents and not a channel, so the "
            "deliberate print is refused rather than relied on to be masked."
        ),
        rewrite=(
            "Use the value in the call that needs it "
            "(`requests.get(url, headers={'Authorization': f'Bearer {token}'})`). "
            "To identify a secret without its value, "
            '`secret` tool with `op="describe"`.'
        ),
        examples=(
            'token = secrets["GITHUB_TOKEN"]\nprint(token)',
            'print(secrets["GITHUB_TOKEN"])',
            'token = secrets["GITHUB_TOKEN"]\nprint(f"token={token}")',
            'import sys\ntoken = secrets["GITHUB_TOKEN"]\nsys.stdout.write(token)',
            'import logging\ntoken = secrets["GITHUB_TOKEN"]\nlogging.info("got %s", token)',
            'token = secrets["GITHUB_TOKEN"]\nprint(repr(token))',
            'token = secrets["GITHUB_TOKEN"]\nprint(token[:8])',
        ),
        counterexamples=(
            'token = secrets["GITHUB_TOKEN"]\n'
            'requests.get(url, headers={"Authorization": f"Bearer {token}"})',
            'print(len(secrets["GITHUB_TOKEN"]))',
            'print("GITHUB_TOKEN" in secrets)',
            "print([name for name in secrets])",
        ),
    ),
    Rule(
        label="python.result-of-source",
        verdict="printing",
        lang="python",
        question="Is the cell's own result the value?",
        why=(
            "`eval` returns the trailing expression's `repr` to the model, so a "
            "cell that ends in the value (or in an f-string holding it) prints "
            "it without calling anything. It is the bare `lop secret get` of the "
            "eval surface, and it is easy to write by accident: a cell whose "
            "last line is `token` after a retrieval."
        ),
        rewrite=(
            "End the cell on a statement (an assignment, a call) rather than on "
            "the value, and hand the value to the call that needs it inside the "
            "cell."
        ),
        examples=(
            'secrets["GITHUB_TOKEN"]',
            'token = secrets["GITHUB_TOKEN"]\ntoken',
            'token = secrets["GITHUB_TOKEN"]\nf"{token[:4]}…"',
            'secrets.get("GITHUB_TOKEN")',
        ),
        counterexamples=(
            'token = secrets["GITHUB_TOKEN"]\nlen(token)',
            '"GITHUB_TOKEN" in secrets',
            'requests.get(url, headers={"Authorization": f"Bearer {secrets[\'GITHUB_TOKEN\']}"})',
        ),
    ),
    Rule(
        label="python.write-then-read-of-source",
        verdict="printing",
        lang="python",
        question="Does the cell write the value to a path and then read it back?",
        why=(
            "The same contained-but-debt shape as the shell half: materialising "
            "a value is not itself the leak, the read is. Caught only when both "
            "halves are in one cell, which is the case a single scan can see; "
            "the cross-call half is the design doc's session ledger."
        ),
        rewrite=(
            "Give the path to the library that needs it "
            "(`google.auth.load_credentials_from_file(path)`), or materialise it "
            "with `lop secret file NAME -- …` from the shell, and never read the "
            "copy back."
        ),
        examples=(
            'token = secrets["GITHUB_TOKEN"]\n'
            'open("/tmp/t", "w").write(token)\n'
            'print(open("/tmp/t").read())',
            'token = secrets["GITHUB_TOKEN"]\np = Path("/tmp/t")\n'
            "p.write_text(token)\np.read_text()",
        ),
        counterexamples=(
            'token = secrets["GITHUB_TOKEN"]\nPath("/tmp/t").write_text(token)',
            'Path("/tmp/t").write_text("hello")\nprint(Path("/tmp/t").read_text())',
        ),
    ),
    Rule(
        label="python.source-into-shell-string",
        verdict="printing",
        lang="python",
        question="Is the value interpolated into a shell string a printer will run?",
        why=(
            "`os.system(f'echo {token}')` and "
            "`subprocess.run(f'printf %s {token}', shell=True)` move the leak one "
            "process down, where the text the model wrote no longer looks like "
            "printing. The scan follows the interpolation: a tainted value going "
            "into a string whose literal half contains a printing command word "
            "is refused. A value going into a string whose literal half is a "
            "consumer (`curl …`, `gcloud …`) is the sanctioned form and is not."
        ),
        rewrite=(
            "Pass arguments as a list rather than through a shell "
            '(`subprocess.run(["gcloud", "--key-file", path])`) or give the '
            "value to the child's environment — "
            "`lop secret run --secret NAME -- …` in bash."
        ),
        examples=(
            'import os\ntoken = secrets["GITHUB_TOKEN"]\nos.system(f"echo {token}")',
            'import subprocess\ntoken = secrets["GITHUB_TOKEN"]\n'
            'subprocess.run(f"printf %s {token}", shell=True)',
            'import os\nos.system("echo " + secrets["GITHUB_TOKEN"])',
        ),
        counterexamples=(
            'import os\ntoken = secrets["GITHUB_TOKEN"]\n'
            "os.system(f\"curl -H 'Bearer {token}' https://x\")",
            "import os\nos.system('echo hello')",
        ),
    ),
    Rule(
        label="python.consumer-of-source",
        verdict="consumer",
        lang="python",
        question="Is the value handed to the call that needs it?",
        why=(
            'Non-finding (i) on the eval surface: `token = secrets["NAME"]` then '
            "a request header or a client argument is the documented form, and "
            "the lookup itself is one round trip and one audit entry."
        ),
        rewrite="Nothing to rewrite: this form is allowed.",
        examples=(
            'token = secrets["GITHUB_TOKEN"]\n'
            'requests.get(url, headers={"Authorization": f"Bearer {token}"})',
            'client.get_secret_value(SecretId=secrets["AWS_SECRET"])',
            'len(secrets["GITHUB_TOKEN"])',
            '"GITHUB_TOKEN" in secrets',
        ),
        counterexamples=(
            'print(secrets["GITHUB_TOKEN"])',
            'secrets["GITHUB_TOKEN"]',
        ),
    ),
    Rule(
        label="python.unresolved-source-region",
        verdict="unresolved",
        lang="python",
        question="Does the cell mention `secrets` but not parse?",
        why=(
            "The cell cannot be parsed, so nothing about where the value goes is "
            "knowable: the same fail-closed asymmetry as the shell half, and "
            "subject to the same condition — a cell that never mentions the "
            "store is not refused, and a syntax error there is the kernel's "
            "ordinary error message, not ours."
        ),
        rewrite=(
            "Fix the syntax error and re-send the retrieval, keeping the value "
            "inside the call that needs it."
        ),
        examples=(
            'token = secrets["GITHUB_TOKEN"]\nprint(token',
            'secrets["GITHUB_TOKEN"]\ndef f(:',
        ),
        counterexamples=(
            "x = 1\nprint(x",
            "token = secrets['GITHUB_TOKEN']\nprint(len(token))",
        ),
    ),
)

#: Rule labels as a set, so a test can assert the code's labels and the table's
#: labels agree in both directions (an unreachable rule is dead policy that
#: drifts).
RULE_LABELS: frozenset[str] = frozenset(rule.label for rule in RULES)


def rule(label: str) -> Rule:
    """The rule with this label. Raises ``KeyError`` for an unknown label."""
    for candidate in RULES:
        if candidate.label == label:
            return candidate
    raise KeyError(label)


@dataclass(frozen=True)
class Finding:
    """One refusal: the rule that fired, where, and for which secret."""

    rule: str
    span: tuple[int, int]
    secret_name: str
    rewrite: str
    reason: str = ""


@dataclass(frozen=True)
class ScanResult:
    """What one scan of one command or cell concluded."""

    verdict: Verdict = "none"
    findings: tuple[Finding, ...] = ()
    sources: tuple[str, ...] = ()
    #: Set when the scanner hit an internal fault it could not attribute — the
    #: text is still refused rather than passed (see ``unresolved``).
    fault: str = ""

    @property
    def refused(self) -> bool:
        """``True`` when the tool must return an error instead of running."""
        return self.verdict in ("printing", "unresolved")

    @property
    def labels(self) -> tuple[str, ...]:
        """The distinct rule labels that fired, in first-seen order."""
        seen: list[str] = []
        for finding in self.findings:
            if finding.rule not in seen:
                seen.append(finding.rule)
        return tuple(seen)


# ---------------------------------------------------------------------------
# Shell tokenizer
# ---------------------------------------------------------------------------
#
# Why a tokenizer and not a regex over the text: the discriminating
# counterexample — a heredoc whose BODY holds the literal pattern — is a
# property of the REGION the pattern sits in. `cat <<'EOF'` does not expand and
# `cat <<EOF` does; single quotes are literal, double quotes are not; `#`
# starts a comment in one place and a word in another. A pattern match cannot
# tell those apart, so "a source inside a literal region is not a source" has to
# come out of the classification. The tokenizer is therefore the real work of
# this module, and it is deliberately small: it classifies, it does not evaluate.


class _LexFault(Exception):
    """Internal: the text could not be classified at this position."""

    def __init__(self, message: str, pos: int) -> None:
        super().__init__(message)
        self.message = message
        self.pos = pos


@dataclass(frozen=True)
class _Piece:
    """A run of word text with its region kind.

    ``literal`` — not expanded (single quotes, a quoted heredoc body, a
    comment, an escaped character): a source spelling inside it is DATA.
    ``expand``  — expanded by the shell (bare text, double quotes, an unquoted
    heredoc body): variables and substitutions here are live.
    ``subst``   — the text of a ``$( )`` or backtick substitution, which is a
    command list of its own.
    """

    text: str
    kind: str
    span: tuple[int, int]
    heredoc: bool = False


@dataclass(frozen=True)
class _Word:
    pieces: tuple[_Piece, ...]
    span: tuple[int, int]


@dataclass(frozen=True)
class _Op:
    """An operator or a redirection, kept in the stream so stages can be built."""

    text: str
    span: tuple[int, int]


@dataclass(frozen=True)
class _Body:
    """A heredoc body, attached to the stage whose delimiter introduced it."""

    piece: _Piece

    @property
    def span(self) -> tuple[int, int]:
        """Where the body is, so the stage code can speak of "its" span."""
        return self.piece.span


_SHELL_OPERATORS = (
    "<<<",
    "<<-",
    ">>",
    ">&",
    "<&",
    "&&",
    "||",
    "|&",
    ";;",
    "<<",
    "<>",
    ">|",
    "|",
    "&",
    ";",
    "(",
    ")",
    "<",
    ">",
    "\n",
)

#: Operators that end a stage without opening a pipeline.
#: `{` and `}` are NOT here, and not operators: bash treats them as reserved
#: words only in command position, while `xargs -I{} echo {}` is a single word
#: either side. Splitting on them cost a real case (`xargs -I{} … < <(…)`).
_STAGE_ENDS = frozenset({"\n", ";", "&", "&&", "||", ";;", "(", ")"})

#: Redirection operators whose next word is a path (or a delimiter, for here-docs).
_REDIRECTS = frozenset({">", ">>", ">|", "<", "<>", "<<", "<<-", "<<<", ">&", "<&"})


def _read_parens(text: str, open_index: int) -> tuple[str, int]:
    """Read a balanced ``( … )`` group at ``open_index``; return ``(inner, end)``.

    Shared by ``$( )`` and by the process substitutions ``<( )`` / ``>( )``,
    which differ only in what consumes the group's output — a substitution's
    stdout is captured as text, a process substitution's as a path — so the body
    is read the same way and classified by the caller. Nesting and inner quoting
    are honoured because a value is often fetched inside a nested group, and
    miscounting the parens would misplace every span after it.
    """
    assert text[open_index] == "("
    i = open_index + 1
    depth = 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "'":
            end = text.find("'", i + 1)
            if end < 0:
                raise _LexFault("unterminated single quote inside a substitution", i)
            i = end + 1
            continue
        if ch == '"':
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    i += 2
                    continue
                if text.startswith("$(", i):
                    _inner, i = _read_substitution(text, i)
                    continue
                i += 1
            if i >= n:
                raise _LexFault("unterminated double quote inside a substitution", i)
            i += 1
            continue
        if text.startswith("$(", i) or text.startswith("<(", i) or text.startswith(">(", i):
            _inner, i = _read_parens(text, i + 1)
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i], i + 1
        i += 1
    raise _LexFault("unbalanced ( with no closing parenthesis", open_index)


def _read_substitution(text: str, start: int) -> tuple[str, int]:
    """Read ``$( … )`` at ``start``; return ``(inner, index after ')')``."""
    assert text[start : start + 2] == "$("
    return _read_parens(text, start + 1)


def _read_backtick(text: str, start: int) -> tuple[str, int]:
    """Read ``\\` … \\``` at ``start``; return ``(inner, index after the close)``."""
    i = start + 1
    n = len(text)
    while i < n:
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "`":
            return text[start + 1 : i], i + 1
        i += 1
    raise _LexFault("unbalanced backtick substitution", start)


def _read_word(text: str, start: int) -> tuple[_Word, int]:
    """Read one word (quoting and substitutions included) from ``start``.

    Ends at unquoted whitespace or an operator character. ``$`` is kept in the
    ``expand`` text rather than resolved: variable *references* are taint to be
    tracked, not text to be replaced.
    """
    pieces: list[_Piece] = []
    buf: list[str] = []
    buf_start = start
    i = start
    n = len(text)

    def flush() -> None:
        nonlocal buf
        if buf:
            pieces.append(_Piece("".join(buf), "expand", (buf_start, i)))
            buf = []

    while i < n:
        ch = text[i]
        if ch in " \t\n":
            break
        if ch in "|&;()<>" or ch == "`":
            break
        if ch == "\\":
            if i + 1 >= n:
                raise _LexFault("trailing backslash", i)
            flush()
            pieces.append(_Piece(text[i + 1], "literal", (i, i + 2)))
            i += 2
            buf_start = i
            continue
        if ch == "'":
            flush()
            end = text.find("'", i + 1)
            if end < 0:
                raise _LexFault("unterminated single quote", i)
            pieces.append(_Piece(text[i + 1 : end], "literal", (i, end + 1)))
            i = end + 1
            buf_start = i
            continue
        if ch == '"':
            flush()
            i += 1
            buf_start = i
            while True:
                if i >= n:
                    raise _LexFault("unterminated double quote", start)
                if text[i] == '"':
                    flush()
                    i += 1
                    buf_start = i
                    break
                if text[i] == "\\" and i + 1 < n and text[i + 1] in '$`"\\\n':
                    buf.append(text[i + 1])
                    i += 2
                    continue
                if text.startswith("$(", i):
                    flush()
                    inner, end = _read_substitution(text, i)
                    pieces.append(_Piece(inner, "subst", (i, end)))
                    i = end
                    buf_start = i
                    continue
                buf.append(text[i])
                i += 1
            continue
        if text.startswith("$(", i):
            flush()
            inner, end = _read_substitution(text, i)
            pieces.append(_Piece(inner, "subst", (i, end)))
            i = end
            buf_start = i
            continue
        if ch == "`":
            flush()
            inner, end = _read_backtick(text, i)
            pieces.append(_Piece(inner, "subst", (i, end)))
            i = end
            buf_start = i
            continue
        if not buf:
            buf_start = i
        buf.append(ch)
        i += 1
    flush()
    if not pieces:
        raise _LexFault("empty word", start)
    return _Word(tuple(pieces), (start, i)), i


def _read_heredoc_body(
    text: str, start: int, delimiter: str, *, strip_tabs: bool, quoted: bool
) -> tuple[_Piece, int]:
    """Read a here-doc body up to its terminator line.

    A QUOTED delimiter (``<<'EOF'``) makes the body literal — which is the
    non-finding (iv) shape: an agent writing a script that CONTAINS the
    pattern, e.g. ``ADMIN_API_KEY="$(lop secret get MINERVA_API_KEY_DEV)"``. An
    unquoted delimiter makes it expanded, and then a substitution inside it
    really does run.
    """
    i = start
    n = len(text)
    lines: list[str] = []
    while i <= n:
        end = text.find("\n", i)
        line_end = n if end < 0 else end
        line = text[i:line_end]
        probe = line.lstrip("\t") if strip_tabs else line
        if probe == delimiter:
            return (
                _Piece(text[start:i], "literal" if quoted else "expand", (start, i), heredoc=True),
                min(line_end + 1, n),
            )
        lines.append(text[i : min(line_end + 1, n)])
        if end < 0:
            break
        i = line_end + 1
    raise _LexFault(f"unterminated heredoc body (looking for {delimiter!r})", start)


def _tokenize_shell(text: str) -> list[_Word | _Op | _Body]:
    """Classify ``text`` into words, operators and heredoc bodies."""
    items: list[_Word | _Op | _Body] = []
    pending: list[tuple[str, bool, bool]] = []  # (delimiter, quoted, strip_tabs)
    i = 0
    n = len(text)
    at_word_start = True
    while i < n:
        ch = text[i]
        if ch in " \t\r":
            i += 1
            at_word_start = True
            continue
        if ch == "\n":
            items.append(_Op("\n", (i, i + 1)))
            i += 1
            for delimiter, quoted, strip_tabs in pending:
                body, i = _read_heredoc_body(
                    text, i, delimiter, strip_tabs=strip_tabs, quoted=quoted
                )
                items.append(_Body(body))
            pending.clear()
            at_word_start = True
            continue
        if ch == "#" and at_word_start:
            end = text.find("\n", i)
            end = n if end < 0 else end
            items.append(_Word((_Piece(text[i:end], "literal", (i, end)),), (i, end)))
            i = end
            continue
        if text.startswith("<(", i) or text.startswith(">(", i):
            # A process substitution is CAPTURED into a path exactly as `$( )`
            # is captured into text, so its body is analysed contained and its
            # output is a flow. It is emitted as a WORD because that is what it
            # is — an argument (`diff <(get X) f`, `cat <(get X)`) — and reading
            # that path is what prints the value.
            inner, end = _read_parens(text, i + 1)
            items.append(_Word((_Piece(inner, "subst", (i, end)),), (i, end)))
            i = end
            at_word_start = False
            continue
        if ch in "|&;()<>":
            op = next(
                (candidate for candidate in _SHELL_OPERATORS if text.startswith(candidate, i)), ch
            )
            items.append(_Op(op, (i, i + len(op))))
            i += len(op)
            if op in ("<<", "<<-"):
                j = i
                while j < n and text[j] in " \t":
                    j += 1
                word, j = _read_word(text, j)
                items.append(word)
                quoted = all(piece.kind != "expand" for piece in word.pieces)
                delimiter = "".join(piece.text for piece in word.pieces if piece.kind != "subst")
                if word.pieces and word.pieces[0].kind == "subst":
                    # A substitution cannot be a here-doc delimiter; the shell
                    # expands nothing there, and guessing it wrong would
                    # mis-place every following body.
                    raise _LexFault("here-doc delimiter is a substitution", i)
                pending.append((delimiter, quoted, op == "<<-"))
                i = j
                continue
            at_word_start = True
            continue
        word, i = _read_word(text, i)
        items.append(word)
        at_word_start = False
    if pending:
        raise _LexFault("unterminated heredoc at end of input", n)
    return items


# ---------------------------------------------------------------------------
# Shell analysis
# ---------------------------------------------------------------------------
#
# One walk, left to right, over the stages the tokenizer produced. For each
# stage the question is narrow and local: does a secret-bearing value reach this
# stage, and does this stage's stdout reach the model? A value whose stdout is
# discarded, written to a file, or consumed by the next stage is NOT printed;
# the same value in argv of a printer IS. That single distinction is what keeps
# the sanctioned `curl -H "Authorization: Bearer $v"` allowed (an argument to a
# consumer) while refusing `echo "$v"` (an argument to a printer).

#: Commands whose stdout carries their operands or their input, so an operand
#: holding a secret is written into this result. Interpreters are deliberately
#: NOT here: `sh -c "curl -H \"…$v\""` is the sanctioned form, so an inline
#: program is followed one level instead (:meth:`_ShellAnalyzer._interpreter`).
_EMITTERS = frozenset(
    {
        "echo",
        "printf",
        "print",
        "cat",
        "tee",
        "head",
        "tail",
        "nl",
        "tac",
        "od",
        "xxd",
        "hexdump",
        "base64",
        "base32",
        "basenc",
        "rev",
        "tr",
        "sed",
        "awk",
        "gawk",
        "cut",
        "fold",
        "sort",
        "uniq",
        "paste",
        "column",
        "strings",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "jq",
        "yq",
        "diff",
        "comm",
        "join",
        "expand",
        "unexpand",
        "iconv",
        "less",
        "more",
        "bat",
        "dd",
        "split",
        "csplit",
        "fmt",
        "pr",
        "logger",
    }
)

#: Sinks that turn a value into something that is not the value, so a pipeline
#: ending here has not printed anything. Deliberately tight: `cmp -l` prints
#: bytes of its subject and `diff` prints lines of it, so both are emitters
#: above rather than length sinks — the design doc's list includes `cmp`, and
#: what it actually prints is why it is not here.
_LENGTH_ONLY = frozenset(
    {
        "wc",
        "shasum",
        "sha1sum",
        "sha256sum",
        "sha512sum",
        "md5",
        "md5sum",
        "cksum",
        "b2sum",
    }
)

#: Programs whose argument is another program.
_INTERPRETERS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})

#: Programs whose argument is a Python program.
_INLINE_PYTHON = frozenset({"python", "python3"})

#: `lop secret …` verbs that hand out a value or a value's path. `list`,
#: `describe`, `set`, `audit` and `--help` produce no value and are not sources
#: (required non-finding (iii)).
_SOURCE_VERBS = frozenset({"get", "file", "run"})

#: Redirect targets that are not a file: the value still reaches this result.
_STDOUT_DEVICES = frozenset({"/dev/stdout", "/dev/fd/1", "/proc/self/fd/1"})

#: Redirect targets that drop the value entirely.
_DISCARD_DEVICES = frozenset({"/dev/null"})

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
#: Shell reserved words are not commands. Without this, `if …; then lop secret
#: get NAME; fi` reads `then` as the command and the source is never seen — the
#: keyword is a word to the tokenizer, and the command word is what the source
#: rule keys on.
_SHELL_KEYWORDS = frozenset(
    {
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "for",
        "while",
        "until",
        "do",
        "done",
        "case",
        "esac",
        "in",
        "function",
        "select",
        "time",
        "coproc",
        "!",
        "{",
        "}",
        "[[",
        "]]",
    }
)
_VAR_REF_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
#: `${#v}` is a length, and a length is not the value (required non-finding ii).
_LENGTH_REF_RE = re.compile(r"\$\{#")
#: Words put on the process table by `ps`-family reads.
_PROC_CMD_RE = re.compile(r"/proc/[^\s]*?/cmdline|/proc/[^\s]*?/environ")
_XTRACE_RE = re.compile(r"^-[a-zA-Z]*x[a-zA-Z]*$|^xtrace$")
#: A printing construct inside an inline program's literal text — the shape
#: `python3 -c "print($v)"` has, where the outer text shows no `echo`.
_INLINE_PRINT_RE = re.compile(
    r"(?:^|[\s;|&(\[,.=])"
    r"(?:print|echo|printf|cat|tee|base64|repr|logging\.\w+"
    r"|sys\.stdout\.write|sys\.stderr\.write)\b"
)
#: A raw-text source spelling, used ONLY to decide the fail-closed question
#: when the tokenizer faults: is there something here that could be a source?
_RAW_SOURCE_RE = re.compile(r"\blop\b[^\n]{0,80}?\bsecret\b")

#: Bound on nested inline programs and substitutions. A refusal is for one
#: call; nothing legitimate nests six programs deep, and the bound keeps a
#: pathological input from being a stack exercise.
_MAX_DEPTH = 6


@dataclass(frozen=True)
class _Flow:
    """What a piece of command text carries, and where it came from."""

    value: bool = False
    path: bool = False
    name: str = ""
    span: tuple[int, int] = (0, 0)


class _ShellAnalyzer:
    """Stateful walk of one command text (and of what it nests)."""

    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.sources: list[str] = []
        #: Variables holding a secret VALUE, and the secret each came from.
        self.value_vars: dict[str, str] = {}
        #: Variables holding a PATH whose content is a secret (`lop secret file`).
        self.file_vars: dict[str, str] = {}
        #: Paths a value was written into during THIS command text.
        self.tainted_paths: dict[str, tuple[int, int]] = {}
        #: The whole-command conditions: recorded during the walk, judged after
        #: it, because `set -x` and `ps` can appear before or after the source.
        self._xtrace: tuple[tuple[int, int], str] | None = None
        self._proc_read: tuple[int, int] | None = None
        self._argv_taint: tuple[tuple[int, int], str] | None = None
        self._seen: set[tuple[str, tuple[int, int]]] = set()

    # -- findings -----------------------------------------------------------

    def _add(self, label: str, span: tuple[int, int], secret_name: str, reason: str = "") -> None:
        key = (label, span)
        if key in self._seen:
            return
        self._seen.add(key)
        self.findings.append(
            Finding(
                rule=label,
                span=span,
                secret_name=secret_name,
                rewrite=rule(label).rewrite,
                reason=reason,
            )
        )

    # -- word helpers -------------------------------------------------------

    @staticmethod
    def _word_text(word: _Word) -> str:
        """The word as the shell will see it, minus expansions.

        A substitution becomes the empty string: its OUTPUT is a flow, not
        text, and pretending otherwise would be the guess this module refuses.
        """
        return "".join(piece.text for piece in word.pieces if piece.kind != "subst")

    @staticmethod
    def _is_assignment(word: _Word) -> bool:
        """``NAME=…`` — a value bound to a variable, not a command."""
        if not word.pieces or word.pieces[0].kind == "subst":
            return False
        return bool(_ASSIGNMENT_RE.match(word.pieces[0].text))

    @staticmethod
    def _assignment_name(word: _Word) -> str:
        return word.pieces[0].text.split("=", 1)[0]

    @classmethod
    def _command_word(cls, stage: list[_Word | _Op | _Body]) -> str:
        """The command this stage runs, assignment prefixes skipped."""
        for item in stage:
            if not isinstance(item, _Word):
                continue
            text = cls._word_text(item).strip()
            if not text or text in _SHELL_KEYWORDS:
                continue
            if cls._is_assignment(item):
                continue
            return text.rsplit("/", 1)[-1]
        return ""

    @staticmethod
    def _redirections(stage: list[_Word | _Op | _Body]) -> list[tuple[str, _Word]]:
        """``(operator, target word)`` for every redirection in the stage."""
        targets: list[tuple[str, _Word]] = []
        pending: str | None = None
        for item in stage:
            if isinstance(item, _Op):
                pending = item.text if item.text in _REDIRECTS else None
                continue
            if isinstance(item, _Word) and pending is not None:
                targets.append((pending, item))
                pending = None
        return targets

    # -- taint --------------------------------------------------------------

    def _value_flow(self, word: _Word, *, depth: int) -> _Flow:
        """Taint carried by one word: a nested source, or a tainted variable.

        This is where the reversed/encoded re-emission is handled without any
        knowledge of the value: whatever the pipeline does to the bytes, the
        word still references the variable that holds them, so the flow is the
        same and the verdict is the same.
        """
        flow = _Flow()
        for piece in word.pieces:
            if piece.kind == "subst":
                if depth >= _MAX_DEPTH:
                    continue
                inner = self._analyze(piece.text, contained=True, depth=depth + 1)
                if inner.value or inner.path:
                    flow = _Flow(
                        value=flow.value or inner.value,
                        path=flow.path or inner.path,
                        name=flow.name or inner.name,
                        span=flow.span if flow.span != (0, 0) else piece.span,
                    )
                continue
            if piece.kind != "expand":
                continue
            for match in _VAR_REF_RE.finditer(piece.text):
                if _LENGTH_REF_RE.match(piece.text[max(0, match.start() - 1) : match.start() + 1]):
                    continue
                name = match.group(1)
                span = (piece.span[0] + match.start(), piece.span[0] + match.end())
                if name in self.value_vars:
                    flow = _Flow(
                        value=True,
                        name=flow.name or self.value_vars[name],
                        span=flow.span if flow.span != (0, 0) else span,
                    )
                elif name in self.file_vars:
                    flow = _Flow(
                        path=True,
                        name=flow.name or self.file_vars[name],
                        span=flow.span if flow.span != (0, 0) else span,
                    )
        return flow

    def _argv_flow(self, stage: list[_Word | _Op | _Body], *, depth: int) -> _Flow:
        flow = _Flow()
        for item in stage:
            if not isinstance(item, _Word) or self._is_assignment(item):
                continue
            piece = self._value_flow(item, depth=depth)
            flow = _Flow(
                value=flow.value or piece.value,
                path=flow.path or piece.path,
                name=flow.name or piece.name,
                span=flow.span if flow.span != (0, 0) else piece.span,
            )
            if piece.value:
                self._argv_taint = self._argv_taint or (piece.span or item.span, piece.name)
        return flow

    def _literal_path_hit(self, stage: list[_Word | _Op | _Body]) -> tuple[bool, tuple[int, int]]:
        """Does this stage name a file a secret was written into?"""
        for item in stage:
            if not isinstance(item, _Word):
                continue
            text = self._word_text(item).strip()
            if text in self.tainted_paths:
                return True, item.span
            for path, span in self.tainted_paths.items():
                if text.endswith("/" + path.lstrip("./")) and path not in ("", "/dev/null"):
                    return True, span
        return False, (0, 0)

    def _register_path(self, path: str, span: tuple[int, int]) -> None:
        path = path.strip().strip("'\"")
        if path and path not in _DISCARD_DEVICES:
            self.tainted_paths.setdefault(path, span)

    # -- sources ------------------------------------------------------------

    @classmethod
    def _command_index(cls, words: list[_Word]) -> int | None:
        """Index of the word that IS the command: keywords and assignments skip.

        `if …; then lop secret get NAME; fi` reads `then` as the command word
        without this, and then the source rule never sees the fetch.
        """
        for index, word in enumerate(words):
            text = cls._word_text(word).strip()
            if not text or text in _SHELL_KEYWORDS:
                continue
            if cls._is_assignment(word):
                continue
            return index
        return None

    def _source_verb(self, stage: list[_Word | _Op | _Body]) -> tuple[str, str] | None:
        """``(verb, name)`` when this stage IS ``lop secret get|file|run``.

        Command position is required, and that narrowing is deliberate: a text
        search for ``lop secret get`` also matches a `grep` for the pattern, a
        docstring or a comment that names the verb, and every one of those would
        be a false refusal — the class of bug the four required non-findings
        exist to prevent.
        """
        words = [item for item in stage if isinstance(item, _Word)]
        index = self._command_index(words)
        if index is None:
            return None
        first = self._word_text(words[index]).strip().rsplit("/", 1)[-1]
        if first not in ("lop", "lop.exe"):
            return None
        args = [self._word_text(word).strip() for word in words[index + 1 :]]
        if len(args) < 2 or args[0] != "secret":
            return None
        verb = args[1]
        if verb not in _SOURCE_VERBS:
            return None
        if verb == "run":
            for index, arg in enumerate(args):
                if arg.startswith("--secret") and index + 1 < len(args):
                    name = args[index + 1].split("=", 1)[0]
                    if arg == "--secret":
                        return verb, name
                    if arg.startswith("--secret="):
                        return verb, arg.split("=", 1)[1].split("=", 1)[0]
        else:
            name = args[2] if len(args) > 2 else ""
            if name and not name.startswith("-"):
                # `lop secret get --help` is not a source: no value exists.
                return verb, name
        return None

    def _emitting_consumer(self, stage: list[_Word | _Op | _Body], name: str) -> None:
        """``lop secret file NAME -- cat``: the verb's consumer prints the value."""
        words = [self._word_text(item).strip() for item in stage if isinstance(item, _Word)]
        rest: list[str] = []
        for index, word in enumerate(words):
            if word in ("--", "-"):
                rest = words[index + 1 :]
                break
            if word.startswith("--secret"):
                break
        if not rest:
            return
        consumer = rest[0].rsplit("/", 1)[-1]
        if consumer in _EMITTERS or (
            consumer in _INTERPRETERS and _INLINE_PRINT_RE.search(" ".join(rest[1:]))
        ):
            span = next((item.span for item in stage if isinstance(item, _Word)), (0, 0))
            self._add("shell.secret-verb-emitting-consumer", span, name)

    # -- the walk -----------------------------------------------------------

    def _analyze(self, text: str, *, contained: bool, depth: int) -> _Flow:
        """Walk one command list and return the flow ITS stdout carries.

        ``contained`` means the stdout is captured by an enclosing ``$( )`` (or
        written to a file): a printer inside is then not a leak, but its output
        is still the value, which is why the returned flow stays tainted — that
        is what makes ``v=$(echo $(lop secret get X)); echo "$v"`` refuse.
        """
        stages, piped = self._group(_tokenize_shell(text))
        stdin = _Flow()
        out = _Flow()
        for index, stage in enumerate(stages):
            reads_stdin = piped[index]
            writes_stdout = index + 1 < len(stages) and piped[index + 1]
            out = self._stage(
                stage,
                stdin if reads_stdin else _Flow(),
                writes_stdout=writes_stdout,
                contained=contained,
                depth=depth,
            )
            stdin = out
        return out

    @staticmethod
    def _group(
        items: list[_Word | _Op | _Body],
    ) -> tuple[list[list[_Word | _Op | _Body]], list[bool]]:
        """Split tokens into stages; ``piped[i]`` is "stage i reads stdin"."""
        stages: list[list[_Word | _Op | _Body]] = []
        piped: list[bool] = []
        current: list[_Word | _Op | _Body] = []
        reads_stdin = False
        for item in items:
            if isinstance(item, _Body):
                # A here-doc BODY arrives after the newline that ended the stage
                # whose delimiter introduced it, so it belongs to that stage —
                # appending it to `current` would hand the body to the NEXT
                # command, and (iv)'s redirect target would be misread.
                if stages:
                    stages[-1].append(item)
                else:
                    current.append(item)
                continue
            if isinstance(item, _Op) and item.text in ("|", "|&"):
                stages.append(current)
                piped.append(reads_stdin)
                current = []
                reads_stdin = True
            elif isinstance(item, _Op) and item.text in _STAGE_ENDS:
                if current:
                    stages.append(current)
                    piped.append(reads_stdin)
                current = []
                reads_stdin = False
            elif isinstance(item, _Op):
                current.append(item)  # a redirection operator, kept for _redirections
            else:
                current.append(item)
        if current:
            stages.append(current)
            piped.append(reads_stdin)
        return stages, piped

    def _stage(
        self,
        stage: list[_Word | _Op | _Body],
        stdin: _Flow,
        *,
        writes_stdout: bool,
        contained: bool,
        depth: int,
    ) -> _Flow:
        # Reserved words are structure, not operands: dropping them here is what
        # lets `for …; do v=$(…); done` and `if …; then lop secret get X; fi`
        # read as the stages a person sees rather than as keyword soup.
        words = [
            item
            for item in stage
            if isinstance(item, _Word) and self._word_text(item).strip() not in _SHELL_KEYWORDS
        ]
        bodies = [item.piece for item in stage if isinstance(item, _Body)]
        redirections = self._redirections(stage)
        command = self._command_word(stage)

        # Where this stage's stdout goes. This is the whole basis for deciding
        # whether an emitted value is PRINTED: `>/dev/null` drops it, a real
        # path contains it, a pipe hands it on, and anything else IS this tool
        # result.
        stdout_path: str | None = None
        discarded = False
        to_stderr = False
        for op, target in redirections:
            text = self._word_text(target).strip().strip("'\"")
            if op in (">", ">>", ">|"):
                if text in _DISCARD_DEVICES:
                    discarded = True
                elif text in _STDOUT_DEVICES:
                    stdout_path = None
                else:
                    stdout_path = text
            elif op in (">&", "1>&", "&>") and text.lstrip("&") == "2":
                to_stderr = True
        for word in words:
            for piece in word.pieces:
                if piece.kind == "expand" and re.search(r"1?>&\s*2\b", piece.text):
                    to_stderr = True
        reaches_result = (
            not contained and not writes_stdout and not discarded and stdout_path is None
        )

        # -- whole-command conditions, recorded and judged after the walk ---
        # Before the assignment branch, because `PS4='+ '` IS an assignment and
        # would otherwise return early without ever being examined.
        self._note_conditions(command, stage, words)

        # -- a substitution standing alone in command position ---------------
        # `$(lop secret get NAME)` with nothing consuming it: its output becomes
        # a command NAME, so nothing uses it as a value — and the shell prints
        # it straight back in `command not found`. Seen here (rather than in the
        # emitter branch) because the word IS the command, and returned as a
        # flow because a capture around it — a here-doc body, an outer `$( )` —
        # is exactly how the value travels.
        if (
            command == ""
            and not any(self._is_assignment(word) for word in words)
            and any(piece.kind == "subst" for word in words for piece in word.pieces)
        ):
            argv = self._argv_flow(stage, depth=depth)
            if (argv.value or argv.path) and not contained:
                self._add(
                    "shell.bare-source-in-command-position",
                    argv.span or (words[0].span if words else (0, 0)),
                    argv.name,
                    reason=(
                        "the substitution's output is executed as a command name, "
                        "and the failure prints it back"
                    ),
                )
            return argv

        # -- assignments: the value is bound, not printed -------------------
        if command == "" and all(self._is_assignment(word) for word in words) and words:
            for word in words:
                flow = self._value_flow(word, depth=depth)
                name = self._assignment_name(word)
                if flow.value:
                    self.value_vars[name] = flow.name
                elif flow.path:
                    self.file_vars[name] = flow.name
            # A pure assignment's stdout carries nothing; the value is in the
            # variable, which is exactly the sanctioned first half.
            return _Flow()

        argv_flow = self._argv_flow(stage, depth=depth)
        flow_in = argv_flow
        if stdin.value or stdin.path:
            flow_in = _Flow(
                value=argv_flow.value or stdin.value,
                path=argv_flow.path or stdin.path,
                name=argv_flow.name or stdin.name,
                span=argv_flow.span if argv_flow.span != (0, 0) else stdin.span,
            )
        body_flow = self._heredoc_flow(bodies, depth=depth)
        flow_in = _Flow(
            value=flow_in.value or body_flow.value,
            path=flow_in.path or body_flow.path,
            name=flow_in.name or body_flow.name,
            span=flow_in.span if flow_in.span != (0, 0) else body_flow.span,
        )
        span = flow_in.span or (stage[0].span if stage else (0, 0))
        name = flow_in.name
        path_hit, path_span = self._literal_path_hit(stage)

        # -- a length sink ends the value's journey --------------------------
        # `lop secret get NAME | wc -c | tr -d ' '` must not refuse the `tr`:
        # what reaches it is a number. This is the same claim `handlers._set`
        # makes by printing `len(value)` on purpose.
        if command in _LENGTH_ONLY:
            return _Flow()

        # -- the stage IS the source ----------------------------------------
        source = self._source_verb(stage)
        if source is not None:
            verb, secret = source
            self.sources.append(secret)
            src_span = next((item.span for item in stage if isinstance(item, _Word)), (0, 0))
            if verb == "file":
                self._emitting_consumer(stage, secret)
                if to_stderr:
                    self._add("shell.source-to-stderr", src_span, secret)
                for op, target in redirections:
                    if op in (">", ">>", ">|"):
                        self._register_path(self._word_text(target), target.span)
                # The verb hands out a PATH; the value stays in the file until
                # something reads it, which the read rules above then refuse.
                return _Flow(path=True, name=secret, span=src_span)
            if verb == "run":
                self._emitting_consumer(stage, secret)
                return _Flow()
            if to_stderr:
                self._add("shell.source-to-stderr", src_span, secret)
            elif discarded:
                pass
            elif stdout_path is not None:
                self._register_path(stdout_path, src_span)
            elif reaches_result:
                self._add("shell.bare-source-in-command-position", src_span, secret)
            if discarded:
                return _Flow()
            return _Flow(value=True, name=secret, span=src_span)

        # -- the stage prints -------------------------------------------------
        if command in _EMITTERS:
            emits = flow_in.value or flow_in.path or path_hit
            if emits:
                if to_stderr:
                    self._add("shell.source-to-stderr", span, name)
                elif discarded:
                    pass
                elif stdout_path is not None:
                    self._register_path(stdout_path, span)
                elif reaches_result:
                    if path_hit and not flow_in.value:
                        self._add("shell.read-of-secret-file-path", path_span or span, name)
                    elif stdin.value and not argv_flow.value and not body_flow.value:
                        self._add("shell.pipe-of-source", span, name)
                    elif flow_in.path and not flow_in.value:
                        self._add("shell.read-of-secret-file-path", span, name)
                    else:
                        self._add("shell.print-of-source", span, name)
                # A `tee` names another file the same bytes are written to.
                if command == "tee":
                    for word in words[1:]:
                        text = self._word_text(word).strip()
                        if text and not text.startswith("-"):
                            self._register_path(text, word.span)
                            break
            if discarded:
                return _Flow()
            if stdout_path is not None:
                return _Flow()
            # The value is on stdout: captured, piped on, or printed above.
            return _Flow(value=flow_in.value, path=flow_in.path, name=name, span=span)

        # -- the stage runs an inline program ---------------------------------
        if command in _INTERPRETERS or command in _INLINE_PYTHON or command == "xargs":
            self._interpreter(stage, command, flow_in, stdin, span, name, depth)
            return _Flow(value=flow_in.value, path=flow_in.path, name=name, span=span)

        # -- a consumer: the value was used, not printed ----------------------
        # `curl -H "…$v"`, `docker login --password-stdin`, a client argv, a
        # length sink. Nothing here reaches this result, so nothing is added —
        # this branch is the sanctioned form's home, and the reason the guard is
        # a rule table rather than a blanket "no secrets in bash".
        return _Flow()

    def _note_conditions(
        self, command: str, stage: list[_Word | _Op | _Body], words: list[_Word]
    ) -> None:
        """Record `set -x`-shaped and process-table conditions for after the walk.

        Recorded rather than judged here because `set -x` and `ps` can appear
        before OR after the source; the leak is the same either way.
        """
        stage_span = stage[0].span if stage else (0, 0)
        flags = [self._word_text(word).strip() for word in words[1:]]
        if command in ("set", "export", "declare"):
            if any(_XTRACE_RE.match(flag) for flag in flags) or "xtrace" in flags:
                self._xtrace = self._xtrace or (stage_span, "")
        elif command in _INTERPRETERS or command in ("env", "xargs"):
            if any(flag.startswith("-") and "x" in flag and "c" not in flag for flag in flags):
                self._xtrace = self._xtrace or (stage_span, "")
        for word in words:
            if self._is_assignment(word) and self._assignment_name(word) == "PS4":
                self._xtrace = self._xtrace or (word.span, "")
            if _PROC_CMD_RE.search(self._word_text(word)):
                self._proc_read = self._proc_read or word.span
            for piece in word.pieces:
                if piece.kind == "expand" and _PROC_CMD_RE.search(piece.text):
                    self._proc_read = self._proc_read or piece.span
        if command == "ps":
            self._proc_read = self._proc_read or stage_span

    def _heredoc_flow(self, bodies: list[_Piece], *, depth: int) -> _Flow:
        """Taint an unquoted here-doc body carries (a quoted body is literal)."""
        flow = _Flow()
        for piece in bodies:
            if piece.kind == "literal" or depth >= _MAX_DEPTH:
                continue
            inner = self._analyze(piece.text, contained=True, depth=depth + 1)
            if inner.value or inner.path:
                flow = _Flow(
                    value=flow.value or inner.value,
                    path=flow.path or inner.path,
                    name=flow.name or inner.name,
                    span=flow.span if flow.span != (0, 0) else piece.span,
                )
        return flow

    def _interpreter(
        self,
        stage: list[_Word | _Op | _Body],
        command: str,
        flow_in: _Flow,
        stdin: _Flow,
        span: tuple[int, int],
        name: str,
        depth: int,
    ) -> None:
        """Judge an inline program, one level down.

        Two shapes: the value was expanded into the program TEXT
        (``sh -c "echo $v"`` — the outer shell expanded it, so the text now
        holds the value and a printing word), and the source is spelled INSIDE
        the program (``sh -c 'echo $(lop secret get X)'``, whose program text is
        a literal region to the outer lexer). Both are the same leak, so both
        reach the refusal by the same rule.
        """
        words = [item for item in stage if isinstance(item, _Word)]
        if command == "xargs":
            rest = [
                self._word_text(word).strip()
                for word in words[1:]
                if not self._word_text(word).strip().startswith("-")
            ]
            if (not rest or rest[0].rsplit("/", 1)[-1] in _EMITTERS) and (
                flow_in.value or stdin.value
            ):
                self._add(
                    "shell.interpreter-inline-source", span, name, reason="xargs runs a printer"
                )
            return
        inline: _Word | None = None
        for index, word in enumerate(words):
            if self._word_text(word).strip() in ("-c", "--eval"):
                if index + 1 < len(words):
                    inline = words[index + 1]
                break
        if inline is None:
            return
        literal = "".join(piece.text for piece in inline.pieces if piece.kind != "subst")
        static = _INLINE_PRINT_RE.search(literal)
        if (flow_in.value or stdin.value) and static:
            self._add(
                "shell.interpreter-inline-source",
                inline.span,
                name,
                reason="the inline program's own text prints the expansion",
            )
            return
        if flow_in.path and static:
            self._add(
                "shell.read-of-secret-file-path",
                inline.span,
                name,
                reason="the inline program reads a file a secret was written into",
            )
            return
        if flow_in.value or flow_in.path or depth >= _MAX_DEPTH:
            return
        inner = _ShellAnalyzer()
        try:
            inner._analyze(literal, contained=False, depth=depth + 1)
            inner._after_walk()
        except _LexFault:
            return
        if inner.findings:
            first = inner.findings[0]
            # The OUTER rule label is reported rather than the inner one: the
            # offending span is this word (`sh -c '…'`), and the inner rule is
            # named in the reason so a reviewer can follow the chain without the
            # message pointing at an offset in text the model never wrote.
            self._add(
                "shell.interpreter-inline-source",
                inline.span,
                first.secret_name,
                reason=f"the inline program is refused by `{first.rule}`",
            )

    def _after_walk(self) -> None:
        """Judge the whole-command conditions the walk recorded.

        Both are only leaks when a value is in the text at all, which is what
        keeps an ordinary `set -x` debug run and an ordinary `ps` untouched.
        """
        if self._xtrace is not None and self.sources:
            span, _ = self._xtrace
            self._add(
                "shell.xtrace-of-source",
                span,
                self.sources[0],
                reason=(
                    "the shell echoes every expansion to stderr, and stderr is "
                    "part of this result"
                ),
            )
        if self._proc_read is not None and self._argv_taint is not None:
            span, name = self._argv_taint
            self._add(
                "shell.argv-visible-source",
                span,
                name,
                reason="this command's argv holds the value and the command reads a process table",
            )


# ---------------------------------------------------------------------------
# Python analysis (the eval surface)
# ---------------------------------------------------------------------------
#
# A different language, the same question. The parser does the region work that
# the shell tokenizer has to do by hand, so the rules here are about call shapes
# instead of quoting: what retrieves a value, what prints, what writes and reads.

_PY_SOURCE_RE = re.compile(r"\bsecrets\s*(?:\[|\.get\s*\(|\.__getitem__\s*\()")
#: `display` is deliberately absent: it carries output to the OPERATOR, not to
#: the model's context (the eval tool excludes it), so refusing it would break
#: a legitimate "show the user" without closing a transcript path.
_PY_PRINT_CALLS = frozenset({"print"})
_PY_LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
)
_PY_SHELL_CALLS = frozenset(
    {
        "system",
        "run",
        "call",
        "check_call",
        "check_output",
        "Popen",
        "popen",
        "getoutput",
        "getstatusoutput",
    }
)
_PY_WRITE_MODES = frozenset({"w", "a", "x", "w+", "a+", "xb", "wb"})


class _PyAnalyzer:
    """Every call in a parsed cell, classified against the rule table."""

    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.sources: list[str] = []
        self.tainted: set[str] = set()
        self.secret_paths: dict[str, tuple[int, int]] = {}
        #: `p = Path("/tmp/t")` — a name standing for a literal path.
        self.path_handles: dict[str, str] = {}
        self._seen: set[tuple[str, tuple[int, int]]] = set()

    def _add(self, label: str, node: ast.AST, secret_name: str, reason: str = "") -> None:
        span = (getattr(node, "lineno", 0) or 0, getattr(node, "col_offset", 0) or 0)
        key = (label, span)
        if key in self._seen:
            return
        self._seen.add(key)
        self.findings.append(
            Finding(
                rule=label,
                span=span,
                secret_name=secret_name,
                rewrite=rule(label).rewrite,
                reason=reason,
            )
        )

    # -- taint --------------------------------------------------------------

    def _source_name(self, node: ast.AST) -> str | None:
        """``secrets["NAME"]`` / ``secrets.get("NAME")`` → ``NAME``."""
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == "secrets" and isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, str):
                    return node.slice.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "secrets":
                if node.func.attr == "get" and node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        return node.args[0].value
        return None

    def _is_tainted(self, node: ast.AST) -> tuple[bool, str]:
        """Does this expression carry a value from the store?

        ``len(x)`` is not: the length of a secret is not the secret, which is
        what keeps `print(len(secrets["X"]))` on the allowed side.
        """
        name = self._source_name(node)
        if name is not None:
            self.sources.append(name)
            return True, name
        if isinstance(node, ast.Name) and node.id in self.tainted:
            return True, node.id
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("len", "hash", "id"):
                return False, ""
        for child in ast.iter_child_nodes(node):
            found, child_name = self._is_tainted(child)
            if found:
                return True, child_name
        return False, ""

    def _collect_paths(self, tree: ast.Module) -> None:
        """`p = Path("/tmp/t")` — remember the literal a handle stands for."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            path = self._literal(node.value)
            if path is not None:
                self.path_handles[target.id] = path

    def _value_taint(self, node: ast.AST) -> tuple[bool, str]:
        """Taint of an expression's VALUE — not of its arguments.

        The distinction is what keeps a documented counterexample allowed:
        `requests.get(url, headers={"Authorization": f"Bearer {token}"})` has a
        tainted ARGUMENT (the value is used, correctly) and its VALUE is a
        response, so a cell ending on that call must not be refused. A method on
        the value (`token.strip()`) and `str`/`repr`/`format` keep the value; a
        free function call (`len`, `requests.get`) does not.
        """
        name = self._source_name(node)
        if name is not None:
            self.sources.append(name)
            return True, name
        if isinstance(node, ast.Name):
            return (True, node.id) if node.id in self.tainted else (False, "")
        if isinstance(node, ast.Attribute):
            return self._value_taint(node.value)
        if isinstance(node, ast.Subscript):
            return self._value_taint(node.value)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                return self._value_taint(func.value)
            if isinstance(func, ast.Name) and func.id in ("str", "repr", "format", "bytes"):
                for argument in node.args:
                    found, child = self._value_taint(argument)
                    if found:
                        return True, child
            return False, ""
        for child in ast.iter_child_nodes(node):
            found, child_name = self._value_taint(child)
            if found:
                return True, child_name
        return False, ""

    def _collect_taint(self, tree: ast.Module) -> None:
        """Names that hold a value, to a fixed point.

        Iterated because a cell rebinds freely (`a = secrets[...]`,
        `b = a.strip()`, `c = b`), and cells are small enough that a few passes
        cost nothing.
        """
        for _ in range(5):
            before = len(self.tainted)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    continue
                if node.value is None:
                    continue
                found, _ = self._is_tainted(node.value)
                if not found:
                    continue
                targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        self.tainted.add(target.id)
            if len(self.tainted) == before:
                break

    # -- static paths -------------------------------------------------------

    def _literal(self, node: ast.AST) -> str | None:
        """The path behind an expression, when one is statically knowable.

        Covers the three spellings an agent actually writes: a literal, a
        `Path("…")` (or a name bound to one), and an `open("…", "w")` handle.
        Anything else returns ``None``, which the write/read rules treat as
        "unknown path" rather than guessing one.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.path_handles.get(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("Path", "str", "open") and node.args:
                return self._literal(node.args[0])
        return None

    def _open_mode(self, node: ast.Call) -> str:
        mode = ""
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            mode = str(node.args[1].value)
        for keyword in node.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                mode = str(keyword.value.value)
        return mode

    def _write_path(self, node: ast.AST) -> str | None:
        """The literal path of ``open(p, "w")…`` / ``p.write_text(…)`` / ``f.write(…)``."""
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open" and node.args:
            if self._open_mode(node) in _PY_WRITE_MODES:
                return self._literal(node.args[0])
        if isinstance(func, ast.Attribute) and func.attr in ("write_text", "write_bytes"):
            return self._literal(func.value)
        if isinstance(func, ast.Attribute) and func.attr in ("write", "writelines"):
            # `f = open(p, "w"); f.write(v)` — the handle name is the key.
            if isinstance(func.value, ast.Name):
                return "\x00handle:" + func.value.id
            return self._literal(func.value)
        return None

    def _read_path(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open" and node.args:
            if self._open_mode(node) not in _PY_WRITE_MODES:
                return self._literal(node.args[0])
        if isinstance(func, ast.Attribute) and func.attr in ("read_text", "read_bytes", "read"):
            literal = self._literal(func.value)
            if literal is not None:
                return literal
            if isinstance(func.value, ast.Name):
                return "\x00handle:" + func.value.id
            return None
        return None

    # -- classification -----------------------------------------------------

    def _call_label(self, node: ast.Call, *, shell_string: bool) -> str | None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _PY_PRINT_CALLS:
                return "python.print-of-source"
            if func.id == "repr":
                return "python.print-of-source"
        if isinstance(func, ast.Attribute):
            attr = func.attr
            base = func.value
            base_name = base.id if isinstance(base, ast.Name) else ""
            dotted = self._dotted(base)
            # `sys.stdout`/`sys.stderr`, not `os.write(2, …)`: a bare fd write
            # is the channel the ledger's fd-2 scrub exists for (and the R1
            # regression test measures), so refusing the cell would delete the
            # surface that test covers rather than close a path the scrub misses.
            if attr in ("write", "writelines") and dotted in ("sys.stdout", "sys.stderr"):
                return "python.print-of-source"
            if base_name == "logging" or (
                attr in _PY_LOG_METHODS and base_name in ("logger", "log", "LOG", "LOGGER")
            ):
                return "python.print-of-source"
            if attr in ("format", "format_map") and isinstance(base, ast.JoinedStr):
                return "python.print-of-source"
            if attr in _PY_SHELL_CALLS and shell_string:
                return "python.source-into-shell-string"
        return None

    @staticmethod
    def _shell_string_args(node: ast.Call) -> str:
        """The inline command text a subprocess/os call would hand to a shell.

        Follows `+` concatenation and f-strings, because `"echo " + token` and
        `f"echo {token}"` are the same leak written two ways.
        """
        chunks: list[str] = []

        def walk(item: ast.AST) -> None:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                chunks.append(item.value)
            elif isinstance(item, ast.JoinedStr):
                for piece in item.values:
                    walk(piece)
            elif isinstance(item, ast.FormattedValue):
                walk(item.value)
            elif isinstance(item, ast.BinOp):
                walk(item.left)
                walk(item.right)

        for argument in node.args:
            walk(argument)
        return " ".join(chunks)

    @staticmethod
    def _dotted(node: ast.AST) -> str:
        """`sys.stdout` for an attribute chain, or "" when it is not one."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            head = _PyAnalyzer._dotted(node.value)
            return f"{head}.{node.attr}" if head else node.attr
        return ""

    def run(self, tree: ast.Module) -> None:
        self._collect_paths(tree)
        self._collect_taint(tree)
        body = tree.body
        for index, statement in enumerate(body):
            is_last = index == len(body) - 1
            value: ast.AST | None = None
            if isinstance(statement, ast.Expr):
                value = statement.value
                if is_last and self._read_path(value) is None:
                    found, name = self._value_taint(value)
                    if found:
                        self._add(
                            "python.result-of-source",
                            value,
                            name,
                            reason=(
                                "eval returns the trailing expression's repr, " "which is the value"
                            ),
                        )
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                label = self._call_label(node, shell_string=bool(self._shell_string_args(node)))
                arguments = list(node.args) + [kw.value for kw in node.keywords]
                tainted = False
                name = ""
                for argument in arguments:
                    if (
                        self._read_path(argument) is not None
                        or self._write_path(argument) is not None
                    ):
                        continue
                    found, found_name = self._is_tainted(argument)
                    if found:
                        tainted = True
                        name = name or found_name
                if tainted and label is not None:
                    if label == "python.source-into-shell-string":
                        if _INLINE_PRINT_RE.search(self._shell_string_args(node)):
                            self._add(label, node, name)
                    else:
                        self._add(label, node, name)
                write_path = self._write_path(node)
                if write_path is not None and tainted:
                    self.secret_paths.setdefault(write_path, (node.lineno, node.col_offset))
                read_path = self._read_path(node)
                if read_path is not None and read_path in self.secret_paths:
                    self._add(
                        "python.write-then-read-of-source",
                        node,
                        name or "?",
                        reason="the cell wrote a secret to this path and then read it back",
                    )
        # `f = open(p, "w"); f.write(v)` reaches the read rule through the handle
        # key above only if the same handle is read; a plain `open(p).read()`
        # after a `write_text` on the same literal path is the common shape and
        # is covered by the literal key.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SHELL_PREFILTER_CHARS = ("'", '"', "\\", "`")


def may_carry_a_shell_source(text: str) -> bool:
    """Cheap, sound prefilter: can this text hold a source at all?

    A source needs the literal words ``lop secret get|file|run`` at a command
    position. Those words can also be spelled across quoting pieces
    (``lop sec"ret" get X``), which needs a quote or an escape in the text —
    hence the second arm. So text this returns ``False`` for cannot produce a
    value, and ``False`` is what keeps a scan off the path of every ordinary
    command.
    """
    if "secret" in text:
        return True
    return any(char in text for char in _SHELL_PREFILTER_CHARS)


def _verdict_of(findings: Sequence[Finding], sources: Sequence[str]) -> Verdict:
    labels = {finding.rule for finding in findings}
    if any(rule(label).verdict == "printing" for label in labels):
        return "printing"
    if any(rule(label).verdict == "unresolved" for label in labels):
        return "unresolved"
    if sources:
        return "consumer"
    return "none"


def scan_command(command: str) -> ScanResult:
    """Scan shell command text. Never raises, never runs anything.

    ``none``        no source is in the text — the blast radius of this scan is
                    exactly the set of commands that fetch a stored secret,
                    which is what keeps a build, a test suite, a `terraform
                    apply` or a train loop untouched (by a refusal AND by a
                    lexing fault);
    ``consumer``    a source is present and used — the sanctioned form;
    ``printing``    a source reaches a sink that carries it back to the model;
    ``unresolved``  a source is present in text the tokenizer could not
                    classify. Refused, because the two errors are not
                    symmetrical: a false refusal costs one re-spelling (the
                    message hands the model the accepted form), while a false
                    allow is a credential in the transcript, and nothing undoes
                    that.

    The blast radius is exactly "text that fetches from the store": a command
    that never names ``lop secret`` is untouched — including one that prints a
    session credential (``echo $NAME`` for a name the harness injected into the
    child). That second class IS a leak and the design doc's source list
    includes it, but it is out of this change's scope by the delegation's ruling
    on the blast radius, and it is not free: refusing it rewrites the tests of
    two existing defences (``test_bash_injects_session_credentials_and_redacts_
    them_from_output``, whose subject is the mask, and the R1 fd-2 crash tail),
    both of which measure a path the wider rule never lets run. Recorded in the
    PR under "not addressed" rather than half-done here.
    """
    if not may_carry_a_shell_source(command):
        return ScanResult()
    analyzer = _ShellAnalyzer()
    try:
        analyzer._analyze(command, contained=False, depth=0)
        analyzer._after_walk()
    except _LexFault as fault:
        # Fail closed ONLY where something could actually be a source: a lexing
        # fault in a command that never fetches a secret is not our business,
        # and refusing it would make an ordinary broken command unusable.
        if not analyzer.sources and not _RAW_SOURCE_RE.search(command):
            return ScanResult()
        analyzer._add(
            "shell.unresolved-source-region",
            (fault.pos, min(fault.pos + 1, max(len(command), 1))),
            analyzer.sources[0] if analyzer.sources else "?",
            reason=fault.message,
        )
        return ScanResult(
            verdict="unresolved",
            findings=tuple(analyzer.findings),
            sources=tuple(dict.fromkeys(analyzer.sources)),
            fault=fault.message,
        )
    except RecursionError:  # pragma: no cover - `_MAX_DEPTH` is the real guard
        return ScanResult(verdict="unresolved", fault="nesting depth exhausted")
    return ScanResult(
        verdict=_verdict_of(analyzer.findings, analyzer.sources),
        findings=tuple(analyzer.findings),
        sources=tuple(dict.fromkeys(analyzer.sources)),
    )


def scan_python(source: str) -> ScanResult:
    """Scan an eval cell. Never raises.

    Same verdicts and the same asymmetry as :func:`scan_command`. The gate
    condition is a mention of ``secrets`` (the store's mapping), because a cell
    that never names it cannot retrieve a value.
    """
    if not _PY_SOURCE_RE.search(source):
        return ScanResult()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        node = ast.Expr(value=ast.Constant(value=None))
        node.lineno = exc.lineno or 1
        node.col_offset = exc.offset or 0
        analyzer = _PyAnalyzer()
        analyzer._add(
            "python.unresolved-source-region",
            node,
            "?",
            reason=f"the cell does not parse: {exc.msg}",
        )
        return ScanResult(
            verdict="unresolved",
            findings=tuple(analyzer.findings),
            fault=exc.msg,
        )
    analyzer = _PyAnalyzer()
    analyzer.run(tree)
    return ScanResult(
        verdict=_verdict_of(analyzer.findings, analyzer.sources),
        findings=tuple(analyzer.findings),
        sources=tuple(dict.fromkeys(analyzer.sources)),
    )


def _span_context(text: str, span: tuple[int, int]) -> tuple[str, str]:
    """A bounded window on the offending span, with a caret line under it."""
    start, end = span
    if not isinstance(start, int) or not isinstance(end, int):
        return "", ""
    if start < 0 or start > len(text):
        return "", ""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    if line_end - line_start > 200:
        line_start = max(0, start - 60)
        line_end = min(len(text), start + 140)
    line = text[line_start:line_end]
    caret = " " * max(0, start - line_start) + "^" * max(1, min(end, line_end) - start)
    return line, caret


def refusal_text(result: ScanResult, *, text: str, tool_name: str = "") -> str:
    """The tool-level error the model reads: rule, span, why, rewrite.

    It quotes the COMMAND, never a value. That is safe by construction — the
    offending span of a refused call is a spelling like ``lop secret get NAME``
    — and it is the whole point of naming the rule: the refusal is debuggable,
    and the rule table is falsifiable in review ("this came from rule X, here is
    rule X's ``why``").
    """
    if result.verdict == "unresolved":
        head = (
            "refusing to run this call: it reaches a stored secret but part of "
            "the text could not be classified (an unbalanced quote, "
            "substitution or here-doc), so whether the value is printed is "
            "unknown. An unrecoverable leak beats an inconvenient refusal."
        )
    else:
        head = (
            "refusing to run this call: a stored secret would be printed into "
            "this session's transcript, which IS the model's context. Nothing "
            "can undo that afterwards, so it is refused before the command runs."
        )
    lines = [f"[secret sink] {head}"]
    if result.findings:
        finding = result.findings[0]
        spec = rule(finding.rule)
        if spec.lang == "python":
            lines.append(
                f"  rule:   {finding.rule}  (cell line {finding.span[0]}, column {finding.span[1]})"
            )
        else:
            lines.append(
                f"  rule:   {finding.rule}  (span {finding.span[0]}:{finding.span[1]} of "
                f"the command)"
            )
        if finding.secret_name and finding.secret_name != "?":
            lines.append(f"  secret: {finding.secret_name}")
        snippet, caret = _span_context(text, finding.span)
        if snippet:
            lines.append(f"  here:   {snippet}")
            lines.append(f"          {caret}")
        if finding.reason:
            lines.append(f"  why:    {finding.reason}")
        lines.append(f"  do:     {finding.rewrite}")
        lines.append(f"  rule {finding.rule} exists because: {spec.why}")
        if len(result.findings) > 1:
            lines.append(f"  also refused by: {', '.join(result.labels[1:])}")
    lines.append(
        "  still allowed: `lop secret list`, `lop secret get --help`, "
        "`lop secret describe NAME`, `lop secret get NAME | wc -c`, and the "
        'sanctioned `curl -H "Authorization: Bearer $(lop secret get NAME)" …`.'
    )
    if tool_name:
        lines.append(f"  tool: {tool_name}; nothing was executed.")
    lines.append("  guide: guide://credentials — read it before re-spelling this call.")
    return "\n".join(lines)
