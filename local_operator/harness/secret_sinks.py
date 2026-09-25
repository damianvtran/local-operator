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
from collections.abc import Collection, Iterable, Mapping, Sequence
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
    ``examples``/``counterexamples`` are not prose: ``tests/unit/secrets/
    test_secret_sink_scan.py`` iterates this tuple and runs every entry through the
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
            # POSIX backticks are the same substitution as `$( )` and were
            # invisible before R1-1 — the table carried no backtick at all,
            # which is how the hole stayed green.
            'echo "`lop secret get GITHUB_TOKEN`"',
            'v="`lop secret get GITHUB_TOKEN`"; echo "$v"',
            "echo `lop secret get GITHUB_TOKEN`",
            # `read` binds through a builtin, not an `=` (R1-3).
            'read -r l < <(lop secret get GITHUB_TOKEN); echo "$l"',
            'cat "$(lop secret get GITHUB_TOKEN)"',
            "v=$(lop secret get NAME); echo TOKEN=$v",
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
            "V=1 echo TOKEN=literal",
            'v=$(lop secret get NAME); printf "%s\\n" X=$v > /tmp/contained',
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
            "local-operator secret get NAME",
            "command lop secret get NAME",
            "exec lop secret get NAME",
            "timeout 30 lop secret get NAME",
            "env lop secret get NAME",
            "l=lop; $l secret get NAME",
            "v=$(lop secret get NAME); $v",
            "nohup lop secret get NAME >/dev/stdout 2>/dev/null",
            "lop secret get NAME 2>/dev/null",
            # R4-2: the NAME arrives on `xargs`'s stdin, and the fetch is the
            # same fetch — `xargs` runs `lop` with this result as its stdout.
            "echo NAME | xargs lop secret get",
        ),
        counterexamples=(
            "v=$(lop secret get GITHUB_TOKEN)",
            "lop secret get GITHUB_TOKEN | wc -c",
            "lop secret get GITHUB_TOKEN > /tmp/token",
            'curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" https://x',
            "lop secret get --help",
            "command -v lop",
            "timeout 5 sleep 0.1",
            "env | grep -c lop",
            "lop secret get NAME > /tmp/contained 2>/dev/null; echo contained",
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
            'v=$(lop secret get NAME); echo "TOKEN=$v" | rev',
            "echo NAME | xargs lop secret get | rev",
            "xargs lop secret get <<< NAME | rev",
            # R5-5: a `run` whose consumer is itself the fetch is the fetch.
            "lop secret run --secret NAME=TOKEN -- lop secret get NAME | rev",
            "lop secret run --secret NAME=TOKEN -- timeout 5 lop secret get NAME | rev",
            # R5-1's class: bash takes a redirection anywhere in a simple
            # command, so one in FRONT of the fetch is still the fetch.
            "2>/dev/null lop secret get NAME | rev",
            "{fd}>/dev/null lop secret get NAME | rev",
        ),
        counterexamples=(
            "lop secret run --secret NAME=TOKEN -- lop secret get NAME | wc -c",
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
            # The ASSIGNMENT spelling has to be bound inside the body too, or
            # the redirect target is never registered and the read-back is
            # allowed with only the output filter in the way (R1-7).
            'cat > /tmp/creds2 <<EOF\nKEY="$(lop secret get GITHUB_TOKEN)"\nEOF\ncat /tmp/creds2',
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
            # R3-1: `run` always carries a `--secret` BEFORE the separator, and
            # the consumer check used to stop at that word, so none of these four
            # was ever examined. The verb's stdout is this result either way.
            "lop secret run --secret NAME -- printenv NAME",
            "lop secret run --secret NAME=TOKEN -- env",
            "lop secret run --secret NAME=TOKEN -- sh -c 'echo \"$TOKEN\"'",
            "lop secret run --secret NAME=TOKEN -- python3 -c "
            "'import os; print(os.environ[\"TOKEN\"])'",
            # R4-1: a wrapper in front of the consumer is the consumer — the
            # outer stage and this check share one wrapper grammar.
            "lop secret run --secret NAME=TOKEN -- timeout 5 printenv TOKEN",
            "lop secret run --secret NAME=TOKEN -- env printenv TOKEN",
            "lop secret run --secret NAME=TOKEN -- env sh -c 'echo $TOKEN'",
            "lop secret run --secret NAME=TOKEN -- command printenv TOKEN | rev",
            "lop secret run --secret NAME=TOKEN -- stdbuf -oL printenv TOKEN | rev",
            "lop secret run --secret NAME=TOKEN -- timeout 5 python3 -c "
            "'import os;print(os.environ[\"TOKEN\"][::-1])'",
            # R4-2: `xargs` hands its command the environment unchanged, and the
            # other inline interpreters read it as Python does.
            "lop secret run --secret NAME=TOKEN -- xargs printenv TOKEN",
            "lop secret run --secret NAME=TOKEN -- perl -e 'print scalar reverse $ENV{TOKEN}'",
            "lop secret run --secret NAME=TOKEN -- ruby -e 'puts ENV[\"TOKEN\"].reverse'",
            "lop secret run --secret NAME=TOKEN -- node -e 'console.log(process.env.TOKEN)'",
            "lop secret run --secret NAME=TOKEN -- awk 'BEGIN{print ENVIRON[\"TOKEN\"]}'",
            # R4-3: `file`'s inline shell program is walked with the path
            # variable bound, so an emitter the old word list lacked is seen.
            "lop secret file GCP_SA_JSON -- sh -c 'rev \"$GOOGLE_APPLICATION_CREDENTIALS\"'",
            "lop secret file GCP_SA_JSON --env-var KF -- sh -c 'rev \"$KF\"'",
            # R5-1: a redirection's words are not the consumer's argv, so the
            # habitual `2>/dev/null` no longer hides the `env` in front of it.
            "lop secret run --secret NAME=TOKEN -- env 2>/dev/null",
            "lop secret run --secret NAME=TOKEN -- env 2>&1 | grep TOKEN",
            "lop secret run --secret NAME=TOKEN -- env < /dev/null | rev",
            "lop secret run --secret NAME=TOKEN -- time printenv TOKEN | rev",
            "lop secret run --secret NAME=TOKEN -- nice --adjustment 5 printenv TOKEN | rev",
            # R5-2: the `env -i` exemption belongs to `env`'s own options, not to
            # another wrapper's `-i` or an `-u` operand spelled `-i`.
            "lop secret run --secret NAME=TOKEN -- stdbuf -i 0 env printenv TOKEN | rev",
            "lop secret run --secret NAME=TOKEN -- env -u -i printenv TOKEN | rev",
            # R5-3: `-p` clustered with `-e` still prints the program's result.
            "lop secret run --secret NAME=TOKEN -- node -pe "
            "'[...process.env.TOKEN].reverse().join(\"\")'",
            # R5-4: `file`'s `--env-var` before the NAME is still `file`.
            "lop secret file --env-var KF GCP_SA_JSON -- sh -c 'rev \"$KF\"'",
            "lop secret file --env-var=KF GCP_SA_JSON -- sh -c 'base64 < \"$KF\"'",
        ),
        counterexamples=(
            "lop secret file GCP_SA_JSON -- gcloud auth activate-service-account "
            '--key-file "$GOOGLE_APPLICATION_CREDENTIALS"',
            "lop secret file GCP_SA_JSON -- /bin/true",
            # The consumer that NEEDS the value is the whole point of the verb:
            # a client, an uploader, a request — and an attribute-setting builtin.
            "lop secret run --secret NAME=TOKEN -- python3 client.py",
            "lop secret run --secret NAME -- curl -sS -H 'Authorization: Bearer $NAME' https://x",
            "lop secret run --secret NAME -- sed -n 1p /etc/hosts",
            "lop secret run --secret NAME=TOKEN -- declare -x TOKEN",
            "lop secret file GCP_SA_JSON -- env V=1 /bin/true",
            # R4-1/R4-2: the same wrappers around a consumer that NEEDS the value,
            # a program handed in a file, and an interpreter that never reads
            # the environment stay allowed.
            "lop secret run --secret NAME=TOKEN -- timeout 5 curl -sS "
            '-H "Authorization: Bearer $TOKEN" https://x',
            "lop secret run --secret NAME=TOKEN -- nice python3 client.py",
            "lop secret run --secret NAME=TOKEN -- env -i printenv",
            "lop secret run --secret NAME=TOKEN -- timeout 5 env -iv printenv",
            "lop secret run --secret NAME=TOKEN -- node -pe '1 + 1'",
            "lop secret file --env-var KF GCP_SA_JSON -- sh -c 'wc -c < \"$KF\"'",
            "lop secret run --secret NAME=TOKEN -- node app.js",
            "lop secret run --secret NAME=TOKEN -- awk '{print $1}' /etc/hosts",
            "lop secret file GCP_SA_JSON -- sh -c 'wc -c < \"$GOOGLE_APPLICATION_CREDENTIALS\"'",
            "lop secret run --secret NAME=TOKEN -- sh -c "
            "'curl -sS -H \"Authorization: Bearer $TOKEN\" https://x' 2>/dev/null",
            "lop secret run --secret NAME -- echo",
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
            # Attribute flags are not tracing: `declare -x` marks for export.
            "v=$(lop secret get [redacted]); declare -x v",
            "v=$(lop secret get [redacted]); export -n v",
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
            "v=$(lop secret get GITHUB_TOKEN); sh -c 'curl -H \"Bearer $TOKEN\" https://x'",
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
            "lop secret run --secret NAME=TOKEN -- python3 client.py",
            "lop secret run --secret NAME -- sed -n 1p /etc/hosts",
            'lop secret run --secret NAME -- curl -sS -H "Authorization: Bearer $NAME" https://x',
            'v=$(lop secret get NAME); echo "${#v}"',
        ),
        counterexamples=(
            'echo "$(lop secret get GITHUB_TOKEN)"',
            "lop secret get GITHUB_TOKEN",
            "lop secret get GITHUB_TOKEN | base64",
        ),
    ),
    Rule(
        label="shell.environment-dump-of-source",
        verdict="printing",
        lang="shell",
        question="Does the command print the variable or environment a value was bound into?",
        why=(
            "A value does not have to be named as an operand to be printed. "
            "`export V=$(lop secret get NAME); printenv V`, `V=$(…); export V; "
            "env | grep V=` and `V=$(…); set | grep V=` all hand the raw value "
            "back, and none of them puts it in a printer's argv — which is why "
            "the emitter rule cannot see them. The dumper's REACH is what "
            "decides, and it is keyed on what the shell did with the name, not on "
            "which prefix bound it: `printenv` and a command-less `env` (any "
            "flags: `-0`, `-u OTHER`) show the EXPORTED namespace (`export`, "
            "`-x`, `set -a`, a `V=… cmd` prefix); bare `set`, bare "
            "`declare`/`typeset` and `-p` show every shell variable; `readonly` "
            "and `local` bind WITHOUT exporting. An `env` that runs a command "
            "(`env V=… client`) is a consumer handing the value to a child."
        ),
        rewrite=(
            "Let the consumer print its own result: "
            "`lop secret run --secret [redacted] -- client`, or `v=$(lop secret get "
            'NAME); curl -H "Authorization: Bearer $v" …`. Do not export a value '
            "and then dump the environment."
        ),
        examples=(
            "export V=$(lop secret get GITHUB_TOKEN); printenv V",
            "V=$(lop secret get GITHUB_TOKEN); export V; env | grep V=",
            "declare -x V=$(lop secret get GITHUB_TOKEN); printenv",
            "export V=$(lop secret get GITHUB_TOKEN); env",
            "V=$(lop secret get GITHUB_TOKEN); set | grep V=",
            "export V=$(lop secret get GITHUB_TOKEN); export",
            # R2-1: the reach of a dump is the shell's, whatever bound the name.
            "v=$(lop secret get GITHUB_TOKEN); declare",
            "v=$(lop secret get GITHUB_TOKEN); typeset",
            "v=$(lop secret get GITHUB_TOKEN); typeset -p v",
            "export V=$(lop secret get GITHUB_TOKEN); env -0 | tr '\\0' '\\n' | grep '^V='",
            "export V=$(lop secret get GITHUB_TOKEN); env -u HOME | grep '^V='",
            "V=$(lop secret get GITHUB_TOKEN) printenv V",
            "set -a; V=$(lop secret get GITHUB_TOKEN); printenv V",
            "readonly V=$(lop secret get GITHUB_TOKEN); readonly",
            "export V; V=$(lop secret get NAME); printenv V",
            # R5-1's class: a redirection is not the dumper's argument, so
            # `env 2>/dev/null` is still a bare `env`.
            "export V=$(lop secret get NAME); env 2>/dev/null | rev",
            "export V=$(lop secret get NAME); export 2>/dev/null",
        ),
        counterexamples=(
            'v=$(lop secret get GITHUB_TOKEN); env V="$v" some-client --flag',
            "printenv PATH",
            "printenv HOME",
            "set -e",
            "env",
            'v=$(lop secret get GITHUB_TOKEN); curl -H "Authorization: Bearer $v" https://x',
            # R2-4: binding is not exporting (bash 3.2.57: nothing printed, rc 1).
            "f() { local V=$(lop secret get GITHUB_TOKEN); printenv V; }; f",
            "readonly V=$(lop secret get GITHUB_TOKEN); printenv V",
            "declare V=$(lop secret get GITHUB_TOKEN); printenv",
            "export V=$(lop secret get GITHUB_TOKEN); export -n V; printenv V",
            "v=$(lop secret get GITHUB_TOKEN); declare -f",
            "export V=$(lop secret get GITHUB_TOKEN); env -i some-client",
            "export V; printenv V",
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
            # R1-8: deeper than the walk follows, which used to answer `none`
            # where six levels were refused — the opposite polarity.
            'echo "$("$("$("$("$("$("$(lop secret get GITHUB_TOKEN)")")")")")")"',
        ),
        counterexamples=(
            "echo 'unterminated-looking but no source here",
            "cat <<'EOF'\n$(lop secret get X)\nEOF",
            "ls -la /tmp",
            # R1-4: an apostrophe inside an unquoted body is the SCRIPT's
            # text, not an unterminated quote.
            "cat <<EOF > /tmp/m\nIt's fine\nEOF\nv=$(lop secret get GITH"
            'UB_TOKEN); curl -H "Bearer $v" https://x',
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
            "deliberate print is refused rather than relied on to be masked. "
            "The allowed derived observations are a length (`len`/`hash`/`id`) "
            "and anything read off a RESPONSE the value was used to build "
            "(`resp.status_code`, `done.returncode`); a membership or comparison "
            "(`print('x' in token)`) is refused, because a bool the model can "
            "ask for again is an oracle that reads the value one character at "
            "a time (R2-5)."
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
            # R2-2: a method whose value is built from its ARGUMENT carries the
            # value whatever its receiver is, printed directly or via a name.
            'token = secrets["NAME"]\nprint(",".join([token]))',
            'token = secrets["NAME"]\nprint("{}".format(token))',
            'token = secrets["NAME"]\nprint("".replace("", token))',
            'token = secrets["NAME"]\nblob = ",".join([token])\nprint(blob)',
            'import base64\ntoken = secrets["NAME"]\n'
            "enc = base64.b64encode(token.encode())\nprint(enc)",
        ),
        counterexamples=(
            'token = secrets["GITHUB_TOKEN"]\n'
            'requests.get(url, headers={"Authorization": f"Bearer {token}"})',
            'print(len(secrets["GITHUB_TOKEN"]))',
            'print("GITHUB_TOKEN" in secrets)',
            "print([name for name in secrets])",
            # Q1's boundary: a print of something DERIVED from the request is not
            # a print of the value. The response was built WITH it and does not
            # carry it, and `_value_taint` is the test that says so.
            'token = secrets["[redacted]"]\nresp = requests.get(url, head'
            'ers={"Authorization": f"Bearer {token}"})\nprint(resp.status'
            ")",
            'token = secrets["[redacted]"]\nresp = requests.get(url, head'
            'ers={"Authorization": f"Bearer {token}"})\nprint(str(resp.st'
            "atus))",
            'token = secrets["[redacted]"]\nresp = requests.get(url, head'
            'ers={"Authorization": f"Bearer {token}"})\nprint(resp.url)',
            'token = secrets["[redacted]"]\nresp = requests.get(url, head'
            'ers={"Authorization": f"Bearer {token}"})\nprint(len(resp.re'
            "ad()))",
            'token = secrets["[redacted]"]\ndone = subprocess.run(["curl"'
            ', "-H", "Authorization: Bearer " + token, url], capture_outp'
            'ut=True)\nprint("rc", done.returncode)',
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
        label="python.exception-of-source",
        verdict="printing",
        lang="python",
        question="Does the cell raise an exception whose message IS the value?",
        why=(
            "An exception's text is not a side channel — it is the output: the "
            "eval tool returns the traceback, so `raise ValueError(token)` and "
            "`assert False, token` both put the value in this result with no "
            "print anywhere in the cell (R3-4). Refused at the same "
            "pre-execution point as the print rule, because the traceback is "
            "written after the value already exists. A bare `raise` re-raises and "
            "carries nothing, and `assert token` fails with no message, so both "
            "stay allowed."
        ),
        rewrite=(
            "Do not carry the value out in an exception. Check a derived "
            "property (`len(token)`, `token is None`) and raise on that, and keep "
            "the value in the call that consumes it."
        ),
        examples=(
            'token = secrets["NAME"]\nraise ValueError(token)',
            'token = secrets["NAME"]\nraise ValueError(token[::-1])',
            'token = secrets["NAME"]\nassert False, token',
            'token = secrets["NAME"]\nraise RuntimeError(f"token={token}")',
        ),
        counterexamples=(
            "raise ValueError('plain')",
            'token = secrets["NAME"]\nassert token',
            'assert len(secrets["NAME"]) > 0',
            'token = secrets["NAME"]\nassert token is not None',
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
    # `&>` / `&>>` before `&`: bash reads them as ONE redirection of stdout and
    # stderr, so splitting them into a background `&` plus `>` put the target
    # on a stage of its own and the discard decision never saw it (round 6).
    "&>>",
    "&>",
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
_REDIRECTS = frozenset({">", ">>", ">|", "<", "<>", "<<", "<<-", "<<<", ">&", "<&", "&>", "&>>"})
#: `{name}` touching a redirection: bash allocates a fresh descriptor into
#: `name`, so the word is the redirection's, not the command's.
_FD_VARIABLE_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


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
        if ch == "`":
            # A backtick group inside `$( )` is opaque to the paren count: a `)`
            # inside it belongs to the inner command, not to this group.
            _inner, i = _read_backtick(text, i)
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


def _iter_expansions(text: str) -> list[tuple[str, tuple[int, int]]]:
    """The ``$( )`` and backtick groups in a here-doc body, in order.

    Deliberately not a command lexer — see :meth:`_ShellAnalyzer._heredoc_flow`:
    a body is data, so only the two constructs that really run are read out of
    it. An unbalanced group is DROPPED rather than raised: bash fails the whole
    command at expansion time, so there is nothing to print, and raising here is
    what produced R1-4's false refusal in the first place. The scan also STOPS at
    that group, so a balanced `$(…)` after it is never read — deliberately:
    the unbalanced one already fails the expansion of the whole body before any
    later group runs, so there is no later value to follow.
    """
    found: list[tuple[str, tuple[int, int]]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\\":
            i += 2
            continue
        if text.startswith("$(", i):
            try:
                inner, end = _read_parens(text, i + 1)
            except _LexFault:
                break
            found.append((inner, (i, end)))
            i = end
            continue
        if text[i] == "`":
            try:
                inner, end = _read_backtick(text, i)
            except _LexFault:
                break
            found.append((inner, (i, end)))
            i = end
            continue
        i += 1
    return found


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
        # NOT a word terminator: `` ` `` opens a POSIX command substitution, the
        # same thing `$(` is. Leaving it in this break set (R1-1) made
        # `echo "` + backtick + `lop secret get X` + backtick + `"` answer `none`
        # while the child ran and the raw value landed in the result — and made
        # `_read_backtick` below dead code.
        if ch in "|&;()<>":
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
                if text[i] == "`":
                    # The other command substitution, and just as live inside
                    # double quotes as `$(` is.
                    flush()
                    inner, end = _read_backtick(text, i)
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
        if (
            ch == "("
            and items
            and isinstance(items[-1], _Word)
            and items[-1].span[1] == i
            and (
                items[-1].pieces
                and items[-1].pieces[0].kind != "subst"
                and _ASSIGNMENT_RE.match(items[-1].pieces[0].text)
            )
        ):
            # `a=("$v" "$w")` — a compound assignment. The element list belongs
            # to the ASSIGNMENT, not to the shell's grammar: reading `(` as
            # punctuation here split the word into `a=` plus a separate stage,
            # so the elements' taint never reached `a` and `echo "${a[@]}"`
            # printed the value (R3-3). Folded into one word, the elements are
            # exactly what `_value_flow` already reads.
            inner, end = _read_parens(text, i)
            previous = items[-1]
            items[-1] = _Word(
                previous.pieces + (_Piece(text[i:end], "expand", (i, end)),),
                (previous.span[0], end),
            )
            i = end
            at_word_start = False
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

#: Prefixes that BIND an assignment rather than commanding with it. Binding is
#: not exporting (R2-4): only `export`, `declare -x`/`typeset -x`/`local -x` and
#: `set -a` put a value in the child environment that `printenv`/`env` print —
#: `local V=…; printenv V` and `readonly V=…; printenv V` print nothing (bash
#: 3.2.57 measured: empty, rc 1). Which prefix exports is decided per call in
#: :meth:`_ShellAnalyzer._binding_attributes`, not by membership here.
_BINDING_PREFIXES = frozenset({"export", "declare", "local", "readonly", "typeset"})

#: Prefixes whose `-x`/`-r` flags set the export/readonly ATTRIBUTE, and whose
#: bare or `-p` form lists variables with their values.
_ATTRIBUTE_BUILTINS = frozenset({"declare", "typeset", "local"})

#: Commands that can dump a variable's value out of the environment or the shell
#: rather than by naming it as an operand. `typeset` is `declare`'s synonym and
#: prints the same listing (R2-1); `readonly`/`local` list their own variables.
_ENV_DUMPERS = frozenset(
    {"printenv", "env", "set", "export", "declare", "typeset", "readonly", "local"}
)

#: `env` options that take the NEXT word as their argument, so that word is
#: neither an assignment nor the command `env` would run.
_ENV_ARG_OPTIONS = frozenset({"-u", "--unset", "-C", "--chdir", "-P"})

#: `env` options that start a new, empty environment: with no command after
#: them, only the assignments given on the line are printed.
_ENV_CLEARING = frozenset({"-i", "-", "--ignore-environment"})

#: Commands that move a path's contents somewhere else, so a value's copy keeps
#: its debt under a new name.
_PATH_MOVERS = frozenset({"cp", "mv", "ln", "install"})

#: Redirect targets that are not a file: the value still reaches this result.
_STDOUT_DEVICES = frozenset({"/dev/stdout", "/dev/fd/1", "/proc/self/fd/1"})

#: Redirect targets that are this process's STDERR: the value still reaches this
#: result, in its `--- stderr ---` section. Read as ordinary files they made
#: `lop secret get X >/dev/stderr` a contained write (round 6).
_STDERR_DEVICES = frozenset({"/dev/stderr", "/dev/fd/2", "/proc/self/fd/2"})

#: Redirect targets that drop the value entirely.
_DISCARD_DEVICES = frozenset({"/dev/null"})

#: A redirect target carrying expansion or glob syntax — `$LOG`, a backtick,
#: `*.f`. The shell chooses its text at run time, so the guard cannot read it and
#: must not write it down as an ordinary file: `>& $LOG` with `LOG=/dev/stderr`
#: was read as a contained path and put the raw value in this result's
#: `--- stderr ---` section (R7-2).
_UNRESOLVED_WORD_RE = re.compile(r"[$`*?\[\]]")

#: Spelling families that name a descriptor rather than a file whose contents the
#: guard could account for. Only the targets the fd table models
#: (`_DISCARD_DEVICES`, `_STDOUT_DEVICES`, `_STDERR_DEVICES`) are readable; any
#: other word in these families (`/dev/tty`, `/dev/fd/3`) is a device this guard
#: has no rule for, so it is refused rather than absorbed as a file.
_DESCRIPTOR_WORD_PREFIXES = ("/dev/", "/proc/self/fd/")


def _device_spelling(text: str) -> str:
    """``text`` with `.` segments and repeated `/` collapsed, for device lookup.

    Device recognition is an exact-string lookup, so `/dev//stderr`,
    `/dev/./stderr` and `/dev//stdout` — the same files as `/dev/stderr` and
    `/dev/stdout` — were read as ordinary contained paths and the raw value
    landed in the stderr or stdout section unnoticed (R7-3, pre-existing at both
    heads). Only the lexical half of normalisation is done here: `.` and empty
    segments are dropped, while `..` is left alone because resolving it lexically
    would be a guess about symlinks. A word carrying expansion or glob syntax is
    returned untouched — its spelling is not this guard's to resolve.
    """
    if _UNRESOLVED_WORD_RE.search(text):
        return text
    segments = [part for part in text.split("/") if part not in ("", ".")]
    if not segments:
        return "/" if text.startswith("/") else text
    return ("/" if text.startswith("/") else "") + "/".join(segments)


#: `NAME=`, `NAME+=` and `NAME[i]=` all bind a value to NAME (R3-3). The append
#: and element spellings used to fail this match, so `v+=$(lop secret get X)` and
#: `a[0]="$v"` were read as COMMANDS and their names never carried the taint.
_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[[^\]]*\])?\+?=")

#: The console scripts that run the `lop` CLI. `pyproject.toml` registers
#: `local-operator` for the same entry point as `lop` (R3-2), and a source rule
#: that knows one spelling of the program has a documented way around it.
_PROGRAM_NAMES = frozenset({"lop", "lop.exe", "local-operator", "local-operator.exe"})

#: Precommand wrappers that run their operand as THE command, with the same
#: stdout and environment (R3-2): `command lop secret get X`, `timeout 30 lop
#: …` and `env lop …` are `lop secret get X`. Each entry is
#: ``(options that take the next word, positional words the wrapper consumes)``,
#: so `timeout -s KILL 30 lop …` steps over `-s KILL` and the duration.
#:
#: **Why the bound is here.** Every wrapper in this table has a fixed, published
#: option grammar, so stepping over it is exact rather than a guess. A wrapper
#: outside the table (`sudo`, `caffeinate`, a function or alias the call
#: defines, a copy of the binary under a new name) is read as the command
#: itself. That residual is the same one a script invoked by name has, and the
#: PR names it rather than implying a reach this table does not have.
_PRECOMMANDS: dict[str, tuple[frozenset[str], int]] = {
    "command": (frozenset(), 0),
    "builtin": (frozenset(), 0),
    "exec": (frozenset({"-a"}), 0),
    "nohup": (frozenset(), 0),
    # GNU's long spellings take a SEPARATE operand word too (`nice --adjustment
    # 5`): without them the operand was read as the command (R5-2's note).
    "nice": (frozenset({"-n", "--adjustment"}), 0),
    "timeout": (frozenset({"-s", "-k", "--signal", "--kill-after"}), 1),
    "gtimeout": (frozenset({"-s", "-k", "--signal", "--kill-after"}), 1),
    "stdbuf": (frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}), 0),
    # Homebrew's coreutils prefix, as with `gtimeout`: the only GNU stdbuf on
    # macOS, and the one that accepts the long spellings above.
    "gstdbuf": (frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}), 0),
    "env": (frozenset({"-u", "--unset", "-C", "--chdir", "-P"}), 0),
}

#: `env` options after which the rest of the line is a STRING, not words: the
#: wrapper cannot be stepped over exactly, so `env` stays the command.
_ENV_SPLIT = frozenset({"-S", "--split-string"})

#: The consumer of `lop secret run|file … -- CMD`, and a source's own command
#: word, step over the same wrappers plus `xargs` (R4-2). `xargs` hands its
#: command the ENVIRONMENT unchanged, so `run … -- xargs printenv TOK` is
#: `printenv TOK`, and `echo X | xargs lop secret get` is a source whose name
#: arrives on stdin. The OUTER unwrap deliberately does not carry `xargs`: there
#: it also turns stdin into argv, which the dedicated `xargs` branch models
#: (:meth:`_ShellAnalyzer._interpreter`), and stepping over it would lose that.
_CONSUMER_WRAPPERS: dict[str, tuple[frozenset[str], int]] = {
    **_PRECOMMANDS,
    "xargs": (
        frozenset(
            {
                *("-I", "-L", "-n", "-P", "-s", "-d", "-E", "-a", "-J", "-R", "-S"),
                *("--max-args", "--max-procs", "--max-chars", "--delimiter", "--arg-file"),
            }
        ),
        0,
    ),
    # `time` in an OUTER stage is bash's reserved word (skipped as a keyword),
    # but after `run … --` it is `/usr/bin/time`, a program that runs its
    # operand with the same environment: `-- time printenv TOK | rev` leaked.
    "time": (frozenset({"-f", "-o", "--format", "--output"}), 0),
}


def _wrapped_command_index(
    texts: Sequence[str],
    *,
    table: Mapping[str, tuple[frozenset[str], int]] = _PRECOMMANDS,
    opaque: Collection[int] = (),
    assignments: Collection[int] | None = None,
) -> tuple[int, frozenset[int]]:
    """Where the command a wrapper chain runs begins: ``(index, kept)``.

    ``texts`` starts at the first word in command position. The result is the
    index of the command the wrappers run (``0`` when the first word is not a
    wrapper) and the indices of `env`'s own ``NAME=value`` words, which are the
    child's environment rather than wrapper syntax.

    ONE grammar for every caller (R4-1): the outer stage (:meth:`_ShellAnalyzer.
    _unwrap`), the `run`/`file` consumer and a source's command word all ask
    this, because R3-2 taught only the outer stage and `run … -- timeout 5
    printenv TOK` then walked straight past the consumer check. A second copy
    of the wrapper grammar is how that happened, so there is no second copy.

    ``opaque`` marks words that are a substitution: their text is not known, so
    the walk stops there rather than guess. ``assignments`` marks the
    ``NAME=value`` words when the caller has the tokens (the lexer knows a
    quoted `=` from a real one); plain text falls back to the assignment regex.
    A wrapper with nothing after it IS the command (`env` alone dumps), and
    `command -v`/`env -S` look a name up or re-split a string rather than
    running the next word, so none of those is stepped over.
    """

    def is_assignment(position: int) -> bool:
        if assignments is not None:
            return position in assignments
        return bool(_ASSIGNMENT_RE.match(texts[position]))

    cursor = 0
    kept: set[int] = set()
    while cursor < len(texts):
        if cursor in opaque:
            break
        wrapper = texts[cursor].rsplit("/", 1)[-1]
        spec = table.get(wrapper)
        if spec is None:
            break
        takes_argument, positionals = spec
        probe = cursor + 1
        lookup_only = False
        while probe < len(texts):
            option = texts[probe]
            if option == "--":
                probe += 1
                break
            if wrapper == "command" and option[:1] == "-" and set(option[1:]) & {"v", "V"}:
                lookup_only = True
                break
            if wrapper == "env" and (option in _ENV_SPLIT or option.startswith("--split-string")):
                lookup_only = True
                break
            if option in takes_argument:
                probe += 2
                continue
            if option[:1] == "-":
                probe += 1  # `-oL`, `-5`, `--signal=KILL`, env's `-i`/`-`
                continue
            break
        if lookup_only:
            break
        step_kept: set[int] = set()
        if wrapper == "env":
            while probe < len(texts) and is_assignment(probe):
                step_kept.add(probe)
                probe += 1
        probe += positionals
        if probe >= len(texts):
            break  # nothing left to run: the wrapper is the command
        kept |= step_kept
        cursor = probe
    return cursor, frozenset(kept)


def _env_clears_environment(wrappers: Sequence[str]) -> bool:
    """Does an `env` in this wrapper chain start its child with no environment?

    Only an `-i` (or `-`, `--ignore-environment`) that `env` ITSELF parses as an
    option counts (R5-2). Searching every wrapper word accepted `stdbuf -i 0`
    (stdbuf's stdin-buffer option) and `env -u -i` (unset a variable named
    `-i`), and both leaked with the exemption granted. The walk is `env`'s own
    getopt grammar from :data:`_PRECOMMANDS`: operand-taking options skip their
    operand, a short cluster is read letter by letter until a letter that takes
    the rest of the word as its operand, and the first non-option ends it.
    Each wrapper's extent is found by :func:`_wrapped_command_index` itself, so
    this is not a second copy of the chain grammar.
    """
    env_takes, _ = _PRECOMMANDS["env"]
    operand_letters = {option[1] for option in env_takes if len(option) == 2} | {"S"}
    cursor = 0
    while cursor < len(wrappers):
        wrapper = wrappers[cursor].rsplit("/", 1)[-1]
        spec = _CONSUMER_WRAPPERS.get(wrapper)
        if spec is None:
            return False
        # The extent of THIS wrapper only: its name is swapped for a key no
        # other word can equal, so a nested `env env -i` is not swallowed into
        # the first step, and a stand-in consumer follows because without a
        # word after it the grammar reads the wrapper as the command itself.
        step, _ = _wrapped_command_index(["\0", *wrappers[cursor + 1 :], "\0"], table={"\0": spec})
        if step <= 0:
            return False
        if wrapper == "env":
            probe = cursor + 1
            while probe < cursor + step:
                option = wrappers[probe]
                if option in ("-", "-i", "--ignore-environment"):
                    return True
                if option == "--" or option[:1] != "-":
                    break
                if option in env_takes:
                    probe += 2
                    continue
                if not option.startswith("--"):
                    for letter in option[1:]:
                        if letter == "i":
                            return True
                        if letter in operand_letters:
                            break
                probe += 1
        cursor += step
    return False


#: `$l`, `${l}` or `"$l"` — a word that is one variable and nothing else, which
#: is how `l=lop; $l secret get X` spells the program (R3-2).
_BARE_REF_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")

#: Builtins that bind their operands from stdin or a redirected file: the
#: `read` of R1-3, plus the array readers R3-3 found unfollowed.
_READ_BUILTINS = frozenset({"read", "mapfile", "readarray"})
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
#: How a Python program reaches an environment variable it was given. Only
#: the two documented spellings: an unknown one (`dict(os.environ)["TOKEN"]`)
#: is the residual, stated in the rule rather than silently covered.
_PY_ENV_READ_RE = re.compile(r"\b(?:os\.environ|environ|getenv)\b")


@dataclass(frozen=True)
class _InlineLanguage:
    """How one non-shell interpreter's inline program reaches the value (R4-2).

    These languages are not this module's walk, so the test is the coarse one
    the Python arm always used, searched rather than parsed: the program must
    NAME the variable, READ the environment and PRINT (``printer``, or a flag
    in ``print_flags`` that prints the program's result, like `node -p`). A
    `file` consumer must also READ A FILE (``file_read``), because there the
    variable holds a path and naming it alone (`getsize(os.environ["G"])`)
    reads no bytes. Fail-closed in the module's usual direction:
    `print(len(os.environ["T"]))` is refused with the rest.
    """

    printer: re.Pattern[str]
    env_read: re.Pattern[str]
    file_read: re.Pattern[str]
    print_flags: frozenset[str] = frozenset()
    #: Short-option letters that print the result wherever they sit in a
    #: cluster: `node -pe '…'` is `-p -e '…'` (R5-3), and matching whole words
    #: against ``print_flags`` alone let the clustered spelling through.
    print_letters: frozenset[str] = frozenset()


_PYTHON_LANGUAGE = _InlineLanguage(
    printer=_INLINE_PRINT_RE,
    env_read=_PY_ENV_READ_RE,
    file_read=re.compile(r"\bopen\s*\(|\bread_(?:text|bytes)\b|\bcopyfileobj\b"),
)
_AWK_LANGUAGE = _InlineLanguage(
    printer=re.compile(r"\bprintf?\b"),
    env_read=re.compile(r"\bENVIRON\b"),
    file_read=re.compile(r"\bgetline\b"),
)
_NODE_LANGUAGE = _InlineLanguage(
    printer=re.compile(r"\bconsole\.\w+|\bprocess\.std(?:out|err)\.write\b|\bthrow\b"),
    env_read=re.compile(r"\bprocess\.env\b"),
    file_read=re.compile(r"\breadFileSync\b|\breadFile\b|\bcreateReadStream\b"),
    print_flags=frozenset({"-p", "--print"}),
    print_letters=frozenset({"p"}),
)

#: The interpreters whose inline program a `run`/`file` consumer check reads,
#: by command name (a versioned `python3.12` is looked up as `python3`).
#: **The bound**, stated rather than implied: a program handed in a FILE
#: (`perl x.pl`, `node app.js`), a language not listed (`php -r`, `lua -e`,
#: `osascript -e`), a dump of the WHOLE environment that never names the
#: variable (`print(os.environ)`, `print %ENV`, `console.log(process.env)`),
#: and an environment read spelled another way (`system("printenv")`) are the
#: residual — the "child prints on its own initiative" class the PR names.
_INLINE_LANGUAGES: dict[str, _InlineLanguage] = {
    "python": _PYTHON_LANGUAGE,
    "python3": _PYTHON_LANGUAGE,
    "perl": _InlineLanguage(
        printer=re.compile(r"\b(?:print|printf|say|warn|die)\b"),
        env_read=re.compile(r"\$ENV\s*\{|%ENV\b"),
        file_read=re.compile(r"\bopen\b|<\s*\$?\w*\s*>|\bslurp\b"),
    ),
    "ruby": _InlineLanguage(
        printer=re.compile(
            r"\b(?:puts|print|printf|pp|p|warn|abort|raise)\b|\$std(?:out|err)\b|\bSTD(?:OUT|ERR)\b"
        ),
        env_read=re.compile(r"\bENV\b"),
        file_read=re.compile(r"\b(?:File|IO)\.(?:read|open|readlines|foreach|binread)\b"),
    ),
    "node": _NODE_LANGUAGE,
    "nodejs": _NODE_LANGUAGE,
    "awk": _AWK_LANGUAGE,
    "gawk": _AWK_LANGUAGE,
    "mawk": _AWK_LANGUAGE,
    "nawk": _AWK_LANGUAGE,
}

#: `python3.12` and `python3` are one interpreter to the check above.
_VERSIONED_PYTHON_RE = re.compile(r"^(python3?)(?:\.\d+)+$")

#: A shell's inline-program flag, alone or clustered (`-c`, `-lc`, `-ec`).
_SHELL_INLINE_FLAG_RE = re.compile(r"^-[A-Za-z]*c[A-Za-z]*$")

#: `lop secret file`'s default `--env-var`. Mirrored from
#: ``secrets/cli.py``'s ``DEFAULT_FILE_ENV_VAR`` rather than imported: this
#: module stays stdlib-only on the tool path, and a test pins the two equal.
_DEFAULT_FILE_ENV_VAR = "GOOGLE_APPLICATION_CREDENTIALS"

#: A raw-text source spelling, used ONLY to decide the fail-closed question
#: when the tokenizer faults: is there something here that could be a source?
_RAW_SOURCE_RE = re.compile(r"\b(?:lop|local-operator)\b[^\n]{0,80}?\bsecret\b")

#: Bound on nested inline programs and substitutions. A refusal is for one
#: call; nothing legitimate nests six programs deep, and the bound keeps a
#: pathological input from being a stack exercise.
_MAX_DEPTH = 6


#: One stage's token list. Named because the helpers that take it are as long
#: as the annotation and get read far more often than the type is changed.
_Stage = list[_Word | _Op | _Body]


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
        #: The secret behind each of those paths, so a read of one can name it.
        self.path_secrets: dict[str, str] = {}
        #: The secret behind this command list's `< PATH` redirect, if any: a
        #: compound's redirect sits on its LAST stage (`…; done < f`), after the
        #: body that reads it, so it is recorded for the whole list (R3-3).
        self._stdin_taint: str = ""
        #: Names an `export NAME` (no `=`) declared before their assignment, so
        #: the later binding is exported too (R3-3).
        self._pending_exports: set[str] = set()
        #: Names bound to a program name (`l=lop`), so `$l secret get X` still
        #: names the source (R3-2).
        self.program_vars: dict[str, str] = {}
        #: Names this command put into the exported namespace (R1-2), whose
        #: values an environment dump can print without naming the secret.
        #: Only a real export lands here (R2-4): `export`, a `-x` attribute, or
        #: an assignment while `set -a` is on.
        self.exported_vars: set[str] = set()
        #: Names bound by `readonly`/`-r` and by `local`: `readonly` and `local`
        #: with no operand list exactly these, with their values.
        self.readonly_vars: set[str] = set()
        self.local_vars: set[str] = set()
        #: `set -a` / `set -o allexport`: every later plain assignment exports.
        self._allexport = False
        #: The text of a substitution this walk was too deep to follow (R1-8),
        #: and of the here-doc bodies it could not place.
        self._skipped_deep = ""
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
        """`v` for `v=`, `v+=` and `v[0]=` alike: the variable the value lands in."""
        match = _ASSIGNMENT_RE.match(word.pieces[0].text)
        return match.group(1) if match else word.pieces[0].text.split("=", 1)[0]

    @staticmethod
    def _redirect_targets(stage: Sequence[_Word | _Op | _Body]) -> set[int]:
        """Indices of the words that are a redirection's target, not an argument."""
        targets: set[int] = set()
        pending = False
        for index, item in enumerate(stage):
            if isinstance(item, _Op):
                pending = item.text in _REDIRECTS
            elif isinstance(item, _Word) and pending:
                targets.add(index)
                pending = False
        return targets

    @classmethod
    def _redirect_words(cls, stage: Sequence[_Word | _Op | _Body]) -> set[int]:
        """Indices of every word that belongs to a redirection, not to argv.

        That is the target (`/dev/null` in `2>/dev/null`) AND the descriptor
        prefix the lexer hands over as its own word: `2` in `2>&1`, `{fd}` in
        `{fd}>x`. Bash reads a digit (or `{name}`) as a descriptor only when it
        touches the operator — `echo 2 >x` prints `2` — so adjacency is the
        test, as in :meth:`_fd_redirections`.

        Every walk that decides "which word is the command" must skip these
        (R5-1): `run … -- env 2>/dev/null` read `2` as the consumer (the wrapper
        grammar stepped over `env`), so the dump check never saw `env` and the
        raw value came back.
        """
        words = cls._redirect_targets(stage)
        for index, item in enumerate(stage):
            if not isinstance(item, _Op) or item.text not in _REDIRECTS or not index:
                continue
            previous = stage[index - 1]
            if isinstance(previous, _Word) and previous.span[1] == item.span[0]:
                text = cls._word_text(previous).strip()
                if text.isdigit() or _FD_VARIABLE_RE.fullmatch(text):
                    words.add(index - 1)
        return words

    @classmethod
    def _unwrap(cls, stage: list[_Word | _Op | _Body]) -> list[_Word | _Op | _Body]:
        """The stage with its precommand wrappers dropped (R3-2).

        `timeout 30 lop secret get X | rev` is, for every rule in the table,
        `lop secret get X | rev`: the wrapper runs its operand with the same
        stdout and environment. Dropping the wrapper words here, once, is what
        lets every rule below see the real command rather than each rule
        learning its own list of wrappers.

        `env`'s own `NAME=value` words are KEPT: once the wrapper is gone they
        sit before the command, which is exactly what they are to the child —
        a `V=… command` prefix (so `env V="$(…)" printenv V` reaches the dump
        rule). A wrapper with nothing after it IS the command (`env` alone
        dumps, `nice` alone prints a number), and `command -v`/`-V` looks a name
        up rather than running it, so none of those is unwrapped.
        """
        # Every redirection word, the `2` of `2>/dev/null` included, is not a
        # command word (R5-1's class): reading it as one stopped the unwrap.
        targets = cls._redirect_words(stage)
        sequence = [
            index
            for index, item in enumerate(stage)
            if isinstance(item, _Word) and index not in targets
        ]

        def word_at(position: int) -> _Word:
            item = stage[sequence[position]]
            assert isinstance(item, _Word)
            return item

        cursor = 0
        while cursor < len(sequence):
            text = cls._word_text(word_at(cursor)).strip()
            if text and text not in _SHELL_KEYWORDS and not cls._is_assignment(word_at(cursor)):
                break
            cursor += 1
        tail = sequence[cursor:]
        words = [word_at(position) for position in range(cursor, len(sequence))]
        start, kept = _wrapped_command_index(
            [cls._word_text(word).strip() for word in words],
            opaque={
                k for k, word in enumerate(words) if any(p.kind == "subst" for p in word.pieces)
            },
            assignments={k for k, word in enumerate(words) if cls._is_assignment(word)},
        )
        if start == 0:
            return stage
        drop = {tail[k] for k in range(start) if k not in kept}
        return [item for index, item in enumerate(stage) if index not in drop]

    def _resolve_program(self, text: str) -> str:
        """A bare reference standing for a program name resolves to it (R3-2).

        `l=lop; $l secret get X` names the source without the literal word `lop`
        in command position. The variable's value is followed rather than the
        text searched, so an arbitrary re-speller (`f() { lop "$@"; }`, an alias,
        a copy of the binary under a new name) is the residual: this closes the
        spelling the review measured and states the bound instead of implying
        the guard tracks any indirection at all.
        """
        match = _BARE_REF_RE.match(text)
        if match:
            return self.program_vars.get(match.group(1), text)
        return text

    @classmethod
    def _command_word(cls, stage: list[_Word | _Op | _Body]) -> str:
        """The command this stage runs, assignment prefixes skipped.

        Redirection words are skipped as well (R5-1's class): bash accepts a
        redirection anywhere in a simple command, so `2>/dev/null lop secret get
        X | rev` runs `lop` and the `2` in front of it is no command.
        """
        redirects = cls._redirect_words(stage)
        for index, item in enumerate(stage):
            if not isinstance(item, _Word) or index in redirects:
                continue
            text = cls._word_text(item).strip()
            if not text or text in _SHELL_KEYWORDS:
                continue
            if cls._is_assignment(item):
                continue
            return text.rsplit("/", 1)[-1]
        return ""

    @classmethod
    def _fd_redirections(cls, stage: _Stage) -> list[tuple[str, str, _Word]]:
        """``(fd, operator, target)`` for every redirection in the stage.

        The fd matters: `2>/dev/null` drops STDERR and leaves stdout in this
        result, so reading it as an stdout redirect discarded the whole stage and
        `lop secret get X >/dev/stdout 2>/dev/null` came back as a consumer with
        the raw value in the result (R3-2's `nohup` row). The number is its own
        WORD to the tokenizer (`2` `>` `/dev/null`), so the fd is recognised by a
        bare-digit word that ENDS exactly where the operator begins.
        """
        found: list[tuple[str, str, _Word]] = []
        pending: tuple[str, int] | None = None
        for index, item in enumerate(stage):
            if isinstance(item, _Op):
                pending = (item.text, index) if item.text in _REDIRECTS else None
                continue
            if not isinstance(item, _Word) or pending is None:
                continue
            operator, op_index = pending
            pending = None
            fd = "0" if operator.startswith("<") else "1"
            previous = stage[op_index - 1] if op_index else None
            if (
                isinstance(previous, _Word)
                and previous.span[1] == stage[op_index].span[0]
                and (
                    cls._word_text(previous).strip().isdigit()
                    or _FD_VARIABLE_RE.fullmatch(cls._word_text(previous).strip())
                )
            ):
                # `{fd}>/dev/null` opens a NEW descriptor; stdout is untouched,
                # so reading it as fd 1 marked `{fd}>/dev/null lop secret get X
                # | rev` discarded (R5-1's class sweep).
                fd = cls._word_text(previous).strip()
            found.append((fd, operator, item))
        return found

    @staticmethod
    def _resolve_destination(text: str, fds: dict[str, tuple[str, str]]) -> tuple[str, str]:
        """The ``(kind, path)`` a redirect target names, given the fd table.

        A device target is not a file: `/dev/stdout` and `/dev/null` are the
        descriptors they name AT THAT MOMENT, so they resolve through ``fds``
        rather than being written down as paths. It is shared with the `tee`
        operand walk because a device operand is the same question there
        (round 6). The target is read through `_device_spelling` first, because
        the lookup is an exact-string one and `>/dev//stderr` names the same
        file as `>/dev/stderr` (R7-3).
        """
        device = _device_spelling(text)
        if device in _DISCARD_DEVICES:
            return ("null", "")
        if device in _STDOUT_DEVICES:
            return fds["1"]
        if device in _STDERR_DEVICES:
            return fds["2"]
        return ("path", text)

    @classmethod
    def _stdout_destination(cls, stage: _Stage) -> tuple[str, str]:
        """Where this stage's STDOUT finally points: ``(kind, path)``.

        ``kind`` is ``result`` (this tool result's stdout), ``stderr`` (its
        `--- stderr ---` section, which the operator and the model read just the
        same), ``null`` (dropped) or ``path`` (a file, named by ``path``).

        The redirections are applied IN ORDER, the way the shell applies them,
        because each one rebinds a descriptor to wherever its target points AT
        THAT MOMENT. Deciding per redirect with sticky flags (round 6) let an
        earlier `>/dev/null` win over a later `1>&2`, so `get X >/dev/null 1>&2`
        read as discarded while the value landed on stderr — and it let
        `2>&1 >/dev/null` and `>/dev/null 2>&1` read the same only by accident.
        `/dev/stderr`, `/dev/fd/2` and `/proc/self/fd/2` are descriptor 2, not
        files: read as paths they made `get X >/dev/stderr` a contained write.

        Only descriptors 1 and 2 are modelled, since the value is written to
        stdout and only 1 and 2 are captured; a `{fd}>` opens a fresh descriptor
        and touches neither. `&>` / `>&WORD` rebind both: `&>` for a word this
        guard can read, `>&WORD` only for the two shapes
        `_legacy_word_destination` names, since that spelling is where round 7's
        leaks came from. A CLOSED descriptor is not a discarded one (R7-1), and a
        word the guard cannot read is refused rather than written down as a file
        (R7-2). `&>>` is lexed as one operator with `&>` even though it is bash
        ≥4 syntax and this host's `/bin/bash` is 3.2: there it is a syntax error,
        the line aborts and nothing is written (ran=False, measured), so the
        lexing costs nothing locally and is right wherever the tool resolves a
        newer shell.
        """
        fds: dict[str, tuple[str, str]] = {"1": ("result", ""), "2": ("stderr", "")}

        for fd, op, target in cls._fd_redirections(stage):
            text = cls._word_text(target).strip().strip("'\"")
            if op in ("&>", "&>>"):
                # `&>WORD` sends BOTH streams to the word. A word the guard
                # cannot read on its own keeps the pre-round-6 verdict: fd 1 is
                # left on this result, so the stage is refused. That is where
                # `&> $LOG` (`LOG=/dev/stderr`) put the raw value in the
                # `--- stderr ---` section once it was absorbed as a path (R7-2).
                if not cls._word_is_unreadable(text):
                    fds["1"] = fds["2"] = cls._resolve_destination(text, fds)
            elif op in (">", ">>", ">|"):
                if fd in fds:
                    fds[fd] = cls._resolve_destination(text, fds)
            elif op in (">&", "<&"):
                if text == "-":
                    if fd in fds:
                        # A CLOSED descriptor is NOT a discarded one. That was
                        # round 6's premise here and it is false on the shell this
                        # tool resolves for itself (`resolve_bash_shell(None)` →
                        # `/bin/bash` 3.2.57): a BUILTIN keeps writing to the
                        # shell's stream while it reports the close, measured —
                        #   bash -c 'echo AAA >&-' | wc -c        -> 4
                        #   bash -c 'printf BBB >&-'              -> 3
                        #   bash -c '/bin/echo CCC >&-'           -> 0 (external honours it)
                        #   bash -c 'echo DDD 2>&- 1>&-' | wc -c  -> 4 (fd 1 closed)
                        # — so `v=$(lop secret get N); echo "$v" >&-` printed the
                        # raw value in `--- stdout ---` while the scan read the fd
                        # as discarded (R7-1). Each descriptor is therefore read
                        # as still reaching the visible stream it names, the way
                        # an unmodelled source descriptor already is: fd 1 this
                        # result, fd 2 its stderr section. Symmetric, one rule,
                        # and it puts the `2>&-` dups back on the refusal side
                        # rather than resting on the premise this round retired —
                        # they cost nothing real, since bash aborts them anyway
                        # (`echo AAA 2>&- 1>&2` → `bash: 2: Bad file descriptor`,
                        # rc=1, 0 bytes on both streams, like `2>&- >&2`).
                        fds[fd] = ("result", "") if fd == "1" else ("stderr", "")
                elif text.isdigit():
                    if fd in fds:
                        # An unmodelled source descriptor (3, 9…) is unknown
                        # territory; reading it as this result is the safe side.
                        fds[fd] = fds.get(text, ("result", ""))
                elif op == ">&" and fd == "1":
                    both = cls._legacy_word_destination(text, fds)
                    fds["1"] = fds["2"] = both
        return fds["1"]

    @classmethod
    def _word_is_unreadable(cls, text: str) -> bool:
        """Is this redirect target a word the guard cannot read as itself?

        Three shapes: nothing at all, expansion or glob syntax, and a word in a
        descriptor family the fd table does not model (``/dev/tty``,
        ``/dev/fd/3``). A device the guard has no rule for is not a file whose
        contents it could account for, so it earns a refusal rather than the
        benefit of the doubt.
        """
        if not text or _UNRESOLVED_WORD_RE.search(text):
            return True
        spelling = _device_spelling(text)
        if spelling in _DISCARD_DEVICES or spelling in _STDOUT_DEVICES:
            return False
        if spelling in _STDERR_DEVICES:
            return False
        return spelling == "/dev" or spelling.startswith(_DESCRIPTOR_WORD_PREFIXES)

    @classmethod
    def _legacy_word_destination(
        cls, text: str, fds: dict[str, tuple[str, str]]
    ) -> tuple[str, str]:
        """The single destination bash's older `>&WORD` opens for BOTH streams.

        Absorbed for two word shapes only, each with driven evidence: the fd-2
        device (`>& /dev/stderr` is this result's stderr section, and the round-6
        device rule already refuses it as `shell.source-to-stderr`), and a plain
        literal file path (`>& /tmp/f`), whose file the rig has verified holds
        the value. Every other word keeps the verdict every revision before
        round 6 gave — base refused this spelling outright, and round 6's
        loosening is what R7-1 and R7-2 are about — so fd 1 stays on this result
        and the stage is refused. Two of those are worth naming:

        * a word carrying expansion or glob syntax (`>& $LOG`) is not resolved;
        * a device word gets no device resolution under the legacy spelling
          (`>& /dev/null`, `>& /dev/tty`): the modern `&>` gets it, and this
          spelling only earned the two rows above.

        The trade is deliberate. For this guard an unrecognised spelling is
        REFUSED, not allowed: a false refusal costs a user one workaround
        (`&> /dev/null` or `>/dev/null 2>&1`), while a false allow publishes a
        secret into the tool result, which is the whole harm this module exists
        to prevent.
        """
        spelling = _device_spelling(text)
        if spelling in _STDERR_DEVICES:
            return fds["2"]
        if (
            cls._word_is_unreadable(text)
            or spelling in _DISCARD_DEVICES
            or spelling in _STDOUT_DEVICES
        ):
            # Not the current ``fds["1"]``: `>&WORD` OPENS the word, it does not
            # duplicate fd 1, so an earlier `>/dev/null` does not survive it and
            # the refusal has to hold whatever fd 1 was bound to at that point.
            return ("result", "")
        return ("path", text)

    def _tee_operands_to_stderr(self, words: list[_Word], stdout: tuple[str, str]) -> bool:
        """Register every file `tee` writes, and report a device STDERR operand.

        `tee` is the one emitter that writes the value somewhere OTHER than its
        own stdout, and it writes to every operand. Reading only the first
        operand's path (and treating a device as a path) meant
        `get X | tee /dev/stderr >/dev/null` was contained while the value sat
        in this result's stderr section (round 6) — so each operand is resolved
        the way a redirect target is, and a fd-2 device is the leak signal.

        A `-` operand is NOT stdout — measured against GNU tee, which writes a
        file named `-` — so it is registered as a path like any other operand;
        only real flags are skipped.
        """
        fds = {"1": stdout, "2": ("stderr", "")}
        to_stderr = False
        for operand in words[1:]:
            text = self._word_text(operand).strip().strip("'\"")
            if not text or (text.startswith("-") and text != "-"):
                continue
            kind, path = self._resolve_destination(text, fds)
            if kind == "stderr":
                to_stderr = True
            elif kind == "path":
                self._register_path(path, operand.span)
        return to_stderr

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
                    # The bound used to `continue`, which made SEVEN levels of
                    # nesting answer `none` where six were refused (R1-8) — the
                    # opposite polarity from this module's stated asymmetry.
                    self._note_depth(piece.text)
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
        # The assignment skip is POSITIONAL, not stage-wide. Only a word BEFORE
        # the command word is an assignment to the shell (`V=… cmd` binds V in
        # the child's environment and hands cmd nothing in argv). After the
        # command word the same shape is an ordinary argument, and skipping it
        # stage-wide let `echo TOKEN=$v` and `echo "TOKEN=$v" | rev` through as
        # consumers: the incident's own `KEY = value` print, with its `=`
        # moved into the word.
        in_prefix = True
        for item in stage:
            if not isinstance(item, _Word):
                continue
            if in_prefix and (
                self._is_assignment(item) or self._word_text(item).strip() in _SHELL_KEYWORDS
            ):
                continue
            in_prefix = False
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

    def _secret_of(self, names: Iterable[str]) -> str:
        """The secret behind the first of ``names`` that holds one, or ""."""
        for name in names:
            if name in self.value_vars or name in self.file_vars:
                return self.value_vars.get(name) or self.file_vars.get(name) or name
        return ""

    @staticmethod
    def _operands(words: Sequence[_Word]) -> list[_Word]:
        """The words after a builtin's name that are not option flags."""
        return [
            word
            for word in words
            if (text := _ShellAnalyzer._word_text(word).strip()) and text[:1] not in ("-", "+")
        ]

    def _is_listing(self, command: str, flags: list[str], args: Sequence[_Word]) -> bool:
        """Does a binding builtin PRINT here rather than bind?

        Bare `export`/`readonly`/`declare`/`typeset`/`local` list variables with
        their values, and so does `-p` with or without names. Anything carrying
        an operand and no `-p` binds or re-attributes, and prints nothing.
        """
        if any(flag in ("-p", "--print") or (flag[:1] == "-" and "p" in flag) for flag in flags):
            return True
        return not self._operands(args) and command in _BINDING_PREFIXES

    @staticmethod
    def _binding_attributes(command: str, flags: list[str]) -> tuple[bool, bool, bool]:
        """``(exports, unexports, readonly)`` for one binding builtin call (R2-4).

        `local`/`readonly`/`declare` BIND without exporting: measured on bash
        3.2.57, `f() { local V=…; printenv V; }` and `readonly V=…; printenv V`
        print nothing and exit 1. Only `export` (not `export -n`) and a `-x`
        attribute put the value into a child's environment.
        """
        dashed = "".join(flag[1:] for flag in flags if flag.startswith("-") and flag[:2] != "--")
        plussed = "".join(flag[1:] for flag in flags if flag.startswith("+"))
        if command == "export":
            return "n" not in dashed, "n" in dashed, False
        if command == "readonly":
            return False, False, True
        return "x" in dashed, "x" in plussed, "r" in dashed

    @classmethod
    def _command_span(cls, stage: list[_Word | _Op | _Body]) -> tuple[int, int]:
        """The span of the stage's command word, or ``(0, 0)`` when it has none."""
        redirects = cls._redirect_words(stage)
        for index, item in enumerate(stage):
            if not isinstance(item, _Word) or cls._is_assignment(item) or index in redirects:
                continue
            text = cls._word_text(item).strip()
            if text and text not in _SHELL_KEYWORDS:
                return item.span
        return (0, 0)

    def _prefix_exports(self, stage: list[_Word | _Op | _Body], *, depth: int) -> list[str]:
        """Names a `V=… command` prefix exports into THIS command's environment.

        `V=$(lop secret get X) printenv V` binds nothing in the shell, but the
        child runs with `V` in its environment — the same reach as `export`.
        """
        names: list[str] = []
        for item in stage:
            if not isinstance(item, _Word):
                continue
            if not self._is_assignment(item):
                break
            flow = self._value_flow(item, depth=depth)
            name = self._assignment_name(item)
            if flow.value:
                self.value_vars.setdefault(name, flow.name)
                names.append(name)
            elif flow.path:
                self.file_vars.setdefault(name, flow.name)
                names.append(name)
        return names

    def _dumped_name(
        self, command: str, flags: list[str], words: list[_Word], prefix: list[str], *, depth: int
    ) -> str:
        """The secret an environment/variable dump would print, or "".

        Each dumper's REACH is the precision here, and reach is keyed on what
        the SHELL did with the name, not on which prefix bound it (R2-1/R2-4):

        * `printenv` (bare or naming it) and `env` with no command show the
          EXPORTED namespace — `export`, a `-x` attribute, `set -a`, or a
          `V=… command` prefix. `env` flags that only change the format or drop
          other names (`-0`, `--null`, `-u OTHER`) still print it; `-i`/`-`
          start empty, so only the assignments on the line are shown.
        * bare `set`, bare `declare`/`typeset`, and `declare -p`/`typeset -p`
          show every SHELL variable, exported or not (measured: bare `declare`
          prints a non-exported `v`); `declare -x` bare shows only exported
          ones, `readonly`/`declare -r` the readonly ones, `local` the locals.
        * `export` bare or `-p` lists the exported namespace.

        An `env` that runs a command (`env V=… client`) is a CONSUMER handing
        the environment to a child, and is allowed.
        """
        args = words[1:]
        exported = [*self.exported_vars, *prefix]
        operands = [self._word_text(word).strip() for word in self._operands(args)]
        if command == "printenv":
            return self._secret_of([n for n in operands if n in exported] if operands else exported)
        if command == "env":
            return self._env_dump(args, exported, depth=depth)
        if command == "set":
            return "" if operands or flags else self._secret_of(list(self.value_vars))
        if not self._is_listing(command, flags, args):
            return ""
        if operands:
            # `declare -p V` / `export -p V`: the named ones, whatever their
            # attributes — a conservative reading, since a false refusal costs
            # one re-spelling and a false allow is the value in the transcript.
            return self._secret_of(operands)
        dashed = "".join(flag[1:] for flag in flags if flag.startswith("-") and flag[:2] != "--")
        if command == "export" or "x" in dashed:
            return self._secret_of(exported)
        if command == "readonly" or "r" in dashed:
            return self._secret_of(list(self.readonly_vars))
        if dashed.strip("p"):
            # `declare -f`/`-F` list functions, `-a`/`-A`/`-i` list arrays and
            # integers — none of them is a string a source bound.
            return ""
        if command == "local":
            return self._secret_of(list(self.local_vars))
        return self._secret_of([*self.value_vars, *self.file_vars])

    def _env_dump(self, args: Sequence[_Word], exported: list[str], *, depth: int) -> str:
        """`env`'s reach: its assignments, its unsets, and whether it runs a command."""
        # A SET, not a list: `-u`/`--unset` need only membership, and a
        # `list.remove` here reads as a filesystem removal to the AST guard in
        # tests/unit/session/test_no_session_deletion.py — the guard is biased
        # to false positives on purpose, and its allow-list should stay short.
        printed = set(exported)
        pending_arg = False
        for word in args:
            text = self._word_text(word).strip()
            if pending_arg:
                pending_arg = False
                printed.discard(text)
                continue
            if text in _ENV_ARG_OPTIONS:
                pending_arg = True
                continue
            if text.startswith("--unset="):
                printed.discard(text.split("=", 1)[1])
                continue
            if text in _ENV_CLEARING:
                printed = set()
                continue
            if text in ("-S", "--split-string") or text.startswith("--split-string="):
                return ""  # a command string follows: env runs something
            if text.startswith("-"):
                continue  # -0/--null/-v: format only, the dump is unchanged
            if self._is_assignment(word):
                flow = self._value_flow(word, depth=depth)
                name = self._assignment_name(word)
                if flow.value or flow.path:
                    if flow.value:
                        self.value_vars.setdefault(name, flow.name)
                    else:
                        self.file_vars.setdefault(name, flow.name)
                    printed.add(name)
                continue
            return ""  # the first plain word is the command env runs: a consumer
        return self._secret_of(printed)

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

    def _register_path(self, path: str, span: tuple[int, int], name: str = "") -> None:
        path = path.strip().strip("'\"")
        if path and path not in _DISCARD_DEVICES:
            self.tainted_paths.setdefault(path, span)
            if name:
                self.path_secrets.setdefault(path, name)

    @classmethod
    def _redirect_only_read(cls, stage: list[_Word | _Op | _Body]) -> bool:
        """Is this stage bare `$(<PATH)` — a redirect with no command at all?

        Bash's shorthand for reading a file: the bytes come back as the
        substitution's text, so a tainted path here IS the value (R3-3). Without
        this, the redirect's target is the only word, it reads as the command
        NAME, and the stage falls through to the consumer branch.
        """
        operands = 0
        for index, item in enumerate(stage):
            if not isinstance(item, _Word):
                continue
            if index in cls._redirect_targets(stage):
                continue
            if cls._is_assignment(item) or cls._word_text(item).strip() in _SHELL_KEYWORDS:
                continue
            operands += 1
        return operands == 0

    def _note_stdin_redirects(self, stages: list[list[_Word | _Op | _Body]]) -> None:
        """Flag a `< PATH` whose path a secret was written into in this list."""
        for stage in stages:
            for fd, op, target in self._fd_redirections(stage):
                if op not in ("<", "<>"):
                    continue
                text = self._word_text(target).strip().strip("'\"")
                name = self.path_secrets.get(text)
                if name:
                    self._stdin_taint = self._stdin_taint or name

    def _prescan_written_paths(self, stages: list[list[_Word | _Op | _Body]]) -> None:
        """Register the files a source verb writes to BEFORE the walk reaches them.

        The same registration the source branch does, hoisted: a read on a LATER
        stage (`l=$(cat f)` after `… > f`) would otherwise be classified before
        the write is known. Only a source verb's own redirect qualifies, so this
        adds no reach the walk did not already have — it fixes the order.
        """
        for stage in stages:
            source = self._source_verb(stage)
            if source is None:
                continue
            for fd, op, target in self._fd_redirections(stage):
                if op in (">", ">>", ">|") and fd == "1":
                    self._register_path(self._word_text(target), target.span, source[1])

    def _bind_loop_targets(self, stage: list[_Word | _Op | _Body], *, depth: int) -> None:
        """A `for NAME… in WORDS` binds each name to the taint of those words.

        `for x in $(lop secret get NAME); do echo $x; done` printed the value
        with `x` untainted (R3-3): the substitution is the loop's LIST and the
        shell binds its targets before the body runs, so a rule that follows
        only `NAME=$(…)` never sees it. `select` is the same shape.
        """
        words = [item for item in stage if isinstance(item, _Word)]
        texts = [self._word_text(word).strip() for word in words]
        if not words or texts[0] not in ("for", "select") or "in" not in texts:
            return
        stop = texts.index("in")
        names = [text for text in texts[1:stop] if text and not text.startswith("-")]
        flow = _Flow()
        for word in words[stop + 1 :]:
            piece = self._value_flow(word, depth=depth)
            flow = _Flow(
                value=flow.value or piece.value,
                path=flow.path or piece.path,
                name=flow.name or piece.name,
                span=flow.span if flow.span != (0, 0) else piece.span,
            )
        if not (flow.value or flow.path):
            return
        for name in names:
            if flow.value:
                self.value_vars[name] = flow.name
            else:
                self.file_vars[name] = flow.name

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
        # A redirection is not the command or its argv (R5-1's class): with
        # its words left in, `2>/dev/null lop secret get X` had `2` as the
        # command and was no source at all.
        redirects = self._redirect_words(stage)
        words = [
            item
            for position, item in enumerate(stage)
            if isinstance(item, _Word) and position not in redirects
        ]
        index = self._command_index(words)
        if index is None:
            return None
        # A wrapper around the source is the source (R3-2), and that includes
        # `xargs` (R4-2): `echo X | xargs lop secret get` fetches X and prints
        # it, with the NAME arriving on stdin rather than in the text. The
        # outer unwrap has already dropped the ordinary wrappers from a walked
        # stage; this call is what the prescan and `xargs` need.
        texts = [self._resolve_program(self._word_text(word).strip()) for word in words[index:]]
        start, _ = _wrapped_command_index(
            texts,
            table=_CONSUMER_WRAPPERS,
            opaque={
                k
                for k, word in enumerate(words[index:])
                if any(piece.kind == "subst" for piece in word.pieces)
            },
        )
        via_xargs = "xargs" in (text.rsplit("/", 1)[-1] for text in texts[:start])
        first = texts[start].rsplit("/", 1)[-1]
        if first not in _PROGRAM_NAMES:
            return None
        args = texts[start + 1 :]
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
            # `file`'s own `--env-var VAR` may come BEFORE the name (R5-4):
            # argparse accepts `file --env-var KF NAME -- …`, and reading `KF` as
            # the name — then `--env-var` as "not a name" — left the stage
            # unseen as a source. `get` has no option that takes an operand.
            position = 2
            while verb == "file" and position < len(args):
                if args[position] == "--env-var":
                    position += 2
                elif args[position].startswith("--env-var="):
                    position += 1
                else:
                    break
            name = args[position] if len(args) > position else ""
            if name and not name.startswith("-"):
                # `lop secret get --help` is not a source: no value exists.
                return verb, name
            if not name and via_xargs:
                # The name is `xargs`'s stdin, which the text does not show.
                return verb, "?"
        return None

    def _nested_source_stage(
        self, stage: list[_Word | _Op | _Body]
    ) -> list[_Word | _Op | _Body] | None:
        """The consumer of `run … -- CMD` as a stage, when CMD is itself a source.

        `file NAME -- lop secret get NAME | rev` was already refused (its PATH
        flow reaches `rev`), but `run` returns no flow, so the same consumer
        under `run` leaked reversed (R5-5). The consumer is everything after the
        first `--` word, or — separator-less, which argparse also accepts —
        after the last `--secret NAME` pair (each takes exactly one operand, so
        the boundary is exact). Redirections are kept: they apply to the whole
        command line either way.
        """
        redirects = self._redirect_words(stage)
        positions = [
            index
            for index, item in enumerate(stage)
            if isinstance(item, _Word) and index not in redirects
        ]
        texts = [
            self._word_text(item).strip()
            for index in positions
            if isinstance(item := stage[index], _Word)
        ]
        if "run" not in texts:
            return None
        cursor = texts.index("run") + 1
        while cursor < len(texts):
            if texts[cursor] == "--":
                cursor += 1
                break
            if texts[cursor] == "--secret":
                cursor += 2
            elif texts[cursor].startswith("--secret="):
                cursor += 1
            else:
                break
        if cursor >= len(texts):
            return None
        inner = [
            item
            for index, item in enumerate(stage)
            if index >= positions[cursor] or (index in redirects or isinstance(item, _Op))
        ]
        unwrapped = self._unwrap(inner)
        if unwrapped and self._source_verb(unwrapped) is not None:
            return inner
        return None

    def _emitting_consumer(
        self, stage: list[_Word | _Op | _Body], name: str, *, verb: str, depth: int
    ) -> None:
        """``lop secret file NAME -- cat``: the verb's consumer prints the value.

        The two verbs put the value in DIFFERENT places, so they need different
        consumer tests, and one test for both is how R3-1 happened:

        * ``file`` materialises the value at a path handed over in an environment
          variable, so a consumer that prints its arguments or its input prints
          the file's bytes — ``cat``, ``tee``, ``sed``, an inline ``sh -c 'cat'``.
          Any ``_EMITTERS`` name is therefore a leak, which is what this rule
          always said for this verb (``lop secret file F -- cat`` has been refused
          since the first commit).
        * ``run`` puts the value in the child's ENVIRONMENT and nowhere else. So
          the leak is a consumer that READS THE ENVIRONMENT (``printenv``, a
          command-less ``env``, a bare ``declare``), or an inline program that
          prints or references it (``sh -c 'echo "$TOKEN"'``, ``python -c
          'print(os.environ["TOKEN"])'``). A consumer that prints only its own
          arguments does not reach it: the outer shell expanded ``$TOKEN`` before
          ``lop`` ever ran, so the child's argv holds no value — measured, not
          assumed, in the round-3 remediation comment. Reading an emitter as a
          leak for ``run`` as well would refuse ``-- sed -n 1p /etc/hosts`` and
          ``-- cat``, which are the verb's ordinary consumers.

        Both arms are the same shape: find the consumer after the ``--``
        separator, skipping each ``--secret NAME[=VAR]`` pair that precedes it.
        Stopping at the first ``--secret`` word — which this did — meant the
        ``run`` branch could never fire at all (R3-1), because ``run`` always
        carries one.
        """
        # Redirection words are not argv: neither `lop`'s nor the consumer's
        # (R5-1 — `-- env 2>/dev/null` and `-- env < /dev/null` each put a
        # redirect word where the consumer is looked for).
        redirects = self._redirect_words(stage)
        words = [
            self._word_text(item).strip()
            for index, item in enumerate(stage)
            if isinstance(item, _Word) and index not in redirects
        ]
        rest: list[str] = []
        names: list[str] = []
        after_options: int | None = None
        index = 0
        while index < len(words):
            word = words[index]
            if word in ("--", "-"):
                rest = words[index + 1 :]
                break
            if word.startswith("--secret"):
                # `--secret=NAME` carries its operand in the same word; the
                # spaced form takes the NEXT word as the name. Either way the
                # separator is still ahead.
                if word.startswith("--secret="):
                    # `--secret=NAME` carries its operand in the same word, and the
                    # `NAME=VAR` form is the VARIABLE the value is exported as — which is
                    # what an inline program reads.
                    names.append(word.split("=", 1)[1])
                    index += 1
                    continue
                if index + 1 < len(words):
                    names.append(words[index + 1])
                index += 2
                after_options = index
                continue
            index += 1
        # `file`'s value is at a path named by `--env-var` (default
        # `GOOGLE_APPLICATION_CREDENTIALS`); an inline program that reads that
        # variable's file is how the bytes come back (R4-3).
        file_variable = _DEFAULT_FILE_ENV_VAR
        own = words[: index if rest else len(words)]
        for position, word in enumerate(own):
            if word == "--env-var" and position + 1 < len(own):
                file_variable = own[position + 1]
            elif word.startswith("--env-var="):
                file_variable = word.split("=", 1)[1]
        if not rest:
            # No `--` in the TEXT at all: argparse consumed it (or none was
            # written), so the child command is what remains after our OWN
            # words — the `--secret` pairs for `run`, the secret's NAME and any
            # `--env-var VAR` for `file`. The separator-less spelling is real and
            # it leaked while this arm required a literal `--`.
            start = after_options if after_options is not None else words.index(verb) + 1
            if after_options is None and verb == "file":
                while start < len(words) and words[start].startswith("-"):
                    start += 2 if start + 1 < len(words) else 1
                start += 1  # the secret's NAME
            rest = [word for word in words[start:] if not word.startswith("-")]
            if not rest:
                return
        # The consumer is found through the SAME wrapper grammar the outer
        # stage uses (R4-1): `run … -- timeout 5 printenv TOK` runs `printenv
        # TOK` with the value in its environment, and reading `timeout` as the
        # consumer let it through while the unwrapped spelling was refused.
        # `xargs` is stepped over here too (R4-2): it hands its command the
        # environment unchanged.
        resolved = [self._resolve_program(word) for word in rest]
        start, _ = _wrapped_command_index(resolved, table=_CONSUMER_WRAPPERS)
        wrappers = resolved[:start]
        consumer = resolved[start].rsplit("/", 1)[-1]
        arguments = resolved[start + 1 :]
        if verb == "run":
            variables = [name.split("=", 1)[-1] for name in names]
            if _env_clears_environment(wrappers):
                # `env -i` starts the child with an EMPTY environment. A
                # `NAME=…` beside it is text the OUTER shell expanded, which
                # never holds what `run` exported, so nothing of the value
                # survives (measured by the round-4 review: `run -- env -i
                # printenv` prints no value).
                return
            leaked = self._run_consumer_dumps(consumer, arguments) or self._inline_program_reads(
                consumer, arguments, variables, name, verb=verb, depth=depth
            )
        else:
            # `file` hands over a PATH in an environment variable, so an
            # argument-printing consumer reaches the bytes, and an inline
            # program reaches them by reading that path (R4-3: `sh -c 'rev
            # "$GOOGLE_APPLICATION_CREDENTIALS"'` printed them reversed because
            # only a fixed list of printer words was searched for).
            leaked = (
                consumer in _EMITTERS
                or (
                    consumer in _INTERPRETERS and bool(_INLINE_PRINT_RE.search(" ".join(arguments)))
                )
                or self._inline_program_reads(
                    consumer, arguments, [file_variable], name, verb=verb, depth=depth
                )
            )
        if leaked:
            span = next((item.span for item in stage if isinstance(item, _Word)), (0, 0))
            self._add("shell.secret-verb-emitting-consumer", span, name)

    def _inline_program_reads(
        self,
        consumer: str,
        arguments: list[str],
        variables: list[str],
        secret: str,
        *,
        verb: str,
        depth: int,
    ) -> bool:
        """Does this inline program reach a variable the verb handed the value in?

        ``run`` exports the VALUE in those variables; ``file`` exports a PATH to
        it. Either way the question is whether the program's printer reaches it.

        The precision is what keeps the verb's own sanctioned form working:
        ``run --secret N=TOKEN -- sh -c 'curl -H "Authorization: Bearer $TOKEN"
        http://…; echo done'`` is an approved CONSUMER, so a bare "an inline
        program that prints" test refused it the moment the program also wrote a
        `done` marker. What leaks is a program whose PRINTER reaches the value.

        For a shell program that question is already answered by this module's own
        walk, so it is asked with it rather than with a second, coarser rule: the
        program is analysed with the variables pre-bound — as values for ``run``,
        as secret-file paths for ``file`` — and any finding means the value came
        back. ``sh -c 'echo "$TOKEN" | base64'`` finds `shell.pipe-of-source`;
        ``file … -- sh -c 'rev "$GOOGLE_APPLICATION_CREDENTIALS"'`` finds the
        read rule (R4-3), where the fixed printer-word list this arm used to rely
        on did not know `rev`; ``sh -c 'curl -H … $TOKEN; echo done'`` finds
        nothing, which is the point.

        Every other language is not this walk's, so it keeps the coarse test
        :class:`_InlineLanguage` describes (R4-2 widened it from Python alone to
        the table in :data:`_INLINE_LANGUAGES`, whose comment states the bound).
        """
        if not variables:
            return False
        language = _INLINE_LANGUAGES.get(_VERSIONED_PYTHON_RE.sub(r"\1", consumer))
        if language is not None:
            text = " ".join(arguments)
            names = "|".join(re.escape(variable) for variable in variables)
            # A boundary written out rather than a `\b` escape: the name is
            # spliced into the pattern, and `$TOKEN`/`TOKEN=` must both count.
            if not re.search(rf"(?:^|[^A-Za-z0-9_]){names}(?:$|[^A-Za-z0-9_])", text):
                return False
            if not language.env_read.search(text):
                return False
            if verb == "file" and not language.file_read.search(text):
                # The variable holds a PATH: a program that never opens it
                # (`getsize(os.environ["G"])`) prints a fact about the file.
                return False
            # Every half is required: naming the variable and reading the
            # environment is not a leak on its own (an agent may check it is set,
            # or hand it to a request), and a printer that never reads the
            # environment cannot print the value. `print(len(...))` is still
            # refused with the rest — this arm is deliberately the coarse one.
            return bool(
                language.printer.search(text)
                or any(argument in language.print_flags for argument in arguments)
                or any(
                    re.fullmatch(r"-[A-Za-z]+", argument)
                    and set(argument[1:]) & language.print_letters
                    for argument in arguments
                )
            )
        if consumer not in _INTERPRETERS:
            return False
        program = ""
        for index, argument in enumerate(arguments):
            # `-c` alone or clustered (`bash -lc '…'`, `sh -ec '…'`): the NEXT
            # word is the program either way.
            if (argument == "--eval" or _SHELL_INLINE_FLAG_RE.match(argument)) and index + 1 < len(
                arguments
            ):
                program = arguments[index + 1]
                break
        if not program:
            return False
        inner = _ShellAnalyzer()
        for variable in variables:
            if verb == "file":
                inner.file_vars[variable] = secret
            else:
                inner.value_vars[variable] = secret
                inner.exported_vars.add(variable)
        try:
            inner._analyze(program, contained=False, depth=depth + 1)
            inner._after_walk()
        except _LexFault:
            # A program that does not lex is not evidence of a leak, and the
            # fail-closed asymmetry belongs to a SOURCE in the text — a consumer's
            # program cannot carry one (its quotes make the text literal to the
            # outer lexer).
            return False
        return bool(inner.findings)

    @staticmethod
    def _run_consumer_dumps(consumer: str, arguments: list[str]) -> bool:
        """Does this ``run`` consumer read the environment the value is in?

        The reach mirrors the dump rule's: ``printenv`` (bare or naming a
        variable), ``env``/``set`` with no command after them, and a bare
        ``declare``/``typeset``/``export``/``readonly``/``local`` list the
        namespace. ``env V=1 client`` RUNS something and is a consumer, and
        ``declare -x TOKEN`` only sets an attribute.
        """
        if consumer not in _ENV_DUMPERS:
            return False
        plain = [word for word in arguments if word and not word.startswith("-")]
        if consumer == "printenv":
            return True
        if consumer == "env":
            # `env V=1 client` runs something (a consumer); a command-less `env`
            # prints the namespace the value was exported into.
            return not [word for word in plain if "=" not in word]
        if consumer == "set":
            # Bare `set` lists everything; `set -e`/`set -x` print nothing.
            return not arguments
        return not plain

    # -- the walk -----------------------------------------------------------

    def _analyze(self, text: str, *, contained: bool, depth: int) -> _Flow:
        """Walk one command list and return the flow ITS stdout carries.

        ``contained`` means the stdout is captured by an enclosing ``$( )`` (or
        written to a file): a printer inside is then not a leak, but its output
        is still the value, which is why the returned flow stays tainted — that
        is what makes ``v=$(echo $(lop secret get X)); echo "$v"`` refuse.
        """
        stages, piped = self._group(_tokenize_shell(text))
        # Order fix and list-wide context, both BEFORE the left-to-right walk:
        # what a source writes, and whether this list's stdin is one of those
        # files. Neither adds reach; both keep a later read from being judged
        # against a world where the write has not happened yet.
        self._prescan_written_paths(stages)
        self._note_stdin_redirects(stages)
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
        # Wrappers first: `timeout 30 lop secret get X | rev` IS `lop secret get
        # X | rev`, so every rule below sees the real command instead of each
        # rule carrying its own list of wrappers (R3-2).
        stage = self._unwrap(stage)
        # Reserved words are structure, not operands: dropping them here is what
        # lets `for …; do v=$(…); done` and `if …; then lop secret get X; fi`
        # read as the stages a person sees rather than as keyword soup.
        # Redirection words are not operands either (R5-1's class): `export
        # 2>/dev/null` bound a `2` instead of listing, and `cp f g 2>/dev/null`
        # took `/dev/null` as the copy's destination. The whole-command scans
        # (`/proc/…/environ` behind a `<`, a `>&2` in an expansion) still read
        # every word, so they get ``all_words``.
        redirects = self._redirect_words(stage)
        all_words = [
            item
            for item in stage
            if isinstance(item, _Word) and self._word_text(item).strip() not in _SHELL_KEYWORDS
        ]
        words = [
            item
            for index, item in enumerate(stage)
            if isinstance(item, _Word)
            and index not in redirects
            and self._word_text(item).strip() not in _SHELL_KEYWORDS
        ]
        bodies = [item.piece for item in stage if isinstance(item, _Body)]
        command = self._command_word(stage)
        # `for x in $(lop secret get X)` binds the value to `x` (R3-3), and the
        # body that reads it is a LATER stage.
        self._bind_loop_targets(stage, depth=depth)

        # Where this stage's stdout goes. This is the whole basis for deciding
        # whether an emitted value is PRINTED: `>/dev/null` drops it, a real
        # path contains it, a pipe hands it on, and anything else IS this tool
        # result.
        stdout_kind, stdout_target = self._stdout_destination(stage)
        stdout_path = stdout_target if stdout_kind == "path" else None
        discarded = stdout_kind == "null"
        to_stderr = stdout_kind == "stderr"
        for word in all_words:
            for piece in word.pieces:
                if piece.kind == "expand" and re.search(r"1?>&\s*2\b", piece.text):
                    to_stderr = True
        reaches_result = (
            not contained and not writes_stdout and not discarded and stdout_path is None
        )

        # -- whole-command conditions, recorded and judged after the walk ---
        # Before the assignment branch, because `PS4='+ '` IS an assignment and
        # would otherwise return early without ever being examined.
        self._note_conditions(command, stage, all_words)

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
        plain_assignment = bool(words) and all(self._is_assignment(word) for word in words)
        rest = [self._word_text(word).strip() for word in words[1:]]
        flags = [text for text in rest if text[:1] in ("-", "+") and len(text) > 1]
        if command in _BINDING_PREFIXES and not self._is_listing(command, flags, words[1:]):
            # A binding builtin that lists nothing: it binds, re-attributes, or
            # both. `export V` with no `=` re-exports a variable an earlier
            # statement bound (R1-2's second spelling), and `declare -x V`
            # does the same without printing (R1-2's adjacent defect (a)).
            exports, unexports, readonly = self._binding_attributes(command, flags)
            for word in words[1:]:
                text = self._word_text(word).strip()
                if not text or text[:1] in ("-", "+"):
                    continue
                name = self._assignment_name(word) if self._is_assignment(word) else text
                if self._is_assignment(word):
                    flow = self._value_flow(word, depth=depth)
                    if flow.value:
                        self.value_vars[name] = flow.name
                    elif flow.path:
                        self.file_vars[name] = flow.name
                if name not in self.value_vars and name not in self.file_vars:
                    # `export V` BEFORE the assignment: the export is declared
                    # first and applies to the binding that follows (R3-3), so
                    # the name is remembered rather than dropped.
                    if exports:
                        self._pending_exports.add(name)
                    continue
                if exports:
                    self.exported_vars.add(name)
                elif unexports:
                    self.exported_vars.discard(name)
                if readonly:
                    self.readonly_vars.add(name)
                if command == "local":
                    self.local_vars.add(name)
            return _Flow()
        if plain_assignment:
            for word in words:
                flow = self._value_flow(word, depth=depth)
                name = self._assignment_name(word)
                if flow.value:
                    self.value_vars[name] = flow.name
                elif flow.path:
                    self.file_vars[name] = flow.name
                if (flow.value or flow.path) and (self._allexport or name in self._pending_exports):
                    self.exported_vars.add(name)
                # `l=lop; $l secret get X`: a name bound to the PROGRAM is how
                # the call can spell a source without the word `lop` in command
                # position (R3-2). Only a value that IS a program name records.
                bound = word.pieces[0].text.split("=", 1)[-1].strip()
                bound = " ".join(
                    [bound] + [piece.text for piece in word.pieces[1:] if piece.kind == "literal"]
                ).strip()
                program = self._resolve_program(bound).rsplit("/", 1)[-1]
                if not flow.value and not flow.path and program in _PROGRAM_NAMES:
                    self.program_vars[name] = program
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
        # `(0, 0)` is the "no span" sentinel and is truthy, so `span or …`
        # never fell back and the caret sat under column 0 (R2-3).
        span = flow_in.span if flow_in.span != (0, 0) else (stage[0].span if stage else (0, 0))
        name = flow_in.name
        path_hit, path_span = self._literal_path_hit(stage)
        # Which path the hit is, so a read of one can name the secret it holds.
        _path_text = next(
            (
                self._word_text(item).strip()
                for item in stage
                if isinstance(item, _Word) and self._word_text(item).strip() in self.tainted_paths
            ),
            "",
        )

        # -- `$(<PATH)` : a redirect with no command at all -------------------
        # The bytes come back as the substitution's text, so a tainted path here
        # IS the value (R3-3). Checked before the emitter branch because there is
        # no command word for that branch to key on.
        if path_hit and self._redirect_only_read(stage):
            return _Flow(
                value=True,
                name=self.path_secrets.get(_path_text, ""),
                span=path_span or (stage[0].span if stage else (0, 0)),
            )

        # -- `read` binds through a builtin, not an `=` -----------------------
        # `read -r l < <(lop secret get X)` puts the value in `l` with no
        # assignment word anywhere, so `echo "$l"` looked like an ordinary
        # consumer (R1-3).
        # -- `read`/`mapfile` bind through a builtin, not an `=` ---------------
        # `read -r l < <(lop secret get X)` puts the value in `l` with no
        # assignment word anywhere, so `echo "$l"` looked like an ordinary
        # consumer (R1-3). `mapfile`/`readarray` read the same bytes into an
        # array, and a `< PATH` on the READ's own stage or on the compound it
        # sits in (`while read …; done < f`) is a file a source wrote (R3-3).
        if command in _READ_BUILTINS and (
            flow_in.value or flow_in.path or path_hit or self._stdin_taint
        ):
            value_tainted = flow_in.value or path_hit or bool(self._stdin_taint)
            name = flow_in.name or self.path_secrets.get(_path_text, "") or self._stdin_taint
            for word in words[1:]:
                text = self._word_text(word).strip()
                if not text or text.startswith("-"):
                    continue
                if value_tainted:
                    self.value_vars[text] = name
                elif flow_in.path:
                    self.file_vars[text] = name
            return _Flow()

        # -- a variable holding the value, in command position ---------------
        # `v=$(lop secret get X); $v` does not RUN anything: bash fails the
        # lookup and prints `v's value: command not found` — the value itself,
        # in this result. That is the same leak the `$( )` spelling below is
        # refused for, spelled through a variable.
        command_is_ref = bool(
            _BARE_REF_RE.match(self._word_text(words[0]).strip() if words else "")
        )
        if command_is_ref and (argv_flow.value or argv_flow.path) and not contained:
            self._add(
                "shell.bare-source-in-command-position",
                argv_flow.span or words[0].span,
                argv_flow.name,
                reason=(
                    "the variable holds the value and is run as a command name, "
                    "and the failure prints it back"
                ),
            )
            return argv_flow

        # -- a dump of the shell's own variables or environment ---------------
        # A value bound to a name is not out of reach just because no printer
        # names it: `printenv V`, `env`, `set | grep V=` and `export` (bare) all
        # print it back. Fired only when this command really did put a value in
        # that namespace, and only for the bare/no-operand spellings — `env
        # V=1 client` is a CONSUMER handing an environment to a child.
        if command == "set":
            # `set -a` is the one `set` spelling that changes what later
            # assignments do: each is exported, so `printenv V` then reaches it.
            for flag in flags:
                if flag.startswith("-") and "a" in flag and not flag.startswith("--"):
                    self._allexport = True
                elif flag.startswith("+") and "a" in flag:
                    self._allexport = False
            if "-o" in rest and "allexport" in rest:
                self._allexport = True
        if command in _ENV_DUMPERS:
            # A `V=…` prefix is not the dumper's argument: slice from the
            # command word so `V=$(…) printenv V` reads `V` as the operand.
            prefix = self._prefix_exports(stage, depth=depth)
            dumper = [word for word in words if not self._is_assignment(word)] or words
            dumper = words[words.index(dumper[0]) :] if dumper else []
            dumper_flags = [
                text
                for text in (self._word_text(word).strip() for word in dumper[1:])
                if text[:1] in ("-", "+") and len(text) > 1
            ]
            dumped = self._dumped_name(command, dumper_flags, dumper, prefix, depth=depth)
            if dumped:
                # The caret goes under the DUMPER: nothing in its argv carries
                # the value, so the flow's span is empty by construction (R2-3).
                self._add(
                    "shell.environment-dump-of-source",
                    self._command_span(stage) or span,
                    dumped,
                    reason=f"`{command}` prints the value back out of the shell's own variables",
                )
                return _Flow()

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
                self._emitting_consumer(stage, secret, verb=verb, depth=depth)
                if to_stderr:
                    self._add("shell.source-to-stderr", src_span, secret)
                for fd, op, target in self._fd_redirections(stage):
                    if op in (">", ">>", ">|") and fd == "1":
                        self._register_path(self._word_text(target), target.span, secret)
                # The verb hands out a PATH; the value stays in the file until
                # something reads it, which the read rules above then refuse.
                return _Flow(path=True, name=secret, span=src_span)
            if verb == "run":
                nested = self._nested_source_stage(stage)
                if nested is not None:
                    # `run … -- lop secret get N` hands the child's stdout — a
                    # second fetch — straight to this stage's stdout (R5-5).
                    # Walking the consumer AS this stage is what lets every
                    # source rule judge it: `| rev` refuses, `| wc -c` and
                    # `> /dev/null` stay allowed, exactly as the bare fetch.
                    return self._stage(
                        nested,
                        stdin,
                        writes_stdout=writes_stdout,
                        contained=contained,
                        depth=depth,
                    )
                self._emitting_consumer(stage, secret, verb=verb, depth=depth)
                return _Flow()
            if stdout_path is not None:
                # Checked BEFORE the stderr flag: a `2>` beside a redirect changes
                # where STDERR goes, not where this value went, so
                # `… > /tmp/o 2>/dev/null` is contained and was refused as
                # stderr output until this order was fixed.
                self._register_path(stdout_path, src_span)
            elif discarded:
                pass
            elif to_stderr:
                self._add("shell.source-to-stderr", src_span, secret)
            elif reaches_result:
                self._add("shell.bare-source-in-command-position", src_span, secret)
            if discarded:
                return _Flow()
            return _Flow(value=True, name=secret, span=src_span)

        # -- the stage prints -------------------------------------------------
        if command in _EMITTERS:
            emits = flow_in.value or flow_in.path or path_hit
            if emits:
                # Same order as the source branch: where stdout goes decides,
                # and the stderr flag is read only when stdout is this result.
                # `tee` writes the value to every operand, so an operand that
                # is a descriptor re-opens the leak the stage's own redirect
                # may have closed; the files it names are registered here too.
                tee_stderr = command == "tee" and self._tee_operands_to_stderr(
                    words, (stdout_kind, stdout_target)
                )
                # The stage's own stdout decides first, then the stderr signal:
                # a `2>` beside a redirect changes where STDERR goes, not where
                # this value went. A `tee` operand on a fd-2 device is the
                # exception — there the value really is written to stderr as
                # well, so it is checked before `discarded`.
                if stdout_path is not None:
                    self._register_path(stdout_path, span)
                if tee_stderr or (stdout_path is None and to_stderr):
                    self._add("shell.source-to-stderr", span, name)
                elif stdout_path is None and discarded:
                    pass
                elif reaches_result:
                    if path_hit and not flow_in.value:
                        self._add("shell.read-of-secret-file-path", path_span or span, name)
                    elif stdin.value and not argv_flow.value and not body_flow.value:
                        self._add("shell.pipe-of-source", span, name)
                    elif flow_in.path and not flow_in.value:
                        self._add("shell.read-of-secret-file-path", span, name)
                    else:
                        self._add("shell.print-of-source", span, name)
            if discarded:
                return _Flow()
            if stdout_path is not None:
                return _Flow()
            # The value is on stdout: captured, piped on, or printed above.
            # A READ of a file a secret was written into is the value when its
            # output is captured or piped rather than printed (R3-3): then the
            # rule above did not fire, and dropping the flow here is what let
            # `l=$(cat f); echo "$l"` through.
            captured_read = path_hit and (contained or writes_stdout)
            return _Flow(
                value=flow_in.value or captured_read,
                path=flow_in.path,
                name=name or (self.path_secrets.get(_path_text, "") if captured_read else ""),
                span=span,
            )

        # -- the stage runs an inline program ---------------------------------
        if command in _INTERPRETERS or command in _INLINE_PYTHON or command == "xargs":
            self._interpreter(stage, command, flow_in, stdin, span, name, depth)
            return _Flow(value=flow_in.value, path=flow_in.path, name=name, span=span)

        # -- a copy or a rename carries the debt to the new path --------------
        # `cp f g` does not print anything, but the copy is now readable under a
        # name nothing else in this command would recognise — following it is the
        # difference between a read rule and a read rule with an obvious way
        # around it. Only the DESTINATION is registered: the origin already is.
        if command in _PATH_MOVERS and path_hit:
            targets = [
                word for word in words[1:] if not self._word_text(word).strip().startswith("-")
            ]
            if targets:
                self._register_path(self._word_text(targets[-1]), targets[-1].span)

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
        # `set -x` and `set -o xtrace` are the spellings that turn tracing on.
        # `declare -x`/`export -x` only set the export ATTRIBUTE, so counting them
        # here refused `V=$(lop secret get X); declare -x V` — a form that prints
        # nothing — for a mode it never entered.
        if command == "set":
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
        """Taint an unquoted here-doc body carries (a quoted body is literal).

        A body is NOT shell text and must not be lexed as one (R1-4). It is data
        the shell expands, so `cat <<EOF` writing a script that says `It's fine`
        is two words and an apostrophe — lexing it as a command list raised
        "unterminated single quote", a FALSE refusal of a call that lexes, with a
        diagnosis describing nothing the model wrote. What an unquoted body
        really does is exactly two things: run its `$( )`/backtick substitutions,
        and resolve its `$VAR` references. This reads those two and nothing else.
        """
        flow = _Flow()
        for piece in bodies:
            if piece.kind == "literal":
                continue
            # `$VAR` references in the body, read the same way an argument's are.
            references = self._value_flow(
                _Word((_Piece(piece.text, "expand", piece.span),), piece.span), depth=depth
            )
            flow = _Flow(
                value=flow.value or references.value,
                path=flow.path or references.path,
                name=flow.name or references.name,
                span=flow.span if flow.span != (0, 0) else references.span,
            )
            for inner, span in _iter_expansions(piece.text):
                if depth >= _MAX_DEPTH:
                    self._note_depth(inner)
                    break
                inner_flow = self._analyze(inner, contained=True, depth=depth + 1)
                if inner_flow.value or inner_flow.path:
                    flow = _Flow(
                        value=flow.value or inner_flow.value,
                        path=flow.path or inner_flow.path,
                        name=flow.name or inner_flow.name,
                        span=flow.span if flow.span != (0, 0) else span,
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

    def _note_depth(self, text: str) -> None:
        """Record a substitution nested deeper than this walk follows (R1-8)."""
        self._skipped_deep = self._skipped_deep or text

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
        if self._skipped_deep and (self.sources or _RAW_SOURCE_RE.search(self._skipped_deep)):
            self._add(
                "shell.unresolved-source-region",
                (0, 0),
                self.sources[0] if self.sources else "?",
                reason=(
                    f"a substitution nested deeper than {_MAX_DEPTH} levels was not followed, "
                    "so whether it prints a value is unknown"
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

_PY_SOURCE_RE = re.compile(r"\bsecrets\b")
#: `warnings.warn`, `pprint.pprint` and the stdlib's other printing helpers: the
#: same stdout/stderr channels as `print`, so the same rule (R3-4). `display` is
#: still deliberately absent, for the opposite reason (it reaches the OPERATOR's
#: pane, not this result).
_PY_WARN_METHODS = frozenset({"warn", "warn_explicit", "showwarning"})
_PY_PPRINT_CALLS = frozenset({"pprint", "pp", "pformat", "saferepr"})
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

#: Carriers: calls whose VALUE is built from their arguments, so an assignment
#: from one taints its target (R2-2). Free functions that re-spell an argument
#: as a string, bytes or container of it.
_PY_CARRIER_FUNCS = frozenset(
    {"str", "repr", "ascii", "format", "bytes", "bytearray", "list", "tuple", "sorted", "reversed"}
)
#: Methods that splice an argument into the returned value, whatever the
#: receiver is: `sep.join([token])`, `tmpl.format(token)`, `"".replace("", token)`.
_PY_CARRIER_METHODS = frozenset({"join", "format", "format_map", "replace", "__add__", "__mod__"})
#: Modules whose functions are encoders: the value comes back re-spelled, which
#: is exactly the incident's second half (`rev`, `base64`). A module outside
#: this list that re-spells a value into a NEW name is not followed — the named
#: boundary a static scan has, recorded in the PR's "What this does not catch".
_PY_CARRIER_MODULES = frozenset(
    {"str", "bytes", "base64", "binascii", "codecs", "json", "urllib.parse", "shlex", "html"}
)


class _PyAnalyzer:
    """Every call in a parsed cell, classified against the rule table."""

    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.sources: list[str] = []
        self.tainted: set[str] = set()
        self.secret_paths: dict[str, tuple[int, int]] = {}
        #: `p = Path("/tmp/t")` — a name standing for a literal path.
        self.path_handles: dict[str, str] = {}
        #: Names bound to the `secrets` mapping itself (`s = secrets`), so
        #: `s["NAME"]` is still a source (R3-4).
        self.store_aliases: set[str] = set()
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
        """``secrets["NAME"]`` / ``secrets.get("NAME")`` → ``NAME``.

        A name the cell bound to the mapping counts as the mapping (R3-4):
        `s = secrets; print(s["NAME"][::-1])` is the same read, and treating the
        alias as an ordinary name made the whole cell a consumer.
        """
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if self._is_store_name(node.value.id) and isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, str):
                    return node.slice.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if isinstance(value, ast.Name) and self._is_store_name(value.id):
                if node.func.attr == "get" and node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        return node.args[0].value
        return None

    def _is_store_name(self, name: str) -> bool:
        """Is this name the store mapping, under its own name or an alias?"""
        return name == "secrets" or name in self.store_aliases

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
                found, receiver = self._value_taint(func.value)
                if found:
                    return True, receiver
            # A CARRIER builds its value out of its arguments, so the value
            # comes along whatever the receiver is (R2-2): `",".join([token])`,
            # `"{}".format(token)` and `base64.b64encode(token.encode())` are
            # the value re-spelled, and dropping them here let
            # `blob = ",".join([token]); print(blob)` through. The argument test
            # is the argument-walking one, so `"".join(reversed(token))` counts.
            if self._is_carrier(func):
                for argument in list(node.args) + [kw.value for kw in node.keywords]:
                    found, child = self._is_tainted(argument)
                    if found:
                        return True, child
            return False, ""
        for child in ast.iter_child_nodes(node):
            found, child_name = self._value_taint(child)
            if found:
                return True, child_name
        return False, ""

    def _is_carrier(self, func: ast.AST) -> bool:
        """Does a call through ``func`` return a value built from its arguments?

        A deliberately NAMED set rather than "every call": the whole point of
        :meth:`_value_taint` is that `requests.get(url, headers=…token…)` and
        `subprocess.run([…token…])` return a response, and QA Q1 measured what
        treating them as carriers costs (`print(resp.status)` refused). What IS
        in the set is the str/bytes surface that splices an argument into its
        result, any method called on a string literal or f-string, the
        re-spelling builtins, and the encoder modules — the `base64` re-spelling
        the incident is about. An unknown function that re-spells the value
        into a NEW name is outside it (see :data:`_PY_CARRIER_MODULES`); a
        direct `print(mylib.scramble(token))` is still refused, because the sink
        test walks the argument.
        """
        if isinstance(func, ast.Name):
            return func.id in _PY_CARRIER_FUNCS
        if not isinstance(func, ast.Attribute):
            return False
        if func.attr in _PY_CARRIER_METHODS:
            return True
        receiver = func.value
        if isinstance(receiver, ast.JoinedStr) or (
            isinstance(receiver, ast.Constant) and isinstance(receiver.value, (str, bytes))
        ):
            return True
        return self._dotted(receiver) in _PY_CARRIER_MODULES

    def _is_tainted(self, node: ast.AST) -> tuple[bool, str]:
        """Does this expression carry a value from the store ANYWHERE in it?

        The SINK-argument test (R2-2), as opposed to :meth:`_value_taint`, which
        is the assignment collector's. A printed argument is judged on
        everything it contains, because a printer shows whatever its argument
        evaluates to and the scan cannot know which calls re-spell their input:
        `print(",".join([token]))` and `print("{}".format(token))` print the
        value. The Q1 boundary survives because it never depended on this test:
        `resp` and `done` are not tainted names (the collector does not follow
        a response), so `print(resp.status_code)` has nothing in it to find.

        ``len``/``hash``/``id`` are exempt: the length of a secret is not the
        secret, which is what keeps `print(len(secrets["X"]))` allowed.
        Membership and comparison are NOT exempt (`print("x" in token)` stays
        refused, R2-5): a bool the model can ask for repeatedly is an oracle
        that reads the value one character at a time, where a length is one
        fixed number.
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

    def _collect_taint(self, tree: ast.Module) -> None:
        """Names that hold a value, to a fixed point.

        Iterated because a cell rebinds freely (`a = secrets[...]`,
        `b = a.strip()`, `c = b`), and cells are small enough that a few passes
        cost nothing.
        """
        for _ in range(5):
            before = len(self.tainted) + len(self.store_aliases)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    continue
                if node.value is None:
                    continue
                # `s = secrets` binds the MAPPING, not a value out of it: that is
                # an alias a later read resolves through, tracked apart from the
                # value taint (R3-4).
                if isinstance(node.value, ast.Name) and self._is_store_name(node.value.id):
                    for target in self._assign_targets(node):
                        if isinstance(target, ast.Name):
                            self.store_aliases.add(target.id)
                    continue
                found, _ = self._value_taint(node.value)
                if not found:
                    continue
                for target in self._assign_targets(node):
                    if isinstance(target, ast.Name):
                        self.tainted.add(target.id)
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.AsyncFor)):
                    # A loop's target is bound by the ITERATION, not by `=`: (R3-4)
                    # `for c in token: print(c)` printed the value one character
                    # per line, which the output scrub does not catch. Inside the
                    # fixed point, because the iterable is usually a name tainted
                    # on the pass before. `enumerate(token)` and a tuple target
                    # (`for i, c in …`) are the same binding read twice.
                    # `_is_tainted`, not `_value_taint`: `enumerate(token)` and
                    # `zip(token, x)` carry the value without BEING it, and a
                    # call's value was read as untainted, so the loop target went
                    # unbound. Conservative in this module's usual direction.
                    if self._is_tainted(node.iter)[0]:
                        self.tainted.update(self._bound_names(node.target))
            if len(self.tainted) + len(self.store_aliases) == before:
                break

    @classmethod
    def _bound_names(cls, target: ast.AST) -> set[str]:
        """Every NAME a binding target binds, unpacking tuple/list targets."""
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            names: set[str] = set()
            for element in target.elts:
                names |= cls._bound_names(element)
            return names
        return set()

    @staticmethod
    def _assign_targets(node: ast.AST) -> list[ast.expr]:
        """The bound name(s) of one binding node, whatever its spelling."""
        if isinstance(node, ast.Assign):
            return list(node.targets)
        target = getattr(node, "target", None)
        return [target] if target is not None else []

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
            if base_name == "warnings" and attr in _PY_WARN_METHODS:
                return "python.print-of-source"
            if dotted in ("pprint", "pprint.pprint") and attr in _PY_PPRINT_CALLS:
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
    def _exception_text(node: ast.AST) -> list[ast.expr]:
        """The expressions a raised exception would print into this result (R3-4).

        `raise ValueError(token)` and `assert False, token` both end the cell
        with the value in the traceback text the tool returns, so the sink is
        the exception itself. A `Message` on a bare `raise` re-raises and carries
        nothing of its own.
        """
        found: list[ast.expr] = []
        if isinstance(node, ast.Raise) and node.exc is not None:
            found.append(node.exc)
        elif isinstance(node, ast.Assert) and node.msg is not None:
            found.append(node.msg)
        return found

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
                for carried in self._exception_text(node):
                    found, name = self._is_tainted(carried)
                    if found:
                        self._add(
                            "python.exception-of-source",
                            node,
                            name,
                            reason=(
                                "an uncaught exception is rendered into this result, "
                                "message and all"
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
                    # ARGUMENT-walking (R2-2), not value-based: the Q1 fix made
                    # this `_value_taint`, and that let `print(",".join([token]))`
                    # and `print("{}".format(token))` through, because a method's
                    # value was read off its receiver alone. Q1's boundary lives
                    # in the COLLECTOR instead — `resp = requests.get(url,
                    # headers={…token…})` does not taint `resp` — so
                    # `print(resp.status)` and `print(done.returncode)` stay
                    # allowed with the stricter sink test.
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


def _span_context(text: str, span: tuple[int, int], *, line_based: bool = False) -> tuple[str, str]:
    """A bounded window on the offending span, with a caret line under it.

    ``line_based`` is for the eval surface: a Python finding's span is
    ``(lineno, col_offset)`` from the AST, not an offset into the cell, so
    reading it as an offset showed the model line 1 of its own cell with a
    caret in a meaningless column (QA Q2) — worse than showing nothing, because
    the line the refusal names is then wrong.
    """
    start, end = span
    if not isinstance(start, int) or not isinstance(end, int):
        return "", ""
    if line_based:
        lines = text.splitlines() or [""]
        if start < 1 or start > len(lines):
            return "", ""
        line = lines[start - 1]
        caret = " " * max(0, min(end, len(line))) + "^"
        return line.rstrip(), caret.rstrip()
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
                "the command)"
            )
        if finding.secret_name and finding.secret_name != "?":
            lines.append(f"  secret: {finding.secret_name}")
        snippet, caret = _span_context(text, finding.span, line_based=spec.lang == "python")
        if snippet:
            lines.append(f"  here:   {snippet}")
            lines.append(f"          {caret}")
        if finding.reason:
            lines.append(f"  why:    {finding.reason}")
        lines.append(f"  do:     {finding.rewrite}")
        lines.append(f"  rule {finding.rule} exists because: {spec.why}")
        # Guarded on the distinct LABELS, not the findings: `print(repr(token))`
        # is two findings of one rule (the `print` and the `repr`), and the old
        # `len(findings)` guard printed an empty "also refused by:" (NIT-1).
        if len(result.labels) > 1:
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
