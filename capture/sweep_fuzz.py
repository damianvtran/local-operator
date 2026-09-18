"""Depth sweep (6 depths x 10 arms) + differential fuzz vs a longest-valid-prefix oracle.

Sweep: every arm's extraction must equal EXACTLY what was written.
Fuzz:  _body_end compared against an oracle that finds the body a different way --
       enumerate candidate cuts LONGEST-FIRST and accept the first the predicate
       below allows, with balance decided by REWRITING pairs away to a fixed
       point (no depth counters, so an off-by-one or a bad reset in the scan
       shows up as a disagreement instead of being shared).
"""
import random
import re
import sys

WT = "/Users/damian/lo-wt/open-links-r3"
sys.path.insert(0, WT)
from local_operator.tui.link_targets import _BODY_STOP, _body_end, extract_links  # noqa: E402

DEPTHS = [0, 1, 2, 3, 4, 8]


def nested(depth: int) -> str:
    s = "x"
    for _ in range(depth):
        s = f"({s})"
    return "https://a.test/a_" + s


def arms(depth):
    u = nested(depth)
    return {
        "bare": (u, [u]),
        "bare-in-prose": (f"See {u} for details.", [u]),
        "bare-in-parens": (f"({u})", [u]),
        "markdown": (f"[the docs]({u})", [u]),
        "markdown-in-prose": (f"Read [the docs]({u}) first.", [u]),
        "markdown-with-title": (f'[the docs]({u} "title")', [u]),
        "autolink": (f"<{u}>", [u]),
        "image": (f"![alt]({u})", [u]),
        "label-is-url": (f"[{u}]({u})", [u]),
        "label-holds-url": (f"[see {u}](https://b.test/y)", [u, "https://b.test/y"]),
    }


sweep_pass = sweep_total = 0
for depth in DEPTHS:
    for name, (text, want) in arms(depth).items():
        got = extract_links(text)
        sweep_total += 1
        if got == want:
            sweep_pass += 1
        else:
            print(f"SWEEP FAIL depth={depth} arm={name}\n  text={text!r}\n  got ={got}\n  want={want}")
print(f"sweep: {sweep_pass}/{sweep_total} (10 arms x depths {DEPTHS})")


# --- the oracle ---------------------------------------------------------------

def rewrite_balanced(pairs: str) -> bool:
    """``()`` (or ``[]``) removed to a fixed point leaves nothing and never
    goes negative. Decided by rewriting, not by a counter."""
    pairs = "".join(ch for ch in pairs if ch in "()")
    while "()" in pairs:
        pairs = pairs.replace("()", "")
    return pairs == ""


def never_negative(pairs: str) -> bool:
    for ch in ("()", "[]"):
        depth = 0
        for c in pairs:
            if c == ch[0]:
                depth += 1
            elif c == ch[1]:
                depth -= 1
                if depth < 0:
                    return False
    return True


def oracle_body(text: str, start: int) -> int:
    """Where the URL body ends at ``start``, by candidate enumeration.

    A candidate cut is accepted when, over ``text[start:cut]``:
      * no :data:`_BODY_STOP` character and no whitespace;
      * brackets never negative and BALANCED -- or, when a ``[`` is still open,
        the next two characters are a ``](`` seam, which is the one cut that is
        allowed to leave a bracket open (the seam rule the rounds settled);
      * the run does not END on the ``]`` of a ``](`` seam (that ``]`` belongs
        to the markdown label, not to the URL);
      * parens never negative and balanced -- a dangling ``(`` is prose, and
        since a prefix ending just before it is itself balanced, the longest
        accepted prefix is that cut-back without a second rule.
    """
    limit = len(text)
    for i in range(start, len(text)):
        if text[i] in _BODY_STOP or text[i].isspace():
            limit = i
            break
    for cut in range(limit, start, -1):
        body = text[start:cut]
        if any(c in _BODY_STOP or c.isspace() for c in body):
            continue
        if not never_negative(body):
            continue
        if not rewrite_balanced(body):
            continue
        # The seam's own `]` is not URL text: a cut that ends on it is the
        # same cut taken one character later.
        if body.endswith("]") and cut < len(text) and text[cut] == "(":
            continue
        brackets_open = body.count("[") - body.count("]")
        if brackets_open:
            seam = cut < len(text) - 1 and text[cut] == "]" and text[cut + 1] == "("
            if not seam:
                continue
        return cut
    return start


# --- the generator ------------------------------------------------------------

WORDS = ["See", "docs", "at", "here", "the", "ref", "and", "so", "on", "(*note*)"]
TAILS = ["", "?q=1", "#frag", "?a=b&c=d", "/wiki/Foo_(bar)", ":8080/x"]
SPACES = [" ", " ", "\u00a0", "\u2002"]
SEAMS = [None, "[{u}]({u})", "[see {u}](https://b.test/y)", "![{u}](https://b.test/i.png)",
         "<{u}>", "[Source: {u}]", "[[{u}]]", "({u})", "[docs]({u})", "|{u}|", "`{u}`"]


def gen(rng: random.Random) -> str:
    u = "https://a.test/" + "".join(rng.choice("abc/_-") for _ in range(rng.randint(1, 6))) + rng.choice(TAILS)
    # An IPv6 authority is the ONE shape where a `]` is URL text rather than
    # prose, so a corpus without one is blind to the character-set decision the
    # `]`-in-_BODY_STOP mutant makes (review round 4 says the same about the
    # arms). Drawn from {[2001:db8::1], [::1]} so both the whole-authority and
    # the pseudo-closer shapes occur.
    if rng.random() < 0.3:
        u = u.replace("https://a.test", "https://" + rng.choice(["[2001:db8::1]", "[::1]", "[2001:db8::1]"]))
    depth = rng.choice(DEPTHS)
    if depth:
        u = u + "_" + "(" * depth + "y" + ")" * depth
    if rng.random() < 0.15:
        u = u + rng.choice(["", "]", "]]", ")", ")", "',", ".,"])
    template = rng.choice(SEAMS)
    body = template.format(u=u) if template else u
    parts = [rng.choice(WORDS) for _ in range(rng.randint(0, 3))]
    parts.append(body)
    parts += [rng.choice(WORDS) for _ in range(rng.randint(0, 3))]
    return rng.choice(SPACES).join(parts)


rng = random.Random(20260917)
N = 40000
texts = [gen(rng) for _ in range(N)]

START = re.compile(r"https?://")
checked = disagree = 0
examples = []
for text in texts:
    for m in START.finditer(text):
        start = m.start()
        checked += 1
        got = _body_end(text, start)
        want = oracle_body(text, start)
        if got != want:
            disagree += 1
            if len(examples) < 8:
                examples.append((text, text[start:got], text[start:want]))
print(f"texts: {N}   url occurrences checked: {checked}   disagreements: {disagree}")
for text, got, want in examples:
    print(f"  text={text!r}\n    impl={got!r}\n    oracle={want!r}")
