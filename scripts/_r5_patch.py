"""Round-5: implement the manager's exact cursor algorithm over source LINES."""

import re
from pathlib import Path

path = Path("local_operator/saver/provenance.py")
src = path.read_text()


def replace_fn(name, new):
    global src
    start = src.index("\ndef %s(" % name) + 1
    match = re.search(r"\n(def |class |_[A-Z_]+ = )", src[start:])
    end = start + (match.start() + 1 if match else len(src) - start)
    src = src[:start] + new.rstrip("\n") + "\n\n\n" + src[end:]


MAP = '''def _map_fragments(out: Sequence[str], record: _Rec) -> bool:
    """THE INVARIANT, implemented exactly: order, distinctness, real deletion.

    For a record whose source lines are ``src[0..n-1]`` and whose emitted
    content lines are ``out[0..m-1]``, in order::

        cursor = 0
        for line in out:
            found = -1
            for j in range(cursor, n):
                S = src[j]
                if line == S or (len(line) >= MIN_CUT
                                 and (S.startswith(line) or S.endswith(line))):
                    found = j
                    break
            if found < 0:
                reject
            cursor = found + 1
        if m >= n:
            reject

    THREE PROPERTIES, and each one is doing distinct work:
    - ``cursor`` only moves forward, so emitted lines are IN ORDER;
    - ``cursor = found + 1`` means a matched source line is never reconsidered,
      so each source line is spent at most once — DISTINCTNESS. Deleting that
      one line is what let round 4 accept `abcabc` as `bc` + `ab` (9.2x tokens);
    - ``m < n`` means the record really deleted something, so the epoch is a
      reduction rather than a re-rendering.

    ``MIN_CUT`` only decides whether a very short prefix or suffix counts as a
    cut. Three characters is the shortest fragment that is not a re-segmentation
    primitive: a one- or two-character piece costs at least as many tokens as the
    line it replaces once the tab-and-newline quoting is counted (QA's 1.70x and
    the manager's 5.08x cases are both string-poisoning below this length).
    """
    source = [line for line in record.text.split("\\n") if line.strip()]
    if not source:
        return not out
    cursor = 0
    for line in out:
        found = -1
        for index in range(cursor, len(source)):
            candidate = source[index]
            if line == candidate or (
                len(line) >= MIN_CUT
                and (candidate.startswith(line) or candidate.endswith(line))
            ):
                found = index
                break
        if found < 0:
            return False
        cursor = found + 1
    return len(out) < len(source)'''

PIECES = '''def _pieces(elements: Sequence[_Element]) -> list[str]:
    """The emitted CONTENT LINES of a record, in order.

    Lines, not fragments: the algorithm in :func:`_map_fragments` maps one
    emitted line to one source line, so a line that carried an inline marker
    contributes its remaining content as a single line. Splitting it further
    would let one source line answer several emitted lines, which is exactly the
    re-segmentation the invariant forbids.

    Framing lines (the epoch header and its continuation) are removed by
    ``parse_records`` before this point, and disclosure elements (markers, the
    reducer's own notes) are not content and are not returned.
    """
    pieces: list[str] = []
    for kind, text in elements:
        if kind != "run":
            continue
        for line in text.split("\\n"):
            line = line.strip()
            if line:
                pieces.append(line)
    return pieces'''

BOUNDS = '''def _check_bounds(elements: Sequence[_Element], record: _Rec) -> tuple[int, int]:
    """The disclosure budget only. Re-segmentation is the cursor's job now.

    ``chars // 8 + 8`` and the aggregate overhead-vs-saved rule are DELETED:
    round 3 proved neither can bound a token cost (span bytes-per-token is
    unbounded above), and round 4's line-count bound is replaced by the two
    exact halves of the invariant in :func:`_map_fragments` — one emitted line
    per source line, and strictly fewer lines than the source record. What
    remains here is the disclosure accounting: markers and notes are lines, they
    are capped, and their slots are checked in :func:`_check_slots`.
    """
    from local_operator.saver.validate import SaverRejected

    markers = [text for kind, text in elements if kind == "marker"]
    notes = [text for kind, text in elements if kind == "note"]
    overhead = sum(len(text) for kind, text in elements if kind != "run")
    pieces = _pieces(elements)
    # A record that carries no text is one elision (or one elision plus the
    # identifier pool). Thousands of disclosure elements is fragmentation.
    if not pieces and len(markers) + len(notes) > 2:
        raise SaverRejected("provenance")
    if len(markers) + len(notes) > min(len(record.text) // 64 + 2, _marker_ceiling(record)):
        raise SaverRejected("provenance")
    # `[radient:ok N lines]` / `omitted N lines` must agree with the source.
    for text in markers:
        claimed = re.search(r"(?:ok|omitted) (\\d{1,9}) lines?", text)
        if claimed is not None and int(claimed.group(1)) > record.lines + 1:
            raise SaverRejected("provenance")
    return len(pieces), overhead'''

replace_fn("_map_fragments", MAP)
replace_fn("_pieces", PIECES)
replace_fn("_check_bounds", BOUNDS)

# The epoch-level line budget is subsumed by the per-record `m < n` rule.
src = src.replace(
    """    span_lines = sum(record.lines for group in records_by_role.values() for record in group)
    _check_line_budget(records, span_lines)
""",
    "",
)
start = src.index("\ndef _check_line_budget(") + 1
match = re.search(r"\n(def |class |_[A-Z_]+ = )", src[start:])
end = start + (match.start() + 1 if match else len(src) - start)
src = src[:start] + src[end:]

src = src.replace(
    """#: Slack over a source record's own line count: a record may gain a head line
#: and a merged result line. Measured on the 20-snapshot real-session replay:
#: honest records sit at p99 +3, with one JSON-bearing outlier.
_PIECE_SLACK = 8""",
    """#: The shortest prefix or suffix that counts as a documented cut (see
#: :func:`_map_fragments`). Shorter pieces are re-segmentation primitives.
MIN_CUT = 3""",
)

src = src.replace("            if not _map_fragments(pieces, candidate):", "            if not _map_fragments(pieces, candidate):")

path.write_text(src)
print("MIN_CUT:", src.count("MIN_CUT = 3"), "no line budget:", "_check_line_budget" not in src)
