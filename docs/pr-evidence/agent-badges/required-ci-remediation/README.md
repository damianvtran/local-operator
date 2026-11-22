# Probe formatting provenance

The two Python probes in this directory are formatting-normalized derivatives
of the originally executed archive bytes at commit
`c4da615e3f79afc963a282c1b0dc5235c7f0520e`. The original probes remain immutable
at that Git revision; the original execution logs, result JSON, source proof,
receipt and completion receipts in this directory are unchanged.

Normalization follows the actual Black 26.1.0 and isort 5.13.2 diffs from
[required lint job 137168631708](https://github.com/damianvtran/local-operator/actions/runs/45009531226/job/137168631708).
Both probes retain identical complete non-import ASTs and exact import
multisets. Import changes only group existing dependencies; the projection
probe now imports Rich before Local Operator as requested by isort. No
statement after imports changes, and no shipping/test/copy/guard file changes.
Only the two corresponding entries in SHA256SUMS were updated; the checksum
file retains its original scope and does not claim to cover this new note.

The normalized probes were independently executed by the implementer with the
existing bounded command-status runner: projection returned exit 0 with 3696
matching cases and unchanged result digest
`f7c24b47cee51e3f4e5055e3d260156e0c5a594337706bd3b28b34ba811762ab`;
the remove-all mutant returned expected exit 1 at the projection-input
assertion (1 failed, 2 warnings, 1.34s). Fresh run logs live in the implementer
session's `ci-repair/normalized-parity.log` and `normalized-mutant.log`;
these are not substituted for the original archived logs. Wrapper cleanup
remains unknown. No full test suite, native frames, dependency installation,
local Black/isort execution or typing qualification is claimed here.
