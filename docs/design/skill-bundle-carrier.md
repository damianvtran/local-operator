# Skill bundle carrier — content-addressed multi-file hub skills (design)

Status: ACCEPTED WITH AMENDMENTS (§12), not implemented, not committed. Author: architect (lopdev). Date: 2026-10-22.
Client tree: `local-operator/.worktrees/skill-bundles` (base `604803b336`). Server tree:
`agent-server/.worktrees/skill-bundles` (base `f493e349`). Every `file:line` below was read in those
trees; line numbers describe the base, not the implementation.

Evidence labels used throughout: **[read]** = read in the tree, cited; **[derived]** = measured by a small
spike run here (scratchpad, not committed); **[sourced]** = a third-party claim, source named;
**[unverified]** = I did not read it — the coder must check before relying on it (each has the
evidence that would settle it).

---------------------------------------------------------------------------------------------------

## 0. Summary of the decision

1. A skill is published in exactly one of two **forms**: a *document* (today's `files[]` JSON manifest,
   text only) or a *bundle* (a ZIP artifact plus a small manifest). The hub row holds one or the other,
   never both. Existing rows and single-file/text skills stay documents; nothing is migrated.
2. The bundle is a **deterministic-in-structure ZIP**, stored in a new bounded **blob collection**
   (not GridFS), addressed by `sha256(archive bytes)` and scoped to the publish target.
3. The manifest carries the operator's fields **plus one addition, `tree_hash`** (sha256 over the
   canonical sorted file list). `tree_hash`, not the archive hash, is the sync fingerprint. Reason (the
   single most important design call, §3.2): the archive hash depends on the deflate implementation and
   cannot be recomputed from a local directory, so L==B / L==R classification through it is impossible
   without downloading.
4. Install fetches → verifies hash → validates the archive strictly in canonical form (anything not
   byte-for-byte what `build_bundle` would emit is **refused**, which makes symlink/exec/device/dir
   entries fall out of one equality check) → unpacks into `skills/.stage-<op>/` → re-verifies → renames.
5. Sync reuses the existing `classify()` (`check.py:46-64`) unchanged, with `tree_hash` as fingerprint.
   Bundle apply is **fast-forward only** (L==B); every other divergence is a conflict, never a merge.
6. No executables, ever. Mode must be exactly `0100644`. Install never `chmod`s and never runs hooks.

---------------------------------------------------------------------------------------------------

## 1. What the code does today (grounding)

### 1.1 Server (Go)

- A skill row is `models.HubSkill` with `Files []HubSkillFile bson:"files"` — text only, "arbitrary binary
  assets are a declared non-goal of v1" (`internal/models/hub_skill.go:22-51`, `98-104`).
- The wire manifest is `requests.SkillDocument{document_version,name,description,files,version}` with a
  closed field set (`internal/requests/skills.go:142-159`). `ValidateWithinBounds` refuses a document with
  zero files (`skills.go` ~`len(d.Files) == 0`, "must contain at least SKILL.md"). The file comment
  explicitly argues *against* ZIP for the document path: "reintroduce the untrusted-archive parsing
  surface" (`skills.go:21-24`). This design reintroduces it on purpose, in a separate, strictly
  canonical-form-only carrier (§4); that comment's concern is the main reason §4 refuses rather than
  sanitizes.
- Bounds today: 64 files, 5 path segments, 240 path bytes, 1024-char description; public 20 KiB per
  file/sum, org 256 KiB per file/sum; body 2 MiB (`skills.go:67-100`). Path grammar `validateSkillPath`
  (`skills.go:356-402`): relative, no backslash, no colon (drive letters), no `.`/`..`/dot-segments, root
  set `SKILL.md | references/ | scripts/ | assets/`.
- Publish order: claim target (`claimSkillPublishTarget`, `hub_skill_service.go:682-700`, org admin check
  at L692) → decode → public: preview/pin + generalizer; org: `skillSubmittedSubmission` pass-through
  (`generalization_skill_documents.go:167-174`) → `JoinReviewedInstructions(skillReviewedJoin(doc))` →
  `runContentReview` (`agent_publication_service.go:1026-1072`) → `skills.Create/Replace`
  (`hub_skill_service.go:83-200`, `203-317`).
- Org previews already answer "unchanged, no pin, review not run" (`skillUngeneralizedPreview`,
  `hub_skill_service.go:828-849`) — a precedent for a bundle preview with no generalization.
- The review gate chunks reviewed text at 32 KiB per prompt (`moderation_service.go:148`, chunker
  `:599-640`).
- Revision: `revision.ETag(id, generation, n)` = sha256 of `revision-v1\0id\0generation\0n`
  (`internal/revision/revision.go:74-77`). `Replace` goes through `definitionUpdate`
  (`internal/repositories/content_revision.go:102-129`), which `$set`s every bson-tagged struct field and
  **`$$REMOVE`s every field omitted from the new value**, and bumps `content_revision` via
  `revisionUpdate` (`$ifNull +1`, `content_revision.go:96-100`). Consequences used below: (a) any
  republish advances the revision, so the existing ETag derivation needs no widening; (b) adding a bson
  `bundle` field with `omitempty` and making `files` `omitempty` makes a republish *remove* the other
  form's field automatically — mutual exclusivity is enforced by the existing write boundary.
- `GET /v1/skills/:skillid` is `optionalAuthMiddleware`; `GetSkill` (`hub_skill_service.go:380-401`)
  returns 404 for org rows to non-members (`skillRowAddressedAuthzError`), and the public projection
  withholds the publisher email (`responses/skills.go:81-87`).
- GridFS precedent: `internal/services/audioartifact/store.go` — a dedicated bucket, **72 h TTL**,
  hand-rolled seekable chunk reader because the driver gives no `Seek` (`store.go:1-12`, `42-60`,
  `100-140`). Two of its three reasons for existing (TTL, seek) are wrong for skills.
- Legacy zip parser `parseAgentZip` (`internal/services/agent_service.go:134-200`) does an unbounded
  `io.ReadAll` per entry and matches names case-insensitively. **Do not reuse or model on it.**

### 1.2 Client (Python)

- A skill = `SkillDocument` (`hub_sync/skill_document.py`): text-only, `MAX_FILES=64`,
  `MAX_FILE_BYTES=200 KiB`, `MAX_CONTENT_BYTES=1 MiB`, `MAX_PATH_BYTES=240`, `MAX_PATH_SEGMENTS=5`
  (`:26-33`); `validate_path` refuses `/`, `\`, `:`, dot segments, Cf characters, non-root paths
  (`:83-107`); `from_directory` refuses symlinks and non-regular entries and non-UTF-8 (`:410-467`);
  `from_json` requires the key set to be exactly `document_version,name,description,files`
  (`from_json`, `:~305-320`).
- `ManagedSkillStorage` (`skill_storage.py`) is text-only by construction: `_document` →
  `SkillDocument.from_directory` (`:95-114`), `_fingerprint_path` (`:180-183`), `_stage` writes
  `file.content.encode("utf-8")` with `0o600` files / `0o700` dirs under `hub/staging/<op>/content`
  (`:133-157`), `_finalize` re-parses `restored.fields` as a document (`:195-224`), `recover` skips
  transactions whose `payload.adapter != "managed-skill"` (`:400-410`). `apply` does the two-rename
  swap (old → `hub/backups`, staged → `skills/<alias>`) under `config_write_lock`, re-checking the
  destination fingerprint (`:226-398`). It must not be overloaded with bytes (operator requirement; also
  `from_directory` would raise on any binary file).
- The journal accepts any 64-hex fingerprint (`transactions.py:29-30`, `LocalTransaction` `:109-171`) —
  a `tree_hash` fits without schema change.
- Classification is fingerprint-only: `classify(base_fp, local_fp, remote_fp)` (`check.py:46-64`);
  `check_hub_skills` (`:568-805`) computes `local_fp = fingerprint_snapshot("skill", local)`,
  `remote_fp` likewise and `base_fp = fingerprint_fields("skill", record.fields)`. Both
  `fingerprint_*` dispatch to `SkillDocument.from_json(...).fingerprint` (`provenance.py:219-253`).
  → A bundle only needs a different fingerprint function; the engine is already shaped for this.
- Old-client behavior on an unknown manifest shape: `adapters.SkillAdapter.canonical_fields` →
  `SkillDocument.from_remote` → `from_json` raises `SkillValidationError`; `check_hub_skills` converts
  that to a per-item `unavailable` / `invalid-remote-manifest` without crashing the tick
  (`check.py:~698-715`, comment citing round-2 QN1); `_resolve_remote_skill` converts to
  `LinkError("invalid-remote-manifest")` (`service.py:~2490-2500`). Old clients **degrade, not crash**.
- `fork.py` roles-copy discipline: sum file bytes before copying, refuse with the offending path
  (`fork.py:200` `MAX_ROLES_COPY_BYTES = 64 MiB`, pre-count `:631-633`), count the **streamed** bytes not
  the stat size (`:664-674`), stage in a temp sibling and publish with one `os.replace`. §5 adopts the
  discipline, not the 64 MiB number.
- `lop` has **no `skills` CLI**. Skill hub operations exist only as desktop routes
  (`server/routes/desktop_hub.py`: pull `:~669-720`, subscribe `:~722-780`, unpull `:~800`, publish
  `:~815-900`) over `service.install_skill/subscribe_skill/unpull_skill/publish_skill`
  (`service.py:2528/2917/3016/3787`). The CLI hub surface is `hub status|links|link set`
  (`cli.py:703-760`) plus `agents|teams push|pull|sync`.

### 1.3 Spike facts (derived here, small, not committed)

Python 3.12 zlib 1.2.12 (`.venv`) and Python 3.14 zlib 1.2.12 (system), on macOS:

| Probe | Result |
|---|---|
| Same inputs, same settings, built twice | identical bytes |
| `ZipInfo.create_system` 3 vs 0 (Unix vs FAT), everything else equal | **different hashes** → pin `create_system=3` explicitly (Windows' default differs) |
| 200 KiB repetitive text at deflate level 1 / 6 / 9 | three different archives/hashes (43476 / 29185 / 27050 bytes) |
| `zipfile` read of names `../x`, `/abs`, `C:\x`, `a/../b`, `a\b` | returned **verbatim** — the reader sanitizes nothing; refusal must be ours |
| Entry with `external_attr = 0o120777<<16` | `stat.S_ISLNK(external_attr>>16)` is True — symlinks are detectable from the central directory |

[sourced] Python 3.14's official Windows build moved to zlib-ng (conda-forge/python-feedstock#821;
man.com "Faster Python 3.14" summary — search snippets, not read in full). Not reproducible on this
host (`zlib.ZLIBNG_VERSION` is `None` here). If true, two honest builders can emit different archive
bytes for the same tree. That is the reason for §3.2.

Not relied on: the manager verified that Go's `archive/zip` reads a Python-built deterministic zip
(method 8, mode `-rw-r--r--`); I stopped my own cross-parser spike at that point.

---------------------------------------------------------------------------------------------------

## 2. Options considered

### 2.1 Carrier / storage

| Option | For | Against |
|---|---|---|
| **A. GridFS bucket** (audioartifact pattern) | Existing driver; no 16 MB cap | Needs hand-written seekable reader (`store.go:42-60`), a *non*-TTL index set, orphan sweep across `files`+`chunks`, and account-deletion wiring for a bucket (`account_deletion_repository.go:613-660` has bespoke GridFS deletion). All for ≤8 MiB objects |
| **B. Plain collection `hub_skill_bundles`, one doc per artifact, `data: Binary`** (recommended) | Archive cap 8 MiB < 16 MiB BSON limit with margin; read is one `FindOne`; `http.ServeContent` over `bytes.Reader` gives Range/conditional for free; trivially transactional/idempotent by `_id` | Whole object in memory per request (bounded 8 MiB; same as the legacy zip path); hard ceiling forever at ~16 MB (chunked storage would need a new format version — acceptable, caps are in `bundle.format` namespace) |
| C. Embed bytes in the `hub_skills` row | No extra collection | Every list/GET decode drags 8 MiB; `definitionUpdate` rewrites it; breaks list cheapness |
| D. S3 (SDK already in `go.mod:8-13`) | Scales | New infra, IAM, deletion semantics — disproportionate; I found no existing skill/agent object-store wiring |

Recommendation **B**. Switching to GridFS later is a store-interface swap (`Put/Open/Delete`), keep the
interface narrow.

### 2.2 Sync fingerprint

| Option | Verdict |
|---|---|
| Archive hash as L/B/R fingerprint | **Rejected.** Local side would need to rebuild the archive from the directory and hash it; deflate output varies by zlib version/level (§1.3) and by Python build. A byte-identical rebuild is not a contract any runtime guarantees. L==B would false-negative ("local change") after a Python upgrade |
| Download the artifact on every check to compare | Rejected: 8 MiB per link per tick, defeats `If-None-Match`, and a check must stay cheap |
| **`tree_hash` in the manifest, verified by the server on upload** (recommended) | Pure function of path+content bytes; computable from a directory in one pass; classification needs no download; archive hash remains purely the transport/integrity identity |

Also considered: `ZIP_STORED` instead of deflate. It is bit-deterministic across implementations (and I
measured identical bytes for same-input builds), costs ~3-5× size on text, and is still under the 8 MiB
cap for realistic skills. I keep **deflate level 6** as the operator specified, because with `tree_hash`
determinism is no longer load-bearing; switching to stored later is a no-wire-change builder tweak.

### 2.3 Where the bundle lives relative to the generalizer (public scope)

The generalizer windows covered *text fields* and refuses > 24 windows (`hub_skill_service.go:652-672`);
it has no notion of binary content. Running it over only the text subset of a bundle would give a
false impression that the bundle was generalized. Operator decision: **review yes, generalize no**. See
§6.4 for the exact call and the residual risk.

---------------------------------------------------------------------------------------------------

## 3. Wire contract

### 3.1 Archive format (`format: "zip"`, canonical form)

A bundle archive is **exactly** what `build_bundle` emits; anything else is refused (no normalization):

- entries sorted by path bytes (UTF-8), `SKILL.md` included in that sort (it is just a path);
- no directory entries (names never end in `/`); directories are implied;
- every entry: `create_system = 3`, `external_attr = 0o100644 << 16` exactly, `date_time =
  (1980,1,1,0,0,0)`, `extra = b""`, `comment = b""`, `flag_bits ∈ {0, 0x0800}` (0x0800 only if the name
  is non-ASCII), method 8 (deflate, level 6) — method 0 (stored) is **also accepted** so a future builder
  can switch without a wire change; no other method;
- no ZIP64, no encryption, no data descriptors (bit 3), no archive comment, no prepended/appended bytes,
  no gaps/overlaps between entries (§4 step 5);
- paths obey the **existing** grammar (`skill_document.validate_path`, `skills.go:356-402`): NFC-stable,
  relative, `/`-separated, ≤5 segments, ≤240 bytes, no dot segments/backslash/colon/Cf characters, first
  segment `SKILL.md | references | scripts | assets`. (Closed root set kept for parity with documents;
  it also bounds what an installed bundle can place in a skill dir.)
- `SKILL.md` required, UTF-8, ≤200 KiB, strict frontmatter via the existing parser
  (`parse_frontmatter`; Go `requests.ParseSkillFrontmatter`).

`tree_hash` (v1) = `sha256` over the UTF-8 bytes of `"skill-tree-v1\n"` followed, for each file sorted by
path bytes, by `"<path>\0<decimal size>\0<sha256-hex of content>\n"`. Excludes timestamps, modes, order
of archive entries. Lowercase hex. Both sides compute it from *decoded content*, never from archive
headers.

### 3.2 Manifest (published record)

```json
{
  "document_version": 1,
  "name": "pdf-forms",
  "description": "Fill and validate PDF forms.",
  "bundle": {
    "format": "zip",
    "hash": "9f2c…64 lowercase hex…",
    "size": 482113,
    "uncompressed_size": 1204551,
    "file_count": 14,
    "entrypoint": "SKILL.md",
    "tree_hash": "3b7e…64 lowercase hex…"
  }
}
```

- `bundle` and `files` are **mutually exclusive**: both present → 422 `invalid_skill_document`
  (`field: "files", rule: "must not be combined with bundle"`); neither → today's "must contain at least
  SKILL.md" rule.
- `publisher` is **not** a client-sent field. The server adds it to read responses: the existing
  `account_id` and `account_metadata.name` (`responses/skills.go:16-36`; email withheld for non-owners,
  `:81-87`) are the publisher; the client surfaces `{id: account_id, name: account_metadata.name}` in
  the confirmation (§7).
- Optional `version` (existing author version field, `skills.go:148`) is allowed and ignored by sync.
- `bundle.hash`, `tree_hash`: `^[0-9a-f]{64}$`. `size`, `uncompressed_size`, `file_count`: positive
  integers within §3.5. `format` must be `"zip"`. `entrypoint` must be `"SKILL.md"`. Unknown keys inside
  `bundle` → 422 (closed set, same stance as `firstKeyOutside`, `skills.go:190-199`).

Server read shape (detail GET, org/tenant GET, publish result `accepted_document`): the row's
`HubSkillResponse` gains `bundle` (omitted for documents) and drops `files` (already `omitempty`,
`responses/skills.go:~30`; for a bundle it is simply absent). List rows (`includeFiles=false`) carry
`bundle` too (it is small and lets listings show size/file_count). `responses.AcceptedSkillDocument`
(`responses/skill_publish.go:52`) projects `bundle`. `sync_contract` gains an additive
`skill_bundles: 1` (omitempty, like `skills`, `revision.go:~46-55`); the client attests it per source
(`service.py:2122-2141`) and refuses bundle publish/auto-push without it.

### 3.3 Routes

| Route | Auth | Body | Success |
|---|---|---|---|
| `POST /v1/skills/bundles?visibility=&tenant_id=` | JWT; **same target claim as publish** (`claimSkillPublishTarget`, so the sibling lane's member-publish change applies automatically) + `agentWriteLimit` middlewares as `routes.go:505` | raw ZIP, `Content-Type: application/zip` (or `application/octet-stream`), ≤ 8 MiB (`http.MaxBytesReader`) | `201 {hash,size,uncompressed_size,file_count,entrypoint,tree_hash,name,description}`; re-upload of identical bytes in the same scope → `200` same body (idempotent) |
| `GET /v1/skills/:skillid/bundle` | `optionalAuthMiddleware` — **authorization decided exactly as `GetSkill`** (`hub_skill_service.go:380-401`) before any body, ETag or 304 | — | `200 application/zip` (the stored bytes), headers `ETag: <revision.ETag(id,generation,n)>`, `X-Skill-Bundle-Sha256: <hash>`, `Content-Length`; `If-None-Match` match (RFC weak compare via `controllers/if_none_match.go` `ifNoneMatchMatches`) → `304` after authz |
| `POST /v1/skills/publish`, `PUT /v1/skills/:skillid/publish` | unchanged | bundle manifest | unchanged envelope; bundle rows skip preview/pin entirely |
| `POST /v1/skills/publish/preview`, `PUT /v1/skills/:skillid/publish/preview` | unchanged | `{preview_version, document: <bundle manifest>}` | `200 status:"unchanged"`, no pin, `review:"not_run"` — same shape as `skillUngeneralizedPreview` (`hub_skill_service.go:828-849`), for **both** scopes, so a client has one flow |

Body bound for publish/preview stays 2 MiB (a bundle manifest is < 1 KiB).

ETag coordination with hub lane `bc62a4db26ca`: bundle republish goes through the same `Replace` →
`definitionUpdate` → `revisionUpdate`, so `content_revision` advances (§1.1) and `revision.ETag(...)` is
unchanged — their derivation needs no widening. The bundle GET uses the *row* revision as its ETag
(operator requirement). Alternative I considered and do **not** recommend now: ETag = `"sha256-<hash>"`
(truly content-addressed, immutable cache). Row-revision costs only a few false misses (any republish
bumps it) and avoids a second tag vocabulary on the same routes while the lane is mid-change. If their
change adds `Last-Modified`/`Cache-Control` helpers on skill reads, the bundle GET should call the same
helper rather than set its own. Collision surface: both PRs edit `routes.go` (skill block `:499-514`) and
`controllers/hub_skill.go`; keep our additions as new functions/new route lines to keep rebases trivial.

### 3.4 Error codes

Existing codes reused: `invalid_skill_document` 422 (`errors.go:291-295`), `payload_too_large` 413
(`errors.go:71-73`; `details.limit_bytes`), `skill_not_found` 404, `name_taken` 409,
`moderation_rejected` 422, `moderation_unavailable` 503, `not_a_member`/`insufficient_role`,
`preview_*`. New:

| Code | HTTP | When | `details` |
|---|---|---|---|
| `invalid_skill_bundle` | 422 | upload bytes fail §4 (any step) | `{step, field, rule}`; `step` ∈ `size, hash, container, entry, packing, content, entrypoint, tree` — never file contents |
| `skill_bundle_not_uploaded` | 422 | manifest `hash` has no blob in the claimed scope | `{hash}` |
| `skill_bundle_mismatch` | 422 | manifest `size/uncompressed_size/file_count/entrypoint/tree_hash/name/description` ≠ stored artifact metadata | `{field}` |
| `skill_form_change_unsupported` | 409 | republish changes document↔bundle | `{current:"document"|"bundle"}` |
| `skill_bundle_not_found` | 404 | `GET …/bundle` on a document row (**after** authz, so it cannot probe org rows) | — |
| `skill_row_inconsistent` | 500 | row has both `files` and `bundle` (rollback hazard, §9.2); logged at error | — |

Client-side refusal codes (`LinkError.code`, mapped by `_raise_skill_refusal`-style handlers):
`bundle-invalid` (+ `step`), `bundle-hash-mismatch`, `bundle-too-large`,
`bundle-confirmation-required`, `bundle-confirmation-mismatch`, `bundle-local-changes`
(L≠B on update/apply), `bundle-conflict` (L≠B and R≠B), `bundle-alias-rename-unsupported`,
`bundle-unsupported-server`, `bundle-form-changed` (remote flipped form under a link).

### 3.5 Bounds (numbers)

| Bound | Value | Where enforced | Note |
|---|---|---|---|
| Archive bytes | **8 MiB** (8,388,608) | server body reader; client download stream; both count streamed bytes | < 16 MiB BSON limit with margin |
| Uncompressed total | **32 MiB** (33,554,432) | declared-sum pre-check, then streamed count | bounds any zip bomb at 32 MiB absolute; no ratio rule needed (one rule fewer, same safety) |
| Single file | **8 MiB** | declared pre-check + streamed count | |
| File count | **256** (includes `SKILL.md`) | EOCD count read **before** parsing the central directory | documents stay at 64 |
| Path depth | **5** segments incl. filename; **240** path bytes | `validate_path` reuse | same as documents (`skill_document.py:31-30`) |
| `SKILL.md` | 200 KiB; description 1024 chars; name 64 chars kebab | existing validators | |
| Reviewed text (server only) | **256 KiB** of text-classified bytes across the bundle | before review (§6.4) | = `SkillFilesSumMaxBytesOrg`, `skills.go:~91`; over → 422 `invalid_skill_bundle` step `content` rule "text exceeds the review budget" |
| Manifest body | 2 MiB (existing) | | |
| Entries in central directory (EOCD) | ≤ 256 | step 2 | |

Compared with `fork.py`: same *discipline* (pre-count totals, stream-count, refuse naming the path,
stage then one rename); the numbers are deliberately 2× tighter than the 64 MiB roles copy because this
input is untrusted and network-delivered.

---------------------------------------------------------------------------------------------------

## 4. Archive validation algorithm (order of checks)

One algorithm, **implemented twice** (Python `skill_bundle.validate_archive`, Go
`requests.ValidateSkillBundle`) with a shared golden-vector corpus (§10). The server is authoritative at
upload; the client re-runs it on every install/update because it must not trust the hub.
Every refusal is by named step/rule; no refusal ever echoes entry content.

0. **Bound the read.** Read at most 8 MiB + 1 bytes; > 8 MiB → `bundle-too-large` / 413. Empty → refuse.
1. **Hash before parse.** `sha256(all bytes)`. Client: compare to `manifest.bundle.hash` with
   `hmac.compare_digest` — mismatch → `bundle-hash-mismatch`, **nothing parsed, nothing written**
   (requirement 1, first half). Server at upload: no claim yet, so compute and return it; at publish:
   compare to the manifest.
2. **Container frame, by hand, before any zip library.** Last 22 bytes must be an EOCD with signature
   `PK\x05\x06`, comment length 0, disk numbers 0, `entries_on_disk == entries_total`, count ≤ 256,
   no `0xFFFF`/`0xFFFFFFFF` ZIP64 markers, no ZIP64 locator, `cd_offset + cd_size + 22 == len(archive)`.
   Rejects trailing/prepended data and absurd entry counts before the library allocates per-entry
   structures (Go's `zip.NewReader` allocates for every declared record).
3. **Open with the library** (`zipfile`/`archive/zip`) — read the central directory only. Library errors
   → `container`.
4. **Per-entry header checks, no data I/O.** For each entry in central-directory order:
   name is a valid path (grammar above; does **not** end in `/`); name bytes decode as UTF-8 and equal
   their NFC form; method ∈ {0, 8}; `flag_bits & ~0x0800 == 0` (rejects encryption bit 0, data
   descriptor bit 3, patched-data bit 5, strong-encryption bit 6); `create_system == 3`;
   **`external_attr == 0o100644 << 16`** (this single equality refuses symlinks `0o120000`, devices
   `0o060000/0o020000`, FIFOs `0o010000`, sockets, directories `0o040000`, any exec bit
   `0o111`, setuid `0o4000`, setgid `0o2000`, sticky `0o1000`, and a nonzero DOS attribute byte);
   `date_time == (1980,1,1,0,0,0)`; `extra == b""`; `comment == b""`; `file_size ≤ 8 MiB`; running sum of
   declared `file_size` ≤ 32 MiB; no duplicate / NFC-casefold collision / file-vs-ancestor-directory
   collision (reuse the `SkillDocument.__post_init__` rule, `skill_document.py` ~`:215-235`, and its Go
   twin `skills.go` `seenFolded`); central-directory order == sorted-by-path order.
5. **Tight packing.** Sort entries by `header_offset`; first offset must be 0; for each entry read the
   30-byte local header and require signature, flags, method, CRC, sizes, name length/bytes and
   `extra_len == 0` to equal the central record; `next_offset == offset + 30 + name_len + compress_size`;
   the last entry's end `== cd_offset`. Refuses overlapping entries, gaps, polyglot prefixes and
   "two parsers see two archives" tricks (local vs central disagreement).
6. **Entrypoint present.** `SKILL.md` exists; total entries ≤ 256; file count equals
   `manifest.bundle.file_count` (client) / is recorded (server).
7. **Streamed decompression with a counted budget.** For each entry open via the library and read in
   64 KiB chunks; count **actual** decompressed bytes; abort the moment a count exceeds the entry's
   declared `file_size` (declared sizes lie — fork's review measured the same class,
   `fork.py:~598-606`), 8 MiB, or the 32 MiB running total; require final length == declared and CRC ok
   (library verifies CRC at EOF — read to EOF). Compute `sha256(content)` per file (for `tree_hash`).
   Client extraction (§5) happens **in this same pass** into staging; the server discards bytes.
8. **Entrypoint content.** `SKILL.md` strictly UTF-8, no control characters except `\n\r\t` (same rule
   as `_text_bytes`, `skill_document.py:72-73`), ≤200 KiB; frontmatter parsed with the existing strict
   parser; `meta.name == manifest.name`, `meta.description == manifest.description`; name valid.
9. **Tree hash.** Recompute `tree_hash` from step 7; `== manifest.bundle.tree_hash` else
   `bundle-invalid` step `tree`. `uncompressed_size == sum(sizes)` and `size == len(archive)` checked.

Steps 4-5 make non-canonical archives refusals, not warnings. A skill authored with another zip tool is
refused with "rebuild with `lop skills publish`" — acceptable because only the builder publishes.

---------------------------------------------------------------------------------------------------

## 5. Extraction and atomic swap (client)

Never `ZipFile.extract/extractall` (they sanitize rather than refuse). Extraction is our code, in the
step-7 pass.

Precondition (unchanged from `ManagedSkillStorage.apply`, `skill_storage.py:226-330`): alias valid,
alias owner is this local id (`alias_owner`), destination absent or owned by this link, destination's
current `tree_hash` equals `expected_fingerprint`, policy revision CAS, `automatic ⇒ subscribe`.

1. **Journal first.** `journal.prepare(identity, "pull", before_fingerprint=<current tree_hash or None>,
   after_fingerprint=<manifest tree_hash>, payload={"adapter": "managed-skill-bundle", "local_alias",
   "baseline", "policy_revision", "automatic", "archive_hash"})`; claim the lease (same pattern as
   `skill_storage.py:~296-312`).
2. **Fetch** (outside the config lock — slow): stream `GET …/bundle` into
   `hub/downloads/<op>/bundle.zip` (dir `0700`, file `O_EXCL`, `0600`), counting bytes ≤ 8 MiB, hashing
   as it streams. Verify against `manifest.bundle.hash` (**hash check #1**, §4 step 1). Re-fetch only on
   `If-None-Match`-miss logic in `check`, never in apply.
3. **Stage inside the skills root:** `skills/.stage-<op>/` created `mkdir(mode=0o700)`; refuse if
   `skills/` or the stage path is a symlink (`path()` precedent, `skill_storage.py:88-93`). A leading
   dot cannot collide with an alias (`validate_name` is `^[a-z0-9]+(-[a-z0-9]+)*$`). Staging inside the
   root (not `hub/staging`, which existing text apply uses, `skill_storage.py:133-157`) is the operator's
   requirement and guarantees one filesystem for the final `os.replace`.
4. **Validate + unpack in one pass** (§4 steps 2-9): for each file, create parents with
   `mkdir(0o700)` (never following links), open the file with
   `os.open(path, O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW, 0o600)`, write counted chunks, `fsync`. **No
   `chmod`. No mode taken from the archive. No post-install hooks.** Files stay `0600`/`0700` dirs
   (identical to `_stage`, `skill_storage.py:143-150`), so they are never executable.
5. **Verify after (#2).** (a) Re-read `bundle.zip` from disk and re-hash — must equal the manifest hash
   (detects tampering of the temp file between check #1 and unpack by a local writer; second half of
   requirement 1). (b) Walk the staged tree with `lstat`: every entry is a regular file (`S_ISREG`),
   `st_nlink == 1` (**refuses hardlinks**), no symlink, no entry with `st_mode & 0o7111`; recompute
   `tree_hash` from disk bytes == manifest `tree_hash`; file count / total size equal the manifest. Any
   failure → `rmtree(stage)`, journal `aborted`, `LinkError(bundle-invalid, step=…)`. On Windows the
   mode test is skipped and the reparse-point/symlink + regular-file tests remain.
6. **Swap under `config_write_lock`** (copy of the existing two-rename discipline,
   `skill_storage.py:~330-395`): re-read destination `tree_hash` == `before`; if the destination exists
   `os.replace(dest, hub/backups/skill-<local_id>-<op>)`; `fsync(skills/)`; `os.replace(stage, dest)`;
   `fsync(skills/)`; recompute destination `tree_hash` == `after` (post-rename verify, third check);
   journal → `content-committed`; `_finalize` writes the baseline (`fields` = the bundle manifest,
   `fingerprint` = `tree_hash`, `content_state="present"`) and policy; journal → `complete`.
7. **Cleanup.** Delete `hub/downloads/<op>`; the backup is kept per existing policy (not deleted by
   this design).
8. **Crash recovery** (`ManagedBundleStorage.recover`, mirror of `_recover_one`,
   `skill_storage.py:602-627`, comparing `tree_hash`): dest==after → finalize; dest absent ∧ backup ∧
   stage → restore backup if its hash == before else `blocked`; dest==before → abort and `rmtree` stage;
   otherwise `blocked: local-concurrent-edit`. Also remove any `skills/.stage-*` and
   `hub/downloads/*` whose journal entry is `aborted/complete/absent`. `managed_skill_read`
   (`skill_storage.py:630-649`) must call both recoveries.

Update = same path with a different `archive_hash`/`tree_hash`; `before` must equal the baseline's
`tree_hash` (L==B) else `bundle-local-changes`. **A bundle apply never overwrites unrecorded local
bytes**: the destination hash is re-checked under the lock.

---------------------------------------------------------------------------------------------------

## 6. Policies

### 6.1 Executable policy (decided; stated in the product docs and the code comments)

**Bundles contain no executables.** Refused at three layers: (1) builder — `build_bundle` refuses a
source tree containing a file with any `0o111`/setuid/setgid/sticky bit, a symlink, a hardlink
(`st_nlink > 1`), a device, FIFO or socket (on Windows: reparse points; mode bits ignored) and *writes*
`0o100644` regardless; (2) server — §4 step 4 (`external_attr` equality); (3) client — §4 step 4 again
plus §5 step 5 on disk. Install never sets an executable mode and never runs a hook; a file under
`scripts/` is inert data the agent may read, and if it chooses to run `python scripts/x.py` that is an
ordinary, separately-gated tool call. The mode bits in the *source* directory are not published (a
published script loses its `+x`, which is the intended consequence).

### 6.2 Consent

- **Public first install** requires explicit confirmation **naming the publisher and bound to the
  bytes**. The install call without confirmation raises `bundle-confirmation-required` carrying
  `{name, publisher:{id,name}, bundle:{hash,size,file_count,uncompressed_size}}` and writes nothing. The
  confirming call must send `confirm_bundle_hash == manifest.bundle.hash` and `confirm_publisher_id ==
  publisher.id`; the install re-fetches the manifest and refuses `bundle-confirmation-mismatch` if either
  changed (no TOCTOU swap between prompt and install). The publisher *id* is the binding value because a
  display name is user-controlled.
- **Org installs** inherit the org role gate: read = member via `GetSkill`'s `CheckOrgAccess`
  (`hub_skill_service.go:392-398`), publish = whatever `claimSkillPublishTarget` returns (admin today,
  member after lane `bc62a4db26ca`). No separate confirmation.
- **Later updates inherit link consent** — subscribe/auto-update policy is unchanged. A public update can
  only come from the owner (`claimSkillRepublishTarget` owner check, `hub_skill_service.go:735-740`), so
  the publisher cannot change under a link; if the manifest's `publisher.id` ever differs from the
  baseline's, the update is refused `origin-conflict`.
- Alias must equal the manifest name for bundles (`bundle-alias-rename-unsupported` otherwise): renaming
  would require rewriting `SKILL.md` bytes and breaks `tree_hash` equality. Document skills keep alias
  rendering (`SkillDocument.with_name`).

### 6.3 Form rule (document vs bundle)

Selection is **explicit**, never an automatic fallback: `lop skills publish … [--bundle]`.
Default is today's behavior (document). A source that does not fit document rules (binary file, > caps)
fails with a message naming `--bundle`. **Auto-fallback is rejected on purpose:** a public text skill
that today fails the 20 KiB generalizer cap would silently route around the generalizer
(§6.4). Server-side, a republish may not switch forms (`skill_form_change_unsupported`, 409);
`definitionUpdate`'s `$$REMOVE` would otherwise do the switch silently.

### 6.4 Review and the generalizer — the exact call

Bundle publish (both scopes) runs the **existing review** and **not the generalizer**:

```go
// internal/services/generalization_skill_documents.go (new func, beside skillSubmittedSubmission :167)
func skillBundleSubmission(m *requests.SkillBundleManifest, inv BundleInventory) *GeneralizedSubmission {
    base := ModerationSubmission{Name: m.Name, Description: m.Description,
        Instructions: bundleReviewedJoin(inv) /* text files path-labelled; binary files as one
                                                 `== path == [binary N bytes sha256 …]` line */}
    return unengagedSubmission(base) // generalization_publication.go:241, named pass-through
}
```

then `runContentReview(ctx, s.reviewer, content, existing.Moderation, scope)` (`agent_publication_service.go:1026`),
`withGeneralizationRecord` is a no-op (transform nil, `:471-475`) so the row claims no generalization.
`unengagedSubmission` is the right constructor: it carries only the five reviewed slots and no
`values` map (the bundle has no per-file `values`), whereas `skillSubmittedSubmission` requires a
`SkillDocument.Files`. Its doc comment calls itself "a known, dated gap" for legacy writers
(`generalization_publication.go:229-240`); a bundle is **not** a gap waiting for check mode, so wrap it
in a *named* `skillBundleSubmission` with its own comment stating the decision, keeping the compile-time
"one gate type" guarantee and greppability.

What the reviewer sees: all text-classified files (valid UTF-8, no control chars) ≤ 256 KiB total,
chunked by the existing 32 KiB chunker (`moderation_service.go:599-640`) on the `== path ==` headers.
**What nobody reviews:** binary files' contents (the reviewer sees path, size, sha256 only), and any
text file reclassified "binary" by one stray control byte. Over-budget text is refused (§3.5), so
oversize text cannot be smuggled as "unreviewed".

Trade-offs, stated plainly:
- *No generalizer ⇒ public bundles can publish references the generalizer would placeholder* (names,
  paths, hostnames, in text or binary). The generalizer cannot operate on binary and its 24-window
  ceiling would forbid the sizes bundles exist for. Operator accepted.
- Mitigations in this design: bundle publish is never automatic; the client requires explicit
  `--public` + `confirm_public_egress` (the existing flag, `service.py:3354`), shows file list and
  sizes in the preview, and the server review still runs. A **recommended later hardening** (not v1):
  run the generalizer over the text subset when it fits the window ceiling, and refuse public bundles
  whose text exceeds it — recorded in §11.
- Org scope is unchanged in kind: the document path already skips the generalizer for orgs.

---------------------------------------------------------------------------------------------------

## 7. Server change map (agent-server)

New files:

| File | Content |
|---|---|
| `internal/requests/skill_bundle.go` | `SkillBundleRef` (manifest sub-object), `ValidateSkillBundleRef`, constants (§3.5), `ValidateSkillBundle(data []byte) (*SkillBundleInventory, *Violation)` implementing §4, `TreeHash`, text/binary classifier. Uses `validateSkillPath`, `ParseSkillFrontmatter` |
| `internal/models/hub_skill_bundle.go` | `HubSkillBundleBlob{ID "scope:hash", Scope, Hash, Size, UncompressedSize, FileCount, TreeHash, Name, Description, Data []byte, CreatedAt, UploaderAccountID}` |
| `internal/repositories/hub_skill_bundle_repository.go` (+ mock via the existing `go:generate` pattern, `hub_skill_repository.go:19`) | `Put` (idempotent by `_id`, re-hash on write), `Get(scope, hash)`, `DeleteUnreferencedBefore(t)`; `MigrateIndexes` for `created_at` |
| `internal/controllers/hub_skill_bundle.go` | `UploadSkillBundleHandlerEcho`, `GetSkillBundleHandlerEcho` (+swagger comments) |
| tests | §10 |

Edits:

- `internal/models/hub_skill.go:42` — `Files` → `bson:"files,omitempty"`; add `Bundle *HubSkillBundle bson:"bundle,omitempty" json:"bundle,omitempty"` (`{Format,Hash,Size,UncompressedSize,FileCount,Entrypoint,TreeHash}`). `omitempty` on `files` is what lets `definitionUpdate` (`content_revision.go:114-127`) remove the stale field. A helper `(*HubSkill).IsBundle()`; readers must treat "both present" as corrupt (§9.2).
- `internal/requests/skills.go:142-159` — allow `bundle` in `skillDocumentAllowedFields`; `SkillDocument.Bundle *SkillBundleRef`; `Files` tag `omitempty`; `ValidateWithinBounds` branch: bundle ⇒ no files, validate ref, skip text/file-sum caps; document ⇒ unchanged. Decoder test for `files`+`bundle` → 422.
- `internal/responses/skills.go:16-36,45-76` and `responses/skill_publish.go:35-62` — project `bundle`; `AcceptedSkillDocument` emits it.
- `internal/revision/revision.go` (`Contract`, `SkillCapability` `:46-63`) — additive `SkillBundles int json:"skill_bundles,omitempty"`; `SkillCapability()` sets 1 **only** after the bundle routes are registered. (Merge-conflict watch: sibling lane may touch this file.)
- `internal/services/hub_skill_service.go` — interface (`:33-44`) gains `UploadSkillBundle(ctx, data, target, accountID)` and `GetSkillBundle(ctx, skillID, accountID)`; `PublishSkill` (`:83`) and `RepublishSkill` (`:203`) fork **after** `claimSkill*Target` and decode: `if doc.Bundle != nil → publishBundle/republishBundle`. The bundle branch: load blob by `(scope, hash)` → `skill_bundle_not_uploaded`; compare every manifest field with the blob's recorded metadata → `skill_bundle_mismatch`; name check `checkBuiltinReserved`/`checkSkillNameFree` as today; **no** `verifyPreviewIfPresent`/`resolvePublicationContent`; build the carrier per §6.4; `runContentReview`; `buildSkillRowInto` variant sets `Bundle`, clears `Files`; `skills.Create/Replace`. Republish: refuse form change; the existing `revision.Check`/`WithExpected` lines (`:253-260`) stay and give If-Match CAS for free. `GetSkillBundle` = the first 20 lines of `GetSkill` (authz) then blob read; shares one authz helper rather than copying it. Preview funcs (`:744-826`): bundle manifest → `skillUngeneralizedPreview`-style answer.
- `internal/services/generalization_skill_documents.go` — `skillBundleSubmission`, `bundleReviewedJoin` (§6.4).
- `internal/repositories/hub_skill_repository.go` — nothing structural; (`Create/Replace` take the model). Optional `ListReferencedBundleHashes` for GC.
- `internal/routes/routes.go:499-514` — register `POST /skills/bundles` (same middleware stack as `:505`) and `GET /skills/:skillid/bundle` (`optionalAuthMiddleware`).
- `cmd/server/main.go:1081-1089` — construct the bundle repository + GC, pass into `NewHubSkillService`; `internal/services/hub_skill_service_mocks.go` regenerated.
- `internal/errors/errors.go` / `messages.go` — new codes/messages (§3.4).
- `cmd/server/content_revision_fixture_test.go` (seed `skillDocument(...)` at the "Skill seeds" block) — add a seeded public + org **bundle** row so the existing QA fixture/rig exercises 304/revision on bundles. Mind the file's synthetic-only credential posture.
- **GC** — a periodic sweep (reuse whatever scheduler hosts other janitors; if none exists, an operator-invoked `DeleteUnreferencedBefore` like `audioartifact.DeleteExpired`, `store.go:~60`) deleting blobs unreferenced by any `hub_skills` row and older than 24 h. Blobs are scoped `visibility:tenant:hash` so an upload in one scope can never be referenced from another and an "already exists" answer never reveals another tenant's content.
- **[unverified] account deletion:** `internal/repositories/account_deletion_registry.go` (+ `_test.go`) appears to enumerate every collection/bucket holding account data (it references GridFS at `account_deletion_repository.go:613-660`). A new collection with `uploader_account_id` almost certainly needs registering there and a registry test will probably fail until it is. Settle by reading `account_deletion_registry_test.go` before coding.

---------------------------------------------------------------------------------------------------

## 8. Client change map (local-operator)

New files:

| File | Content |
|---|---|
| `local_operator/hub_sync/skill_bundle.py` | constants (§3.5); `SkillBundleManifest` (frozen dataclass; `from_json` strict closed key set, `from_remote` tolerant-envelope exactly like `SkillDocument.from_remote`; `to_json`; `fingerprint -> tree_hash`); `BundleError(code, step)`; `build_bundle(dir) -> BuiltBundle(bytes, manifest)`; `validate_archive(bytes, manifest)`; `extract_bundle(archive_path, stage_dir, manifest)` (§4+§5 pass); `tree_hash_of_dir(path)`; `manifest_form(raw) -> "document"|"bundle"` |
| `local_operator/hub_sync/skill_bundle_storage.py` | `ManagedBundleStorage(config_dir)`: `snapshot(local_id)`, `apply(...)`, `recover(...)`, `_fingerprint_path -> tree_hash_of_dir`, `_stage`, `_locations`. Reuses `TransactionJournal`, `config_write_lock`, `alias_owner`, `read_policy`, `_sync_directory`/`_sync_ancestors`, `write_baseline` (imports from `skill_storage`/`transactions`; no inheritance from `ManagedSkillStorage`) |
| tests | §10 |

Edits (file:function):

- `hub_sync/provenance.py:fingerprint_snapshot` (`:219-238`) and `:fingerprint_fields` (`:240-253`): skill branch dispatches `SkillBundleManifest.from_json(fields).fingerprint` when `"bundle" in fields`, else the document path. `parse_baseline_record` (`:507-604`) works unchanged via these (its outbound-receipt checks call `fingerprint_snapshot`, `:~519-530`).
- `hub_sync/skill_storage.py:ManagedSkillStorage.recover` (`:400-410`) already skips non-`managed-skill` payloads — **leave it**; `managed_skill_read` (`:630-649`) additionally recovers `ManagedBundleStorage`. `unpull` (`:412-491`) looks content-agnostic (rename to tombstone) but **[unverified]** — read it fully; if it never reads content, bundle unpull works unchanged.
- `hub_sync/adapters.py:SkillAdapter` — `snapshot` (`:222-240`): read the baseline; bundle ⇒ `ManagedBundleStorage.snapshot`; `canonical_fields` (`:242-251`) and `validate` (`:253-256`) dispatch on `manifest_form(raw)`; `publish_document` skill branch (`:306-312`) dispatch likewise.
- `hub_sync/check.py:check_hub_skills` (`:568-805`): local snapshot through the storage for the record's form (`:~612-650`); the alias-rename comparison at `:~757-759` (`SkillDocument.from_json(local).with_name(remote_name)`) is skipped for bundles; remote `canonical_fields` unchanged call; `classify` unchanged. `_fetch_skill` (`:792-804`) unchanged (manifest only; the artifact is fetched only at apply).
- `hub_sync/service.py`:
  - `_resolve_remote_skill` (`:2404-2525`): accept either canonical form; returns it.
  - `install_skill` (`:2528-2914`): bundle ⇒ require alias == name, consent gate (§6.2, new params `dry_run`, `confirm_bundle_hash`, `confirm_publisher_id`), `ManagedBundleStorage.apply`; document path untouched. The `skill_pull` response gains `bundle` summary on `dry_run`.
  - `_apply_skill` (`:1189-1390`): early branch `_apply_skill_bundle`: refuse unless `c.classification.state == "remote-only"` (L==B) → apply; `both-changed` → `fail("bundle-conflict", …, "needs-review")`; never calls `merge_skill_document`.
  - `_project_publication` (`:3249-3276`), `preview_skill_publication` (`:3361-3437`), `publish_skill` (`:3787-4063`), `_published_baseline` (`:3495`), `_materialize_published_skill` (`:3536-3726`), `_project_accepted_publication` (`:3729`): `form="bundle"` branches — `build_bundle` instead of `from_directory`; the preview record stores `tree_hash` + archive hash + file list (not bytes), confirm re-builds and requires **tree_hash equality** with the previewed one (not archive-hash equality — §2.2); then `upload_skill_bundle` → `publish_skill_document(manifest)`; materialize installs the *same extracted tree already on disk*? No — publish never moves the author's source directory; it records the baseline and installs by the normal path as a document publish does today (`_materialize_published_skill` mirrors `install_skill`).
  - `_fold_capability` (`:2122-2141`) / `_capability_auto_push` (`:2094-2120`): record/require `skill_bundles`.
- `hub_sync/outbound.py:_project_skill` (`:218-262`) / `project_outbound` (`:300-356`): bundle ⇒ no merge: `fields = local manifest` iff R==B else refusal `bundle-conflict`.
- `hub_sync/push.py`: `HttpPushTransport` routes are unchanged (`:125-170`); the push function must **skip the paid preview and the pin requirement** for bundle manifests (the token check at `~:440-460` would otherwise `unsupported-precondition`) and commit directly with `If-Match` + the manifest. The function containing that code starts before line 330 and I did not capture its name — coder to locate. Bundle auto-push additionally requires an outbound receipt (existing guard, `~:350-360`) and `skill_bundles` attestation.
- `clients/radient.py` (next to `get_skill`/`publish_skill_document`, `:1477-1589`): `upload_skill_bundle(data, *, visibility, tenant_id)` (`POST …/skills/bundles`, `Content-Type: application/zip`, `allow_redirects=False`, `_refuse_redirect`, bounded), `download_skill_bundle(skill_id, *, if_none_match, max_bytes, send_bearer)` streaming to a file object with a byte ceiling and `allow_redirects=False`. Same credential posture as `get_skill` (public row readable unauthenticated, org row bearer).
- `server/routes/desktop_hub.py`: `skill_pull` (`:~669`) body model gains `dry_run`, `confirm_bundle_hash`, `confirm_publisher_id`; `skill_publish` (`:~815`) gains `form`; **no new route** (desktop route needed: none — the existing three suffice). A pull `dry_run` returns the bundle summary and writes nothing (mirrors publish's `dry_run`). Error mapping adds the `bundle-*` codes (422/409).
- `cli.py` — the new surface in §8.1: parser block beside `hub_parser` (`:703`); handlers beside `teams_sync_command` (`:11406`) / `hub_status_command` (`:11514`); dispatch beside `:16216-16250`; help strings via a new i18n group modelled on `_hu`/`_th` (`cli.py:70`).

### 8.1 CLI surface (smallest that exercises publish / pull / subscribe / update)

```
lop skills publish <local-skill> (--public | --org <tenant>) [--bundle] [--alias NAME]
                   [--no-subscribe] [--yes] [--json]      # preview then confirm; --yes skips the prompt
lop skills pull <hub-skill-id> (--public | --org <tenant>) [--alias NAME]
                   [--no-subscribe] [--confirm-publisher ID --confirm-hash HASH | --yes]
lop skills sync [--name NAME | --all] [--json]            # check + fast-forward apply (document: merge as today)
lop skills status [--json]                                # per-link L/B/R state, no network write
```

`pull` is `install_skill` (subscribe default true; `--no-subscribe` is "download" mode); `subscribe` /
`unpull` remain desktop-only in v1 (`lop hub link set <id> --subscribe` already toggles policy,
`cli.py:~729-760`). `--yes` on a public pull still prints the publisher id/name and hash and records
them; non-interactive callers must pass `--confirm-publisher/--confirm-hash` or `--yes` explicitly.
Because these commands also drive documents, they double as the first CLI for existing skill sync.

### 8.2 Whole-tree classification (bundle)

Fingerprints are `tree_hash`. Reusing `classify` (`check.py:46-64`):

| L vs B | R vs B | `classify().state` | Action |
|---|---|---|---|
| L==R | — | `up-to-date` | none |
| L==B | R≠B | `remote-only` → `available` | fast-forward update (§5), gated by `subscribe` for auto |
| L≠B | R==B | `local-only` → `up-to-date` (pull side) | push candidate (needs outbound receipt + attestation) |
| L≠B | R≠B (L≠R) | `both-changed` | **conflict**: report `bundle-conflict`, write nothing, no text merge; operator resolves by `unpull`+`pull` (backup in `hub/backups`) or by pushing after reconciling |
| no baseline | | `baseline-unknown` | unchanged handling |

---------------------------------------------------------------------------------------------------

## 9. Migration and compatibility

### 9.1 Old clients / old rows
- Existing rows: no `bundle` field ⇒ document; no backfill, no index change; responses byte-identical
  (the new fields are `omitempty`; `skills.go` `Files` tag change affects only empty slices, which a
  valid document never has).
- Old client vs bundle row: detail GET returns `{…,"bundle":{…}}` with no `files`; `from_json` refuses
  (key set mismatch) ⇒ **per-link `unavailable: invalid-remote-manifest`** in checks, `invalid-remote-manifest`
  on install, nothing written, no tick crash (§1.2). Public list shows the row (name/description);
  the install fails with that code. This is the intended, non-destructive degradation; the operator
  surface should word it "update lop to install this skill" — a `designer`/`ux-reviewer` item, not
  this doc's.
- New client vs old server: `sync_contract.skill_bundles` absent ⇒ `bundle-unsupported-server`
  refusal before any upload; document skills unchanged. New client vs old *document* rows: unchanged.
- Skill-form change on an existing link (remote flips) is refused by the server (§3.4) and by the
  client (`bundle-form-changed`) if a hand-edited row ever appears.

### 9.2 Rollback hazard (needs an operational rule)
An **older server** binary that is rolled back to after bundle rows exist decodes `hub_skills` rows
without the `bundle` field; its `definitionUpdate` iterates the *old* struct and will not `$$REMOVE`
`bundle` — a republish via the old server of a bundle row would leave `bundle` + new `files`, i.e. both
forms. Rules: (1) new server treats both-present as corrupt (`skill_row_inconsistent`, 500, error log)
and never serves either; (2) the rollout note says "delist (delete) bundle rows or keep the new server
until none exist before rolling back past this change"; (3) the feature is also gated by the
`SkillBundles` capability so a canary can be partial. No data migration is needed to roll *forward*.

### 9.3 Capability gating
`sync_contract.skill_bundles:1` is advertised only when routes are registered and the blob repository
is up. Clients cache it per source (`record_server_capabilities`, `service.py:2144-2171`) — it is a
probe-only fact and must stay unreachable from any request field (that function's SECURITY INVARIANT).

---------------------------------------------------------------------------------------------------

## 10. Test matrix

### 10.1 Shared golden corpus
A directory of malformed/valid archives plus expected `(step, rule)` consumed by **both** the Python and
Go suites, so the two validators cannot drift. Valid-archive goldens also pin `tree_hash` outputs.

### 10.2 Malformed-archive cases (each must be refused by name, nothing written)

path `../x`; `a/../b`; `/abs`; `C:\x`; `C:/x`; `a\b`; leading dot segment; NUL in name; >5 segments;
>240 bytes; non-NFC name; Cf character; root outside `SKILL.md|references|scripts|assets`; duplicate
name; case/NFC collision; `a` and `a/b`; directory entry `x/`; symlink (`0o120777`); symlink whose
target is `/etc/passwd`; exec `0o100755`; setuid `0o104755`; setgid `0o102755`; sticky `0o101644`;
block/char device; FIFO; socket; `create_system=0`; nonzero DOS attrs; non-1980 timestamp; non-empty
extra; entry comment; archive comment; ZIP64; encrypted; data-descriptor flag; method bzip2/lzma/
deflate64; entries out of sorted order; central entry count lies (65535 declared, small actual) and
count 257; local header disagrees with central header; overlapping entries; gap between entries;
trailing bytes; prepended bytes (polyglot); truncated; empty file; non-zip bytes; zip-in-zip as an
entry (allowed content, must not be recursed into); declared size 10 / actual 10 MiB; CRC mismatch;
bomb (≈10 KiB archive → 40 MiB); single file 8 MiB+1; total 32 MiB+1; archive 8 MiB+1; 257 files;
missing `SKILL.md`; `SKILL.md` binary / BOM / control chars / frontmatter alias / unknown key /
name ≠ manifest / description ≠ manifest; manifest hash/size/uncompressed_size/file_count/tree_hash/
entrypoint wrong; manifest `files`+`bundle`.

### 10.3 Client units
`build_bundle`: stable across two runs; `tree_hash` independent of deflate level and of source file
mode/mtime; refuses source symlink/hardlink/exec/FIFO/hidden dir; writes `0o100644` for an `0o755`
source. `extract_bundle`: bounds counted on streamed bytes; `O_EXCL|O_NOFOLLOW` against a planted
symlink in the stage dir; leaves no `chmod` side effect (assert extracted modes `0600`/`0700`);
post-extract hardlink/symlink/nlink detection; temp-file tamper between hash #1 and unpack caught by
hash #2. `ManagedBundleStorage`: kill-before/between/after each rename (checkpoint callbacks like
`apply(checkpoint=…)`, `skill_storage.py:236`), recovery to a single legitimate state; destination
changed under lock ⇒ `local-concurrent-edit`; alias owned by another link ⇒ `origin-conflict`; stage
dirs and downloads cleaned on every failure. `classify` table of §8.2. `fingerprint_*` dispatch for
documents is byte-identical to before (regression guard).

### 10.4 Server units
Validator corpus; `SkillDocument` decode (`bundle`, `files`, both, neither, unknown bundle key);
`definitionUpdate` removing the other form's field on a (forbidden) switch is **not reachable** (409
first) — keep a repository test that proves `$$REMOVE` works so a future relaxation is deliberate;
upload idempotency; scope isolation of blobs (A cannot reference B's hash; no existence oracle);
`bundleReviewedJoin` includes inventory line for binary; `unengagedSubmission` carrier yields no
`Generalization` record on the moderation sub-document; reviewed-text budget refusal.

### 10.5 Server integration (Mongo, harness in `cmd/server`, pattern of
`content_revision_fixture_test.go` + `skills_publish_e2e_test.go`)
Auth matrix for `POST /skills/bundles` (anonymous 401; wrong-tenant org target 403/404; member vs admin
per the lane's role gate; rate-limit headers) and `GET …/bundle` (anonymous public 200; anonymous org
404; non-member 404; wrong-tenant 404; owner 200; document row 404 `skill_bundle_not_found`);
`If-None-Match` hit ⇒ 304 **only after authz** (unauthorized + matching tag ⇒ 404 not 304);
`ETag` header == body `revision`; republish advances `content_revision` and changes ETag; If-Match
stale ⇒ 412; form-change ⇒ 409; moderation reject ⇒ 422 `moderation_rejected` and **no row and the
blob remains unreferenced**; manifest/blob mismatch codes; oversize body ⇒ 413 with `limit_bytes`;
both-forms row ⇒ 500 `skill_row_inconsistent`; GC deletes only unreferenced + stale; account deletion
registry test (once the unverified item is settled).

### 10.6 End-to-end (real path; isolated `HOME`/config; synthetic credentials; no live sessions)
1. Go server fixture (the `cmd/server` e2e harness / `qaFixture` receipt) + Python `lop` in a scratch
   home with a temp skills root containing `SKILL.md`, `references/a.md`, `assets/logo.png`
   (binary), and an `scripts/run.sh` with `+x` set.
2. `lop skills publish demo --org T --bundle` — assert refused-never-silent for the `+x` build
   (`bundle-invalid`), then after `chmod -x` publish succeeds; capture manifest JSON, upload response,
   row revision. Public path: confirm the egress consent flow.
3. Second isolated home: `lop skills pull <id> --org T`; assert tree bytes, modes `0600`/`0700`, no
   `+x` anywhere, baseline `tree_hash`, journal `complete`; public pull without confirmation exits
   non-zero having written nothing; with wrong `--confirm-hash` ⇒ `bundle-confirmation-mismatch`.
4. Republish (changed asset) from home 1; `lop skills sync` on home 2 ⇒ fast-forward; assert old
   backup retained, `If-None-Match` 304 observed in server log on an unchanged re-check.
5. Home 2 edits a file, republish upstream ⇒ `bundle-conflict`, tree unchanged, nothing overwritten;
   home 2 edits, upstream unchanged ⇒ `local-only` (+ push if eligible).
6. Tamper cases against the live server fixture: flip one byte in the served blob (via fixture
   control) ⇒ client `bundle-hash-mismatch`, skills root untouched; server returns a valid zip with an
   `0o755` entry ⇒ client refuses at §4 step 4 even though the hash matches the (hostile) manifest.
7. Old-client simulation (document-only `from_json`): bundle row ⇒ `invalid-remote-manifest`, no crash.
8. `kill -9` mid-install at each checkpoint; next `lop skills status` recovers.
Desktop path (separate, requires the harness browser; not covered by CLI E2E): public pull confirmation
dialog — designer + ux-reviewer rounds per the repo's rules (D-/U-findings), screenshots of
confirm/error/populated states.

---------------------------------------------------------------------------------------------------

## 11. Risks and explicit non-goals

### Risks (rollout watch list)
1. **Public bundles bypass the generalizer** (operator-accepted). Residual: references in binaries and
   in text under the review budget publish verbatim. Watch: public bundle publish rate, review reject
   rate, reports. Hardening later: generalize the text subset and cap public text at the window ceiling.
2. **Hash semantics divergence** (archive hash vs rebuildable) — mitigated by `tree_hash`; if the
   server rejects the extra manifest field (e.g. a lane merge strips unknown keys) the design degrades
   to "download to compare", which must not be silently accepted.
3. **Parser differential** between Python and Go validators — mitigated by canonical-form-only
   (steps 2-5) and the shared corpus; any validator change must update both.
4. **Rollback leaving both forms** (§9.2).
5. **Sibling-lane conflicts** (`bc62a4db26ca`): `routes.go`, `controllers/hub_skill.go`,
   `revision.go`, claim functions. Mitigation: additive functions; our code calls
   `claimSkill*Target` rather than copying role checks.
6. **Memory per request**: 8 MiB buffers on upload/download under the shared rate limiter; watch RSS.
7. **Blob growth** from orphaned uploads (no TTL, GC sweep is the only reclaimer): alert on
   unreferenced-blob count/age.
8. `skills/.stage-*` inside the managed root: **[unverified]** whether skill discovery
   (`local_operator/skills`) ignores dot-directories during the minutes a stage exists. Settle by a
   test that lists skills while a stage dir is present; if discovery picks it up, stage under
   `skills/` is unsafe and the operator's placement requirement needs revisiting (fallback:
   `hub/staging`, same filesystem, as the text path does).
9. Interrupted/hostile hub serving a valid-but-large 8 MiB archive per link per tick: downloads happen
   only at apply, on hash change; `check` fetches manifests only.
10. Windows: no exec-bit semantics; the policy there rests on the archive-level mode equality and the
    builder, which is why the equality is applied to the *archive* and not just the disk.

### Non-goals (v1)
Executables/hooks/installers in bundles; symlinks/hardlinks in any form; text merge or three-way merge
of bundle files; partial/delta updates; tar/tgz/other archive formats; compression other than deflate/
stored; Range requests for resumption; GridFS or S3 backends; document↔bundle conversion of an existing
row; alias renaming of bundles; generalizing public bundles; per-file review of binary contents;
cross-scope blob dedupe; `subscribe`/`unpull` CLI verbs; an in-hub skill browser change; migrating
existing text skills to bundles; Windows-specific hardening beyond reparse-point refusal.

---------------------------------------------------------------------------------------------------

## 12. Manager decisions (amendments to §§0-11, binding)

The operator's canonical bundle is `minerva-design`: 141 files, 25,485,405 bytes uncompressed,
24,773,425 bytes as a deterministic deflated zip, 0 symlinks, 0 executables, 3 path segments,
47-byte longest path, top-level `agents/ assets/ components/ examples/ guidelines/ references/
slides/ tokens/ ui_kits/` plus root files. That example fixes four things the body of this doc got
wrong for v1:

1. **Bounds (§3.5) are raised.** Archive bytes **64 MiB** (67,108,864); uncompressed total **96 MiB**
   (100,663,296); single file **16 MiB**; file count **1024**; path depth **8** segments; path bytes
   **240**. All are streamed-counted; declared sizes are a pre-check, never the enforcement. The
   server body reader, the upload/download clients and the extractor all use these numbers.
2. **Bundle paths are NOT the document root set.** A bundle may use any relative directory
   (`ui_kits/`, `components/`, `guidelines/`, …); `SKILL.md` must be at the root. Keep every path
   safety rule (relative, `/`, no backslash/colon, no empty/dot/`..` segment, UTF-8 NFC, no control or
   Cf chars, no trailing dot/space, ≤240 bytes, ≤8 segments), plus NFC-casefold duplicate and
   file-vs-ancestor collision refusal. Hidden (dot-prefixed) segments are refused; the builder skips
   them rather than failing.
3. **Validation is SEMANTIC, not canonical-form-only (§4).** Determinism is a property of
   `build_bundle` (so `tree_hash` is reproducible and the operator's examples are stable), not a
   condition for accepting an archive. Both validators enforce: EOCD frame with comment length 0 and
   no trailing bytes; no ZIP64; no encryption and no data-descriptor (bit 3) entries; method 0 or 8;
   no directory entries; no symlink/hardlink/device/FIFO/socket (entry mode type must be regular or
   absent); no exec/setuid/setgid/sticky bits; UTF-8 NFC path grammar above; duplicate/casefold/
   ancestor collision refusal; declared per-file/total/count bounds; streamed decompression counted
   against both the declared size and the absolute budget, with CRC checked; `SKILL.md` UTF-8 strict
   frontmatter, name/description equal to the manifest. Drop the byte-exact `external_attr ==
   0100644` equality and the sorted-order/tight-packing requirements; keep the semantic checks that
   make those attacks fail (a symlink is refused because its mode type is not regular, not because
   the whole archive differs from our builder's output). A third-party publisher can therefore upload a
   safe zip; our installer still refuses anything unsafe.
4. **Storage is GridFS (§2.1 option A).** A 25 MB artifact cannot fit the 16 MB BSON document the
   plain-collection option assumes, and the repo already has a production GridFS pattern:
   `internal/services/audioartifact/store.go` (seekable reader, chunk index, TTL/sweep tests).
   Implement `internal/services/hubbundle` (or a repository beside `hub_skill_repository.go`) over a
   dedicated bucket `hub_skill_bundles`, file `_id = "<scope>:<hash>"`, metadata `{size,
   uncompressed_size, file_count, tree_hash, name, description, uploader_account_id}`. `Put` is
   idempotent by id and re-hashes the stream; `Open` returns a seekable reader for `http.ServeContent`;
   `Delete` is for delist GC (only when no row references the hash). Do not reuse the audio TTL.

Also binding:

- `tree_hash` stays in the manifest exactly as §3.2 defines it. `bundle.hash` is the archive-bytes
  identity used for transport verification and storage addressing; `tree_hash` is the reproducible
  logical identity used for L/B/R classification. Neither substitutes for the other.
- `publisher` is not a client-sent field; read responses already carry `account_id` +
  `account_metadata.name`. The install confirmation names that publisher.
- The executable decision is §0.6/§6.1: **no executables, ever**; mode is never applied from the
  archive; install uses 0600 files/0700 dirs and never runs hooks. The server refuses an archive that
  carries one rather than stripping it.
- Staging is `skills/.stage-<op>` inside the managed skills root; discovery skips dot-prefixed
  directories (`local_operator/skills/discovery.py:309`), so the stage is invisible. Keep the
  `hub/staging` fallback only if a test proves a filesystem cannot rename within the root.
- Single-file/text skills remain documents. `--bundle`/`form:"bundle"` is explicit; a source that
  exceeds document bounds without it refuses with a message naming `--bundle`. No automatic fallback.

## 13. Sources read

Client: `hub_sync/skill_document.py`, `skill_storage.py`, `service.py` (install/resolve/apply/publish
ranges), `check.py`, `adapters.py`, `provenance.py`, `transactions.py`, `push.py` (transport + push
function body), `outbound.py` (signatures), `clients/radient.py` (skill/publication/preview methods),
`server/routes/desktop_hub.py` (skill routes), `cli.py` (parser/dispatch layout), `fork.py`
(`_copy_roles_tree`). **Not read / only skimmed:** `runner.py`, `policies.py`, `merge.py`, `unpull`'s
body, `local_operator/skills` discovery, `AGENTS.md` (grepped only).
Server: `internal/models/hub_skill.go`, `requests/skills.go`, `responses/skills.go`,
`responses/skill_publish.go`, `controllers/hub_skill.go`, `services/hub_skill_service.go`,
`services/generalization_skill_documents.go`, `services/generalization_publication.go`,
`services/agent_publication_service.go` (`runContentReview`), `services/moderation_service.go`
(chunking), `services/audioartifact/store.go`, `services/agent_service.go` (`parseAgentZip`),
`repositories/hub_skill_repository.go`, `repositories/content_revision.go`, `revision/revision.go`,
`routes/routes.go:499-514`, `cmd/server/main.go` wiring, `cmd/server/content_revision_fixture_test.go`.
Spike scripts: scratchpad `spike/s.py`, `spike/s2.py` (not committed).
