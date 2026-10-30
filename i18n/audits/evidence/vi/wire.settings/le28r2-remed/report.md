# vi / wire.settings — remediation delta audit (le28r2-remed)

Remediation commit `4d38a1d4b8e9a36fa0a38f4833ecf86263da45db` (parent `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`). Verdict: **PASS** for the 3 corrected key(s).

Scope is the delta only: the unchanged keys keep the `le28r2` judgement; this round does not re-audit them.

## Keys re-audited

### `wire.settings.web_fetch_render_backend.help` — fixed

Now: auto dùng markdownify khi extra [fetch] được cài.

pip extra stays the literal `extra`, as the glossary requires for all locales.

### `wire.settings.speech_transcription_cleanup.help` — fixed

Now: Cách nhập bằng giọng nói loại bỏ từ đệm và âm thanh phi ngôn ngữ. auto dùng tính năng làm sạch tích hợp sẵn của nhà cung cấp khi có và quy tắc thận trọng phía máy khách trong các trường hợp còn lại; provider và client mỗi bên chỉ dùng một trong hai cơ chế; off giữ nguyên bản ghi. Trường `cleanup` trong từng yêu cầu vẫn được ưu tiên.

filler words → «từ đệm» (not the medial-glide term «âm đệm»); rest of the sentence unchanged.

### `wire.settings.speech_transcription_cleanup.choice.off.description` — fixed

Now: Giữ nguyên; không bỏ từ đệm hay âm thanh phi ngôn ngữ.

Same term fix; now also says non-speech sounds rather than the vague «tín hiệu».

## Checks

- Key set still equals the English source key set; ICU argument parity checked by `scripts/i18n/check.py --catalogues-only` on this tree.
- Routed `read_public_copy` receipt for the corrected catalogue at `4d38a1d4b8e9a36fa0a38f4833ecf86263da45db` (session `c754bf63ab2a`, 2026-10-29T00:24:38Z); its digest was recomputed from the commit blob and matches.
- No English source, context sidecar, key module or generated artifact changed in the remediation commit.
