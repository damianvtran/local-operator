# D3/D4 before -> after (WCAG, computed over themes.generated.css)

| theme | pair | before | after |
| --- | --- | --- | --- |
| gruvboxLight | ink-dim on danger-wash | 4.336 | 4.596 |
| gruvboxLight | ink-dim on accent-wash | 4.328 | 4.588 |
| ayuLight | ink-dim on accent-wash | 4.413 | 4.555 |
| everforestLight | ink-dim on danger-wash | 4.497 | 4.567 |
| dracula | accent on elevated | 4.220 | 4.559 |

before = origin/main themes.generated.css, after = branch (+ `pnpm check-themes`:
"Contrast contract holds: 1209 assertions across 31 themes", 0 failing pairs;
the shipper's own script over the same file: "31 themes; 0 failing pairs").
