# First independent Hindi audit — wire.teams.effective

Date: 2026-11-19 · Revision: e70bc818b8553dca1fad007705257817be49ee37
Auditor: i18n-auditor / session 1a6e70595b8d / actual openai/gpt-6.1-sol (host Model field; no observed fallback). Independent from translator session 47e37aebb87c.

**Verdict: passed for this first textual/semantic audit, 9/9 keys; 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT.** This is not programme completion: the second independent high-visibility audit and client rendered validation remain outstanding.

## Scope and execution
Read Hindi style guide and shared glossary, then frozen English source and sidecar, then all nine candidate values. No MT/translation service used; no target edits, generation, build, install, PR or ledger mutation. Repository validator scripts.i18n.check.check_catalogues ran against a scratch-only isolated en/hi catalogue root: zero problems. A deliberately renamed role placeholder was rejected, then restored control passed. Independent strict JSON duplicate/key checks, exact placeholder names/types/order, technical literal checks and control-character checks also passed. audit.log is actual execution output; this report is audit_notes_not_execution_log.

There are no plural/select constructs in these nine source or target messages; Hindi CLDR one/other arms are not required. No Rich tags, backtick code, commands or flags appear. No unexcepted Latin prose remains: subagents.models., provider/model, model and effort are technical syntax, while role/tier/selector occur only as opaque ICU parameter identifiers. Danda prose endings and polite Hindi imperatives are consistent.

## Per-key audit

### wire.teams.effective.team_missing — PASS
Current: {role} नाम की कोई टीम नहीं मिली।
Assessment: Preserves absence of a team, not absence of an individual agent. Opaque {role} naturally precedes नाम की; no grammatical assumptions about the value.
Finding / suggested replacement: none.

### wire.teams.effective.registry_unreadable — PASS
Current: एजेंट रजिस्ट्री पढ़ी नहीं जा सकी। एजेंट की परिभाषाएँ जाँचें और फिर से कोशिश करें।
Assessment: रजिस्ट्री is a natural developer loan; पढ़ी नहीं जा सकी preserves inability to read without diagnosing corruption. परिभाषाएँ and polite जाँचें/कोशिश करें preserve both remedy steps.
Finding / suggested replacement: none.

### wire.teams.effective.config_unreadable — PASS
Current: मॉडल की सेटिंग पढ़ी नहीं जा सकीं। अपनी सेटिंग जाँचें और फिर से कोशिश करें।
Assessment: Model settings could not be read, rather than declared invalid or missing. सेटिंग is an established Hindi loan; plural agreement सकीं works with the collective settings reading. Both remedy steps retained.
Finding / suggested replacement: none.

### wire.teams.effective.tier_invalid_value — PASS
Current: subagents.models.{tier} का मान अमान्य है। इसे provider/model के रूप में या मॉडल मैपिंग के तौर पर सेट करें।
Assessment: अमान्य marks invalidity at the literal config path. The alternatives provider/model form OR a model mapping remain distinct; मैपिंग denotes the configuration object, not an extra executable command.
Finding / suggested replacement: none.

### wire.teams.effective.tier_invalid_mapping — PASS
Current: subagents.models.{tier} की मैपिंग अमान्य है। model में provider/model के रूप में मान दें और, चाहें तो, effort में समर्थित तर्क-प्रयास का स्तर दें।
Assessment: The required instruction model में ... मान दें precedes और, चाहें तो, effort ...: only effort is optional. Supported reasoning effort is qualified by समर्थित and expressed as a level; literal field names retain exact spelling. Hindi postpositions attach outside the Latin identifiers.
Finding / suggested replacement: none.

### wire.teams.effective.tier_unconfigured — PASS
Current: subagents.models.{tier} में कोई मॉडल सेट नहीं है।
Assessment: कोई मॉडल सेट नहीं है preserves no model configured at the given path; does not conflate this state with an invalid model or a launch failure.
Finding / suggested replacement: none.

### wire.teams.effective.tier_invalid_selector — PASS
Current: subagents.models.{tier} में मॉडल चुनने के लिए दिया गया मान अमान्य है। provider/model के रूप में मान दें।
Assessment: मॉडल चुनने के लिए दिया गया मान is an idiomatic explanatory rendering of selector. The invalid raw value is not exposed, and the literal provider/model remedy remains exact.
Finding / suggested replacement: none.

### wire.teams.effective.model_unresolvable — PASS
Current: मॉडल {selector} पहचाना नहीं जा सका। मॉडल चुनने के लिए दिए गए मान और प्रोवाइडर की सेटिंग जाँचें।
Assessment: पहचाना नहीं जा सका is a reasonable user-facing rendering of could not be resolved in model lookup context, without asserting a specific missing provider. Both selector and provider configuration checks remain explicit; opaque {selector} is untouched.
Finding / suggested replacement: none.

### wire.teams.effective.unavailable — PASS
Current: टीम का यह सदस्य उपलब्ध नहीं है। इसके मॉडल की सेटिंग और एजेंट की परिभाषा जाँचें।
Assessment: टीम का यह सदस्य उपलब्ध नहीं है preserves member unavailability without implying launch. इसके refers naturally to this member; both model configuration and agent definition remain in the remedy.
Finding / suggested replacement: none.

## Fit and evidence limits
The sidecar explicitly allows natural wrapping and has no fixed cell-width hint; no maxCells overflow is established. Devanagari conjunct/combining-mark width and actual warning wrapping still require rendered client evidence. No screenshot, pseudo-localization, fallback execution, runtime envelope/formatter flow, live resolver/config mapping execution or visual validation was run. Hindi is LTR; RTL is not applicable to this locale, and no RTL/bidi runtime validation is claimed. Model/effort semantics were inspected textually against English and the sidecar, not exercised through configuration.

## Retention and admission
This task explicitly authorizes scratch-only output. These are staging artifacts, not tracked evidence and not on main. Manager must preserve this exact bundle under i18n/audits/evidence/hi/wire.teams.effective/h1-1a6e/ before appending the supplied receipt. The note references that intended canonical path; no append command was executed. Receipt object schema admission is checked with the repository pure parser, not a ledger mutation.
