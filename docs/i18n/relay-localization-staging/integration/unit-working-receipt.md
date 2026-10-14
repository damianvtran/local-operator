# Working-only three counted unit correction — 2026-10-14
Latest parent disposition overrides earlier commit/push instruction: NO COMMIT/PUSH until all fifteen locale files and audited ledger batch supplied; one unified wave. Local and remote HEAD remain de9b9a490c839d6de2c2e41ea03ec819f66be489. No locale integration yet.

Exactly3 existing EN values:
- relay.shell.usage.unit_credits: {count, plural, one {credit} other {credits}}
- relay.shell.usage.unit_tokens: {count, plural, one {token} other {tokens}}
- relay.shell.usage.unit_requests: {count, plural, one {request} other {requests}}
Typed count:number (Python int|float), kindplural. NO outerplural remaining/used change. Unit_usd invariant/noarg; all other39ENshellvalues retained exactly, session/login raw source byteidentical. 44shell+17session+4login=65keyinventory preserved.

Final working EN shell API/raw SHA256 f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560; session a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129; login 9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5. API hashes raw bytes, not normalizedJSON. ENshellcontext rawSHA c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1.

Context3notes: actualfinite numericcount, selectedcatalogue CLDR forzero/one/two/fraction/negative/million, mandatoryother; nounonly and coordinate entireamountgrammar avoiding doublede inFR/ESmillions. Native auditors own forms; synthetic categorymarkers do not certify translations.

Caller: measurednumericvalue first; finiteguard before countedunit rendering; passactualmeasuredcount to3unitmessages. USD/percent/unknown/nonfinite/fraction/date/row code semantics retained. Ownscripts/i18n/generate.py only; no handeditedgenerated artifacts. Pyproject/ledger/shipped unchanged. Sevenworkingfiles +87/-24 (EN/context,3generated artifacts,caller,format tests).

Execution: env-i isolatedHOME/config/TMPDIR stripsCMUX/LOP/XPC; Python-B/no-bytecode directprimaryinterpreter PYTHONPATHworkingtree; TZUTCfixtures, existingdeps only. Narrowserial60s/512MiB, no browser/heavy/fullsuite/type retry.
- Before defect control de9caller+bundle:singularEnglishunit test FAIL1/12skipped, actual1credits instead1credit; ownchangedfiles restoredfinally.
- After working13format tests +8usage+4shell+7store+3runtime:35PASS/5files in3.85s. Exact0/1/2/1.5/-1/1e6 acrossEN/FR/ES/RU/VI/HI, threeunitnouns. Categorymarkers synthetic; byteexact NBSP/narrowNBSP expected withnormalizeridentity. Unknownunits/nonfinite/percent/USD untouchedcases asserted; existingdecimal/date/fallback tests retain.
- First newfixture run2fails FR/RUmillion:TestingLibrary whitespacecollapse normalized actualNBSP groups while expected retainedbytes. ActualDOM had correctcategory+grouping. Fixedfixture matcher to identitynormalizer, not product/source/output; reran35PASS.
- Owngenerator--check32artifacts byteidentical.
- check.py--catalogues-only0problems.
- test_generate/test_check/test_catalogues54PASS11.78s. No packaging/fullsuite rerun forworkingunitdelta; CIterminalafterunifiedcommit.
- A chainedgenerator command initiallycdoneparenttoo-far causedscriptpathsetupfailure; reranabsoluteworktreecommand PASS, notproductfinding.
- gitdiff--checkclean; protectedartifacts unchanged; oldpreservation/row/format proof artifacts retainedhistorical.

Working changes ready for parent-supplied audited15files/ledger mechanicalintegration+generator ONEcommit/push. ActualFR/ESmillion composed 'de' proof remains owed against finalnativecandidates, not replaced bycategorymarkers. Independent review/QA/renderedStageA andfutureheadCI remain separately required; no selfsignoff/merge/release.
