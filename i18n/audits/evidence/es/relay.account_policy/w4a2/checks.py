#!/usr/bin/env python3
"""Read-only reproduction of the mechanical checks for es audit w4a2 / relay.account_policy.

Writes nothing. Run:  python3 checks.py
Recomputes every input sha256, so a later edit to any input invalidates the
binding recorded in record.json, then re-derives each structural claim the
record makes. Exit 0 = every check passed.
"""
import json, re, hashlib, pathlib, sys

here = pathlib.Path(__file__).resolve()
WT = here
while WT.name != "lo-i18n-es" and WT.parent != WT:
    WT = WT.parent
CAT = WT / "local_operator/i18n/catalogues"
NS = "relay.account_policy"
WANT = {"en": "0019b08106e6f7762194b164b9b9b37054a8cfade759aa2f6490787a73dea890", "es": "074787ece57a27646c766deb3b7691979c233259c1f05987c5c79770e8f31204", "context": "7e55e4fadebcde0a6b93d545a62b2743fbb16009498cdf4e2122cb309ecc07f4"}

fails = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ((" :: " + detail) if detail else ""))
    if not ok:
        fails.append(name)

en_p, es_p, ctx_p = CAT/"en"/(NS+".json"), CAT/"es"/(NS+".json"), CAT/"en"/(NS+".context.json")
en, es, ctx = (json.loads(p.read_text()) for p in (en_p, es_p, ctx_p))

for p, tag in ((en_p, "en"), (es_p, "es"), (ctx_p, "context")):
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    check("sha256 " + tag, got == WANT[tag], got)

check("key parity en/es", list(en) == list(es), "en=%d es=%d" % (len(en), len(es)))
check("no duplicate keys", len(es) == len(set(es)))
check("no 'redacted' literal in target bytes", b"redacted" not in es_p.read_bytes())

BR = re.compile(r"\{[^{}]*\}")
CATR = re.compile(r"(zero|one|two|few|many|other)\s*\{")
for k in en:
    a, b = en[k], es[k]
    pa, pb = BR.findall(a), BR.findall(b)
    check("placeholders " + k, (not pa) == (not pb) or pa == pb, "en=%s es=%s" % (pa, pb))
    ca, cb = CATR.findall(a), CATR.findall(b)
    check("icu categories " + k, ca == cb, "en=%s es=%s" % (ca, cb))
    check("no forbidden es category " + k, not set(cb) & {"zero", "two", "few"}, str(cb))

if NS == "relay.account_policy":
    shipped = json.loads((CAT/"es"/"wire.settings.account_policy.json").read_text())
    labels = [v for k, v in shipped.items() if k.endswith(".label")]
    hits = [(k, lab) for k, v in es.items() for lab in labels if lab in v]
    check("no second rendering of shipped account-policy labels", not hits, str(hits))
    stems = ["Reinicio", "margen", "Reparto", "equitativo", "heredada", "cuentas hermanas"]
    sh = [(k, s) for k, v in es.items() for s in stems if s.lower() in v.lower()]
    check("no policy-option vocabulary at all", not sh, str(sh))
    print("INFO  'computer' rendered as 'equipo' in:", [k for k, v in es.items() if re.search(r"\bequipos?\b", v, re.I)])
    print("INFO  English 'relay' kept in:", [k for k, v in es.items() if re.search(r"\brelay\b", v)])
    check("error states unconfirmed, not failed", "no se pudo confirmar" in es["relay.account_policy.error"].lower(), es["relay.account_policy.error"])
    check("rejected states not-saved", "no se guardó" in es["relay.account_policy.rejected"].lower(), es["relay.account_policy.rejected"])

if NS == "relay.library":
    uses_many = "many" in es["relay.library.count"]
    check("es plural carries `many` (locale convention 57/58)", uses_many,
          "present" if uses_many else "MISSING - exact 10^6 falls back to `other`")

check("no formal address", not any(re.search(r"\b(usted|ustedes|vosotros)\b", v) for v in es.values()))
check("sentence-final punctuation preserved", all(
    (v.rstrip().endswith((".", "?", "…")) == v_en.rstrip().endswith((".", "?", "…")))
    for v, v_en in ((es[k], en[k]) for k in en)))

print()
print("RESULT:", "ALL CHECKS PASS" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
