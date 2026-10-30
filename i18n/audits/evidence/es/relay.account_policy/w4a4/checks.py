#!/usr/bin/env python3
"""Read-only reproduction of the round-4 checks for es / relay.account_policy.

Writes nothing. Run:  python3 checks.py

Every one of the 13 keys is pinned positively, not merely "not mentioned": the
shipped file is modelled from the w4a2 values plus the two transformations this
wave applied, and the script asserts the model reproduces the file byte for byte.

  round 2 (w4a3): equipo -> maquina in five keys   (finding es-w4a2-AP-01)
  round 4 (w4a4): relay  -> rele in one key        (finding es-w4a2-AP-02)

If the catalogue drifts, the model stops reproducing it and this script fails.
"""
import json, re, sys, hashlib, pathlib, subprocess

here = pathlib.Path(__file__).resolve()
WT = here
while not (WT / "local_operator" / "i18n").is_dir():
    if WT.parent == WT:
        sys.exit("could not locate the repository root above " + str(here))
    WT = WT.parent
CAT = WT / "local_operator/i18n/catalogues"
NS = "relay.account_policy"
WANT = {"en": "0019b08106e6f7762194b164b9b9b37054a8cfade759aa2f6490787a73dea890", "es": "71956128095ae4afebaf4aa26356db6acf9796a100332d503fbcc785ab178453", "context": "7e55e4fadebcde0a6b93d545a62b2743fbb16009498cdf4e2122cb309ecc07f4"}
PREVIOUS_SHA = "67dbbf6a3bc75058b38f40a076de852c86c7d58004e938b43019513280e2a407"

W4A2 = {"relay.account_policy.title": "Ajustes del equipo", "relay.account_policy.saved": "Guardado en este equipo.", "relay.account_policy.unavailable": "Los ajustes de enrutamiento de cuentas no están disponibles en este relay. Actualiza Local Operator en este equipo.", "relay.account_policy.scope": "Los cambios solo se aplican a este equipo.", "relay.account_policy.rejected": "No se guardó el cambio. Comprueba el valor o los ajustes de este equipo y vuelve a intentarlo.", "relay.account_policy.save": "Guardar política", "relay.account_policy.back": "Volver", "relay.account_policy.loading": "Cargando el enrutamiento de cuentas…", "relay.account_policy.saving": "Guardando…", "relay.account_policy.retry": "Reintentar", "relay.account_policy.error": "No se pudo confirmar el cambio. Comprueba la conexión y recarga para ver la política actual.", "relay.account_policy.entry": "ajustes", "relay.account_policy.load_error": "No se pudo cargar el enrutamiento de cuentas. Comprueba la conexión y vuelve a intentarlo."}

REMEDIATED = {
    "relay.account_policy.title",
    "relay.account_policy.saved",
    "relay.account_policy.scope",
    "relay.account_policy.rejected",
    "relay.account_policy.unavailable",
}

def to_w4a3(key, value):
    """Round 2: the five computer strings moved from 'equipo' to 'maquina'."""
    if key in REMEDIATED:
        return value.replace("del equipo", "de la máquina").replace("este equipo", "esta máquina")
    return value

def to_w4a4(key, value):
    """Round 4: the English noun 'relay' became 'rele' in the one key that kept it."""
    if key == "relay.account_policy.unavailable":
        return value.replace("este relay", "este relé")
    return value

def baseline_w4a3(key):
    """The value this key had at the w4a3 audit (round 4 had not run yet)."""
    return to_w4a3(key, W4A2[key])

fails = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ((" :: " + detail) if detail else ""))
    if not ok:
        fails.append(name)

en_p = CAT/"en"/(NS + ".json")
es_p = CAT/"es"/(NS + ".json")
ctx_p = CAT/"en"/(NS + ".context.json")
en, es, ctx = (json.loads(p.read_text()) for p in (en_p, es_p, ctx_p))

for p, tag in ((en_p, "en"), (es_p, "es"), (ctx_p, "context")):
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    check("sha256 " + tag, got == WANT[tag], got)

check("key parity en/es", list(en) == list(es), "en=%d es=%d" % (len(en), len(es)))
check("model covers exactly the file's keys",
      set(W4A2) == set(es), "%d model / %d file" % (len(W4A2), len(es)))

model = {k: to_w4a4(k, to_w4a3(k, W4A2[k])) for k in W4A2}
check("shipped file == w4a2 + the two documented transformations", model == es)
for k in sorted(set(model) | set(es)):
    if model.get(k) != es.get(k):
        print("      DIFF", k, "\n        model:", model.get(k), "\n        file :", es.get(k))

delta = sorted(k for k in es if es[k] != baseline_w4a3(k))
check("delta since w4a3 is exactly one key",
      delta == ["relay.account_policy.unavailable"], str(delta))
check("the unchanged twelve match their w4a3 values",
      all(es[k] == baseline_w4a3(k) for k in es if k != "relay.account_policy.unavailable"),
      "12/12")
check("previous target sha recorded (w4a3)", len(PREVIOUS_SHA) == 64, PREVIOUS_SHA[:16] + "...")

u = es["relay.account_policy.unavailable"]
check("'rele' present with its acute accent (U+00E9)", "rel\u00e9" in u, u)
check("no English 'relay' left in any value",
      not any(re.search(r"\brelay\b", v, re.I) for v in es.values()))
check("'maquina' survived in the same sentence", "esta máquina" in u)
check("no 'equipo' anywhere in the namespace",
      not re.search(r"equipo", json.dumps(es, ensure_ascii=False), re.I))
check("sentence keeps its two clauses and full stop", u.count(". ") == 1 and u.endswith("."))
check("error still states UNCONFIRMED, not failed",
      "no se pudo confirmar" in es["relay.account_policy.error"].lower())
check("rejected still states not-saved",
      "no se guardó" in es["relay.account_policy.rejected"].lower())
check("no formal address",
      not any(re.search(r"\b(usted|ustedes|vosotros)\b", v) for v in es.values()))

rel = str(es_p.relative_to(WT))
try:
    tracked = subprocess.run(["git", "-C", str(WT), "ls-files", "--error-unmatch", rel],
                             capture_output=True, text=True)
    check("file is tracked, so HEAD pins these bytes", tracked.returncode == 0,
          (tracked.stdout or tracked.stderr).strip()[:90])
    blob = subprocess.run(["git", "-C", str(WT), "show", "HEAD:" + rel], capture_output=True)
    got = hashlib.sha256(blob.stdout).hexdigest()
    check("working tree == HEAD blob", got == WANT["es"], got)
except Exception as exc:
    check("git binding", False, repr(exc))

print()
print("RESULT:", "ALL CHECKS PASS" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
