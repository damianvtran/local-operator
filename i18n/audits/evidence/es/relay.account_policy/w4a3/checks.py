#!/usr/bin/env python3
"""Read-only reproduction of the round-2 checks for es audit w4a3 / relay.account_policy.

Writes nothing. Run:  python3 checks.py
Recomputes every input sha256, re-derives the remediation delta and, for
relay.library, renders the ICU message with the repository's own runtime.
Exit 0 = every check passed.
"""
import json, re, sys, hashlib, pathlib

here = pathlib.Path(__file__).resolve()
WT = here
while not (WT / "local_operator" / "i18n").is_dir():
    if WT.parent == WT:
        sys.exit("could not locate the repository root above " + str(here))
    WT = WT.parent
sys.path.insert(0, str(WT))
CAT = WT / "local_operator/i18n/catalogues"
NS = "relay.account_policy"
WANT = {"en": "0019b08106e6f7762194b164b9b9b37054a8cfade759aa2f6490787a73dea890", "es": "67dbbf6a3bc75058b38f40a076de852c86c7d58004e938b43019513280e2a407", "context": "7e55e4fadebcde0a6b93d545a62b2743fbb16009498cdf4e2122cb309ecc07f4"}

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

if NS == "relay.account_policy":
    SUPERSEDED = {"relay.account_policy.title": "Ajustes del equipo", "relay.account_policy.back": "Volver", "relay.account_policy.loading": "Cargando el enrutamiento de cuentas…", "relay.account_policy.save": "Guardar política", "relay.account_policy.saving": "Guardando…", "relay.account_policy.saved": "Guardado en este equipo.", "relay.account_policy.retry": "Reintentar", "relay.account_policy.unavailable": "Los ajustes de enrutamiento de cuentas no están disponibles en este relay. Actualiza Local Operator en este equipo.", "relay.account_policy.error": "No se pudo confirmar el cambio. Comprueba la conexión y recarga para ver la política actual.", "relay.account_policy.scope": "Los cambios solo se aplican a este equipo.", "relay.account_policy.entry": "ajustes", "relay.account_policy.rejected": "No se guardó el cambio. Comprueba el valor o los ajustes de este equipo y vuelve a intentarlo.", "relay.account_policy.load_error": "No se pudo cargar el enrutamiento de cuentas. Comprueba la conexión y vuelve a intentarlo."}
    CLAIMED = ["relay.account_policy.saved", "relay.account_policy.scope", "relay.account_policy.title", "relay.account_policy.unavailable", "relay.account_policy.rejected"]
    changed = sorted(k for k in es if k in SUPERSEDED and es[k] != SUPERSEDED[k])
    check("remediation delta is exactly the five claimed keys", changed == sorted(CLAIMED),
          "changed=%s" % [k.split(".")[-1] for k in changed])
    for k in sorted(CLAIMED):
        check("remediated value " + k.split(".")[-1], "máquina" in es[k], es[k])
    check("no 'equipo' left in the namespace",
          not re.search(r"equipo", json.dumps(es, ensure_ascii=False), re.I))
    team = {}
    for p in sorted(CAT.glob("es/*.json")):
        d = json.loads(p.read_text())
        eq = [k for k, v in d.items() if isinstance(v, str) and re.search(r"\bequipos?\b", v, re.I)]
        if eq:
            team[p.stem] = eq
    check("team sense survives: cli.teams_help=26, tui.team_authoring=14",
          len(team.get("cli.teams_help", [])) == 26 and len(team.get("tui.team_authoring", [])) == 14,
          str({k: len(v) for k, v in sorted(team.items())}))
    gained = []
    for n in ("cli.teams_help", "tui.team_authoring", "tui.palette"):
        d = json.loads((CAT/"es"/(n + ".json")).read_text())
        if re.search(r"máquina", json.dumps(d, ensure_ascii=False)):
            gained.append(n)
    check("no team namespace gained 'máquina'", not gained, str(gained))
    check("error still states UNCONFIRMED, not failed",
          "no se pudo confirmar" in es["relay.account_policy.error"].lower())
    check("rejected still states not-saved",
          "no se guardó" in es["relay.account_policy.rejected"].lower())
    check("AP-02 known-open: 'relay' still English", bool(re.search(r"\brelay\b", es["relay.account_policy.unavailable"])),
          "expected True at this round - the minor is not fixed and is reported open")

if NS == "relay.library":
    from local_operator.i18n import runtime, format as ifmt
    src = es["relay.library.count"]
    expected = {0: "0 recursos", 1: "1 recurso", 2: "2 recursos", 21: "21 recursos",
                1000000: "1.000.000 de recursos", 1500000: "1.500.000 recursos"}
    for n in sorted(expected):
        got = runtime.render_message(src, {"count": n}, "es")
        check("render %d" % n, got == expected[n],
              "%r (category %s)" % (got, ifmt.plural_category("es", n)))
    old = "{count, plural, one {# recurso} other {# recursos}}"
    check("control: superseded string rendered WRONG at 1000000",
          runtime.render_message(old, {"count": 1000000}, "es") == "1.000.000 recursos",
          runtime.render_message(old, {"count": 1000000}, "es"))
    check("control: old and new agree at a non-multiple of 10^6",
          runtime.render_message(old, {"count": 1500000}, "es") == runtime.render_message(src, {"count": 1500000}, "es"))
    check("`count` argument preserved", "{count," in src.replace(" ", ""))
    check("no forbidden category",
          not set(re.findall(r"(zero|one|two|few|many|other)\s*\{", src)) & {"zero", "two", "few"})
    total = withmany = 0
    for p in sorted(CAT.glob("es/*.json")):
        for k, v in json.loads(p.read_text()).items():
            if isinstance(v, str) and "plural," in v:
                total += 1
                if re.search(r"many\s*\{", v):
                    withmany += 1
    check("every es plural message carries `many`", total == withmany, "%d/%d" % (withmany, total))

check("no formal address", not any(re.search(r"\b(usted|ustedes|vosotros)\b", v) for v in es.values()))
print()
print("RESULT:", "ALL CHECKS PASS" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
