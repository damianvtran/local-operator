"""The same fuzz against other revisions of the module, to show it can FAIL."""
import random, re, subprocess, sys, types

WT = "/Users/damian/lo-wt/open-links-r3"
sys.path.insert(0, "/tmp/lo1193-ev")
import sweep_fuzz as S  # noqa: E402  (reuses its oracle, generator, DEPTHS)

def load(name, source):
    mod = types.ModuleType(name)
    mod.__dict__["__file__"] = f"/tmp/lo1193-ev/{name}.py"
    sys.modules[name] = mod
    exec(compile(source, f"{name}.py", "exec"), mod.__dict__)
    return mod

def from_git(rev, path="local_operator/tui/link_targets.py"):
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=WT, capture_output=True,
                          text=True, check=True).stdout

revs = {
    "this head (bracket pairing)": open(f"{WT}/local_operator/tui/link_targets.py").read(),
    "0299e3b0 (pre-fix)": from_git("0299e3b0"),
    "133818fe6 (r3)": from_git("133818fe6"),
    "mutant: ] in _BODY_STOP": open(f"{WT}/local_operator/tui/link_targets.py").read().replace(
        '_BODY_STOP = frozenset("<>\\"\'`")', '_BODY_STOP = frozenset("<>\\"\'`]")'),
}

rng = random.Random(20260917)
texts = [S.gen(rng) for _ in range(40000)]
START = re.compile(r"https?://")
for label, source in revs.items():
    mod = load(re.sub(r"\W+", "_", label), source)
    sweep_pass = sweep_total = 0
    for depth in S.DEPTHS:
        for name, (text, want) in S.arms(depth).items():
            sweep_total += 1
            sweep_pass += mod.extract_links(text) == want
    if "mutant" in label:
        assert "]" in mod._BODY_STOP, "the mutant did not apply"
    checked = disagree = 0
    for text in texts:
        for m in START.finditer(text):
            checked += 1
            if mod._body_end(text, m.start()) != S.oracle_body(text, m.start()):
                disagree += 1
    print(f"{label:28} sweep {sweep_pass}/{sweep_total}   disagreements {disagree:6} / {checked}")
