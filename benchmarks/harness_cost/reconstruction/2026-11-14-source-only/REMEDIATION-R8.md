# Agent review remediation — R8, source-only integrated root repair

Base: d9c5784bce6e00dd95de4d0bdf5321450ad2d8ec.
Round 7 reviewer f2df8b75a5f5 marked R6/R7 FIXED and found MAJOR R8.
R8 is fixed in this unified commit; exact remediation SHA is in the parent
handoff. No finding is rejected/deferred. Review scope: base..new head.
NOPULL remains the current disposition; no fresh runtime slot/GO is inferred.

## Reproduction and fix

Actual NativeOps.begin_resolver constructs an isolated CLI HOME and writes the
actual supervisor source. The prior supervisor then derived campaign ROOT from
that HOME, duplicating benchmark-runs under lifecycle/.../docker-home. The RED
runs the emitted supervisor's actual ROOT/admission/artifact-read AST fragment
under the actual adapter launch environment: FileNotFoundError despite the
registered source-only fixture admission being present at the correct root.

The adapter now passes BENCH_REGISTERED_ROOT explicitly, retaining the same
isolated CLI HOME, anonymous auth configuration and whitelisted environment.
validated_registered_root checks it against the independent native account home
(pwd.getpwuid(os.getuid()).pw_dir), NOT another caller-controlled HOME variable.
It requires the exact registered campaign suffix, an absolute existing canonical
directory, no parent traversal/symlink escape and operator ownership. The adapter
validates before preflight work and again before supervisor launch; the supervisor
validates the explicit variable before reading admission or artifacts. Missing or
invalid roots have no fallback to CLI HOME or an alternate campaign directory.

## Targeted integration evidence

* R8-RED.txt: original actual-adapter environment -> emitted supervisor read
  fragment fails at the duplicated isolated-HOME admission path.
* R8-GREEN.txt: 29 tests pass across test_registered_root.py and the three direct
  safety/auth/lifecycle regression files. Tests exercise the actual adapter,
  captured launch environment and emitted supervisor ROOT/admission/lifecycle/W
  fragment, not a reimplemented root-selection algorithm.
* New negatives reject missing/wrong/relative/parent-traversal roots and a
  symlink escape. The adapter rejects the latter before any process launch.
* Account-home lookup is mocked to the test's owned canonical scratch directory;
  admission/startup records explicitly say source-only-fixture and are NOT a GO.
  The artifact fixture is inert text named source-fixture.txt, never a wheel.
* Existing anonymous-helper negative/positive sentinel and bounded cleanup tests
  retain R6/R7 coverage. Source-only fragments are NOT a full supervisor run or
  native Docker/credential/cap/cleanup qualification.

No Desktop, Docker CLI, actual credential helper/keychain, network runtime,
install, build, resolver manager or model was invoked. The registered HOLD wheel
and its validated staging receipt are preserved; no artifact was acquired again.
Source receipts remain append-only. No campaign freeze, old outcome recovery,
optional-pull signoff or runtime-success claim is made.
