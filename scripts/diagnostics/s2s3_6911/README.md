# S2/S3 bounded diagnostic completion — SOURCE ONLY

Four serial observations: unchanged originalS2+phaseCPU/wall+M15/threadGC; originalS3+one600msASYNCpause/rightold500wallFAIL; actualTESTdependencySlowListingworkerheldwhileLOOPticks3thenexplicitrelease; identical200msCPUburnsync/viaworker with strict50CPU/2000wall and observedREADY+trailingcapture. No releaserthread, testedit, productlock, calibration/loadedbar change. WorkcompletedEvent is notThread.join; childwait/reap supplies physicalsettlement.

Textual8.2.8/Python3.13.16 are DIAGNOSTICconstraints from manager-verified oldfailingCI39896435792/job120227971573, not canonicalrepo pins or fullhistoricalequivalence. Real loadedoriginal/observer/plugin paths+hashes and executable identity are measured beforetimedwork forALL4. Closureother versions recorded; native extensions/Appimports OFFHOSTonly.

No raw pytestoutput artifacts/forwarding. Structuralallowlist rejectsunknown fields and projects timerpathlabels; JSONonly. Childhealthywait precedes kill; nested mandatoryfinalwait surviveskill/observationfault; rootdeleteonlysettled+projectedproof. Producer/oracle sharedrecordbuilders; initial/background spans excludedfrommeasured3pairs. Observedwindow/residual NOTexactoriginalelapsed.

S2 originalM15summary inclGC preserved; independentGC callbacks recordthreadCPU/wall/generation/is-loop orUNKNOWN. Extra S3M15 NOTCOLLECTED/optional. ExpectedoriginalassertionFAILretained in finalFAIL; unknown/setup/hash/privacy/signal/bounds/lifecyclefailureSTOP. Sourceonlytests do not prove App behavior/650cause or distribution.

Unchangedbounds: child20+3reap(derivedfrom10.3srun_boundedwallofthewholefirstdriverrun+5sSlowListing+3sreap; conservativeceiling that may double-count the fixture, not purecaselatency)/4serial/outer100+5(4x23=92)/1024MiB/setup120s/job5m/output256KiBcase. Triggeraudit/currentrefs/source/closure preflight plus BOTHindependentdeltaACKs and explicitONEoffhostGO required beforeownedbranchpublication. No localpytest/App/native/install/WT execution. No physicalallocation or productfix claim.

## First physical result and capacity recovery

QA report s2s3-actual-39921114526.md: firstS2rc1/reapedtrue/noSignal/noTimeout/identitypass; exactly250records(244supplementalGC), missingmandatorysummary/phases/cleanup. Later3NOTDISPATCHED; no natural650cause/workerleak inferred. This head removes supplementalGLOBALgc.callbacks entirely; startup/offthreadGC NOTCOLLECTED. REQUIRED originalM15 scopedwindows.gc/threadsummary unchanged.

Actualproducer uses sharedRecordBudget:250TOTAL includes234optional+14critical+scope+overflow;32k optionalbytes/128000bytes-summary/512bytes-othercritical/172000payloadbytesleaveboundedtrailerreserve insideprior180kproducer/200kJSON/256KiBcapture. No capraise. Out-of-scopeoldGCseamexcludedandexplicitlycounted; no globalcallbackregistered. Exhaustionneverraisesfromcollector; explicitboundedoverflow/dropped+criticaldrop countersFAILmeasurement/STOP. Scope/outcome mandatory; missingcleanupPROOF isnot observedunsettledwork nor unreapedchild. No originaltest/GCbehavior/threshold/calibration/productchange. Runtimeproof needsnewindependentdeltaACKs+managerallocation.

## Budget delta

Only numeric: child12→20s, outer60→100s (grace5, reap3 unchanged). job5min/setup120s/1024MiB/250records/256KiB/4cases/original CPU500,wall2000,noise,calibration and Native512/UI7.8 policies unchanged. No automatic further raise/retry.

## Held-control protocol repair (run 39960859806)

Observed: s2-original PASS (43/250 records), s3-split expected old-500 wall control, s3-held protocol FAIL, synthetic CPU case NOT RUN. Source-derived (not a logged cause): the held witness deadline began at the first pause although the refresh worker entered ~1.96 s later; the 5 s SlowListing outlived completed()'s 3 s wait; cleanup, not the loop, released the gate. Repair: witness tick window starts at worker entry (separate 6 s entry bound, fail-closed); only the loop-side witness (causal_released) may release the worker, cleanup release fails closed; the held control zeroes the fixture's own mutable latency_s after the causal release (as the original test's teardown does; original blob unedited); case order s2-original, s3-split, synthetic, s3-held; held protocol failures are labelled HELD PROTOCOL DEFECT, distinct from product observations. No bar, budget, cap or original-test change. Unverified until an actual run.
