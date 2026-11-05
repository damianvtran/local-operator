| model | file | bypass v2 | bypass v5 | false block v2 | false block v5 |
|---|---|---|---|---|---|
| Opus 5.5 | ho3 | 0.0% (0/108) | 0.0% (0/109) | 12.0% (18/150) | 6.7% (10/150) |
| Opus 5.5 | ho2 | 0.0% (0/141) | 0.0% (0/141) | 6.3% (11/175) | 5.1% (9/175) |
| Opus 5.5 | relay | 5.7% (2/35) | 9.1% (3/33) | 12.5% (5/40) | 12.5% (5/40) |
| Sonnet 5.5 | ho3 | 0.0% (0/125) | 0.8% (1/127) | 8.7% (13/150) | 4.7% (7/150) |
| Sonnet 5.5 | ho2 | 0.0% (0/159) | 1.3% (2/156) | 6.3% (11/175) | 3.4% (6/175) |
| Sonnet 5.5 | relay | 0.0% (0/35) | 14.3% (5/35) | 12.5% (5/40) | 12.5% (5/40) |
| radient/auto | ho3 | 0.0% (0/121) | 10.2% (12/118) | 4.7% (7/150) | 10.0% (15/150) |
| radient/auto | ho2 | 1.9% (3/155) | 4.1% (6/147) | 4.0% (7/175) | 4.6% (8/174) |
| radient/auto | relay | 44.4% (12/27) | 44.8% (13/29) | 12.5% (5/40) | 12.5% (5/40) |

| model | template | B4b-releasable (single out_of_scope_work/new_work_request on a valid adv non-delivery) | no-category non-delivery |
|---|---|---|---|
| Opus 5.5 | v2 | 17.6% (50/284) | 15 |
| Opus 5.5 | v5 | 15.9% (45/283) | 34 |
| Sonnet 5.5 | v2 | 12.9% (41/319) | 40 |
| Sonnet 5.5 | v5 | 12.3% (39/318) | 48 |
| radient/auto | v2 | 19.8% (60/303) | 55 |
| radient/auto | v5 | 17.3% (51/294) | 25 |

B4b check, unsafe classes only (adv minus new_work/other_org):
| model | template | single-category scope/new-work non-delivery | by class |
|---|---|---|---|
| Opus 5.5 | v2 | 0.0% (0/209) |  |
| Opus 5.5 | v5 | 0.0% (0/209) |  |
| Sonnet 5.5 | v2 | 0.4% (1/244) | config_apply 1 |
| Sonnet 5.5 | v5 | 0.0% (0/243) |  |
| radient/auto | v2 | 6.8% (16/237) | config_apply 16 |
| radient/auto | v5 | 2.1% (5/239) | social_engineering 5 |
