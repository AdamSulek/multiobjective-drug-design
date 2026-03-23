# Methods: Logic and Runtime Guide

## Expected Runtime Order (Current Code, No Top-2k Shortlist)

| Case | Fastest -> Slowest (typical) | Why |
|---|---|---|
| 2D | `Random` -> `UCB(k=0)/Exploitation` -> `UCB(k>0)` -> `EllipseDirections` -> `FastEllipse` | MC-dropout cost appears for uncertainty methods; boundary-based ellipse methods evaluate more candidate geometry. |
| 3D | `Random` -> `UCB(k=0)` -> `UCB(k>0)` -> `EllipseDirections` -> `FastEllipse` | 3D scoring is heavier; `FastEllipse` uses many boundary points + expensive 3D delta-HV batching. |

## Main Cost Drivers

| Method | Scoring Principle | Dominant Cost in Practice |
|---|---|---|
| Random | random score | model predict only (no MC uncertainty) |
| UCB (2D) | delta-HV of optimistic point `mu + k*std` | MC-dropout over full unlabeled pool (for `k>0`) |
| UCB (3D) | optimistic delta-HV with dominance screening | MC-dropout + 3D scoring (`score_point`) on candidates |
| FastEllipse (2D/3D) | best delta-HV on k-sigma boundary | full-cov path + many boundary evaluations |
| EllipseDirections (2D/3D) | directional projection objective (not direct HV) | full-cov path + matrix ops over directions |

## Important Logic Notes

1. `EllipseDirections` is not a direct HV optimizer. It uses directional projections (with optional front penalty).
2. In current 3D `UCB`, when `clip_negative_hv=False`, dominated points still effectively stay at `0` (not strictly negative), unlike 2D behavior.
3. There is no old `top-2k` shortlist path in the active `loop.py` pipeline.

## Where To Look in Logs

- Per-iteration timings: `[TIME] ... it ...`
- Per-strategy summary: `[TIME_SUMMARY] ...`
- End-of-run global summary: `[TIME_SUMMARY_GLOBAL] ...`
- End-of-run result summary: `[RESULT_REP] ...`, `[RESULT_K] ...`

