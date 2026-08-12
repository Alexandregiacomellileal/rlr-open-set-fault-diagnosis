# Experimental protocol

## Primary evaluation

The first complete day of each station is reserved exclusively for local robust-reference statistics. Days 2--7 are used for source model development or external testing. Reference/evaluation overlap is zero.

The primary benchmark uses 13 fault types with sufficient event support. Each outer evaluation holds out one station and one fault type, producing 3 x 13 = 39 matched external open-set scenarios.

The RLR transform uses the Day-1 median as location and a robust hierarchical scale: IQR when positive, otherwise 1.4826 x MAD, otherwise standard deviation, with unit scale only as a final zero-variance fallback.

Final diagnosis is event-level. Unequal-duration fault events are summarized by the median so that long events do not receive disproportionate weight.

## Secondary sensitivity studies

The repository also includes the four reduced-complexity sensitivity analyses reported in the manuscript:

1. reference duration;
2. reference estimator (mean/std, median/MAD, median/IQR);
3. natural reference contamination, clean oracle, and unsupervised filtering;
4. classifier-family robustness (LR, RF, LightGBM).

These secondary analyses are sensitivity studies and should not be used to replace the primary benchmark results.

## Interpretation

RLR is intended as a **domain-deconfounding strategy**, not a universal novelty detector. Its clearest effect is reducing domain-induced false novelty and preserving known events at shifted sites. Unknown-fault sensitivity remains dependent on the downstream classifier and rejection rule.
