# Benchmark integrity report

**Verdict: UNRELIABLE**

| Probe | Grade | Detail |
|---|---|---|
| seed_spread | FAIL | 3/3 cells unstable; worst spread 0.0295 (breastmnist/resnet18) |
| scaffold_sensitivity | PASS | rankings identical across aggregations |
| contamination | FAIL | train/test near-duplicate overlap 35.256% |
