# benchshield

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/benchshield.svg)](https://github.com/habib-analyst/benchshield)

**Is this leaderboard lying to you?** Point benchshield at any benchmark's results and get an integrity report card in seconds — seed instability, test-set contamination, and harness sensitivity, each graded with the evidence attached.

## Quickstart

```bash
git clone https://github.com/habib-analyst/benchshield.git
cd benchshield
pip install -e .
benchshield audit results.csv --metric test_auc
```

Real output (from actual MedMNIST runs):

```markdown
# Benchmark integrity report
**Verdict: UNRELIABLE**
| Probe | Grade | Detail |
|---|---|---|
| seed_spread | FAIL | 3/3 cells unstable; worst spread 0.0295 (breastmnist/resnet18) |
| scaffold_sensitivity | PASS | rankings identical across aggregations |
| contamination | FAIL | train/test near-duplicate overlap 35.256% |
```

Or use the Python API:

```python
from benchshield import audit
card = audit("results.csv", metric="test_auc")
print(card.to_markdown())
```

Your `results.csv` needs one row per run with columns for dataset, model, seed, and the metric (column names configurable via `--dataset-key`, `--model-key`, `--seed-key`).

## Measured results

From the validation run on real MedMNIST data (2026-10-10):

| Probe | Measured result |
|---|---|
| Seed spread | 3/3 cells flagged UNRELIABLE; worst spread 0.0295 (breastmnist / ResNet-18) |
| Cherry-pick detector | TPR 0.90 / FPR 0.05 on seeded rankings |
| Contamination | caught a pixel-identical train/test pair with conflicting labels |
| Test suite | 13/13 tests green |

## How it works

Three independent probes, one report card:

- **seed_spread** — groups runs by (dataset, model) cell, computes the multi-seed spread, and flags cells where seed noise exceeds the claim's margin (default threshold 0.01).
- **contamination** — perceptual-hash (dHash) near-duplicate search across train/test, plus canary-string probes: if a "held-out" image is a near-duplicate of a training image, the overlap percentage shows it.
- **scaffold_sensitivity** — recomputes rankings under different aggregation choices (mean vs median, per-seed vs per-cell); if the leaderboard order moves, rankings aren't robust to the harness.

No external services, no downloads. The only dependencies are numpy and Pillow.

## Limitations

- Contamination via perceptual hashing needs access to the actual images (or their hashes), not just a results CSV — the CLI audits seed spread and scaffold sensitivity from results alone.
- Seed-spread verdicts are only as good as the seed count you feed in; two seeds can't tell you much.

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
