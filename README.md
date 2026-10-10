# benchshield 🛡️

**Is this leaderboard lying to you?** Point benchshield at any benchmark's results and get an integrity report card — in 30 seconds.

```bash
pip install benchshield
benchshield audit results.csv --metric test_auc
```

```markdown
# Benchmark integrity report
**Verdict: UNRELIABLE**
| Probe | Grade | Detail |
|---|---|---|
| seed_spread | FAIL | 3/3 cells unstable; worst spread 0.0295 (breastmnist/resnet18) |
| scaffold_sensitivity | PASS | rankings identical across aggregations |
| contamination | FAIL | train/test near-duplicate overlap 35.256% |
```

![seed spread on real MedMNIST runs](seed_spread.png)

## Why you want this

Leaderboards lie in three boring, common ways — and nobody checks:

1. **Seed luck.** The same model, same data, different seed → different rank. We measured >2.5pp spreads on MedMNIST with ResNet-18. If a paper reports one seed, you're reading noise.
2. **Test-set leakage.** Near-duplicate (sometimes *identical*) images sitting in both train and test. We found a pixel-identical image with *contradictory labels* across the split in a public medical benchmark.
3. **Harness lottery.** "Best seed" reporting inflates scores ~1.4pp on real data. Change mean→best-seed aggregation and watch rankings move.

benchshield runs three independent probes and gives one verdict: **TRUSTWORTHY / QUESTIONABLE / UNRELIABLE**.

## The three probes

| Probe | What it does | Detection (measured) |
|---|---|---|
| `seed_spread` | multi-seed spread, instability flags (>1pp), rank stability, best-seed inflation estimate | flags 3/3 unstable cells on real MedMNIST data |
| `contamination` | dHash near-duplicate train/test overlap + canary memorization probe (permutation p-value) | duplicate recall 1.00; canary TPR 0.83 / FPR 0.00 |
| `scaffold` | leaderboard rankings under mean/median/best-seed/worst-seed aggregation (Kendall τ) | cherry-pick probe TPR 0.90 / FPR 0.05 (n=2000 sims) |

## Quickstart

```python
from benchshield import audit

card = audit("results.csv", metric="test_auc")  # one row per run
print(card.to_markdown())

# with contamination probes on your own data
card = audit("results.csv", train_images=train, test_images=test,
             canary_stats=dict(canary_correct=18, canary_total=20,
                               heldout_correct=170, heldout_total=200))
```

Your `results.csv` needs one row per run with columns like `dataset, arch, seed, test_auc`
(JSONL works too). See [`examples/audit_medmnist.py`](examples/audit_medmnist.py) for the
full flagship run on real measured data.

## Measured results (not claims)

Full numbers in [`PAPER.md`](https://github.com/habib-analyst/benchshield) and `eval/results.json`:

- **Seed spread (real MedMNIST batch-1, 15 runs):** breast 0.9074±0.0103 (spread 2.95pp), pneumonia 0.9464±0.0106 (2.72pp), retina 0.7290±0.0081 (2.55pp) — all flagged UNSTABLE at the 1pp bar.
- **Near-duplicates (real breastmnist pixels):** 55/156 test images (35.3%) within dHash ≤5 of a train image; 6 exact hash matches, incl. one pixel-identical cross-split pair with *conflicting labels*.
- **Cherry-pick probe:** catches best-seed reporting with TPR 0.90 / FPR 0.05 over 2000 simulated leaderboards built from real per-seed values; mean best-seed inflation measured at 1.41pp.
- **Duplicate injection:** 100% recall of injected exact duplicates at 1%/5%/10% rates.
- **Canary probe:** 10/12 memorized models caught, 0/12 false alarms on clean models (sklearn, real training).

## Scope & honesty

- benchshield *audits* benchmarks; it doesn't run them (that's [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)'s job — complementary, not competing).
- dHash near-duplicate detection is tuned for natural/medical images; on very low-entropy image sets, verify flagged pairs visually — the report tells you exactly which pairs.
- The cherry-pick probe needs a reproduction (or the full per-seed data); from a single reported number alone, nobody can detect cherry-picking — which is the point: *demand all seeds*.

## Install

```bash
pip install benchshield        # numpy + Pillow only
```

## License

MIT — see [LICENSE](LICENSE).
