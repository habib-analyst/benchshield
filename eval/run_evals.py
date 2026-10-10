"""Measured probe detection rates on deliberately corrupted leaderboards.

Three experiments, all CPU, all deterministic (fixed seeds):

E1 cherry-pick probe:  simulate published leaderboards from the REAL
   MedMNIST per-seed values. Each cell reports mean (honest, 70%) or
   best-seed (cherry-picked, 30%). An independent reproducer bootstraps
   5 seeds; the probe flags a cell when
       reported > repro_mean + 2 * repro_std
   Measures TPR (caught cherry-picks) and FPR (honest cells flagged).

E2 duplicate-injection probe: on real pneumoniamnist pixels, inject d
   exact test->train duplicates at rates {0, 1%, 5%, 10%}; the
   near-duplicate probe (dHash, threshold 0 = exact) must recover them.
   Measures recall of injected duplicates + false-flag rate on clean data.

E3 canary probe: train sklearn logistic regression on breastmnist pixels
   (flattened) WITH K canary samples in train (memorizing condition) vs
   canaries held OUT of train (clean condition). Probe = canary_gap
   verdict. Measures TPR/FPR over repeated trials.

Writes eval/results.json. All numbers measured, nothing simulated beyond
the stated corruptions.
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchshield.contamination import (
    dhash, hamming_distance, near_duplicate_overlap, canary_gap, make_canaries,
)

HERE = os.path.dirname(os.path.abspath(__file__))
GOAL = "/home/hatch/workspace/goals/github-portfolio-linkedin-presence-buildout"
RESULTS = os.path.join(GOAL, "hidden_files/medmnist_results_current.jsonl")
rng_global = np.random.default_rng(0)


def load_cells():
    cells = {}
    with open(RESULTS) as f:
        for line in f:
            r = json.loads(line)
            key = (r["dataset"], r["arch"])
            cells.setdefault(key, []).append(float(r["test_auc"]))
    return cells


def eval_cherry_pick(n_sims=400, seed=0):
    """E1: TPR/FPR of the reproduction-gap cherry-pick probe."""
    rng = np.random.default_rng(seed)
    cells = load_cells()
    tp = fp = p_total = n_total = 0
    inflations = []
    for _ in range(n_sims):
        key = list(cells.keys())[rng.integers(len(cells))]
        vals = np.array(cells[key])
        cherry = rng.random() < 0.30
        reported = vals.max() if cherry else vals.mean()
        if cherry:
            inflations.append(float(vals.max() - vals.mean()))
        # reproducer: bootstrap 5 seeds from the 5 real ones
        repro = rng.choice(vals, size=5, replace=True)
        rm = repro.mean()
        se = repro.std(ddof=1) / math.sqrt(len(repro)) if len(repro) > 1 else 0.0
        # probe: is `reported` plausible as a MEAN of 5 seeds? Under honest
        # reporting, |reported - repro_mean| ~ SE*sqrt(2); cherry-picked
        # best-seed sits ~3 SE above. Flag one-sided at 2 SE.
        flagged = reported > rm + 2 * se
        if cherry:
            p_total += 1
            tp += flagged
        else:
            n_total += 1
            fp += flagged
    return {
        "n_sims": n_sims,
        "tpr": round(tp / p_total, 4) if p_total else None,
        "fpr": round(fp / n_total, 4) if n_total else None,
        "n_cherry": p_total, "n_honest": n_total,
        "mean_best_seed_inflation": round(float(np.mean(inflations)), 5),
        "max_best_seed_inflation": round(float(np.max(inflations)), 5),
    }


def eval_duplicate_injection(rates=(0.0, 0.01, 0.05, 0.10), seed=1):
    """E2: recall of injected exact duplicates + clean false-flag rate."""
    rng = np.random.default_rng(seed)
    d = np.load("/home/hatch/.medmnist/pneumoniamnist.npz")
    train = d["train_images"]
    test = d["test_images"][:400]  # subsample for speed
    out = {}
    for rate in rates:
        n_inject = int(len(test) * rate)
        idx = rng.choice(len(test), size=n_inject, replace=False)
        # exact pixel copies
        train_aug = list(train) + [test[i] for i in idx]
        # exact-match probe: threshold 0
        res = near_duplicate_overlap(train_aug, test, threshold=0,
                                     max_pairs=10**9)
        # how many of the injected ones were found?
        th = [dhash(im) for im in train_aug]
        found = 0
        for i in idx:
            h = dhash(test[i])
            if any(hamming_distance(h, x) == 0 for x in th):
                found += 1
        out[str(rate)] = {
            "n_injected": n_inject,
            "recall": round(found / n_inject, 4) if n_inject else None,
            "flagged_total": res["n_flagged"],
            "flagged_fraction": round(res["fraction"], 4),
        }
    return out


def eval_canary(n_trials=12, n_canaries=20, seed=2):
    """E3: canary probe TPR/FPR with real trained models (sklearn, CPU)."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    d = np.load("/home/hatch/.medmnist/breastmnist.npz")
    Xtr = d["train_images"].reshape(len(d["train_images"]), -1).astype(np.float64) / 255.0
    ytr = d["train_labels"].ravel().astype(int)
    Xte = d["test_images"].reshape(len(d["test_images"]), -1).astype(np.float64) / 255.0
    yte = d["test_labels"].ravel().astype(int)
    n_classes = int(ytr.max()) + 1
    tp = fp = 0
    gaps_mem, gaps_clean = [], []
    for t in range(n_trials):
        cimgs, clabs = make_canaries(n_canaries, (28, 28), n_classes,
                                     seed=1000 + t, label_mode="fixed")
        Xc = cimgs.reshape(n_canaries, -1).astype(np.float64) / 255.0
        # memorizing condition: canaries IN train
        Xm = np.vstack([Xtr, Xc]); ym = np.concatenate([ytr, clabs])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xm, ym)
        canary_acc = float(clf.score(Xc, clabs))
        held_acc = float(clf.score(Xte, yte))
        stat = canary_gap(int(canary_acc * n_canaries), n_canaries,
                          int(held_acc * len(yte)), len(yte),
                          n_perm=500, seed=t)
        gaps_mem.append(stat["gap"])
        tp += stat["gap"] > 0.2 and stat["p_value"] < 0.05
        # clean condition: fresh canaries NOT in train
        cimgs2, clabs2 = make_canaries(n_canaries, (28, 28), n_classes,
                                       seed=2000 + t, label_mode="fixed")
        Xc2 = cimgs2.reshape(n_canaries, -1).astype(np.float64) / 255.0
        canary_acc2 = float(clf.score(Xc2, clabs2))
        stat2 = canary_gap(int(canary_acc2 * n_canaries), n_canaries,
                           int(held_acc * len(yte)), len(yte),
                           n_perm=500, seed=t)
        gaps_clean.append(stat2["gap"])
        fp += stat2["gap"] > 0.2 and stat2["p_value"] < 0.05
    return {
        "n_trials": n_trials,
        "n_canaries": n_canaries,
        "tpr": round(tp / n_trials, 4),
        "fpr": round(fp / n_trials, 4),
        "mean_gap_memorized": round(float(np.mean(gaps_mem)), 4),
        "mean_gap_clean": round(float(np.mean(gaps_clean)), 4),
    }


def main():
    t0 = time.time()
    out = {"cherry_pick": eval_cherry_pick(n_sims=2000, seed=7)}
    print("E1 done", flush=True)
    out["duplicate_injection"] = eval_duplicate_injection()
    print("E2 done", flush=True)
    out["canary"] = eval_canary()
    print("E3 done", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"elapsed {out['elapsed_s']}s")


if __name__ == "__main__":
    main()
