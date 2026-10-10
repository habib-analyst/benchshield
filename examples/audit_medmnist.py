"""Flagship demo: audit Habib's measured MedMNIST instability data.

Uses the REAL batch-1 rerun results (15 runs: breast/pneumonia/retina x
5 seeds, ResNet-18, native 28x28) from the MedMNIST reproducibility project,
plus a train/test near-duplicate probe on the actual breastmnist pixels.

Run:  python examples/audit_medmnist.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchshield import audit, near_duplicate_overlap

HERE = os.path.dirname(os.path.abspath(__file__))
GOAL = "/home/hatch/workspace/goals/github-portfolio-linkedin-presence-buildout"
RESULTS = os.path.join(GOAL, "hidden_files/medmnist_results_current.jsonl")


def main():
    rows = []
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} measured runs")

    # near-duplicate probe on real breastmnist pixels
    d = np.load("/home/hatch/.medmnist/breastmnist.npz")
    train = d["train_images"]
    test = d["test_images"]
    print(f"train {train.shape} test {test.shape} — hashing...")
    nd = near_duplicate_overlap(train, test, threshold=5)
    print(f"near-duplicate overlap: {nd['n_flagged']}/{nd['n_test']} "
          f"({nd['fraction']:.3%})")

    card = audit(rows, metric="test_auc", threshold=0.01,
                 train_images=train, test_images=test)
    print()
    print(card.to_markdown())
    out = os.path.join(HERE, "medmnist_report.md")
    with open(out, "w") as f:
        f.write(card.to_markdown())
    with open(os.path.join(HERE, "medmnist_report.json"), "w") as f:
        f.write(card.to_json())
    print(f"wrote {out}")
    return card


if __name__ == "__main__":
    main()
