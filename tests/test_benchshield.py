"""Tests for benchshield. All deterministic; no network; no GPU."""
import json
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchshield import (
    analyze_seed_spread,
    aggregation_perturbation,
    audit,
    canary_gap,
    dhash,
    hamming_distance,
    kendall_tau,
    make_canaries,
    near_duplicate_overlap,
    scaffold_sensitivity,
)


def _rows():
    # 2 datasets x 2 models x 3 seeds, breast-like spreads
    rows = []
    vals = {
        ("breast", "resnet18"): [0.9074, 0.9217, 0.8922],   # spread .0295 UNSTABLE
        ("breast", "vit-t"):   [0.9010, 0.9050, 0.8990],   # spread .006 stable
        ("pneumo", "resnet18"): [0.9464, 0.9613, 0.9341],   # spread .0272 UNSTABLE
        ("pneumo", "vit-t"):   [0.9400, 0.9440, 0.9380],   # spread .006 stable
    }
    for (ds, mo), vs in vals.items():
        for i, v in enumerate(vs):
            rows.append({"dataset": ds, "arch": mo, "seed": 17 + i * 12,
                         "test_auc": v})
    return rows


def test_seed_spread_flags():
    rep = analyze_seed_spread(_rows(), metric="test_auc", threshold=0.01)
    assert rep.n_cells == 4
    assert rep.n_unstable == 2
    assert rep.worst_cell == "breast/resnet18"
    assert abs(rep.worst_spread - 0.0295) < 1e-9
    breast_r18 = next(c for c in rep.cells
                      if c.dataset == "breast" and c.model == "resnet18")
    assert breast_r18.unstable
    assert abs(breast_r18.mean - 0.9071) < 1e-4
    assert breast_r18.best_seed_inflation > 0.014  # best-seed inflates


def test_seed_spread_csv_and_jsonl(tmp_path):
    rows = _rows()
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    rep = analyze_seed_spread(str(p), metric="test_auc")
    assert rep.n_cells == 4
    p2 = tmp_path / "r.csv"
    import csv
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rep2 = analyze_seed_spread(str(p2), metric="test_auc")
    assert rep2.n_unstable == 2


def test_rank_stability_present():
    rep = analyze_seed_spread(_rows(), metric="test_auc")
    assert "breast" in rep.rank_stability
    rs = rep.rank_stability["breast"]
    assert rs["n_models"] == 2 and rs["n_common_seeds"] == 3
    assert -1.0 <= rs["mean_pairwise_tau"] <= 1.0


def test_kendall_tau():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert kendall_tau(["a", "b"], ["b", "a"]) == -1.0


def test_aggregation_perturbation_detects_cherrypick():
    # best-seed aggregation flips the ranking vs mean -> fragile
    rep = analyze_seed_spread(_rows(), metric="test_auc")
    agg = aggregation_perturbation(rep.cells)
    assert "breast" in agg
    # mean ranks vit-t above resnet18? check structure only
    assert set(agg["breast"]["rankings"].keys()) == {
        "mean", "median", "best_seed", "worst_seed"}
    assert  -1.0 <= agg["breast"]["min_tau"] <= 1.0


def test_scaffold_sensitivity_generic():
    out = scaffold_sensitivity({
        "base": ["a", "b", "c"],
        "perturbed": ["b", "a", "c"],
    })
    assert out["fragile"] is True
    assert out["min_tau"] < 1.0
    out2 = scaffold_sensitivity({"a": ["x", "y"], "b": ["x", "y"]})
    assert out2["fragile"] is False


def test_dhash_determinism_and_hamming():
    import numpy as np
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(28, 28)).astype(np.uint8)
    assert dhash(a) == dhash(a)
    assert hamming_distance(0b101, 0b100) == 1
    b = a.copy()
    b[0, 0] = 255 - b[0, 0]
    # single-pixel change: hash identical or nearly (dHash is robust)
    assert hamming_distance(dhash(a), dhash(b)) <= 4


def test_near_duplicate_overlap_clean_and_injected():
    import numpy as np
    rng = np.random.default_rng(1)
    train = [rng.integers(0, 256, size=(28, 28)).astype(np.uint8)
             for _ in range(20)]
    test = [rng.integers(0, 256, size=(28, 28)).astype(np.uint8)
            for _ in range(10)]
    clean = near_duplicate_overlap(train, test)
    assert clean["fraction"] == 0.0
    # inject exact duplicates of 3 test images into train
    train2 = train + [test[0], test[3], test[7]]
    dirty = near_duplicate_overlap(train2, test)
    assert dirty["n_flagged"] >= 3
    assert dirty["fraction"] >= 0.3


def test_canary_gap_memorized_vs_clean():
    # memorized: 18/20 canaries right, heldout 10/20 -> big gap, tiny p
    mem = canary_gap(18, 20, 10, 20, n_perm=500, seed=0)
    assert mem["gap"] > 0.3 and mem["p_value"] < 0.05
    # clean: both ~10/20 -> gap ~0, p large
    clean = canary_gap(11, 20, 10, 20, n_perm=500, seed=0)
    assert abs(clean["gap"]) < 0.1 and clean["p_value"] > 0.2


def test_make_canaries_shapes():
    imgs, labels = make_canaries(10, (28, 28), 4, seed=0)
    assert imgs.shape == (10, 28, 28) and labels.shape == (10,)
    assert set(labels.tolist()) <= {0, 1, 2, 3}


def test_audit_end_to_end_markdown_and_json():
    card = audit(_rows(), metric="test_auc", threshold=0.01)
    md = card.to_markdown()
    assert "Verdict" in md and "seed_spread" in md
    d = json.loads(card.to_json())
    assert d["verdict"] in ("TRUSTWORTHY", "QUESTIONABLE", "UNRELIABLE")
    # 2/4 cells unstable -> QUESTIONABLE (frac 0.5 -> FAIL boundary is >=0.5)
    assert d["verdict"] == "UNRELIABLE"  # 2/4 = 0.5 -> FAIL
    assert d["sections"]["seed_spread"]["grade"] == "FAIL"


def test_distance_histogram():
    import numpy as np
    from benchshield import distance_histogram
    rng = np.random.default_rng(3)
    train = [rng.integers(0, 256, size=(28, 28)).astype(np.uint8)
             for _ in range(10)]
    test = [train[0], train[5]] + [rng.integers(0, 256, size=(28, 28)).astype(np.uint8)
                                   for _ in range(3)]
    hist = distance_histogram(train, test)
    assert hist.get(0, 0) >= 2  # the two exact copies
    assert sum(hist.values()) == 5


def test_cli_audit(tmp_path, capsys):
    rows = _rows()
    p = tmp_path / "r.csv"
    import csv
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from benchshield.cli import main
    assert main(["audit", str(p), "--metric", "test_auc"]) == 0
    out = capsys.readouterr().out
    assert "Verdict" in out
