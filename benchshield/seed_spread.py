"""Seed-spread analysis: does the leaderboard survive its own randomness?

Ingests multi-seed benchmark results (one row per run) and measures:
  - spread: range (max-min), std, per (dataset, model) cell
  - instability flags: spread above a threshold (default 0.01 = 1pp)
  - rank stability: do model rankings change across seeds?
  - cherry-pick suspicion: when a leaderboard reports ONE number per cell,
    how far could it be from the honest mean? (best-seed inflation estimate)

Input formats: list of dicts, or path to .jsonl / .csv with columns like
run_id, dataset, model/arch, seed, <metric>.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CellResult:
    dataset: str
    model: str
    seeds: list
    values: list
    mean: float = 0.0
    std: float = 0.0
    spread: float = 0.0  # max - min
    min_v: float = 0.0
    max_v: float = 0.0
    n: int = 0
    unstable: bool = False
    best_seed_inflation: float = 0.0  # max - mean: how much best-seed reporting inflates


@dataclass
class SeedSpreadReport:
    metric: str
    threshold: float
    cells: list = field(default_factory=list)
    n_cells: int = 0
    n_unstable: int = 0
    worst_spread: float = 0.0
    worst_cell: str = ""
    rank_stability: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "n_cells": self.n_cells,
            "n_unstable": self.n_unstable,
            "worst_spread": self.worst_spread,
            "worst_cell": self.worst_cell,
            "rank_stability": self.rank_stability,
            "cells": [
                {
                    "dataset": c.dataset,
                    "model": c.model,
                    "n": c.n,
                    "mean": round(c.mean, 5),
                    "std": round(c.std, 5),
                    "spread": round(c.spread, 5),
                    "min": round(c.min_v, 5),
                    "max": round(c.max_v, 5),
                    "unstable": c.unstable,
                    "best_seed_inflation": round(c.best_seed_inflation, 5),
                }
                for c in self.cells
            ],
        }


def _load_rows(source):
    """Accept list[dict] or path to .jsonl/.csv."""
    if isinstance(source, (list, tuple)):
        return list(source)
    if isinstance(source, str) and os.path.exists(source):
        if source.endswith(".jsonl"):
            rows = []
            with open(source) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        if source.endswith(".csv"):
            with open(source, newline="") as f:
                return list(csv.DictReader(f))
    raise ValueError(f"cannot load results from {source!r}")


def _get(row, *keys, default=None):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _kendall_tau(a, b):
    """Kendall tau between two rankings (lists of labels). Pure python."""
    n = len(a)
    if n < 2:
        return 1.0
    pos = {label: i for i, label in enumerate(b)}
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = pos[a[i]], pos[a[j]]
            if ai < aj:
                conc += 1
            elif ai > aj:
                disc += 1
    denom = conc + disc
    return (conc - disc) / denom if denom else 1.0


def analyze_seed_spread(source, metric="test_auc", threshold=0.01,
                        dataset_key=("dataset",), model_key=("arch", "model"),
                        seed_key=("seed",)):
    """Analyze multi-seed spread. Returns SeedSpreadReport."""
    rows = _load_rows(source)
    cells = defaultdict(list)
    for r in rows:
        ds = str(_get(r, *dataset_key, default="dataset"))
        mo = str(_get(r, *model_key, default="model"))
        sd = _get(r, *seed_key, default=None)
        v = _get(r, metric, default=None)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        cells[(ds, mo)].append((sd, v))

    report = SeedSpreadReport(metric=metric, threshold=threshold)
    worst = 0.0
    for (ds, mo), pts in sorted(cells.items()):
        seeds = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n
        std = math.sqrt(var)
        spread = max(vals) - min(vals)
        cell = CellResult(
            dataset=ds, model=mo, seeds=seeds, values=vals,
            mean=mean, std=std, spread=spread,
            min_v=min(vals), max_v=max(vals), n=n,
            unstable=spread > threshold,
            best_seed_inflation=max(vals) - mean,
        )
        report.cells.append(cell)
        if spread > worst:
            worst = spread
            report.worst_cell = f"{ds}/{mo}"
    report.n_cells = len(report.cells)
    report.n_unstable = sum(1 for c in report.cells if c.unstable)
    report.worst_spread = worst

    # Rank stability: for each dataset, rank models by mean, then per-seed;
    # measure agreement between per-seed rankings (mean pairwise Kendall tau).
    by_ds = defaultdict(lambda: defaultdict(list))  # ds -> model -> [(seed,val)]
    for (ds, mo), pts in cells.items():
        for sd, v in pts:
            by_ds[ds][mo].append((sd, v))
    for ds, models in by_ds.items():
        # seeds present in ALL models of this dataset
        seed_sets = [set(s for s, _ in v) for v in models.values()]
        common = set.intersection(*seed_sets) if seed_sets else set()
        if len(models) < 2 or len(common) < 2:
            report.rank_stability[ds] = {"n_models": len(models),
                                        "n_common_seeds": len(common),
                                        "mean_pairwise_tau": None,
                                        "note": "insufficient overlap"}
            continue
        rankings = []
        for sd in sorted(common, key=str):
            order = sorted(models.keys(),
                           key=lambda m: dict(models[m])[sd], reverse=True)
            rankings.append(order)
        taus = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                taus.append(_kendall_tau(rankings[i], rankings[j]))
        report.rank_stability[ds] = {
            "n_models": len(models),
            "n_common_seeds": len(common),
            "mean_pairwise_tau": round(sum(taus) / len(taus), 4),
        }
    return report
