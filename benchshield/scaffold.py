"""Scaffold-sensitivity: does the leaderboard survive the harness?

A benchmark number is produced by an *eval scaffold*: preprocessing
(resize/interpolation/normalization), batching, metric aggregation
(mean vs median vs "best seed"), tie-breaking, etc. If swapping a
defensible scaffold choice flips the ranking, the leaderboard is
scaffold-fragile — the winner won the harness lottery, not the task.

Two tools:
1. aggregation_perturbation(cells): given per-seed values per (dataset,
   model) cell (as produced by seed_spread), build leaderboards under
   different aggregation rules — mean, median, best-seed, worst-seed —
   and report pairwise Kendall tau between the resulting rankings.
2. scaffold_sensitivity(rankings): generic — given a dict
   {perturbation_name: ranking_list}, return the pairwise tau matrix and
   the minimum tau (the fragility score).
"""
from __future__ import annotations

import math
from collections import defaultdict


def kendall_tau(rank_a, rank_b):
    """Kendall tau between two rankings (lists of labels, best-first)."""
    n = len(rank_a)
    if n < 2:
        return 1.0
    pos = {label: i for i, label in enumerate(rank_b)}
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = pos[rank_a[i]], pos[rank_a[j]]
            if ai < aj:
                conc += 1
            elif ai > aj:
                disc += 1
    denom = conc + disc
    return (conc - disc) / denom if denom else 1.0


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


AGGREGATIONS = {
    "mean": lambda xs: sum(xs) / len(xs),
    "median": _median,
    "best_seed": max,
    "worst_seed": min,
}


def aggregation_perturbation(cells, higher_is_better=True):
    """Leaderboards under mean/median/best-seed/worst-seed aggregation.

    cells: iterable of objects with .dataset, .model, .values (list of
    per-seed floats). Returns dict with per-dataset rankings + pairwise taus.
    """
    by_ds = defaultdict(list)
    for c in cells:
        by_ds[c.dataset].append(c)
    out = {}
    for ds, cs in by_ds.items():
        rankings = {}
        for name, fn in AGGREGATIONS.items():
            scored = [(c.model, fn(c.values)) for c in cs]
            scored.sort(key=lambda t: t[1], reverse=higher_is_better)
            rankings[name] = [m for m, _ in scored]
        names = list(rankings)
        taus = {}
        min_tau = 1.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                t = kendall_tau(rankings[names[i]], rankings[names[j]])
                taus[f"{names[i]}_vs_{names[j]}"] = round(t, 4)
                min_tau = min(min_tau, t)
        out[ds] = {
            "rankings": rankings,
            "pairwise_tau": taus,
            "min_tau": round(min_tau, 4),
            "fragile": min_tau < 0.8,
        }
    return out


def scaffold_sensitivity(rankings):
    """Generic fragility score from {perturbation_name: ranking_list}."""
    names = list(rankings)
    taus = {}
    min_tau = 1.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            t = kendall_tau(rankings[names[i]], rankings[names[j]])
            taus[f"{names[i]}_vs_{names[j]}"] = round(t, 4)
            min_tau = min(min_tau, t)
    return {
        "pairwise_tau": taus,
        "min_tau": round(min_tau, 4),
        "fragile": min_tau < 0.8,
    }
