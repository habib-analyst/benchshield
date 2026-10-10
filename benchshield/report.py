"""Report card: one grade per integrity dimension + overall verdict."""
from __future__ import annotations

import json

from . import contamination, scaffold, seed_spread


def _grade_seed_spread(rep):
    if rep.n_cells == 0:
        return "N/A", "no multi-seed data"
    frac = rep.n_unstable / rep.n_cells
    if frac == 0:
        return "PASS", f"all {rep.n_cells} cells stable (spread <= {rep.threshold})"
    if frac < 0.5:
        return "WARN", (f"{rep.n_unstable}/{rep.n_cells} cells unstable; "
                        f"worst spread {rep.worst_spread:.4f} ({rep.worst_cell})")
    return "FAIL", (f"{rep.n_unstable}/{rep.n_cells} cells unstable; "
                    f"worst spread {rep.worst_spread:.4f} ({rep.worst_cell})")


def _grade_contamination(nd_result, canary_result):
    notes = []
    if nd_result is not None:
        f = nd_result["fraction"]
        notes.append(f"train/test near-duplicate overlap {f:.3%}")
        if f > 0.05:
            return "FAIL", "; ".join(notes)
        if f > 0.01:
            return "WARN", "; ".join(notes)
    if canary_result is not None:
        g, p = canary_result["gap"], canary_result["p_value"]
        notes.append(f"canary gap {g:+.3f} (p={p})")
        if g > 0.2 and p < 0.05:
            return "FAIL", "; ".join(notes) + " — memorization detected"
        if g > 0.1 and p < 0.05:
            return "WARN", "; ".join(notes)
    if not notes:
        return "N/A", "no contamination probes run"
    return "PASS", "; ".join(notes)


def _grade_scaffold(agg_result):
    if not agg_result:
        return "N/A", "no scaffold perturbations run"
    worst = min(v["min_tau"] for v in agg_result.values())
    fragile = [k for k, v in agg_result.items() if v["fragile"]]
    if fragile:
        return "FAIL", (f"ranking flips under aggregation change "
                        f"(min tau {worst:.2f}): {', '.join(fragile)}")
    if worst < 1.0:
        return "WARN", f"minor ranking movement (min tau {worst:.2f})"
    return "PASS", "rankings identical across aggregations"


_GRADES = {"PASS": 0, "WARN": 1, "FAIL": 2, "N/A": -1}


def _overall(grades):
    scored = [g for g in grades if g != "N/A"]
    if not scored:
        return "N/A"
    worst = max(scored, key=lambda g: _GRADES[g])
    return {"PASS": "TRUSTWORTHY", "WARN": "QUESTIONABLE",
            "FAIL": "UNRELIABLE"}[worst]


class ReportCard:
    def __init__(self, title="Benchmark integrity report"):
        self.title = title
        self.sections = {}

    def add(self, name, grade, detail, data=None):
        self.sections[name] = {"grade": grade, "detail": detail,
                               "data": data or {}}

    @property
    def verdict(self):
        return _overall([s["grade"] for s in self.sections.values()])

    def to_dict(self):
        return {"title": self.title, "verdict": self.verdict,
                "sections": self.sections}

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self):
        lines = [f"# {self.title}", "",
                 f"**Verdict: {self.verdict}**", "",
                 "| Probe | Grade | Detail |",
                 "|---|---|---|"]
        for name, s in self.sections.items():
            lines.append(f"| {name} | {s['grade']} | {s['detail']} |")
        return "\n".join(lines) + "\n"


def audit(source, metric="test_auc", threshold=0.01,
          train_images=None, test_images=None, canary_stats=None):
    """One-call audit: seed spread (+ optional contamination probes).

    source: results file (.jsonl/.csv) or list of dicts (one row per run).
    train_images/test_images: arrays for the near-duplicate probe.
    canary_stats: dict(canary_correct, canary_total, heldout_correct,
        heldout_total) for the canary probe.
    """
    card = ReportCard()
    rep = seed_spread.analyze_seed_spread(source, metric=metric,
                                         threshold=threshold)
    g, d = _grade_seed_spread(rep)
    card.add("seed_spread", g, d, rep.to_dict())

    agg = scaffold.aggregation_perturbation(rep.cells)
    g, d = _grade_scaffold(agg)
    card.add("scaffold_sensitivity", g, d, agg)

    nd_result = None
    if train_images is not None and test_images is not None:
        nd_result = contamination.near_duplicate_overlap(train_images,
                                                         test_images)
    canary_result = None
    if canary_stats is not None:
        canary_result = contamination.canary_gap(**canary_stats)
    g, d = _grade_contamination(nd_result, canary_result)
    card.add("contamination", g, d,
             {"near_duplicate": nd_result, "canary": canary_result})
    return card
