"""benchshield — benchmark integrity toolkit.

"Is this leaderboard lying to you?"

Three independent probes, one report card:
  - seed_spread:   multi-seed spread analysis + instability flags
  - contamination: canary-string + near-duplicate probes for test-set leakage
  - scaffold:      ranking movement under eval-harness perturbations

Usage:
    from benchshield import audit
    card = audit("results.csv", metric="test_auc")
    print(card.to_markdown())
"""

__version__ = "0.1.0"

from .seed_spread import SeedSpreadReport, analyze_seed_spread
from .contamination import (
    dhash,
    hamming_distance,
    near_duplicate_overlap,
    distance_histogram,
    canary_gap,
    canary_probe,
    make_canaries,
)
from .scaffold import (
    kendall_tau,
    scaffold_sensitivity,
    aggregation_perturbation,
)
from .report import ReportCard, audit

__all__ = [
    "__version__",
    "SeedSpreadReport",
    "analyze_seed_spread",
    "dhash",
    "hamming_distance",
    "near_duplicate_overlap",
    "distance_histogram",
    "canary_gap",
    "canary_probe",
    "make_canaries",
    "kendall_tau",
    "scaffold_sensitivity",
    "aggregation_perturbation",
    "ReportCard",
    "audit",
]
