"""benchshield CLI: `benchshield audit results.csv` -> integrity report card."""
import argparse
import sys

from .report import audit


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="benchshield",
        description="Benchmark integrity toolkit: is this leaderboard lying to you?")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="audit a results file")
    a.add_argument("results", help="results .csv or .jsonl (one row per run)")
    a.add_argument("--metric", default="test_auc",
                   help="metric column to analyze (default: test_auc)")
    a.add_argument("--threshold", type=float, default=0.01,
                   help="instability flag threshold in metric units (default 0.01)")
    a.add_argument("--format", choices=["markdown", "json"], default="markdown")
    a.add_argument("--dataset-key", default="dataset")
    a.add_argument("--model-key", default="arch")
    a.add_argument("--seed-key", default="seed")
    a.add_argument("-o", "--output", default=None, help="write report to file")

    args = ap.parse_args(argv)

    from . import seed_spread
    # re-run analysis with custom keys for the audit path
    import benchshield.report as _rep
    rep = seed_spread.analyze_seed_spread(
        args.results, metric=args.metric, threshold=args.threshold,
        dataset_key=(args.dataset_key,), model_key=(args.model_key,),
        seed_key=(args.seed_key,))
    from .scaffold import aggregation_perturbation
    card = _rep.ReportCard()
    g, d = _rep._grade_seed_spread(rep)
    card.add("seed_spread", g, d, rep.to_dict())
    agg = aggregation_perturbation(rep.cells)
    g, d = _rep._grade_scaffold(agg)
    card.add("scaffold_sensitivity", g, d, agg)
    g, d = _rep._grade_contamination(None, None)
    card.add("contamination", g, d, {})

    out = card.to_json() if args.format == "json" else card.to_markdown()
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
