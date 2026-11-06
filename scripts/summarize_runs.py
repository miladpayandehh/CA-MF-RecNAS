"""Summarize per-seed CA-MF-RecNAS run summaries and optionally compare two folders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from camf_recnas.objectives.statistics import (
    benjamini_hochberg,
    mean_confidence_interval,
    wilcoxon_paired,
)


def read_runs(root: Path) -> dict[int, dict[str, Any]]:
    paths = sorted(root.glob("run_*/run_summary.json"))
    if not paths and (root / "run_summary.json").exists():
        paths = [root / "run_summary.json"]
    runs: dict[int, dict[str, Any]] = {}
    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        index = int(obj.get("run_index", 0))
        if index in runs:
            raise ValueError(f"Duplicate run_index={index} under {root}")
        runs[index] = obj
    if not runs:
        raise FileNotFoundError(f"No run_summary.json files found under {root}")
    return runs


def metric_value(run: dict[str, Any], name: str) -> float:
    if name == "final_full_hypervolume":
        return float(run["final_full_hypervolume"])
    if name == "best_validation_ndcg10":
        records = [entry["full_record"] for entry in run.get("pareto_front", [])]
        return float(max((r["ndcg10"] for r in records), default=float("nan")))
    if name == "best_validation_hr10":
        records = [entry["full_record"] for entry in run.get("pareto_front", [])]
        return float(max((r["hr10"] for r in records), default=float("nan")))
    if name == "n_low_evaluations":
        return float(run["n_low_evaluations"])
    if name == "n_full_evaluations":
        return float(run["n_full_evaluations"])
    if name == "wall_time_days_24h":
        return float(run["cost_accounting"]["run_wall_time_days_24h"])
    raise ValueError(f"Unsupported metric: {name}")


def summarize(root: Path, metric: str) -> tuple[dict[str, Any], dict[int, float]]:
    runs = read_runs(root)
    values = {idx: metric_value(run, metric) for idx, run in runs.items()}
    ordered = np.asarray([values[idx] for idx in sorted(values)], dtype=np.float64)
    if not np.all(np.isfinite(ordered)):
        raise ValueError(f"Metric {metric} contains non-finite values under {root}")
    mean, low, high = mean_confidence_interval(ordered, 1.96)
    return ({"folder": str(root), "metric": metric, "n_runs": len(ordered),
             "run_indices": sorted(values), "mean": mean,
             "ci95_low": low, "ci95_high": high,
             "values_by_run": {str(k): float(v) for k, v in values.items()}}, values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Folder containing run_XX directories")
    parser.add_argument("--compare", type=Path, default=None, help="Optional second result folder for paired comparison")
    parser.add_argument("--metric", default="final_full_hypervolume", choices=[
        "final_full_hypervolume", "best_validation_ndcg10", "best_validation_hr10",
        "n_low_evaluations", "n_full_evaluations", "wall_time_days_24h",
    ])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    first, values_a = summarize(args.results, args.metric)
    report: dict[str, Any] = {"primary": first}
    if args.compare is not None:
        second, values_b = summarize(args.compare, args.metric)
        common = sorted(set(values_a) & set(values_b))
        if len(common) < 2:
            p_value = None
        else:
            p_value = wilcoxon_paired([values_a[k] for k in common],
                                      [values_b[k] for k in common])
        adjusted = (float(benjamini_hochberg([p_value])[0])
                    if p_value is not None else None)
        report["comparison"] = {
            "secondary": second,
            "paired_run_indices": common,
            "wilcoxon_p_value": p_value,
            "bh_adjusted_p_value_single_comparison": adjusted,
            "note": "BH over a single comparison equals the unadjusted p-value; combine p-values from the complete planned family for a multi-test correction.",
        }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
