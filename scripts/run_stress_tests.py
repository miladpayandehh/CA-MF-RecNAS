"""Run few-shot-user and controlled-sparsity stress evaluations for one architecture."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from camf_recnas.config import load_config
from camf_recnas.data.dataset import load_processed_dataset
from camf_recnas.data.protocols import controlled_sparsity, few_shot_user_protocol
from camf_recnas.models.encoding import Architecture
from camf_recnas.training import choose_device, fit_and_evaluate


def load_architecture(path: Path, index: int) -> Architecture:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "pareto_front" in payload:
        candidates = payload["pareto_front"]
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = [payload]
    if not 0 <= index < len(candidates):
        raise IndexError(f"Architecture index {index} is out of range [0, {len(candidates)-1}]")
    selected = candidates[index]
    return Architecture.from_dict(selected.get("architecture", selected))


def evaluate_case(architecture: Architecture, dataset, config, seed: int) -> dict[str, Any]:
    _, record = fit_and_evaluate(
        architecture,
        dataset,
        config.training,
        config.training.full_epochs,
        choose_device(config.device),
        seed=seed,
        split="test",
        measure_inference_latency=False,
        parameter_reference=config.search.parameter_reference,
        latency_reference_seconds=config.search.latency_reference_seconds,
        fidelity_label="stress-test",
    )
    return {
        "seed": seed,
        "n_users": dataset.n_users,
        "n_items": dataset.n_items,
        "train_interactions": len(dataset.train),
        "test_interactions": len(dataset.test),
        "ndcg_at_10": record.ndcg10,
        "hr_at_10": record.hr10,
        "training_time_seconds": record.training_time_seconds,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Processed dataset NPZ")
    parser.add_argument("--architecture", required=True, type=Path,
                        help="pareto_front.json or run_summary.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stress"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of user samples / random sparsity masks per condition")
    parser.add_argument("--fewshot-k", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--user-fraction", type=float, default=0.20)
    parser.add_argument("--sparsity-levels", nargs="+", type=float,
                        default=[0.0, 0.90, 0.95, 0.99, 0.995])
    parser.add_argument("--skip-fewshot", action="store_true")
    parser.add_argument("--skip-sparsity", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if any(k < 1 for k in args.fewshot_k):
        parser.error("all --fewshot-k values must be at least 1")
    if any(not 0 <= x < 1 for x in args.sparsity_levels):
        parser.error("sparsity levels must be in [0, 1)")

    dataset = load_processed_dataset(args.data)
    config = load_config(args.config)
    architecture = load_architecture(args.architecture, args.index)
    args.output.mkdir(parents=True, exist_ok=True)

    fewshot_rows: list[dict[str, Any]] = []
    if not args.skip_fewshot:
        for k in args.fewshot_k:
            for repeat in range(args.repeats):
                seed = args.seed + 10_000 * k + repeat
                stressed = few_shot_user_protocol(dataset, k=k,
                                                  user_fraction=args.user_fraction, seed=seed)
                row = evaluate_case(architecture, stressed, config, seed)
                fewshot_rows.append({"k": k, "repeat": repeat, **row})
        write_csv(args.output / "fewshot.csv", fewshot_rows)

    sparsity_rows: list[dict[str, Any]] = []
    if not args.skip_sparsity:
        for level_index, level in enumerate(args.sparsity_levels):
            for repeat in range(args.repeats):
                seed = args.seed + 100_000 + 1_000 * level_index + repeat
                stressed = controlled_sparsity(dataset, remove_fraction=level, seed=seed)
                row = evaluate_case(architecture, stressed, config, seed)
                sparsity_rows.append({"removed_fraction": level, "repeat": repeat, **row})
        write_csv(args.output / "sparsity.csv", sparsity_rows)

    summary: dict[str, Any] = {
        "dataset": dataset.name,
        "architecture_id": architecture.fingerprint,
        "architecture": architecture.to_dict(),
        "seed": args.seed,
        "repeats_per_condition": args.repeats,
        "fewshot": {},
        "sparsity": {},
        "note": ("Few-shot is a user-present proxy, not true unseen-user cold-start. "
                 "The manuscript specifies five masks for sparsity; the script defaults to "
                 "five repeats for both protocols, while few-shot repeat count is configurable."),
    }
    for k in args.fewshot_k:
        values = [row["ndcg_at_10"] for row in fewshot_rows if row["k"] == k]
        if values:
            summary["fewshot"][str(k)] = {
                "ndcg_at_10_mean": float(np.mean(values)),
                "ndcg_at_10_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "hr_at_10_mean": float(np.mean([row["hr_at_10"] for row in fewshot_rows if row["k"] == k])),
            }
    for level in args.sparsity_levels:
        values = [row["ndcg_at_10"] for row in sparsity_rows
                  if np.isclose(row["removed_fraction"], level)]
        if values:
            summary["sparsity"][str(level)] = {
                "ndcg_at_10_mean": float(np.mean(values)),
                "ndcg_at_10_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "hr_at_10_mean": float(np.mean([row["hr_at_10"] for row in sparsity_rows
                                                 if np.isclose(row["removed_fraction"], level)])),
            }
    (args.output / "stress_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
