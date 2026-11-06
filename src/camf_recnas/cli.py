from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from camf_recnas.config import load_config
from camf_recnas.data.dataset import InteractionDataset, leave_one_out_split, load_processed_dataset, save_processed_dataset
from camf_recnas.data.download import download_dataset
from camf_recnas.data.preprocess import _netflix_target_sample, load_raw_ratings, preprocess_ratings
from camf_recnas.models.encoding import Architecture
from camf_recnas.objectives.diversity import catalog_coverage, intra_list_similarity
from camf_recnas.search.engine import run_search
from camf_recnas.training import choose_device, fit_and_evaluate


def _prepare(args: argparse.Namespace) -> None:
    source = Path(args.source) if args.source else download_dataset(args.dataset, args.raw_dir)
    ratings = load_raw_ratings(args.dataset, source)
    if args.dataset == "netflix" and args.sample_netflix:
        ratings = _netflix_target_sample(ratings, args.max_users, args.max_items,
                                         args.max_interactions, args.seed)
    cleaned = preprocess_ratings(ratings, args.positive_threshold,
                                 args.min_user_interactions, args.min_item_interactions)
    dataset = leave_one_out_split(cleaned, name=args.dataset)
    output_path = args.output or str(Path("data/processed") / f"{args.dataset}.npz")
    save_processed_dataset(dataset, output_path)
    print(json.dumps({"dataset": args.dataset, "n_users": dataset.n_users,
                      "n_items": dataset.n_items, "train": len(dataset.train),
                      "validation": len(dataset.validation), "test": len(dataset.test),
                      "processed_file": output_path}, indent=2))


def _download(args: argparse.Namespace) -> None:
    print(download_dataset(args.dataset, args.raw_dir))


def _search(args: argparse.Namespace) -> None:
    dataset = load_processed_dataset(args.data)
    config = load_config(args.config)
    runs = args.runs if args.runs is not None else config.search.runs
    for offset in range(runs):
        run_index = args.start_run + offset
        output_dir = Path(args.output) / f"run_{run_index:02d}"
        result = run_search(dataset, config, output_dir, run_index=run_index,
                            calibration_cache_dir=Path(args.output) / "shared_calibration")
        print(f"Saved {output_dir}; final full HV={result['final_full_hypervolume']:.6f}")


def _make_synthetic_dataset(seed: int = 42) -> InteractionDataset:
    """Small deterministic dataset for unit/smoke testing only (not a paper benchmark)."""
    rng = np.random.default_rng(seed)
    n_users, n_items, events_per_user = 24, 48, 10
    rows = []
    for user in range(n_users):
        # A shifted cycle covers the item catalog with a roughly balanced degree.
        chosen = [(user * 7 + j * 13) % n_items for j in range(events_per_user)]
        chosen = sorted(set(chosen))
        while len(chosen) < events_per_user:
            item = int(rng.integers(n_items))
            if item not in chosen:
                chosen.append(item)
        for timestamp, item in enumerate(chosen):
            rows.append((user, item, 5.0 if rng.random() > 0.1 else 4.0, timestamp))
    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    return leave_one_out_split(frame, "synthetic-smoke-test")


def _smoke(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dataset = _make_synthetic_dataset(config.seed)
    result = run_search(dataset, config, args.output, run_index=0)
    print("SMOKE TEST COMPLETED")
    print(json.dumps({"n_users": dataset.n_users, "n_items": dataset.n_items,
                      "final_full_hypervolume": result["final_full_hypervolume"],
                      "low_evaluations": result["n_low_evaluations"],
                      "full_evaluations": result["n_full_evaluations"]}, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    dataset = load_processed_dataset(args.data)
    config = load_config(args.config)
    source = json.loads(Path(args.architecture).read_text(encoding="utf-8"))
    if isinstance(source, dict) and "pareto_front" in source:
        candidates = source["pareto_front"]
    elif isinstance(source, list):
        candidates = source
    else:
        candidates = [source]
    architecture_dict = candidates[args.index].get("architecture", candidates[args.index])
    architecture = Architecture.from_dict(architecture_dict)
    device = choose_device(config.device)
    model, record = fit_and_evaluate(
        architecture, dataset, config.training, config.training.full_epochs,
        device, seed=args.seed, split="test", measure_inference_latency=True,
        parameter_reference=config.search.parameter_reference,
        latency_reference_seconds=config.search.latency_reference_seconds,
        fidelity_label="test-final",
    )
    record.metrics["CatalogCoverage@10"] = catalog_coverage(
        model, dataset, k=10, device=str(device),
        max_sequence_length=config.training.max_sequence_length)
    record.metrics["IntraListSimilarity@10"] = intra_list_similarity(
        model, dataset, k=10, device=str(device),
        max_sequence_length=config.training.max_sequence_length)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": architecture.to_dict(), "state_dict": model.state_dict(),
                "n_users": dataset.n_users, "n_items": dataset.n_items}, out.with_suffix(".pt"))
    out.write_text(json.dumps({"architecture": architecture.to_dict(),
                               "test_record": record.to_dict()}, indent=2), encoding="utf-8")
    print(f"Saved test evaluation to {out} and model weights to {out.with_suffix('.pt')}")
    print(json.dumps(record.to_dict(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="camf-recnas", description="CA-MF-RecNAS research code")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Download a source dataset")
    download.add_argument("dataset", choices=["ml-100k", "ml-1m", "amazon-movies-tv", "netflix"])
    download.add_argument("--raw-dir", default="data/raw")
    download.set_defaults(func=_download)

    prepare = sub.add_parser("prepare", help="Preprocess ratings and create leave-one-out splits")
    prepare.add_argument("dataset", choices=["ml-100k", "ml-1m", "amazon-movies-tv", "netflix"])
    prepare.add_argument("--source", default=None, help="Source folder/file; downloads automatically if omitted")
    prepare.add_argument("--raw-dir", default="data/raw")
    prepare.add_argument("--output", default=None,
                         help="Processed output path (default: data/processed/<dataset>.npz)")
    prepare.add_argument("--positive-threshold", type=float, default=4.0)
    prepare.add_argument("--min-user-interactions", type=int, default=5)
    prepare.add_argument("--min-item-interactions", type=int, default=5)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--sample-netflix", action="store_true",
                         help="Use a deterministic repository-defined sample, not the undisclosed paper sample")
    prepare.add_argument("--max-users", type=int, default=10_000)
    prepare.add_argument("--max-items", type=int, default=15_000)
    prepare.add_argument("--max-interactions", type=int, default=482_100)
    prepare.set_defaults(func=_prepare)

    search = sub.add_parser("search", help="Run one or more CA-MF-RecNAS searches")
    search.add_argument("--data", required=True, help="Processed .npz dataset")
    search.add_argument("--config", default="configs/paper.yaml")
    search.add_argument("--output", default="results")
    search.add_argument("--runs", type=int, default=None)
    search.add_argument("--start-run", type=int, default=0)
    search.set_defaults(func=_search)

    smoke = sub.add_parser("smoke", help="Run a tiny end-to-end synthetic smoke test")
    smoke.add_argument("--config", default="configs/smoke.yaml")
    smoke.add_argument("--output", default="results/smoke")
    smoke.set_defaults(func=_smoke)

    evaluate = sub.add_parser("evaluate", help="Train one selected architecture and evaluate once on test")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--architecture", required=True, help="pareto_front.json or run_summary.json")
    evaluate.add_argument("--index", type=int, default=0)
    evaluate.add_argument("--config", default="configs/paper.yaml")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--output", default="results/final_test.json")
    evaluate.set_defaults(func=_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
