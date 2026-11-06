from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import joblib

from camf_recnas.config import ExperimentConfig
from camf_recnas.data.dataset import InteractionDataset
from camf_recnas.models.encoding import Architecture, SearchSpace
from camf_recnas.objectives.hypervolume import hypervolume_3d
from camf_recnas.objectives.pareto import crowding_distance, nondominated_sort, nsga2_select
from camf_recnas.search.automata import CostAwareLearningAutomata
from camf_recnas.search.fuzzy import FuzzyRateController
from camf_recnas.search.operators import OPERATOR_NAMES, crossover, mutate
from camf_recnas.search.promotion import MultiFidelityCalibrator, architecture_features
from camf_recnas.training import EvaluationRecord, choose_device, fit_and_evaluate, make_training_view


@dataclass
class Candidate:
    architecture: Architecture
    low_record: EvaluationRecord
    low_objective: np.ndarray
    full_record: EvaluationRecord | None = None
    full_objective: np.ndarray | None = None


def normalize_objective(objective: list[float] | np.ndarray, lower: list[float],
                        upper: list[float]) -> np.ndarray:
    value = np.asarray(objective, dtype=np.float64)
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    return np.clip((value - lo) / (hi - lo), 0.0, 1.0)


def _levenshtein(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def architecture_tokens(architecture: Architecture) -> list[str]:
    d_bin = {16: "d0", 32: "d1", 64: "d2", 128: "d3", 256: "d4"}[architecture.embedding_dim]
    width_bin = {64: "w0", 128: "w1", 256: "w2", 512: "w3"}[architecture.head_width]
    dropout_bin = {0.0: "r0", 0.1: "r1", 0.2: "r2", 0.3: "r3", 0.5: "r4"}[architecture.head_dropout]
    tokens = [d_bin, f"enc:{architecture.interaction_encoder}", width_bin, dropout_bin]
    for gene in architecture.blocks:
        tokens.extend([f"block:{gene.kind}", f"activation:{gene.activation}"])
    return tokens


def population_diversity(population: list[Candidate]) -> float:
    if len(population) < 2:
        return 0.0
    tokens = [architecture_tokens(c.architecture) for c in population]
    distances = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            denom = max(len(tokens[i]), len(tokens[j]), 1)
            distances.append(_levenshtein(tokens[i], tokens[j]) / denom)
    return float(np.mean(distances)) if distances else 0.0


def _front(candidates: list[Candidate], objective_attribute: str) -> list[Candidate]:
    if not candidates:
        return []
    matrix = np.vstack([getattr(c, objective_attribute) for c in candidates])
    indexes = nondominated_sort(matrix)[0]
    return [candidates[i] for i in indexes]


def _tournament(population: list[Candidate], rng: random.Random) -> Candidate:
    if len(population) == 1:
        return population[0]
    objectives = np.vstack([c.low_objective for c in population])
    fronts = nondominated_sort(objectives)
    ranks = {}
    distances = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank
        distances.update(crowding_distance(objectives, front))
    i, j = rng.sample(range(len(population)), 2)
    if ranks[i] < ranks[j] or (ranks[i] == ranks[j] and distances[i] >= distances[j]):
        return population[i]
    return population[j]


def _record_full_candidate(candidate: Candidate, dataset: InteractionDataset,
                           config: ExperimentConfig, device: torch.device,
                           seed: int) -> None:
    _, record = fit_and_evaluate(
        candidate.architecture, dataset, config.training, config.training.full_epochs,
        device, seed=seed, split="validation", measure_inference_latency=True,
        parameter_reference=config.search.parameter_reference,
        latency_reference_seconds=config.search.latency_reference_seconds,
        fidelity_label="full",
    )
    candidate.full_record = record
    candidate.full_objective = normalize_objective(
        record.objective, config.search.normalization_lower,
        config.search.normalization_upper,
    )


def fit_calibrator(dataset: InteractionDataset, config: ExperimentConfig,
                   space: SearchSpace, device: torch.device, run_seed: int,
                   output_dir: Path) -> tuple[MultiFidelityCalibrator, dict[str, Any]]:
    """Build the isolated 100-architecture (or smoke-config) calibration set."""
    # Calibration architectures are fixed per dataset/configuration and reused across runs.
    rng = random.Random(config.seed + 100_003)
    sample: list[Architecture] = []
    seen = set()
    while len(sample) < config.search.calibration_size:
        architecture = space.sample(rng)
        if architecture.fingerprint not in seen:
            seen.add(architecture.fingerprint)
            sample.append(architecture)
    low_rows = dataset.low_fidelity_train(config.data.low_fidelity_fraction, run_seed)
    low_dataset = make_training_view(dataset, low_rows)
    feature_rows, full_objectives, full_costs, log_rows = [], [], [], []
    for idx, architecture in enumerate(sample):
        _, low_record = fit_and_evaluate(
            architecture, low_dataset, config.training, config.training.low_fidelity_epochs,
            device, seed=run_seed + idx, split="validation", measure_inference_latency=True,
            parameter_reference=config.search.parameter_reference,
            latency_reference_seconds=config.search.latency_reference_seconds,
            fidelity_label="low",
        )
        _, full_record = fit_and_evaluate(
            architecture, dataset, config.training, config.training.full_epochs,
            device, seed=run_seed + 50_000 + idx, split="validation", measure_inference_latency=True,
            parameter_reference=config.search.parameter_reference,
            latency_reference_seconds=config.search.latency_reference_seconds,
            fidelity_label="full",
        )
        feature_rows.append(architecture_features(architecture, low_record))
        full_objectives.append(normalize_objective(
            full_record.objective, config.search.normalization_lower,
            config.search.normalization_upper))
        full_costs.append(max(full_record.training_time_seconds / 3600.0, 1e-8))
        log_rows.append({"index": idx, "architecture": architecture.to_dict(),
                         "low": low_record.to_dict(), "full": full_record.to_dict(),
                         "full_cost_hours": full_costs[-1]})
    calibrator = MultiFidelityCalibrator(config.search.ridge_alphas).fit(
        np.vstack(feature_rows), np.vstack(full_objectives), np.asarray(full_costs))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "calibration.jsonl").open("w", encoding="utf-8") as stream:
        for row in log_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"n_calibration_architectures": len(sample),
               "selected_alphas": [float(model.alpha_) for model in calibrator.models],
               "residual_std": calibrator.residual_std.tolist(),
               "cost_residual_std_log": calibrator.cost_residual_std,
               "low_training_seconds": float(sum(row["low"]["training_time_seconds"] for row in log_rows)),
               "full_training_seconds": float(sum(row["full"]["training_time_seconds"] for row in log_rows)),
               "evaluation_wall_seconds": float(sum(row["low"]["wall_time_seconds"] +
                                                     row["full"]["wall_time_seconds"] for row in log_rows))}
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return calibrator, summary


def _calibration_cache_fingerprint(dataset: InteractionDataset,
                                   config: ExperimentConfig) -> str:
    payload = {
        "dataset": dataset.name,
        "n_users": dataset.n_users,
        "n_items": dataset.n_items,
        "train_shape": dataset.train.shape,
        "validation_shape": dataset.validation.shape,
        "config": {
            "seed": config.seed,
            "data": config.data.__dict__,
            "training": config.training.__dict__,
            "search": config.search.__dict__,
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    for array in (dataset.train, dataset.validation):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _load_or_fit_calibrator(dataset: InteractionDataset, config: ExperimentConfig,
                            space: SearchSpace, device: torch.device, run_seed: int,
                            cache_dir: Path) -> tuple[MultiFidelityCalibrator, dict[str, Any], bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "calibrator.joblib"
    summary_path = cache_dir / "calibration_summary.json"
    manifest_path = cache_dir / "calibration_manifest.json"
    fingerprint = _calibration_cache_fingerprint(dataset, config)
    if model_path.exists() and summary_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Calibration cache at {cache_dir} does not match this dataset/config. "
                "Choose a new results folder or remove the stale shared_calibration directory."
            )
        calibrator = joblib.load(model_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return calibrator, summary, True
    if any(path.exists() for path in (model_path, summary_path, manifest_path)):
        raise ValueError(
            f"Incomplete calibration cache at {cache_dir}; remove it and rerun calibration."
        )
    calibrator, summary = fit_calibrator(dataset, config, space, device, run_seed, cache_dir)
    joblib.dump(calibrator, model_path)
    manifest_path.write_text(json.dumps({"fingerprint": fingerprint,
                                         "dataset": dataset.name,
                                         "n_users": dataset.n_users,
                                         "n_items": dataset.n_items}, indent=2), encoding="utf-8")
    return calibrator, summary, False


def run_search(dataset: InteractionDataset, config: ExperimentConfig,
               output_dir: str | Path, run_index: int = 0,
               calibration_cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Run one CA-MF-RecNAS search and persist all per-run records."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    run_seed = config.seed + run_index * 10_007
    rng = random.Random(run_seed)
    device = choose_device(config.device)
    space = SearchSpace(max_blocks=config.search.max_blocks, allow_side_info=False)
    automata = CostAwareLearningAutomata(config.search.learning_automata_rate,
                                         config.search.reward_ema_beta)
    fuzzy = FuzzyRateController()
    pc = config.search.crossover_probability_initial
    pm = config.search.mutation_probability_initial
    reference = tuple(config.search.reference_point)

    if calibration_cache_dir is None:
        calibrator, calibration_summary = fit_calibrator(
            dataset, config, space, device, run_seed, output_dir)
        calibration_reused = False
    else:
        calibrator, calibration_summary, calibration_reused = _load_or_fit_calibrator(
            dataset, config, space, device, run_seed, Path(calibration_cache_dir))
    calibration_summary = {**calibration_summary, "reused_from_shared_cache": calibration_reused}
    low_rows = dataset.low_fidelity_train(config.data.low_fidelity_fraction, run_seed)
    low_dataset = make_training_view(dataset, low_rows)

    low_cache: dict[str, Candidate] = {}
    low_evaluation_rows: list[dict[str, Any]] = []
    full_evaluation_rows: list[dict[str, Any]] = []
    full_candidates: dict[str, Candidate] = {}

    def evaluate_low(architecture: Architecture) -> tuple[Candidate, float]:
        key = architecture.fingerprint
        if key in low_cache:
            return low_cache[key], 0.0
        _, record = fit_and_evaluate(
            architecture, low_dataset, config.training, config.training.low_fidelity_epochs,
            device, seed=run_seed + len(low_cache), split="validation",
            measure_inference_latency=True,
            parameter_reference=config.search.parameter_reference,
            latency_reference_seconds=config.search.latency_reference_seconds,
            fidelity_label="low",
        )
        objective = normalize_objective(record.objective, config.search.normalization_lower,
                                        config.search.normalization_upper)
        candidate = Candidate(architecture, record, objective)
        low_cache[key] = candidate
        cost_hours = max(record.training_time_seconds / 3600.0, 0.0)
        low_evaluation_rows.append({"architecture_id": key,
                                    "architecture": architecture.to_dict(),
                                    "record": record.to_dict(),
                                    "training_cost_hours": cost_hours})
        return candidate, cost_hours

    population_archs: list[Architecture] = []
    population_seen = set()
    while len(population_archs) < config.search.population_size:
        arch = space.sample(rng)
        if arch.fingerprint not in population_seen:
            population_seen.add(arch.fingerprint)
            population_archs.append(arch)
    population = [evaluate_low(arch)[0] for arch in population_archs]

    trace_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    previous_low_hv = 0.0
    for generation in range(1, config.search.generations + 1):
        old_low_front = _front(population, "low_objective")
        old_front_matrix = np.vstack([c.low_objective for c in old_low_front])
        slots = automata.allocate_slots(config.search.population_size, automata.probabilities, minimum=2)
        operator_children: list[list[Architecture]] = [[] for _ in OPERATOR_NAMES]
        for operator_idx, (operator_name, n_slots) in enumerate(zip(OPERATOR_NAMES, slots)):
            while len(operator_children[operator_idx]) < int(n_slots):
                parent_a = _tournament(population, rng).architecture
                parent_b = _tournament(population, rng).architecture
                if rng.random() < pc:
                    child_a, child_b = crossover(parent_a, parent_b, operator_name, rng, space)
                else:
                    child_a, child_b = parent_a, parent_b
                child_a = mutate(child_a, rng, space, pm)
                child_b = mutate(child_b, rng, space, pm)
                for child in (child_a, child_b):
                    if len(operator_children[operator_idx]) < int(n_slots):
                        operator_children[operator_idx].append(child)

        union_by_id: dict[str, Candidate] = {c.architecture.fingerprint: c for c in population}
        union_candidates: list[Candidate] = list(population)
        child_cost_by_id: dict[str, float] = {}
        producers_by_id: dict[str, set[int]] = {}
        for operator_idx, group in enumerate(operator_children):
            for architecture in group:
                candidate, cost_hours = evaluate_low(architecture)
                union_by_id[candidate.architecture.fingerprint] = candidate
                union_candidates.append(candidate)
                key = candidate.architecture.fingerprint
                producers_by_id.setdefault(key, set()).add(operator_idx)
                # Duplicate genotypes are evaluated once; their measured cost is
                # shared among the operators that independently generated them.
                child_cost_by_id[key] = max(child_cost_by_id.get(key, 0.0), cost_hours)
        # Keep duplicate individuals in the NSGA-II population while caching their
        # expensive evaluation by genotype. Duplicate genotypes still have zero GD distance.
        union = union_candidates
        baseline_front = old_front_matrix
        raw_rewards = []
        for operator_idx, group in enumerate(operator_children):
            unique_keys = list(dict.fromkeys(a.fingerprint for a in group))
            group_candidates = [union_by_id[k] for k in unique_keys if k in union_by_id]
            group_objectives = np.vstack([c.low_objective for c in group_candidates]) if group_candidates else np.empty((0, 3))
            if len(group_objectives):
                gain = max(0.0, hypervolume_3d(np.vstack([baseline_front, group_objectives]), reference)
                           - hypervolume_3d(baseline_front, reference))
            else:
                gain = 0.0
            cost = sum(child_cost_by_id.get(key, 0.0) / max(len(producers_by_id.get(key, ())), 1)
                       for key in unique_keys)
            raw_rewards.append(gain / (config.search.epsilon + cost))
        automata_state = automata.update(np.asarray(raw_rewards))

        current_low_front = _front(union, "low_objective")
        current_front_matrix = np.vstack([c.low_objective for c in current_low_front])
        current_hv = hypervolume_3d(current_front_matrix, reference)
        hv_progress = float(np.clip((current_hv - previous_low_hv) /
                                    (previous_low_hv + config.search.epsilon), -1.0, 1.0))
        # Population diversity is measured after NSGA-II environmental selection.
        selected_idx = nsga2_select(np.vstack([c.low_objective for c in union]),
                                    config.search.population_size)
        population = [union[i] for i in selected_idx]
        gd = population_diversity(population)
        pc, pm = fuzzy.update(hv_progress, gd)
        pc = float(np.clip(pc, config.search.min_crossover_probability,
                           config.search.max_crossover_probability))
        pm = float(np.clip(pm, config.search.min_mutation_probability,
                           config.search.max_mutation_probability))

        # Promote candidates on the current low-fidelity Pareto front using calibrated LCB / cost.
        full_front_candidates = _front(list(full_candidates.values()), "full_objective") if full_candidates else []
        full_front_matrix = (np.vstack([c.full_objective for c in full_front_candidates])
                             if full_front_candidates else np.empty((0, 3)))
        promotion_pool = [c for c in current_low_front
                          if c.architecture.fingerprint not in full_candidates]
        scored = []
        for candidate in promotion_pool:
            x = architecture_features(candidate.architecture, candidate.low_record)
            score = calibrator.promotion_score(x, full_front_matrix, reference,
                                               config.search.promotion_z, config.search.epsilon)
            scored.append((score, candidate))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        promoted = []
        for rank, (score, candidate) in enumerate(scored[:config.search.promotion_budget]):
            _record_full_candidate(candidate, dataset, config, device,
                                   run_seed + generation * 1000 + rank)
            full_candidates[candidate.architecture.fingerprint] = candidate
            full_evaluation_rows.append({"architecture_id": candidate.architecture.fingerprint,
                                         "architecture": candidate.architecture.to_dict(),
                                         "record": candidate.full_record.to_dict(),
                                         "training_cost_hours": candidate.full_record.training_time_seconds / 3600.0,
                                         "generation": generation})
            promoted.append({"architecture_id": candidate.architecture.fingerprint,
                             "score": float(score), "full_record": candidate.full_record.to_dict()})

        previous_low_hv = current_hv
        trace_rows.append({
            "generation": generation,
            "low_hypervolume": current_hv,
            "full_hypervolume": hypervolume_3d(
                np.vstack([c.full_objective for c in _front(list(full_candidates.values()), "full_objective")]),
                reference) if full_candidates else 0.0,
            "hvp": hv_progress,
            "genotypic_diversity": gd,
            "crossover_probability": pc,
            "mutation_probability": pm,
            "la_probabilities": automata.probabilities.tolist(),
            "la_ema_rewards": automata.ema_rewards.tolist(),
            "low_candidates_cached": len(low_cache),
            "full_candidates": len(full_candidates),
            "promoted_this_generation": len(promoted),
        })
        operator_rows.append({"generation": generation, "raw_rewards": raw_rewards,
                              "ema_rewards": automata_state["ema_rewards"].tolist(),
                              "probabilities": automata_state["probabilities"].tolist(),
                              "rewarded_operators": automata_state["rewarded_operators"],
                              "slots": slots.tolist(), "promotions": promoted})
        print(f"[run {run_index} | gen {generation:03d}/{config.search.generations}] "
              f"HV_low={current_hv:.4f} GD={gd:.3f} pc={pc:.3f} pm={pm:.3f} "
              f"promoted={len(promoted)}")

    final_front = _front(list(full_candidates.values()), "full_objective") if full_candidates else []
    run_wall_seconds = time.perf_counter() - run_started
    cost_accounting = {
        "calibration_low_training_seconds": (0.0 if calibration_reused else calibration_summary["low_training_seconds"]),
        "calibration_full_training_seconds": (0.0 if calibration_reused else calibration_summary["full_training_seconds"]),
        "calibration_evaluation_wall_seconds": (0.0 if calibration_reused else calibration_summary["evaluation_wall_seconds"]),
        "calibration_reused_from_shared_cache": calibration_reused,
        "search_low_training_seconds": float(sum(row["record"]["training_time_seconds"]
                                                   for row in low_evaluation_rows)),
        "promotion_full_training_seconds": float(sum(row["record"]["training_time_seconds"]
                                                      for row in full_evaluation_rows)),
        "run_wall_time_seconds": run_wall_seconds,
        "run_wall_time_days_24h": run_wall_seconds / 86400.0,
        "device": str(device),
    }
    result = {
        "dataset": dataset.name,
        "run_index": run_index,
        "seed": run_seed,
        "device": str(device),
        "calibration": calibration_summary,
        "n_low_evaluations": len(low_cache),
        "n_full_evaluations": len(full_candidates),
        "cost_accounting": cost_accounting,
        "final_full_hypervolume": hypervolume_3d(
            np.vstack([c.full_objective for c in final_front]), reference) if final_front else 0.0,
        "pareto_front": [
            {"architecture": c.architecture.to_dict(),
             "architecture_id": c.architecture.fingerprint,
             "full_record": c.full_record.to_dict(),
             "normalized_objective": c.full_objective.tolist()}
            for c in final_front
        ],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "search_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        if trace_rows:
            writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]))
            writer.writeheader(); writer.writerows(trace_rows)
    (output_dir / "operator_history.json").write_text(
        json.dumps(operator_rows, indent=2), encoding="utf-8")
    for filename, rows in (("low_evaluations.jsonl", low_evaluation_rows),
                           ("full_evaluations.jsonl", full_evaluation_rows)):
        with (output_dir / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "pareto_front.json").write_text(
        json.dumps(result["pareto_front"], indent=2), encoding="utf-8")
    return result
