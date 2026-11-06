# CA-MF-RecNAS

**Cost-Aware Multi-Fidelity Neural Architecture Search for Recommender Systems with Fuzzy-Adaptive Control and Cost-Aware Operator Reward**

This repository contains a research implementation of the CA-MF-RecNAS framework described in the accompanying manuscript. It provides typed recommender architectures, BPR training, leave-one-out evaluation, a three-objective NSGA-II search loop, progress--diversity fuzzy control, cost-aware learning-automata rewards, ridge-based multi-fidelity calibration, dataset preparation, stress-test utilities, and experiment logging.

> **Reproducibility statement.** The supplied manuscript contains a method description and aggregate result tables, but not the original source code, per-seed logs, raw datasets, the Netflix subset sampling recipe, or the referenced `Fig1.png`--`Fig12.png` assets. The code here is an executable research implementation based on the manuscript, not a claim that the original private implementation has been recovered. The paper's reported numbers are stored separately under [`paper_reference/`](paper_reference/); new runs write to `results/`.

## Repository map

```text
CA-MF-RecNAS/
├── configs/
│   ├── paper.yaml                 # manuscript hyperparameters (expensive)
│   └── smoke.yaml                 # tiny CPU configuration for a wiring check
├── data/
│   ├── raw/                       # downloaded source data (git-ignored)
│   └── processed/                 # leave-one-out NPZ files (git-ignored)
├── docs/
│   ├── DATASETS.md                # source links and dataset restrictions
│   ├── METHODOLOGY.md             # equations-to-code map
│   └── REPRODUCIBILITY_AUDIT.md   # assumptions, missing details, manuscript checks
├── paper_reference/               # manuscript-reported tables only; not fresh results
├── results/                       # generated experiments (git-ignored)
├── scripts/                       # convenience command wrappers
├── src/camf_recnas/
│   ├── data/                      # download, parse, k-core, split, stress protocols
│   ├── models/                    # typed grammar and PyTorch recommender
│   ├── objectives/                # ranking, latency, diversity, HV, Pareto, statistics
│   ├── search/                    # fuzzy FIS, crossover/mutation, LA, calibration, engine
│   ├── config.py
│   ├── training.py                # BPR training and evaluation
│   └── cli.py
└── tests/                         # unit and model-forward tests
```

## 1. Installation

Python 3.10 or newer is required. A CUDA-enabled PyTorch build is recommended for full experiments; CPU is supported for tests and smoke runs.

```bash
git clone https://github.com/<YOUR-USERNAME>/CA-MF-RecNAS.git
cd CA-MF-RecNAS
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For a CUDA installation, install the PyTorch build appropriate for your CUDA runtime from [pytorch.org](https://pytorch.org/get-started/locally/) before installing this project.

## 2. Quick verification (no external dataset required)

The smoke test creates a small **synthetic** interaction matrix only to verify that the code paths are connected. It is not one of the paper benchmarks and must never be reported as a benchmark result.

```bash
camf-recnas smoke --config configs/smoke.yaml --output results/smoke
# or
python scripts/run_smoke_test.py --config configs/smoke.yaml
```

A successful run writes a calibration summary, generation trace, operator history, run summary, and Pareto-front JSON under `results/smoke/`.

## 3. Download and prepare benchmark data

### MovieLens 100K

```bash
camf-recnas download ml-100k
camf-recnas prepare ml-100k \
  --source data/raw/ml-100k \
  --output data/processed/ml-100k.npz
```

### MovieLens 1M

```bash
camf-recnas download ml-1m
camf-recnas prepare ml-1m \
  --source data/raw/ml-1m \
  --output data/processed/ml-1m.npz
```

### Amazon Movies and TV

```bash
camf-recnas download amazon-movies-tv
camf-recnas prepare amazon-movies-tv \
  --source data/raw/reviews_Movies_and_TV_5.json.gz \
  --output data/processed/amazon-movies-tv.npz
```

### Netflix Prize subset

The Netflix Prize files are **not** downloaded by the command because acquisition and use must follow the provider's terms. Place one or more `combined_data_*.txt` files under `data/raw/netflix/`, then run:

```bash
camf-recnas prepare netflix \
  --source data/raw/netflix \
  --sample-netflix --seed 42 \
  --max-users 10000 --max-items 15000 --max-interactions 482100 \
  --output data/processed/netflix-subset.npz
```

The sampling operation above is a deterministic repository-defined convenience, **not** a recovered description of the manuscript's Netflix sample. See [`docs/DATASETS.md`](docs/DATASETS.md) and [`docs/REPRODUCIBILITY_AUDIT.md`](docs/REPRODUCIBILITY_AUDIT.md).

### Preprocessing protocol

The preparation pipeline applies the manuscript's stated rules: ratings `>= 4.0` are treated as positive; duplicate user--item records are removed; users/items below the configured five-interaction threshold are iteratively filtered; then each user's interactions are ordered by timestamp and split as last interaction = test, second-to-last = validation, earlier interactions = train. IDs are remapped to contiguous integer indices and saved in a compressed NPZ file.

```bash
# Override the preprocessing thresholds when auditing alternate interpretations.
camf-recnas prepare ml-1m --source data/raw/ml-1m \
  --positive-threshold 4 --min-user-interactions 5 --min-item-interactions 5 \
  --output data/processed/ml-1m.npz
```

The command prints the *actual* post-preprocessing counts for each downloaded version. Compare these with `paper_reference/dataset_statistics.csv`; a discrepancy should be investigated rather than silently patched.

## 4. Run architecture search

The default paper configuration matches the manuscript's main settings: population 40, 80 generations, 100 calibration architectures, five low-fidelity epochs on a per-user 5% subset, 300 full-training epochs, promotion budget 10 per generation, SGD momentum 0.9, learning rate 0.025 with cosine annealing, batch size 256, weight decay 0.0003, and ten seeds.

```bash
camf-recnas search \
  --data data/processed/ml-1m.npz \
  --config configs/paper.yaml \
  --output results/ml-1m \
  --runs 10 --start-run 0
```

Each run is saved to `results/ml-1m/run_XX/`. The calibration set and fitted Ridge predictor are created once under `results/ml-1m/shared_calibration/` and reused by subsequent seeds, consistent with the paper's per-dataset calibration protocol. The cache is fingerprinted against the processed data and configuration; a mismatch raises an error rather than reusing stale calibration. To run only one seed while debugging:

```bash
camf-recnas search --data data/processed/ml-1m.npz \
  --config configs/smoke.yaml --output results/debug --runs 1
```

**Compute warning:** the paper configuration is deliberately expensive. Calibration trains every calibration architecture at both fidelities; the search then performs low-fidelity population evaluations and up to ten full-fidelity promotions per generation. Start with `smoke.yaml` or a reduced custom YAML configuration before launching a complete ten-run study.

## 5. Evaluate a selected architecture on the held-out test split

The search uses validation metrics for calibration, promotion, and Pareto selection. The following command trains the selected architecture and evaluates it on the test split once:

```bash
camf-recnas evaluate \
  --data data/processed/ml-1m.npz \
  --architecture results/ml-1m/run_00/pareto_front.json \
  --index 0 \
  --config configs/paper.yaml \
  --output results/ml-1m/final_test.json
```

It writes a JSON report and a PyTorch checkpoint. Select the architecture using validation objectives only; do not use the test split to pick among Pareto candidates.

## 6. Few-shot and sparsity stress tests

After choosing an architecture from the validation Pareto front, run the paper's RQ5 stress protocols without re-running architecture search:

```bash
python scripts/run_stress_tests.py \
  --data data/processed/ml-1m.npz \
  --architecture results/ml-1m/run_00/pareto_front.json \
  --index 0 --config configs/paper.yaml \
  --output results/ml-1m/stress --repeats 5
```

The default few-shot levels are `k={1,2,3,4}` with 20% of users sampled per repeat. Sparsity masks default to removing `{0%,90%,95%,99%,99.5%}` of training interactions, with five repeats per level. The script writes per-repeat `fewshot.csv` and `sparsity.csv` plus `stress_summary.json`. The paper specifies five masks for sparsity but does not specify a repeated few-shot sampling count; five is an explicit repository default. This protocol is user-present few-shot, not true cold-start.

The final `evaluate` command also reports catalog coverage and intra-list similarity along with held-out test ranking metrics.

## 7. Core methodological elements

| Manuscript element | Implementation location |
|---|---|
| Typed grammar and architecture validation | `models/encoding.py` |
| Embeddings, interaction encoders, GCN/Attention/DWC/MLP blocks, prediction head | `models/recommender.py` |
| BPR pairwise training and cosine schedule | `training.py` |
| Leave-one-out all-item NDCG@10 and HR@10 | `objectives/metrics.py` |
| Three minimization objectives and fixed normalization | `search/engine.py`, `config.py` |
| Exact 3-D hypervolume for minimization | `objectives/hypervolume.py` |
| Non-dominated sorting and crowding distance | `objectives/pareto.py` |
| HVP--GD Mamdani FIS and centroid defuzzification | `search/fuzzy.py` |
| Three crossover operators and two mutation types | `search/operators.py` |
| Cost-aware operator reward, EMA and probability update | `search/automata.py` |
| Ridge calibration, residual uncertainty and LCB/HV-per-cost score | `search/promotion.py` |
| Evolutionary loop and output logs | `search/engine.py` |
| Few-shot, controlled sparsity and statistical utilities | `data/protocols.py`, `objectives/statistics.py` |

## 8. Outputs and interpretation

A typical run directory contains:

- `shared_calibration/calibration.jsonl`: architecture-level low/full calibration records;
- `shared_calibration/calibration_summary.json`: selected ridge penalties and residual uncertainty;
- `shared_calibration/calibrator.joblib`: fitted predictor reused across the dataset's runs;
- `search_trace.csv`: per-generation hypervolume, diversity, fuzzy rates, and evaluation counts;
- `operator_history.json`: operator slots, raw/EMA reward, and learning-automata probabilities;
- `pareto_front.json`: full-fidelity Pareto architectures with metrics and normalized objectives;
- `run_summary.json`: run metadata, final Pareto summary, and timing breakdown;
- `low_evaluations.jsonl` / `full_evaluations.jsonl`: candidate-level records for audit and downstream analysis.

The `paper_reference/` directory contains only aggregate values transcribed from the manuscript. It is not populated by the training code. The `results/` directory is the destination for genuinely executed experiments and is ignored by Git by default so that large outputs are not committed accidentally.

## 9. Aggregate ten-run results

After completing repeated runs, summarize mean and 95% normal confidence interval from per-run values. For a paired comparison, point `--compare` to a second folder containing matching `run_XX` indices:

```bash
python scripts/summarize_runs.py --results results/ml-1m --metric final_full_hypervolume
python scripts/summarize_runs.py --results results/proposed --compare results/nsga2 \
  --metric final_full_hypervolume --output results/paired_hv_comparison.json
```

For validation ranking quality, use `best_validation_ndcg10` or `best_validation_hr10`. These are validation-selected quantities, not a substitute for the final one-time held-out test evaluation. The script applies the Wilcoxon signed-rank test to matched run indices; the Benjamini--Hochberg adjustment for a single supplied comparison is identical to the raw p-value. For a planned family of multiple comparisons, collect all p-values and apply the correction jointly.

## 10. Tests and quality checks

```bash
pytest -q
ruff check src tests
```

## 11. Data and code licensing

The MIT license in this repository applies to the repository's original code only. It does **not** grant rights to redistribute MovieLens, Amazon, or Netflix source data. Review each provider's terms before downloading, storing, publishing, or redistributing data. The downloader intentionally keeps raw data outside version control.

## 12. Push the prepared repository to GitHub

Create an empty repository named `CA-MF-RecNAS` in your GitHub account (do not initialize it with a README or license, because these files are already present here), then run:

```bash
git init
git add .
git commit -m "Initial release: CA-MF-RecNAS"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/CA-MF-RecNAS.git
git push -u origin main
```

Raw and processed benchmark data, model checkpoints, and generated experiment results are excluded by `.gitignore`; users reproduce them locally with the documented download and run commands.

## 13. Citation

If this implementation supports your research, cite the manuscript using the metadata in [`CITATION.cff`](CITATION.cff). Please also cite each external dataset and its original provider/paper as described in [`docs/DATASETS.md`](docs/DATASETS.md).
