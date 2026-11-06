# Reproducibility audit of the supplied manuscript

This document distinguishes manuscript-specified details from choices required to make a runnable public implementation. It should be updated if the authors provide the original scripts, run logs, data manifests, or a corrected manuscript.

## Specified sufficiently to implement

- Positive feedback threshold: rating `>= 4`.
- Minimum interaction count: five for users and items.
- Leave-one-out split: last event test, second-to-last validation, earlier events train.
- Four data families: MovieLens 100K, MovieLens 1M, Amazon Movies and TV, Netflix Prize subset.
- Low-fidelity view: uniform 5% per user, retaining at least one interaction; five epochs.
- Full training: 300 epochs; SGD, momentum 0.9, learning rate 0.025 with cosine annealing, batch 256, weight decay 0.0003, embedding dropout 0.2.
- Search defaults: population 40, 80 generations, maximum 8 blocks, 100 calibration architectures, promotion budget 10 per generation. The repository caches the 100-architecture calibration set and Ridge models once per processed dataset/configuration and reuses them across runs.
- Three objectives, fixed normalization bounds, and reference point `(1.05,1.05,1.05)`.
- Fuzzy input memberships and qualitative rule table; three crossover operators, two mutation families; LA rate 0.1 and reward EMA beta 0.3.
- RTX 3060 Ti latency protocol: batch size 1, 100 warm-up runs, 1,000 synchronized forward passes.
- Aggregate results and reported ablations are transcribed in `paper_reference/`.

## Details not fully specified by the manuscript

1. **No original code or run logs.** The source provided for this repository request was the manuscript text only. The paper does not include the authors' actual Python package, dependency lockfile, individual random seeds, per-candidate records, or raw outputs.
2. **Duplicate tie-breaking.** The manuscript says duplicate user--item records are removed but does not define which copy to retain. This implementation retains the latest timestamp using a stable sort; ties are resolved deterministically by the sorted user/item keys.
3. **Netflix subset construction.** The paper gives the final dimensions (10,000 users, 15,000 items, 482,100 interactions) but not the sampled file IDs, sampling order, seed, filtering sequence, or checksum. The repository's sampler is an explicitly labeled alternative.
4. **Amazon release/version.** The paper says Amazon Movies and TV but does not fully pin the release file/version or provide a checksum. The downloader points to the historical UCSD/SNAP 5-core file; its final counts may differ after positive-rating filtering and k-core filtering.
5. **K-core semantics.** “Users and items with fewer than 5 interactions are removed” does not specify whether the rule is applied once or iteratively. This code uses iterative k-core filtering and records the resulting counts.
6. **Side information.** The grammar lists optional demographics/categories, but the manuscript does not define the feature encoding, missing-value treatment, or exact integration into each block. The default executable search therefore uses `side_info='none'` rather than claiming an unreported feature pipeline.
7. **Structural repair.** The prose reports a projection-based dimension repair and a 3.1% repair rate, but the grammar also states that every block preserves dimension `d`. The typed implementation enforces the stated domains and attention divisibility; the exact projection-repair frequency and 4,200-parameter average cannot be reproduced without the original genotype/phenotype details.
8. **Promotion accounting.** The algorithm lists a budget of 10 promotions per generation over 80 generations (up to 800 promotions, excluding calibration), while the paper reports 1.10 GPU-days of full training. No per-architecture training-time distribution, early-stop rule, or cap on cumulative full promotions is provided, so the reported end-to-end budget cannot be independently verified from the manuscript alone.
9. **Cost units.** The abstract/conclusion and cost-accounting table report search cost as `0.32 GPU-days`, while the “five closest methods” table labels the CA-MF-RecNAS search value `0.32` under a `Search (GPU-h)` column. These differ by a factor of 24. The text also contains 17.0 GPU-hour scalability measurements for the ML-1M-sized case. Confirm the intended unit before final publication.
10. **Scalability range.** The scalability table's largest listed configuration is 6,040 users and 3,706 items, whereas the accompanying text says the tested maximum is 10,000 users and 15,000 items. The additional configurations are not tabulated in the supplied text.
11. **Figure assets.** The TeX references `Fig1.png` through `Fig12.png`, but those files were not part of the supplied attachment. They are not fabricated in this repository.
12. **Statistical reconstruction.** The manuscript gives means, confidence intervals, and test procedure but not paired per-seed values. Statistical test outputs cannot be reconstructed from aggregate tables alone.
13. **Fuzzy-controller tuning.** The paper says the membership supports/rule base were selected using 30 short ML-1M validation searches. The repository implements the reported selected memberships/rules, but does not recreate the original 30-search tuning exercise because its trial logs and exact support-scaling candidates are absent.
14. **Comparative baselines.** Several contextual methods were adapted/re-implemented. Their exact code versions, hyperparameter grids, seeds, and equal-budget implementation details are not fully disclosed.

## Claims that must remain carefully worded

- The user stress condition is a **few-shot proxy**, not true cold-start. The user is present in training with `k` interactions.
- Controlled sparsity removes users/items that have no remaining training interactions; reported high-sparsity values are therefore subject to survivor-selection bias.
- Coverage and intra-list similarity are secondary reported metrics, not dimensions of the three-objective Pareto vector.
- Aggregate manuscript values in `paper_reference/` are reported values, not results reproduced by this code.

## Recommended author-supplied artifacts before claiming exact reproduction

- Exact raw data manifests and checksums, plus Netflix sample IDs and sampling script.
- Original source code or a tagged reference commit.
- `pip freeze`/Conda environment and CUDA/PyTorch versions.
- Ten per-seed metrics and run times for every table row, plus raw paired values used for Wilcoxon/BH tests.
- Clarification of the unit of the `0.32` search cost and the promotion budget's cumulative interpretation.
- The twelve figure files and any supplementary implementation notes.
