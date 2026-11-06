# Methodology and code-level conventions

## Objective vector

For each candidate architecture, the code records three minimized quantities:

\[
F(A)=\left[1-\mathrm{NDCG@10}(A),\;\log_{10}(P(A)/10^6),\;
\log_{10}(T_{p95}(A)/10^{-3})\right].
\]

The fixed component bounds are `(0.20, 0, 0)` and `(0.60, 1, 1)` and each component is clipped to `[0,1]` before Pareto sorting and hypervolume computation. The hypervolume reference point is `(1.05,1.05,1.05)`.

## Architecture grammar

An architecture contains embedding dimension, an interaction encoder (`product`, `concat`, `attn`, `gcn`), one to eight typed blocks (`gcn`, `attention`, `dwc`, `mlp`), and a prediction head. Search domains are copied from the manuscript:

- embedding dimension `{16,32,64,128,256}`;
- activations `{ReLU, LeakyReLU, PReLU, ELU, SELU}`;
- attention heads `{1,2,4,8}`;
- depthwise-convolution kernel `{1,3,5}`;
- head width `{64,128,256,512}`;
- head dropout `{0,.1,.2,.3,.5}`.

All implemented blocks preserve embedding width. Attention heads must divide the selected embedding dimension. Since the manuscript does not define a complete side-feature data schema or encoder, the `side_info` field is retained in the serialized architecture format but non-`none` values are disabled in the default search to avoid pretending that demographic/category features are implemented.

## Training and ranking evaluation

Positive user--item interactions are optimized with pairwise BPR. A negative item is uniformly sampled from the item catalog and resampled if it is already a positive interaction for that user. At validation/test time the model scores all catalog items, masks items present in the training history, and computes NDCG@10, HR@10, and Recall@10. With the ordinary leave-one-out split (one held-out positive per user), Recall@10 equals HR@10. The few-shot protocol holds out multiple positive items for selected users; for this case the evaluator computes multi-relevant-item NDCG and distinguishes hit rate from recall.

The graph block uses the symmetrically normalized adjacency `D^(-1/2) (A+I) D^(-1/2)` over the training user--item bipartite graph. GCN operations update the entity embedding table, which is then used by later graph blocks and the prediction layer. The sequential Attention and DWC blocks operate on the left-padded user interaction sequence; MLP blocks operate on the pooled user representation.

## Multi-fidelity calibration and promotion

For each calibration architecture, the runner evaluates a 5%-per-user low-fidelity training view and a full-fidelity model. Ridge regressors predict each normalized full-fidelity objective using low-fidelity NDCG, parameter count, block count, block-type histogram, activation histogram, and p95 latency. `RidgeCV` selects the regularization parameter by five-fold cross-validation. Residual standard deviation and a ridge leverage term estimate prediction uncertainty.

The promotion score is the lower confidence bound on predicted 3-D hypervolume improvement divided by estimated full-training cost. The standard deviation of the hypervolume improvement is estimated with a first-order finite-difference delta approximation. `promotion_z` defaults to the 95% one-sided normal quantile used in the paper.

## Cost-aware operator credit

All three crossover operators generate offspring within every generation. Slots are allocated proportional to the learning-automata probability vector subject to at least two offspring per operator. Each operator receives positive hypervolume gain per low-fidelity evaluation cost. Rewards are smoothed with EMA (`beta=.3`), and a multi-reward Linear Reward-Inaction update shifts probability mass toward operators whose EMA reward is greater than the generation mean and positive.

## Fuzzy-adaptive rate controller

The Mamdani controller uses the manuscript's piecewise HVP and genotypic-diversity memberships, product t-norm, max aggregation, triangular output memberships, and centroid defuzzification. Genotypic edit distance is computed on typed tokens; embedding dimension, head width, and dropout are mapped to discrete gene tokens.

## Cost accounting

The runner logs wall-clock seconds for each evaluation and converts them to hours. When run on an RTX 3060 Ti, these are directly comparable to the manuscript's GPU-hour accounting only if the same GPU, software environment, data splits, batch shapes, synchronization and workload are used. A CPU smoke run is only a software check; it is not a comparable GPU-cost measurement.
