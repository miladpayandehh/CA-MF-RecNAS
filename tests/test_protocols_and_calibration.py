import numpy as np
import pytest
import torch

from camf_recnas.data.dataset import InteractionDataset, leave_one_out_split
from camf_recnas.data.protocols import controlled_sparsity, few_shot_user_protocol
from camf_recnas.data.preprocess import preprocess_ratings
from camf_recnas.search.promotion import MultiFidelityCalibrator
from camf_recnas.objectives.metrics import evaluate_ranking
from tests.test_core import make_ratings


def test_few_shot_protocol_is_explicitly_user_present():
    dataset = leave_one_out_split(preprocess_ratings(make_ratings(), 4.0, 5, 5), "unit")
    few = few_shot_user_protocol(dataset, k=2, user_fraction=0.5, seed=10)
    assert few.n_users == dataset.n_users
    assert few.n_items == dataset.n_items
    assert "fewshot" in few.name


def test_controlled_sparsity_keeps_valid_remapped_ids():
    dataset = leave_one_out_split(preprocess_ratings(make_ratings(), 4.0, 5, 5), "unit")
    sparse = controlled_sparsity(dataset, remove_fraction=0.25, seed=5)
    assert sparse.n_users <= dataset.n_users
    assert sparse.n_items <= dataset.n_items
    assert sparse.train[:, 0].min() >= 0
    assert sparse.train[:, 0].max() < sparse.n_users
    assert sparse.train[:, 1].max() < sparse.n_items


def test_ridge_calibrator_predicts_and_scores():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(24, 11))
    y = np.column_stack([
        0.5 + 0.05 * x[:, 0],
        0.3 + 0.02 * x[:, 1],
        0.4 + 0.03 * x[:, 2],
    ])
    costs = np.exp(-2.0 + 0.1 * x[:, 3])
    calibrator = MultiFidelityCalibrator([0.01, 0.1, 1.0]).fit(x, y, costs)
    prediction = calibrator.predict(x[0])
    assert prediction.objectives.shape == (3,)
    assert np.all(prediction.standard_deviation > 0)
    assert prediction.full_cost_hours > 0
    score = calibrator.promotion_score(x[1], np.empty((0, 3)))
    assert np.isfinite(score)


def test_multi_positive_ranking_distinguishes_hit_rate_from_recall():
    class FixedRanker:
        _evaluation_adjacency = None

        def eval(self):
            return self

        def encode_users(self, users, histories, adjacency):
            return torch.zeros((len(users), 1), dtype=torch.float32)

        def score_from_hidden(self, hidden, target_items):
            item_scores = torch.tensor([100.0, 10.0, 7.0, 8.0, 6.0])
            return item_scores[target_items]

    dataset = InteractionDataset(
        1, 5,
        np.asarray([[0, 0, 5.0, 1.0]]),
        np.empty((0, 4)),
        np.asarray([[0, 1, 5.0, 2.0], [0, 2, 5.0, 3.0]]),
        ["u0"], [f"i{i}" for i in range(5)], "multi-positive"
    )
    metrics = evaluate_ranking(FixedRanker(), dataset, split="test", k=2)
    assert metrics["HR@2"] == pytest.approx(1.0)
    assert metrics["Recall@2"] == pytest.approx(0.5)
    expected_ndcg = 1.0 / (1.0 + 1.0 / np.log2(3.0))
    assert metrics["NDCG@2"] == pytest.approx(expected_ndcg)
