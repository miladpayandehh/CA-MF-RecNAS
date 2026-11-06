import random

import numpy as np
import pandas as pd
import pytest
import torch

from camf_recnas.data.dataset import InteractionDataset, leave_one_out_split
from camf_recnas.data.preprocess import preprocess_ratings
from camf_recnas.models.encoding import Architecture, BlockGene, SearchSpace
from camf_recnas.models.recommender import Recommender
from camf_recnas.objectives.hypervolume import hypervolume_3d
from camf_recnas.objectives.pareto import nondominated_sort, nsga2_select
from camf_recnas.search.automata import CostAwareLearningAutomata
from camf_recnas.search.fuzzy import FuzzyRateController
from camf_recnas.search.operators import OPERATOR_NAMES, crossover, mutate
from camf_recnas.objectives.statistics import benjamini_hochberg, mean_confidence_interval


def make_ratings(n_users=8, n_items=8, per_user=6):
    rows = []
    for user in range(n_users):
        for step in range(per_user):
            rows.append((f"u{user}", f"i{(user + step) % n_items}", 5.0,
                         1000 + step))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_preprocessing_and_leave_one_out_protocol():
    ratings = make_ratings()
    # Every user-item pair is unique; include one rating below the positive threshold.
    ratings.loc[len(ratings)] = ["u0", "negative", 2.0, 9999]
    cleaned = preprocess_ratings(ratings, 4.0, 5, 5)
    assert (cleaned.rating >= 4.0).all()
    dataset = leave_one_out_split(cleaned, "unit")
    assert dataset.n_users == 8
    assert len(dataset.validation) == dataset.n_users
    assert len(dataset.test) == dataset.n_users
    assert len(dataset.train) == dataset.n_users * 4


def test_low_fidelity_is_deterministic_and_keeps_each_train_user():
    dataset = leave_one_out_split(make_ratings(), "unit")
    a = dataset.low_fidelity_train(0.05, 123)
    b = dataset.low_fidelity_train(0.05, 123)
    np.testing.assert_array_equal(a, b)
    assert set(a[:, 0].astype(int)) == set(dataset.train[:, 0].astype(int))


def test_hypervolume_and_pareto_selection():
    ref = np.array([1.0, 1.0, 1.0])
    one = hypervolume_3d(np.array([[0.5, 0.5, 0.5]]), ref)
    assert one == pytest.approx(0.125)
    points = np.array([[0.2, 0.8, 0.8], [0.8, 0.2, 0.8], [0.8, 0.8, 0.2],
                       [0.9, 0.9, 0.9]])
    assert hypervolume_3d(points, ref) == pytest.approx(0.08)
    fronts = nondominated_sort(points)
    assert set(fronts[0]) == {0, 1, 2}
    selected = nsga2_select(points, 3)
    assert len(selected) == 3


def test_fuzzy_controller_and_learning_automata_invariants():
    controller = FuzzyRateController(resolution=301)
    pc, pm = controller.update(-0.2, 0.1)
    assert 0.4 <= pc <= 0.95
    assert 0.005 <= pm <= 0.30

    automata = CostAwareLearningAutomata(learning_rate=0.1, beta=0.3)
    slots = automata.allocate_slots(40, automata.probabilities)
    assert slots.sum() == 40
    assert (slots >= 2).all()
    state = automata.update(np.array([0.0, 1.0, 0.0]))
    assert np.isclose(automata.probabilities.sum(), 1.0)
    assert np.all(automata.probabilities >= 0)
    assert len(state["rewarded_operators"]) >= 1


def test_search_space_crossover_and_mutation_are_valid():
    space = SearchSpace(max_blocks=8)
    rng = random.Random(7)
    a, b = space.sample(rng), space.sample(rng)
    for operator in OPERATOR_NAMES:
        child_a, child_b = crossover(a, b, operator, rng, space)
        child_a.validate(max_blocks=8)
        child_b.validate(max_blocks=8)
    mutated = mutate(a, rng, space, mutation_probability=1.0)
    mutated.validate(max_blocks=8)
    assert 1 <= len(mutated.blocks) <= 8

    # A field that is not used by a block family must not change its genotype ID.
    base = Architecture(32, "product", (BlockGene("gcn", "ReLU"),), 64, 0.1)
    altered_unused_gene = Architecture(
        32, "product", (BlockGene("gcn", "ReLU", heads=8, kernel_size=5, residual=False),),
        64, 0.1, repair_count=99, metadata={"note": "audit-only"})
    assert base.fingerprint == altered_unused_gene.fingerprint


def test_all_block_families_forward_and_bpr_gradient():
    architecture = Architecture(
        embedding_dim=16,
        interaction_encoder="concat",
        blocks=(
            BlockGene("gcn", "ReLU"),
            BlockGene("attention", "ELU", heads=4),
            BlockGene("dwc", "SELU", kernel_size=3),
            BlockGene("mlp", "LeakyReLU"),
        ),
        head_width=64,
        head_dropout=0.1,
    )
    model = Recommender(3, 6, architecture, embedding_dropout=0.0)
    train = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 2], [2, 3], [2, 4]],
                         dtype=torch.float32)
    adjacency = model.build_normalized_adjacency(train, 3, 6, torch.device("cpu"))
    users = torch.tensor([0, 1, 2], dtype=torch.long)
    items_pos = torch.tensor([2, 3, 5], dtype=torch.long)
    items_neg = torch.tensor([4, 5, 0], dtype=torch.long)
    histories = torch.tensor([[-1, 0, 1], [-1, 1, 2], [-1, 3, 4]], dtype=torch.long)
    pos = model(users, items_pos, histories, adjacency)
    neg = model(users, items_neg, histories, adjacency)
    loss = -torch.nn.functional.logsigmoid(pos - neg).mean()
    loss.backward()
    assert pos.shape == (3,)
    assert torch.isfinite(loss)
    assert model.parameter_count > 0


def test_confidence_interval_and_bh_adjustment():
    mean, low, high = mean_confidence_interval([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert low < mean < high
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert adjusted.shape == (3,)
    assert np.all(adjusted >= np.array([0.01, 0.04, 0.03]))
    assert np.all(adjusted <= 1.0)
