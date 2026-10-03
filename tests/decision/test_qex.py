"""Checks for the independently implemented published QEX equations."""

import dataclasses
import itertools
import json

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import QED

from s2s_decision.qex import (
    PROPERTY_NAMES,
    QEXFitError,
    QEXModel,
    _ads,
    _entropy_weights,
    _fit_curve,
    fit_qex,
    predict_qex,
    qed_properties,
)


def synthetic_properties(n=80):
    rng = np.random.default_rng(138)
    result = np.column_stack([
        rng.normal(350, 50, n), rng.normal(2, 1, n), rng.integers(0, 9, n),
        rng.integers(0, 5, n), rng.normal(75, 15, n), rng.integers(0, 8, n),
        rng.integers(0, 5, n), rng.integers(0, 4, n),
    ])
    return result


def test_actual_rdkit_qed_properties_and_no_input_mutation():
    smiles = ['CCO', 'c1ccccc1O', 'C[N+](=O)[O-]']
    expected = np.array([QED.properties(Chem.MolFromSmiles(s)) for s in smiles])
    assert PROPERTY_NAMES == QED.QEDproperties._fields
    np.testing.assert_array_equal(qed_properties(smiles), expected)
    assert smiles == ['CCO', 'c1ccccc1O', 'C[N+](=O)[O-]']
    assert qed_properties([]).shape == (0, 8)


@pytest.mark.parametrize('smiles', [['not smiles'], [''], [None], 'CCO'])
def test_smiles_input_guard(smiles):
    with pytest.raises(ValueError):
        qed_properties(smiles)


def test_ads_matches_published_equation_and_extreme_tails():
    p = (0.1, 2.0, 0.3, 1.2, 0.4, 0.7)
    x = np.linspace(-5, 5, 20)
    a, b, c, d, e, f = p
    expected = a + b / (1 + np.exp(-(x-c+d/2)/e)) * (
        1 - 1 / (1 + np.exp(-(x-c-d/2)/f)))
    np.testing.assert_allclose(_ads(x, p), expected, rtol=1e-13)
    np.testing.assert_allclose(_ads(np.array([-1e300, 1e300]), p), [a, a])


def test_entropy_weight_search_matches_exhaustive_independent_calculation():
    # 390,624 nonzero weight vectors, including all published quarter steps.
    desirability = np.array([[.15, .31, .51, .72, .92, .1, .35, .6],
                            [.9, .8, .7, .6, .5, .4, .3, .2]])
    grid = np.array(list(itertools.product((0, .25, .5, .75, 1), repeat=8)))[1:]
    log_s = np.log(desirability) @ grid.T / grid.sum(axis=1)
    scores = np.exp(log_s)
    entropy = -(scores * np.log2(scores)).sum(axis=0)
    selected = np.lexsort((np.arange(len(grid)), -entropy))[:1000]
    expected = grid[selected].mean(axis=0)
    np.testing.assert_allclose(_entropy_weights(np.log(desirability)), expected, atol=0)


def test_curve_normalizer_is_global_maximum_and_constant_is_unavailable():
    curve = _fit_curve(synthetic_properties()[:, 0], 'MW')
    raw = _ads(np.linspace(-80, 80, 100000), curve.parameters)
    assert raw.max() <= curve.normalizer * (1 + 1e-10)
    assert raw.max() >= curve.normalizer * .999
    assert curve.converged_starts > 0
    with pytest.raises(QEXFitError, match='constant'):
        _fit_curve(np.ones(20), 'HBD')


def test_fit_prediction_immutable_roundtrip_and_external_independence():
    train = synthetic_properties()
    original = train.copy()
    model = fit_qex(train)
    scores = predict_qex(model, train[:10])
    np.testing.assert_array_equal(train, original)
    assert scores.shape == (10,)
    assert np.isfinite(scores).all() and (scores > 0).all() and (scores <= 1).all()
    assert model.training_count == len(train)
    assert len(model.curves) == len(model.weights) == 8
    assert model.to_dict()['weight_candidates'] == 390624
    state = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = QEXModel.from_dict(state)
    np.testing.assert_array_equal(scores, predict_qex(restored, train[:10]))
    expanded = np.vstack([train[:10], np.full((1, 8), 1e6)])
    np.testing.assert_array_equal(scores, predict_qex(restored, expanded)[:10])
    assert predict_qex(restored, np.empty((0, 8))).shape == (0,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.training_count = 100
    state['weights'][0] = 0
    assert model.weights[0] != 0


@pytest.mark.parametrize('data', [np.ones((0, 8)), np.ones((3, 8)), np.ones((10, 7)),
                                  np.full((20, 8), np.nan), np.full((20, 8), np.inf)])
def test_training_input_guard(data):
    with pytest.raises(ValueError):
        fit_qex(data)


def test_negative_physical_property_and_fractional_counts_rejected():
    data = synthetic_properties()
    bad = data.copy()
    bad[0, 0] = -1
    with pytest.raises(ValueError, match='nonnegative'):
        fit_qex(bad)
    bad = data.copy()
    bad[0, 7] = .5
    with pytest.raises(ValueError, match='integer'):
        fit_qex(bad)


def test_untrusted_model_validation():
    with pytest.raises(ValueError):
        QEXModel.from_dict({})
    with pytest.raises(ValueError):
        QEXModel.from_dict({'method': 'arbitrary'})


def test_loaded_models_reject_tampered_parameters_and_mutable_containers():
    model = fit_qex(synthetic_properties())
    for key, value in [('training_count', 0), ('training_sha256', 'bad'),
                       ('rdkit_version', ''), ('weights', [0] * 8),
                       ('weight_candidates', 12), ('properties', list(reversed(PROPERTY_NAMES)))]:
        state = {**model.to_dict(), key: value}
        with pytest.raises(ValueError):
            QEXModel.from_dict(state)
    for key, value in [('parameters', [0] * 6), ('normalizer', 999),
                       ('histogram_bins', 1), ('converged_starts', 0), ('rmse', np.inf)]:
        state = model.to_dict()
        state['curves'][0][key] = value
        with pytest.raises(ValueError):
            QEXModel.from_dict(state)
    with pytest.raises(ValueError, match='immutable'):
        dataclasses.replace(model, weights=list(model.weights))
    with pytest.raises(ValueError, match='structure'):
        QEXModel.from_dict({'method': model.to_dict()['method']})


def test_fit_failure_is_explicit(monkeypatch):
    from s2s_decision import qex
    def failed(*args, **kwargs):
        raise ValueError('synthetic optimizer failure')
    monkeypatch.setattr(qex, 'least_squares', failed)
    with pytest.raises(QEXFitError, match='no converged'):
        _fit_curve(synthetic_properties()[:, 0], 'MW')
