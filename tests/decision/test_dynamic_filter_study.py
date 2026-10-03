"""B14 software checks use synthetic chemistry, never biological evidence."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s2s_decision.schema import CONTEXT_NAMES, PROPERTY_NAMES


def runner():
    path = Path(__file__).parents[2] / "benchmarks/dynamic_filter_study.py"
    spec = importlib.util.spec_from_file_location("dynamic_filter_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(prefix="dev", n=48):
    y = np.arange(n) % 2
    frame = pd.DataFrame({name: np.ones(n) for name in PROPERTY_NAMES})
    frame = frame.assign(**dict.fromkeys(CONTEXT_NAMES, np.nan))
    frame = frame.assign(
        record_id=np.arange(n), identity=[f"{prefix}-{i}" for i in range(n)],
        murcko_scaffold=[f"{prefix}-scaf-{i // 2}" for i in range(n)],
        fingerprint_hex=[("01" if value else "00") + "00" * 255 for value in y],
        split=np.where(np.arange(n) < 40, "train", "validation"),
        y_active=y, mol_wt=200 + 20 * y, qed=0.2 + 0.5 * y,
        lipinski_violations=np.where(np.arange(n) % 7 == 0, 2, 0),
        veber_violations=0, pains_count=np.where(np.arange(n) % 5 == 0, 1, 0),
        brenk_count=0, rule_mw=(np.arange(n) % 3 == 0).astype(float),
        alert_alpha=(np.arange(n) % 5 == 0).astype(float),
    )
    return frame


def configuration(alerts=True):
    return {"feature_groups": {
        "counts": list(PROPERTY_NAMES[-4:]), "rules": ["rule_mw"],
        "alerts": ["alert_alpha"] if alerts else [],
    }}


def test_count_and_context_features_cannot_enter_baseline():
    m = runner()
    frame = fixture().query("split == 'train'")
    pre = m.Preprocessor.fit(frame)
    first = m.base_features(frame, pre)
    changed = frame.assign(**dict.fromkeys((*PROPERTY_NAMES[-4:], *CONTEXT_NAMES), 99))
    np.testing.assert_array_equal(first, m.base_features(changed, pre))
    assert first.shape == (40, 2080)


@pytest.mark.parametrize("change,match", [
    (lambda f: f.assign(y_active=np.nan), "binary"),
    (lambda f: f.assign(y_active=2), "binary"),
    (lambda f: f.assign(identity="same"), "unique"),
    (lambda f: f.assign(identity=""), "nonempty"),
    (lambda f: f.assign(record_id=1), "unique"),
    (lambda f: f.assign(qed=np.inf), "finite"),
    (lambda f: f.assign(rule_mw=np.nan), "finite"),
    (lambda f: f.assign(murcko_scaffold="shared"), "scaffold"),
    (lambda f: f.assign(split="test"), "train.*validation"),
    (lambda f: f.iloc[:0], "nonempty"),
    (lambda f: f.assign(fingerprint_hex="ff"), "fingerprint"),
    (lambda f: f.assign(record_id=[str(i) for i in range(len(f))]), "integers"),
    (lambda f: f.assign(murcko_scaffold=np.nan), "scaffold"),
    (lambda f: f.assign(qed=2), "QED"),
    (lambda f: f.assign(pains_count=-1), "nonnegative"),
    (lambda f: f.assign(y_active=1), "Both classes"),
])
def test_input_guards(change, match):
    with pytest.raises(ValueError, match=match):
        runner().validate_frame(change(fixture()), configuration(), development=True)


@pytest.mark.parametrize("groups", [
    {"wrong": []}, {"counts": [], "rules": ["y_active"], "alerts": []},
    {"counts": [], "rules": ["split"], "alerts": []},
    {"counts": [], "rules": ["identity"], "alerts": []},
    {"counts": [], "rules": ["rule_mw", "rule_mw"], "alerts": []},
    {"counts": [], "rules": "rule_mw", "alerts": []},
    {"counts": ["rule_mw"], "rules": [], "alerts": []},
])
def test_config_rejects_leaky_and_ambiguous_features(groups):
    with pytest.raises(ValueError):
        runner().validate_frame(fixture(), {"feature_groups": groups}, development=True)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("dynamic")
    dev, config = root / "development.parquet", root / "config.json"
    fixture().to_parquet(dev)
    config.write_text(json.dumps(configuration()), encoding="utf-8")
    m = runner()
    out = root / "trained"
    m.train(dev, config, out)
    return root, out, m


def test_training_fold_groups_preprocessors_and_replay(trained):
    root, out, m = trained
    model = m.load_training(out)
    assert model["baseline"]["coefficient_count"] == 2080
    folds = pd.read_parquet(out / "oof.parquet")
    dev = fixture().query("split == 'train'")
    for seed in m.SEEDS:
        rows = folds.query("seed == @seed")
        assert set(rows.fold) == set(range(5))
        assert rows.identity.tolist() == dev.identity.tolist()
        assert rows.groupby("murcko_scaffold").fold.nunique().max() == 1
        for fold in range(5):
            saved = model["folds"][f"{seed}:{fold}"]["preprocessor"]
            fit = dev.loc[~dev.identity.isin(rows.query("fold == @fold").identity)]
            assert saved == m.Preprocessor.fit(fit).to_dict()
    predicted, contributions = m.predict(fixture(), model)
    assert set(predicted) == {f"{head}:{seed}" for head in m.HEADS for seed in m.SEEDS}
    for key, scores in predicted.items():
        np.testing.assert_allclose(contributions[key].sum(axis=1), scores, atol=1e-12)
    reloaded = m.load_training(out)
    replay, _ = m.predict(fixture(), reloaded)
    for key in predicted:
        np.testing.assert_array_equal(predicted[key], replay[key])
    grid = pd.read_parquet(out / "validation-grid.parquet")
    assert len(grid) == 3 + 3 * 5 * 3
    assert grid.average_precision.between(0, 1).all()
    assert model["baseline"]["C"] == 0.1  # identical AP ties choose smaller C
    assert not (out / "evaluation").exists()


def test_evaluation_uses_common_universe_and_preserves_requested_n(trained):
    root, training, m = trained
    external = root / "external.parquet"
    frame = fixture("external", 12).drop(columns="split")
    frame.to_parquet(external)
    out = root / "evaluated"
    m.evaluate(training, external, out)
    result = pd.read_parquet(out / "selection-metrics.parquet")
    assert len(result) == 3 * 9 * 3 * 2
    assert set(result.requested_n) == {20, 50, 100}
    assert (result.shortfall == result.requested_n - result.final_count).all()
    assert (result.total_positives == 6).all()
    assert (result.final_count <= 12).all()
    np.testing.assert_allclose(result.enrichment_selected, result.precision_selected / 0.5)
    np.testing.assert_allclose(result.enrichment_requested_n, result.hits_per_requested_n / 0.5)
    nofilter = result.query("arm == 'nofilter_control'")
    assert (nofilter.final_count == 12).all()
    default = result.query("arm == 'default_activity'")
    assert (default.filter_removed_positives == 1).all()
    ranking = pd.read_parquet(out / "rankings.parquet")
    assert len(ranking) == 3 * 9 * 12
    assert ranking.groupby(["seed", "arm"]).identity.nunique().eq(12).all()
    primary = pd.read_parquet(out / "paired-comparisons.parquet")
    assert set(primary.comparator) == {"nofilter_control", "default_activity"}
    assert len(primary) == 3 * 2 * 3 * 2
    assert (out / "completion.json").exists()


def test_external_overlap_and_unknown_label_rejected_before_output(trained):
    root, training, m = trained
    for name, frame, message in (
        ("overlap", fixture().drop(columns="split"), "identity overlap"),
        ("unknown", fixture("new").assign(y_active=np.nan), "binary"),
    ):
        path = root / f"{name}.parquet"
        frame.to_parquet(path)
        with pytest.raises(ValueError, match=message):
            m.evaluate(training, path, root / name)
        assert not (root / name).exists()


def test_empty_alert_group_and_one_class_metric():
    m = runner()
    m.validate_frame(fixture(), configuration(False), development=True)
    assert m.head_columns(configuration(False), "alerts") == []
    assert m.ranking_metrics(np.zeros(3), np.arange(3)) == {
        "average_precision": None, "roc_auc": None,
        "status": "one_class", "observed": 3, "positives": 0,
    }


def test_selection_empty_pool_quota_and_extreme_logit_order():
    m = runner()
    frame = fixture("quota", 12).assign(murcko_scaffold="one")
    metrics, selected = m.selection_result(frame, np.arange(12) + 1000,
                                           np.ones(12, dtype=bool), 5, 3)
    assert metrics["final_count"] == 3
    assert metrics["shortfall"] == 2
    assert set(selected.manifest["final_ids"]) == {9, 10, 11}
    metrics, _ = m.selection_result(frame, np.arange(12), np.zeros(12, dtype=bool), 5, 3)
    assert metrics["final_count"] == 0
    assert metrics["precision_selected"] is None
    assert metrics["shortfall"] == 5
    with pytest.raises(ValueError, match="positive integer"):
        m.selection_result(frame, np.arange(12), np.ones(12, dtype=bool), 0, None)


def test_crossfit_support_and_control_orientation_fail_closed(monkeypatch):
    m = runner()
    frame = fixture().query("split == 'train'")
    with pytest.raises(ValueError, match="five.*scaffolds"):
        m.crossfit(frame.assign(murcko_scaffold="one"), 0.1, 42)
    monkeypatch.setattr(m, "select_regularization", lambda *args:
                        ({"weights": [-0.5], "intercept": 0}, []))
    with pytest.raises(ValueError, match="not positive"):
        m.fit_heads(frame, frame, np.arange(40), np.arange(40), configuration(), 42)


def test_nonconvergence_is_explicit(monkeypatch):
    import warnings
    m = runner()

    class Nonconverged:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            warnings.warn("synthetic convergence failure", m.ConvergenceWarning)
            return self

    monkeypatch.setattr(m, "LogisticRegression", Nonconverged)
    with pytest.raises(ValueError, match="did not converge"):
        m.fit_logistic(np.zeros((2, 1)), [0, 1], 0.1)


def test_null_and_unicode_inputs_and_large_feature_batch():
    m = runner()
    with pytest.raises(ValueError, match="JSON object"):
        m.validate_config(None)
    with pytest.raises(ValueError, match="dataframe"):
        m.validate_frame(None, configuration())
    frame = fixture("molécula-α-🧪", 10002).drop(columns="split")
    m.validate_frame(frame, configuration())
    pre = m.Preprocessor.fit(frame.iloc[:100])
    values = m.base_features(frame, pre)
    assert values.shape == (10002, 2080)
    assert np.isfinite(values).all()


def test_corrupt_training_manifest_rejected(trained, tmp_path):
    _, original, m = trained
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(original, copy)
    with (copy / "models.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="changed"):
        m.load_training(copy)


def test_cli_dispatch(monkeypatch, tmp_path):
    m = runner()
    calls = []
    monkeypatch.setattr(m, "train", lambda *args: calls.append(("train", args)))
    monkeypatch.setattr(m, "evaluate", lambda *args: calls.append(("evaluate", args)))
    m.main(["train", "--development", "d", "--config", "c", "--output", "o"])
    m.main(["evaluate", "--training", "t", "--external", "e", "--output", "o"])
    assert calls == [("train", (Path("d"), Path("c"), Path("o"))),
                     ("evaluate", (Path("t"), Path("e"), Path("o")))]
