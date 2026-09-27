import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from s2s_decision.context import build_context, training_context
from s2s_decision.splits import available_dates


def rows():
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return pd.DataFrame(
        [
            {
                "identity": f"m{i}",
                "murcko_scaffold": f"s{i}",
                "fingerprint_hex": DataStructs.BitVectToBinaryText(
                    generator.GetFingerprint(Chem.MolFromSmiles(s))
                ).hex(),
                "pactivity": 7 if i % 2 else 5,
                "y_active": i % 2,
                "split": "train",
                "year": 2000 + i,
            }
            for i, s in enumerate(["CCO", "CCN", "CCC", "CCCC", "CCCl", "COC"])
        ]
    )


def test_exact_context_excludes_identity_and_missing_class():
    data = rows()
    context = build_context(data.iloc[:1], data.iloc[:1])
    assert context.isna().all().all()
    context = build_context(data.iloc[:1], data.iloc[1:2])
    assert np.isnan(context.iloc[0].similarity_inactive_max)
    assert context.iloc[0].similarity_active_max < 1
    assert context.iloc[0].novelty == 1 - context.iloc[0].similarity_active_max


def test_oof_excludes_same_scaffold():
    data = rows()
    data["murcko_scaffold"] = "same"
    assert training_context(data).isna().all().all()


def test_temporal_uses_only_past_train():
    data = rows()
    result = training_context(data, temporal=True)
    assert result.iloc[0].isna().all()
    assert np.isnan(result.iloc[1].similarity_active_max)
    assert np.isfinite(result.iloc[1].similarity_inactive_max)


def test_reference_conflicts_and_dimensions_are_rejected_or_excluded():
    import pytest

    data = rows()
    refs = pd.concat([data.iloc[1:2], data.iloc[1:2].assign(y_active=0)])
    assert build_context(data.iloc[:1], refs).isna().all().all()
    refs = data.iloc[1:2].assign(fingerprint_hex="01")
    with pytest.raises(ValueError, match="dimensions"):
        build_context(data.iloc[:1], refs)


def test_reference_fallback_does_not_label_missing():
    data = rows().drop(columns="y_active")
    refs = data.iloc[1:3].copy()
    refs.loc[refs.index[0], "pactivity"] = np.nan
    context = build_context(data.iloc[:1], refs)
    assert np.isnan(context.iloc[0].similarity_active_max)
    assert np.isfinite(context.iloc[0].similarity_inactive_max)


def test_identity_leakage_is_rejected():
    import pytest

    data = pd.concat([rows(), rows().iloc[:1].assign(split="test")], ignore_index=True)
    with pytest.raises(ValueError, match="Identity leakage"):
        training_context(data)


def test_context_rejects_mixed_tasks_and_censored_fallback():
    import pytest

    data = rows().assign(target_id="T", endpoint="Ki")
    with pytest.raises(ValueError, match="target|endpoint"):
        build_context(data.iloc[:1], data.iloc[1:].assign(target_id="Other"))
    censored = data.iloc[1:2].drop(columns="y_active").assign(relation=">")
    assert build_context(data.iloc[:1], censored).isna().all().all()


def test_temporal_groups_dates_without_changing_rowwise_context(monkeypatch):
    import s2s_decision.context as context_module

    data = rows()
    data["year"] = [2000, 2001, 2001, 2003, 2004, 2004]
    data.loc[4:, "split"] = ["calibration", "test"]
    # Replicate identity availability is its latest date, not its first appearance.
    data = pd.concat([data, data.iloc[:1].assign(year=2001)], ignore_index=True)
    data.index = [41, 3, 7, 20, 11, 2, 9]
    original = data.copy(deep=True)
    dates = available_dates(data)
    train = data.loc[data.split.eq("train")]
    expected = pd.concat(
        [
            build_context(data.loc[[index]], train.loc[dates.loc[train.index].lt(dates.loc[index])])
            for index in data.index
        ]
    )
    calls = []

    def counted(queries, references, threshold=6.0):
        calls.append(len(queries))
        return build_context(queries, references, threshold)

    monkeypatch.setattr(context_module, "build_context", counted)
    actual = training_context(data, temporal=True)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(data, original)
    assert actual.loc[41].isna().all()  # Same-date evidence and self remain excluded.
    assert len(calls) == dates.nunique()
    assert sum(calls) == len(data)


def test_batched_context_matches_individual_queries_and_preserves_inputs():
    data = rows()
    data = pd.concat([data, data.iloc[:1]], ignore_index=True)
    original = data.copy(deep=True)
    expected = pd.concat([build_context(data.loc[[index]], data) for index in data.index])
    pd.testing.assert_frame_equal(build_context(data, data), expected)
    pd.testing.assert_frame_equal(data, original)
