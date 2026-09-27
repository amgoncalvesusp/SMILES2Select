import pandas as pd
import pytest

from s2s_decision.splits import assign_splits


def frame():
    return pd.DataFrame(
        [
            {"identity": f"m{i // 2}", "murcko_scaffold": f"s{i // 4}", "year": 2000 + i // 2}
            for i in range(40)
        ]
    )


def test_grouped_scaffolds_and_reproducibility():
    result = assign_splits(frame())
    assert result.groupby("identity").split.nunique().max() == 1
    assert result.groupby("murcko_scaffold").split.nunique().max() == 1
    assert set(result.split) == {"train", "validation", "calibration", "test"}
    pd.testing.assert_frame_equal(result, assign_splits(frame()))


def test_temporal_never_splits_identity_or_looks_forward():
    data = frame()
    data.loc[1, "year"] = 2050
    result = assign_splits(data, method="temporal")
    assert result.groupby("identity").split.nunique().max() == 1
    assert result.loc[result.identity.eq("m0"), "split"].eq("test").all()
    assert (
        result.loc[result.split.eq("train"), "year"].max()
        < result.loc[result.split.eq("test"), "year"].max()
    )


def test_temporal_requires_dates():
    with pytest.raises(ValueError, match="date|year"):
        assign_splits(frame().drop(columns="year"), method="temporal")


def test_insufficient_groups_is_explicit():
    with pytest.raises(ValueError, match="four independent"):
        assign_splits(frame().assign(murcko_scaffold="same"))


def test_random_keeps_identity_grouped():
    result = assign_splits(frame(), method="random")
    assert result.groupby("identity").split.nunique().max() == 1
