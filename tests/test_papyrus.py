"""Local Papyrus evidence must preserve identity and endpoint boundaries."""

import gzip
import json
import lzma
import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from smiles2select.reference.papyrus import (
    annotate_candidates,
    import_index,
    lookup_evidence,
    main,
    read_index_metadata,
)

KEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
OTHER = "AAAAAAAAAAAAAA-UHFFFAOYSA-N"


def source_file(tmp_path, rows=None, compressed=False):
    row = {
        "InChIKey": KEY,
        "connectivity": KEY[:14],
        "target_id": "P12345_WT",
        "Quality": "high",
        "type_IC50": "1",
        "type_EC50": "0",
        "type_KD": "0",
        "type_Ki": "0",
        "type_other": "0",
        "relation": "=",
        "pchembl_value_Mean": "7.2",
        "source": "chembl",
        "Activity_ID": "activity-1",
    }
    data = pd.DataFrame([dict(row, **changes) for changes in (rows or [{}])])
    path = tmp_path / ("papyrus.tsv.gz" if compressed else "papyrus.tsv")
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            data.to_csv(stream, sep="\t", index=False)
    else:
        data.to_csv(path, sep="\t", index=False)
    return path


def build(tmp_path, rows=None, **kwargs):
    return import_index(
        source_file(tmp_path, rows),
        tmp_path / "evidence.sqlite",
        target_id="P12345_WT",
        endpoint="IC50",
        dataset_version="05.7",
        **kwargs,
    )


def test_streaming_filters_preserve_identity_endpoint_and_missing_evidence(tmp_path):
    progress = []
    report = build(
        tmp_path,
        [
            {},
            {"pchembl_value_Mean": "6.1"},
            {"target_id": "OTHER"},
            {"Quality": "low"},
            {"type_Ki": "1"},
            {"relation": "<"},
        ],
        chunk_size=2,
        progress=lambda seen, kept: progress.append((seen, kept)),
    )
    assert report.rows_read == 6
    assert report.rows_indexed == 2
    assert report.mixed_endpoint_rows == 1
    assert report.censored_rows == 1
    assert progress[-1] == (6, 2)
    result = lookup_evidence(report.index_path, [KEY, OTHER, None])
    assert result.papyrus_records.tolist() == [2, 0, 0]
    assert result.iloc[0].papyrus_pchembl_min == 6.1
    assert result.iloc[0].papyrus_pchembl_max == 7.2
    assert pd.isna(result.iloc[1].papyrus_pchembl_min)
    assert result.iloc[0].papyrus_identity_level == "connectivity"
    metadata = read_index_metadata(report.index_path)
    assert metadata["dataset_version"] == "05.7"
    assert metadata["endpoint"] == "IC50"
    with sqlite3.connect(report.index_path) as conn:
        assert (
            conn.execute("SELECT activity_id FROM evidence LIMIT 1").fetchone()[0] == "activity-1"
        )


def test_full_identity_and_annotations_preserve_input(tmp_path):
    report = build(tmp_path, identity_level="inchikey")
    candidates = pd.DataFrame({"inchikey": [KEY, KEY[:14] + "-AAAAAAAAAA-N"]}, index=[8, 3])
    annotations = annotate_candidates(candidates, report.index_path)
    assert annotations.index.tolist() == [8, 3]
    assert annotations.papyrus_records.tolist() == [1, 0]
    assert candidates.columns.tolist() == ["inchikey"]


def test_gzip_and_semicolon_pure_endpoint(tmp_path):
    source = source_file(tmp_path, [{"type_IC50": "1;1", "relation": "=;="}], True)
    report = import_index(
        source,
        tmp_path / "e.sqlite",
        target_id="P12345_WT",
        endpoint="IC50",
        dataset_version="05.7",
    )
    assert report.rows_indexed == 1


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"pchembl_value_Mean": "inf"}, "pChEMBL"),
        ({"InChIKey": "bad", "connectivity": "bad"}, "identity"),
        ({"type_IC50": "oops"}, "endpoint"),
    ],
)
def test_malformed_matching_rows_fail_without_partial_index(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        build(tmp_path, [{}, changes], chunk_size=1)
    assert not (tmp_path / "evidence.sqlite").exists()
    assert not list(tmp_path.glob("*.partial*"))


def test_schema_validation_and_cancel_preserve_existing_index(tmp_path):
    report = build(tmp_path)
    previous = report.index_path.read_bytes()
    with pytest.raises(FileExistsError):
        build(tmp_path)
    assert report.index_path.read_bytes() == previous
    report.index_path.unlink()
    with pytest.raises(InterruptedError):
        build(tmp_path, cancelled=lambda: True)
    assert not report.index_path.exists()
    malformed = tmp_path / "bad.csv"
    malformed.write_text("target_id,Quality\nP12345_WT,high\n")
    with pytest.raises(ValueError, match="columns"):
        import_index(
            malformed,
            report.index_path,
            target_id="P12345_WT",
            endpoint="IC50",
            dataset_version="05.7",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_id": ""},
        {"endpoint": "other"},
        {"dataset_version": ""},
        {"qualities": ()},
        {"chunk_size": 0},
        {"identity_level": "smiles"},
    ],
)
def test_requires_explicit_valid_scope(tmp_path, kwargs):
    params = {"target_id": "P12345_WT", "endpoint": "IC50", "dataset_version": "05.7"}
    params.update(kwargs)
    with pytest.raises(ValueError):
        import_index(source_file(tmp_path), tmp_path / "e.sqlite", **params)


def test_read_only_and_metadata_validation(tmp_path):
    absent = tmp_path / "absent.sqlite"
    with pytest.raises(FileNotFoundError):
        lookup_evidence(absent, [KEY])
    assert not absent.exists()
    bad = tmp_path / "bad.sqlite"
    with sqlite3.connect(bad) as conn:
        conn.execute("CREATE TABLE metadata (payload TEXT)")
        conn.execute("INSERT INTO metadata VALUES (?)", (json.dumps({"schema_version": 999}),))
    with pytest.raises(ValueError, match="index"):
        read_index_metadata(bad)


def test_xz_csv_lookup_across_batches_and_read_connections_close(tmp_path):
    source = source_file(tmp_path)
    compressed = tmp_path / "subset.csv.xz"
    with lzma.open(compressed, "wt", encoding="utf-8") as stream:
        pd.read_csv(source, sep="\t").to_csv(stream, index=False)
    report = import_index(
        compressed,
        tmp_path / "e.sqlite",
        target_id="P12345_WT",
        endpoint="IC50",
        dataset_version="05.7",
    )
    result = lookup_evidence(report.index_path, [KEY, OTHER] * 601)
    assert result.papyrus_records.tolist() == [1, 0] * 601
    assert lookup_evidence(report.index_path, []).empty
    assert lookup_evidence(report.index_path, [pd.NA]).papyrus_records.tolist() == [0]
    with closing(sqlite3.connect(report.index_path)) as connection:
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM evidence WHERE identity=?", (KEY[:14],)
        ).fetchall()
        assert any("evidence_identity" in str(row) for row in plan)
    report.index_path.unlink()  # Windows requires all reader connections to close.


def test_cli_and_input_boundaries(tmp_path, capsys):
    source = source_file(tmp_path)
    index = tmp_path / "cli.sqlite"
    assert (
        main(
            [
                str(source),
                str(index),
                "--target-id",
                "P12345_WT",
                "--endpoint",
                "IC50",
                "--dataset-version",
                "05.7",
            ]
        )
        == 0
    )
    assert '"rows_indexed": 1' in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(
            [
                str(source),
                str(index),
                "--target-id",
                "P12345_WT",
                "--endpoint",
                "IC50",
                "--dataset-version",
                "05.7",
            ]
        )
    assert exc.value.code == 1
    with pytest.raises(ValueError, match="Candidate"):
        annotate_candidates(pd.DataFrame(), index)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"pchembl_value_Mean": "bad"}, "pChEMBL"),
        ({"relation": "bad"}, "relation"),
        ({"source": ""}, "source"),
    ],
)
def test_invalid_numeric_relation_and_provenance(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        build(tmp_path, [changes])


def test_fallback_identity_and_nonmatching_endpoint(tmp_path):
    source = source_file(tmp_path, [{}, {"type_IC50": "0", "type_Ki": "1"}])
    frame = pd.read_csv(source, sep="\t").drop(columns="connectivity")
    frame.to_csv(source, sep="\t", index=False)
    report = import_index(
        source,
        tmp_path / "e.sqlite",
        target_id="P12345_WT",
        endpoint="IC50",
        dataset_version="05.7",
    )
    assert report.rows_indexed == 1
