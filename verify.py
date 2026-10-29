import argparse
from core import *
from evaluate import check_gate, verify_saved


def main(outputs_only=False):
    gate = None if outputs_only else check_gate()
    t = verify_saved()
    r = json.loads((ROOT / "results/test_results.json").read_text())
    assert len(t) == r["queries"]
    assert t.query_id.is_unique
    for v in VARIANTS:
        for f in ["ndcg", "recall"]:
            np.testing.assert_allclose(
                t[v + "_ensemble_" + f].mean(),
                r["models"][v]["ensemble"][f],
                rtol=0,
                atol=1e-15,
            )
    for name, left, right in [
        ("auto_both_minus_manual", "auto_both", "manual"),
        ("auto_both_minus_random_matched", "auto_both", "random_matched"),
    ]:
        b = bootstrap(t, left + "_ensemble_ndcg", right + "_ensemble_ndcg")
        for k in ["difference", "ci95"]:
            np.testing.assert_allclose(
                b[k], r["contrasts"][name][k], rtol=0, atol=1e-15
            )
    selection = json.loads((ROOT / "results/selection.json").read_text())
    assert selection["selection_split"] == "validation October21-24 only"
    for f in ["windows", "combinations"]:
        assert len(selection["selected"][f]) == len(selection["random_matched"][f])
    diag = np.load(ROOT / "results/selection_diagnostics.npz")
    p = np.exp(diag["manual_scores"] - diag["manual_scores"].max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        diag["labels"] - p, diag["residuals"], rtol=0, atol=1e-15
    )
    source = json.loads((ROOT / "results/full_history_verification.json").read_text())
    assert source["passed"] and source["users"] == 5000
    manifest = json.loads((ROOT / "data/extraction_manifest.json").read_text())
    assert (
        source["source_events_replayed"] == manifest["source_events"]
        and source["events"] == manifest["selected_events"]
    )
    screening = pd.read_csv(ROOT / "results/feature_screening.csv.gz")
    for f in ["windows", "combinations"]:
        assert set(
            screening.loc[screening.family.eq(f) & screening.selected, "index"]
        ) == set(selection["selected"][f])
    qv = pd.read_csv(
        ROOT / "results/validation_queries.csv.gz", dtype={"query_id": str}
    )
    assert (
        diag["validation_query_ids"].tolist()
        == qv.loc[qv.candidate_covered, "query_id"].tolist()
    )
    assert not set(t.query_id) & set(diag["validation_query_ids"])
    gzips = 0
    for path in (ROOT / "data").rglob("*.gz"):
        with gzip.open(path, "rb") as f:
            while f.read(1048576):
                pass
        gzips += 1
    status = dict(
        passed=True,
        queries=len(t),
        users=int(t.user_id.nunique()),
        ensemble_and_heuristic_top20_lists=len(t) * 7,
        individual_ensemble_and_heuristic_rank_lists=len(t) * 22,
        frozen_files=0 if gate is None else len(gate["sha256"]),
        model_and_data_hashes_checked=not outputs_only,
        models=15,
        full_source_replay=source,
        validation_selection_recorded=True,
        matched_feature_counts=True,
        primary_and_random_bootstraps_reproduced=True,
        gzip_files_verified=gzips,
    )
    if outputs_only:
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    else:
        write_json(ROOT / "results/verification.json", status)
    print("Final checks passed", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-only", action="store_true")
    main(ap.parse_args().outputs_only)
