"""Frozen final test. Same users, cutoffs and candidates across all variants."""

import argparse, datetime, shutil, time
from catboost import CatBoostRanker, Pool
from core import *
from encoding import encode_manual, encode_library


def check_gate():
    g = json.loads((ROOT / "test_gate.json").read_text())
    for p, h in g["sha256"].items():
        assert sha(ROOT / p) == h, p
    return g


def freeze():
    assert not (ROOT / "test_gate.json").exists()
    assert json.loads((ROOT / "results/full_history_verification.json").read_text())[
        "passed"
    ]
    out = ROOT / "results"
    cache = ROOT / "cache"
    for name in [
        "feature_names.json",
        "library_names.json",
        "train_queries.csv.gz",
        "train_fit_queries.csv.gz",
        "validation_queries.csv.gz",
        "train_query_indices.npy",
        "train_negative_positions.npy",
    ]:
        shutil.copyfile(cache / name, out / name)
    vy = np.load(cache / "validation_labels_candidates.npz")["labels"]
    validation = {}
    for v in VARIANTS:
        pred = np.mean(
            [np.load(cache / f"validation_scores_{v}_{s}.npy") for s in SEEDS], axis=0
        )
        validation[v] = summarize(per_query(pred, vy))
    hn = json.loads((cache / "feature_names.json").read_text())
    hs = heuristic_scores(np.load(cache / "validation_X.npy", mmap_mode="r"), hn)
    write_json(
        out / "validation_heuristics.json",
        {
            h: summarize(per_query(score.reshape(vy.shape), vy))
            for h, score in hs.items()
        },
    )
    write_json(out / "validation_ensembles.json", validation)
    names = json.loads((cache / "feature_names.json").read_text())
    sel = json.loads((out / "selection.json").read_text())
    models = sorted((ROOT / "models").glob("*.cbm"))
    assert len(models) == 15
    data_files = (
        sorted((ROOT / "data").glob("*.csv.gz"))
        + sorted((ROOT / "data").glob("*.json"))
        + sorted((ROOT / "data/catalog_daily").glob("*.csv.gz"))
    )
    write_json(
        ROOT / "data/dataset_hashes.json",
        {
            p.relative_to(ROOT / "data").as_posix(): sha(p)
            for p in data_files
            if p.name != "dataset_hashes.json"
        },
    )
    for p in models:
        m = CatBoostRanker()
        m.load_model(str(p))
        variant = p.stem.rsplit("_", 1)[0]
        expected = (
            names["numeric"]
            + [sel["names"][f][j] for f, j in extra_parts(variant, sel)]
            + names["categorical"]
        )
        assert m.feature_names_ == expected and m.tree_count_ == 261
        rec = json.loads((out / (p.stem + ".json")).read_text())
        assert (
            rec["sha256"] == sha(p)
            and rec["actual_parameters"]["counter_calc_method"] == "SkipTest"
        )
    files = (
        sorted(ROOT.glob("*.py"))
        + [ROOT / "PROTOCOL.md"]
        + models
        + sorted(out.glob("*.json"))
        + sorted(out.glob("*.npy"))
        + sorted(out.glob("*.npz"))
        + sorted(out.glob("*.csv.gz"))
        + sorted((ROOT / "data").glob("*.json"))
        + sorted((ROOT / "data").glob("*.csv.gz"))
        + [
            p
            for p in sorted((ROOT / "data/catalog_daily").glob("*.csv.gz"))
            if "2019-10-02" <= p.name[:10] <= "2019-10-30"
        ]
    )
    # Report, verification, packaging and replay orchestration cannot affect predictions.
    files = [
        p
        for p in files
        if p.name
        not in [
            "make_report.py",
            "verify.py",
            "package.py",
            "finish_run.py",
            "run_pipeline.py",
        ]
    ]
    write_json(
        ROOT / "test_gate.json",
        dict(
            frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            primary="auto_both_minus_manual",
            models=15,
            sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        ),
    )
    print("Frozen", len(files), "files and15models", flush=True)


def rank_metrics(rank):
    r = np.asarray(rank, dtype="float64")
    rec = (r >= 1) & (r <= 20)
    n = np.zeros(r.shape)
    n[rec] = 1 / np.log2(r[rec] + 1)
    return n, rec.astype(float)


def verify_saved():
    t = pd.read_csv(
        ROOT / "results/test_per_query.csv.gz", dtype={"query_id": str, "user_id": str}
    )
    z = np.load(ROOT / "results/test_top20.npz")
    assert z["query_ids"].tolist() == t.query_id.tolist()
    for v in VARIANTS:
        top = z[v]
        assert top.shape == (len(t), 20)
        assert all(len(set(row)) == 20 for row in top)
        match = top == t.target_product_id.to_numpy(dtype="int64")[:, None]
        rec = match.any(axis=1).astype(float)
        n = (match / np.log2(np.arange(2, 22))).sum(axis=1)
        np.testing.assert_array_equal(rec, t[v + "_ensemble_recall"])
        np.testing.assert_allclose(n, t[v + "_ensemble_ndcg"], rtol=0, atol=1e-15)
        for method in [v + "_ensemble"] + [v + "_" + str(s) for s in SEEDS]:
            nn, rr = rank_metrics(t[method + "_rank"])
            np.testing.assert_array_equal(rr, t[method + "_recall"])
            np.testing.assert_allclose(nn, t[method + "_ndcg"], rtol=0, atol=1e-15)
    for h in HEURISTICS:
        match = z[h] == t.target_product_id.to_numpy(dtype="int64")[:, None]
        rec = match.any(axis=1).astype(float)
        n = (match / np.log2(np.arange(2, 22))).sum(axis=1)
        nn, rr = rank_metrics(t[h + "_rank"])
        np.testing.assert_array_equal(rec, t[h + "_recall"])
        np.testing.assert_allclose(n, t[h + "_ndcg"], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(rr, rec)
        np.testing.assert_allclose(nn, n, rtol=0, atol=1e-15)
    return t


def main():
    check_gate()
    out = ROOT / "results"
    assert not (out / "test_access.json").exists()
    write_json(
        out / "test_access.json",
        dict(
            opened_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            gate_sha256=sha(ROOT / "test_gate.json"),
        ),
    )
    started = time.time()
    names = json.loads((out / "feature_names.json").read_text())
    sel = json.loads((out / "selection.json").read_text())
    q = (
        pd.read_csv(ROOT / "data/queries_test.csv.gz", dtype=str)
        .sort_values(["cutoff_time", "cutoff_source_row"])
        .reset_index(drop=True)
    )
    e = load_events()
    e = e[e.user_id.isin(q.user_id)]
    hist = {u: g.to_dict("records") for u, g in e.groupby("user_id", sort=False)}
    times = {u: np.array([r["seconds"] for r in h]) for u, h in hist.items()}
    N = len(q)
    models = {}
    pred_seconds = {v: 0.0 for v in VARIANTS}
    for v in VARIANTS:
        models[v] = []
        for s in SEEDS:
            m = CatBoostRanker()
            m.load_model(str(ROOT / "models" / f"{v}_{s}.cbm"))
            models[v].append(m)
    methods = (
        [v + "_" + str(s) for v in VARIANTS for s in SEEDS]
        + [v + "_ensemble" for v in VARIANTS]
        + HEURISTICS
    )
    stats = {
        m: {
            f: np.zeros(N, dtype="int16" if f == "rank" else "float64")
            for f in ["rank", "ndcg", "recall"]
        }
        for m in methods
    }
    tops = {v: np.empty((N, 20), dtype="int64") for v in VARIANTS + HEURISTICS}
    flags = []
    lastday = ""
    cat = retrieval = None
    encode_seconds = 0.0
    generation_seconds = 0.0
    for lo in range(0, N, 200):
        hi = min(N, lo + 200)
        bs = []
        cs = []
        ws = []
        cos = []
        ys = []
        idsall = []
        for row in q.iloc[lo:hi].itertuples(index=False):
            if row.catalog_date != lastday:
                retrieval, cat = load_catalog(row.catalog_date)
                lastday = row.catalog_date
            cut = pd.Timestamp(
                row.cutoff_time.replace(" UTC", ""), tz="UTC"
            ).timestamp()
            h = hist[row.user_id][
                : int(np.searchsorted(times[row.user_id], cut, side="right"))
            ]
            assert len(h) == int(row.history_events) and h[-1]["source_row"] == int(
                row.cutoff_source_row
            )
            assert row.target_time > row.cutoff_time
            t = time.perf_counter()
            ids = retrieval.retrieve(pd.DataFrame(h), 200)
            assert len(set(ids)) == len(ids) == 200
            b, c, nn, ff = encode_manual(h, ids, cat, cut)
            encode_seconds += time.perf_counter() - t
            assert nn == names["numeric"]
            t = time.perf_counter()
            w, co, wn, cn = encode_library(h, ids, cat, cut, b, nn)
            generation_seconds += time.perf_counter() - t
            assert wn == sel["names"]["windows"] and cn == sel["names"]["combinations"]
            y = np.asarray(ids) == row.target_product_id
            ff.update(
                candidate_covered=bool(y.any()),
                target_in_catalog=row.target_product_id in cat.rows,
            )
            flags.append(ff)
            bs.append(b)
            cs.append(c)
            ws.append(w)
            cos.append(co)
            ys.append(y)
            idsall.append(np.array(ids, dtype="int64"))
        b = np.vstack(bs)
        c = np.vstack(cs)
        w = np.vstack(ws)
        co = np.vstack(cos)
        y = np.stack(ys)
        cand = np.stack(idsall)
        for h, scores in heuristic_scores(b, names).items():
            pred = scores.reshape(y.shape)
            for f, val in per_query(pred, y).items():
                stats[h][f][lo:hi] = val
            order = np.argsort(-pred, axis=1, kind="stable")[:, :20]
            tops[h][lo:hi] = np.take_along_axis(cand, order, axis=1)
        for v in VARIANTS:
            t = time.perf_counter()
            d = make_frame(b, c, w, co, names, v, sel)
            pool = Pool(d, cat_features=names["categorical"])
            preds = []
            for s, m in zip(SEEDS, models[v]):
                assert m.feature_names_ == d.columns.tolist()
                pred = m.predict(pool, thread_count=4).reshape(y.shape)
                preds.append(pred)
                for f, val in per_query(pred, y).items():
                    stats[v + "_" + str(s)][f][lo:hi] = val
            pred = np.mean(preds, axis=0)
            pred_seconds[v] += time.perf_counter() - t
            for f, val in per_query(pred, y).items():
                stats[v + "_ensemble"][f][lo:hi] = val
            top = np.argsort(-pred, axis=1, kind="stable")[:, :20]
            tops[v][lo:hi] = np.take_along_axis(cand, top, axis=1)
        if hi % 1000 == 0 or hi == N:
            print(
                "test queries",
                hi,
                "of",
                N,
                "seconds",
                round(time.time() - started),
                flush=True,
            )
    t = pd.concat(
        [
            q[
                [
                    "query_id",
                    "user_id",
                    "catalog_date",
                    "target_product_id",
                    "target_seen_in_prefix",
                ]
            ],
            pd.DataFrame(flags),
            pd.DataFrame(
                {m + "_" + f: a for m, ss in stats.items() for f, a in ss.items()}
            ),
        ],
        axis=1,
    )
    save_csv(out / "test_per_query.csv.gz", t)
    save_npz(out / "test_top20.npz", query_ids=q.query_id.to_numpy(dtype=str), **tops)
    summarize_saved(
        dict(
            elapsed_seconds=time.time() - started,
            manual_feature_and_retrieval_seconds=encode_seconds,
            automatic_library_seconds=generation_seconds,
            frame_and_three_model_scoring_seconds=pred_seconds,
        )
    )
    check_gate()


def summarize_saved(cost=None):
    t = verify_saved()
    models = {
        v: dict(
            ensemble={
                f: float(t[v + "_ensemble_" + f].mean()) for f in ["ndcg", "recall"]
            },
            seeds={
                str(s): {
                    f: float(t[v + "_" + str(s) + "_" + f].mean())
                    for f in ["ndcg", "recall"]
                }
                for s in SEEDS
            },
        )
        for v in VARIANTS
    }
    contrasts = {}
    for a, b in [
        ("auto_both", "manual"),
        ("auto_windows", "manual"),
        ("auto_combinations", "manual"),
        ("random_matched", "manual"),
        ("auto_both", "random_matched"),
    ]:
        contrasts[a + "_minus_" + b] = bootstrap(
            t, a + "_ensemble_ndcg", b + "_ensemble_ndcg"
        )
    daily = {}
    for day, g in t.groupby("catalog_date"):
        daily[day] = dict(
            queries=len(g),
            users=int(g.user_id.nunique()),
            models={
                v: {
                    f: float(g[v + "_ensemble_" + f].mean()) for f in ["ndcg", "recall"]
                }
                for v in VARIANTS
            },
            difference=float(
                (g.auto_both_ensemble_ndcg - g.manual_ensemble_ndcg).mean()
            ),
        )
    groups = {}
    for name, mask in {
        "category_switch": t.category_switch.astype(bool),
        "price_shift": t.price_shift.astype(bool),
        "repeats": t.prefix_has_repeats.astype(bool),
        "target_new": t.target_seen_in_prefix.eq(0),
        "target_seen": t.target_seen_in_prefix.eq(1),
    }.items():
        g = t[mask]
        groups[name] = dict(
            queries=len(g),
            models={
                v: {
                    f: float(g[v + "_ensemble_" + f].mean()) for f in ["ndcg", "recall"]
                }
                for v in VARIANTS
            },
            contrast=(
                bootstrap(g, "auto_both_ensemble_ndcg", "manual_ensemble_ndcg")
                if len(g)
                else None
            ),
        )
    primary = contrasts["auto_both_minus_manual"]
    status = (
        "improved"
        if primary["ci95"][0] > 0
        else "worse" if primary["ci95"][1] < 0 else "no_clear_difference"
    )
    r = dict(
        queries=len(t),
        users=int(t.user_id.nunique()),
        candidate_coverage=float(t.candidate_covered.mean()),
        catalog_coverage=float(t.target_in_catalog.mean()),
        models=models,
        heuristics={
            h: {f: float(t[h + "_" + f].mean()) for f in ["ndcg", "recall"]}
            for h in HEURISTICS
        },
        contrasts=contrasts,
        primary_status=status,
        primary_recall=bootstrap(
            t, "auto_both_ensemble_recall", "manual_ensemble_recall"
        ),
        daily=daily,
        groups=groups,
        cost=cost or {},
        primary_variant="auto_both",
        user_sampling="complete histories of5000 fixed users",
        test_period="2019-10-25 through2019-10-30",
        pilot_results_combined=False,
    )
    write_json(ROOT / "results/test_results.json", r)
    print(json.dumps(r, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze()
    elif a.summarize:
        summarize_saved()
    else:
        main()
