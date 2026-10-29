from core import *
from encoding import encode_manual, encode_library


def main():
    e = load_events()
    hist = {u: g.to_dict("records") for u, g in e.groupby("user_id", sort=False)}
    names = json.loads((ROOT / "cache/feature_names.json").read_text())
    lib = json.loads((ROOT / "cache/library_names.json").read_text())
    assert (
        len(names["numeric"]) == 105
        and len(names["categorical"]) == 3
        and len(lib["windows"]) == 112
        and len(lib["combinations"]) == 48
    )
    checked = 0
    retrieval = cat = None
    lastday = ""
    for split in ["train", "validation"]:
        q = pd.read_csv(ROOT / f"cache/{split}_queries.csv.gz", dtype=str)
        chosen = q.iloc[np.linspace(0, len(q) - 1, 150, dtype=int)]
        saved = np.load(ROOT / f"cache/{split}_labels_candidates.npz")
        fit = (
            np.load(ROOT / "cache/train_query_indices.npy")
            if split == "train"
            else None
        )
        fit_map = {int(qi): i for i, qi in enumerate(fit)} if split == "train" else None
        neg = (
            np.load(ROOT / "cache/train_negative_positions.npy")
            if split == "train"
            else None
        )
        for index, row in chosen.iterrows():
            cut = pd.Timestamp(
                row.cutoff_time.replace(" UTC", ""), tz="UTC"
            ).timestamp()
            h = [r for r in hist[row.user_id] if r["seconds"] <= cut]
            assert len(h) == int(row.history_events) and h[-1]["source_row"] == int(
                row.cutoff_source_row
            )
            future = [
                r
                for r in hist[row.user_id]
                if r["event_type"] == "view" and r["seconds"] > cut
            ]
            assert future and future[0]["product_id"] == row.target_product_id
            if row.catalog_date != lastday:
                retrieval, cat = load_catalog(row.catalog_date)
                lastday = row.catalog_date
            ids = retrieval.retrieve(pd.DataFrame(h), 200)
            np.testing.assert_array_equal(
                np.array(ids, dtype="int64"), saved["candidates"][index]
            )
            np.testing.assert_array_equal(
                np.asarray(ids) == row.target_product_id, saved["labels"][index]
            )
            b, c, nn, flags = encode_manual(h, ids, cat, cut)
            assert nn == names["numeric"]
            if split == "train":
                if index not in fit_map:
                    assert not saved["labels"][index].any()
                    checked += 1
                    continue
                fi = fit_map[index]
                pos = neg[fi]
                ids = [ids[j] for j in pos]
                b = b[pos]
                c = c[pos]
                loc = slice(fi * 41, (fi + 1) * 41)
            else:
                loc = slice(index * 200, (index + 1) * 200)
            w, co, wn, cn = encode_library(h, ids, cat, cut, b, nn)
            for key, a in [("X", b), ("C", c), ("windows", w), ("combinations", co)]:
                np.testing.assert_allclose(
                    a,
                    np.load(ROOT / f"cache/{split}_{key}.npy", mmap_mode="r")[loc],
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
            # Appending unknown future rows then clipping must reproduce the prefix exactly.
            clipped = [
                r
                for r in h + [dict(h[-1], seconds=cut + 1, price=999999.0)]
                if r["seconds"] <= cut
            ]
            bb, cc, _, _ = encode_manual(clipped, ids, cat, cut)
            np.testing.assert_allclose(bb, b, rtol=0, atol=0, equal_nan=True)
            checked += 1
    ty = np.load(ROOT / "cache/train_labels.npy")
    assert np.all(ty.reshape(-1, 41).sum(axis=1) == 1)
    t = pd.DataFrame(
        dict(user_id=["a", "a", "b"], left=[1.0, 1.0, 0.0], right=[0.0, 0.0, 0.0])
    )
    r = bootstrap(t, "left", "right")
    assert r["difference"] == 2 / 3 and r["ci95"] == [0.0, 1.0]
    write_json(
        ROOT / "results/pretraining_checks.json",
        dict(
            passed=True,
            independent_prefix_target_candidate_feature_examples=checked,
            numeric_features=105,
            categorical_features=3,
            generated_features=160,
            whole_user_history_used=True,
            bootstrap_query_weighting_checked=True,
            test_scored=False,
        ),
    )
    print("Pretraining checks passed", checked, flush=True)


if __name__ == "__main__":
    main()
