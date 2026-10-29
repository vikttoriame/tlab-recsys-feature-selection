import gc, time
from core import *
from encoding import encode_manual, encode_library, CAT_NAMES


def main():
    started = time.time()
    e = load_events()
    hist = {u: g.to_dict("records") for u, g in e.groupby("user_id", sort=False)}
    times = {u: np.array([r["seconds"] for r in h]) for u, h in hist.items()}
    manifest = {}
    cache = ROOT / "cache"
    lastday = None
    cat = retrieval = None
    for split, n in [("train", 10000), ("validation", 3000)]:
        t = time.time()
        q = select_queries(
            pd.read_csv(
                ROOT / f"data/queries_{split}.csv.gz", dtype=str, keep_default_na=False
            ),
            n,
            "final-query-v1|",
        )
        blocks = {k: [] for k in ["X", "C", "windows", "combinations"]}
        labels = []
        candidates = []
        fit_indices = []
        flags = []
        offsets = []
        for i, row in enumerate(q.itertuples(index=False)):
            if row.catalog_date != lastday:
                del cat, retrieval
                gc.collect()
                retrieval, cat = load_catalog(row.catalog_date)
                lastday = row.catalog_date
            cut = pd.Timestamp(
                row.cutoff_time.replace(" UTC", ""), tz="UTC"
            ).timestamp()
            h = hist[row.user_id][
                : int(np.searchsorted(times[row.user_id], cut, side="right"))
            ]
            assert len(h) == int(row.history_events)
            assert h[-1]["source_row"] == int(row.cutoff_source_row)
            ids = retrieval.retrieve(pd.DataFrame(h), 200)
            assert len(set(ids)) == len(ids) == 200
            y = (np.asarray(ids) == row.target_product_id).astype("uint8")
            labels.append(y)
            candidates.append(np.array(ids, dtype="int64"))
            b, c, names, ff = encode_manual(h, ids, cat, cut)
            assert b.shape == (200, 105)
            ff.update(
                candidate_covered=bool(y.any()),
                target_in_catalog=row.target_product_id in cat.rows,
            )
            flags.append(ff)
            if split == "train":
                if not y.any():
                    continue
                pos = int(np.flatnonzero(y)[0])
                neg = np.flatnonzero(y == 0)
                seed = int.from_bytes(
                    hashlib.blake2b(
                        ("final-negatives-v1|" + row.query_id).encode(), digest_size=8
                    ).digest(),
                    "big",
                )
                chosen = np.sort(
                    np.r_[
                        pos, np.random.default_rng(seed).choice(neg, 40, replace=False)
                    ]
                )
                fit_indices.append(i)
                offsets.append(chosen)
                ids = [ids[j] for j in chosen]
                b = b[chosen]
                c = c[chosen]
            w, co, wn, cn = encode_library(h, ids, cat, cut, b, names)
            for key, a in [("X", b), ("C", c), ("windows", w), ("combinations", co)]:
                blocks[key].append(a)
            if (i + 1) % 1000 == 0:
                print(
                    json.dumps(
                        dict(
                            split=split,
                            queries=i + 1,
                            total=len(q),
                            seconds=round(time.time() - t, 1),
                        )
                    ),
                    flush=True,
                )
        for key, parts in blocks.items():
            array = np.concatenate(parts)
            save_npy(cache / f"{split}_{key}.npy", array)
            del array
            blocks[key] = []
            gc.collect()
        y = np.stack(labels)
        save_npz(
            cache / f"{split}_labels_candidates.npz",
            labels=y,
            candidates=np.stack(candidates),
        )
        selected = pd.concat([q, pd.DataFrame(flags)], axis=1)
        save_csv(cache / f"{split}_queries.csv.gz", selected)
        if split == "train":
            fi = np.array(fit_indices, dtype="int64")
            off = np.array(offsets, dtype="int64")
            ty = np.take_along_axis(y[fi], off, axis=1)
            assert np.all(ty.sum(axis=1) == 1)
            save_npy(cache / "train_labels.npy", ty.reshape(-1))
            save_npy(cache / "train_query_indices.npy", fi)
            save_npy(cache / "train_negative_positions.npy", off)
            save_csv(
                cache / "train_fit_queries.csv.gz",
                selected.iloc[fi].reset_index(drop=True),
            )
        manifest[split] = dict(
            queries=len(q),
            users=int(q.user_id.nunique()),
            covered=int(y.any(axis=1).sum()),
            candidate_coverage=float(y.any(axis=1).mean()),
            rows_saved=int(np.load(cache / f"{split}_X.npy", mmap_mode="r").shape[0]),
            feature_seconds=time.time() - t,
        )
        print("Features complete", split, json.dumps(manifest[split]), flush=True)
    write_json(cache / "feature_names.json", dict(numeric=names, categorical=CAT_NAMES))
    write_json(cache / "library_names.json", dict(windows=wn, combinations=cn))
    write_json(
        ROOT / "results/feature_manifest.json",
        dict(
            splits=manifest,
            numeric_features=len(names),
            categorical_features=len(CAT_NAMES),
            generated_windows=len(wn),
            generated_combinations=len(cn),
            elapsed_seconds=time.time() - started,
            test_scored=False,
        ),
    )


if __name__ == "__main__":
    main()
