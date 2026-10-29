import argparse, gc, resource, subprocess, time
from catboost import CatBoostRanker, Pool
from core import *


def load_frame(split, variant):
    cache = ROOT / "cache"
    names = json.loads((cache / "feature_names.json").read_text())
    sel = (
        json.loads((ROOT / "results/selection.json").read_text())
        if variant != "manual"
        else dict(selected={}, random_matched={}, names={})
    )
    return (
        make_frame(
            np.load(cache / f"{split}_X.npy", mmap_mode="r"),
            np.load(cache / f"{split}_C.npy", mmap_mode="r"),
            (
                np.load(cache / f"{split}_windows.npy", mmap_mode="r")
                if variant != "manual"
                else None
            ),
            (
                np.load(cache / f"{split}_combinations.npy", mmap_mode="r")
                if variant != "manual"
                else None
            ),
            names,
            variant,
            sel,
        ),
        names["categorical"],
    )


class Progress:
    def __init__(self, key):
        self.key = key
        self.start = time.time()

    def after_iteration(self, info):
        if info.iteration == 1 or info.iteration % 50 == 0:
            print(
                json.dumps(
                    dict(
                        job=self.key,
                        iteration=info.iteration,
                        seconds=round(time.time() - self.start, 1),
                    )
                ),
                flush=True,
            )
        return True


def one(variant, seed):
    key = f"{variant}_{seed}"
    out = ROOT / "results" / f"{key}.json"
    assert not out.exists()
    start = time.time()
    x, cat = load_frame("train", variant)
    v, _ = load_frame("validation", variant)
    ty = np.load(ROOT / "cache/train_labels.npy")
    vy = np.load(ROOT / "cache/validation_labels_candidates.npz")["labels"]
    covered = vy.any(axis=1)
    assert len(x) == len(ty) and np.all(ty.reshape(-1, 41).sum(axis=1) == 1)
    pt = Pool(
        x, label=ty, group_id=np.repeat(np.arange(len(ty) // 41), 41), cat_features=cat
    )
    pv = Pool(
        v.loc[np.repeat(covered, 200)].reset_index(drop=True),
        label=vy[covered].reshape(-1),
        group_id=np.repeat(np.arange(covered.sum()), 200),
        cat_features=cat,
    )
    del x
    gc.collect()
    model = CatBoostRanker(**PARAMS, random_seed=seed)
    t = time.time()
    model.fit(pt, eval_set=pv, use_best_model=False, callbacks=[Progress(key)])
    fit = time.time() - t
    assert model.tree_count_ == PARAMS["iterations"]
    path = ROOT / "models" / f"{key}.cbm"
    tmp = path.with_suffix(".writing")
    model.save_model(str(tmp))
    tmp.replace(path)
    loaded = CatBoostRanker()
    loaded.load_model(str(path))
    t = time.time()
    pred = model.predict(v, thread_count=4).reshape(vy.shape)
    predict = time.time() - t
    np.testing.assert_array_equal(
        loaded.predict(v.iloc[:1000], thread_count=4), pred.reshape(-1)[:1000]
    )
    save_npy(ROOT / "cache" / f"validation_scores_{key}.npy", pred)
    pd.DataFrame(
        dict(
            feature=model.feature_names_,
            importance=model.get_feature_importance(type="PredictionValuesChange"),
        )
    ).sort_values("importance", ascending=False).to_csv(
        ROOT / "results" / f"importance_{key}.csv", index=False
    )
    rec = dict(
        variant=variant,
        seed=seed,
        features=len(v.columns),
        feature_names=model.feature_names_,
        trees=int(model.tree_count_),
        train_queries=len(ty) // 41,
        train_rows=len(ty),
        validation=summarize(per_query(pred, vy)),
        fit_seconds=fit,
        predict_validation_seconds=predict,
        elapsed_seconds=time.time() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        sha256=sha(path),
        actual_parameters=model.get_all_params(),
        test_scored=False,
    )
    assert rec["actual_parameters"]["counter_calc_method"] == "SkipTest"
    write_json(out, rec)
    print(json.dumps(rec), flush=True)


def controller(variants):
    jobs = [(v, s) for s in SEEDS for v in variants]
    active = []
    while jobs or active:
        while jobs and len(active) < 2:
            v, s = jobs.pop(0)
            key = f"{v}_{s}"
            f = (ROOT / "logs" / f"{key}.log").open("w")
            env = dict(
                os.environ,
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="1",
            )
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--variant",
                    v,
                    "--seed",
                    str(s),
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
            active.append((p, f, key))
            print("launched", key, flush=True)
        rest = []
        for p, f, key in active:
            code = p.poll()
            if code is None:
                rest.append((p, f, key))
            else:
                f.close()
                print("finished", key, code, flush=True)
                if code:
                    for p2, f2, _ in active:
                        if p2.poll() is None:
                            p2.terminate()
                            f2.close()
                    raise RuntimeError(key + " failed")
        active = rest
        if active:
            time.sleep(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["manual", "auto"])
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.variant:
        one(a.variant, a.seed)
    else:
        controller(["manual"] if a.stage == "manual" else VARIANTS[1:])
