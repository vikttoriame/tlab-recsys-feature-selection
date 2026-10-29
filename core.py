import gzip, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
from atomic_gzip import atomic_gzip

SEEDS = [20261022, 20261023, 20261024]
VARIANTS = [
    "manual",
    "auto_windows",
    "auto_combinations",
    "auto_both",
    "random_matched",
]
HEURISTICS = ["global_popularity", "recent_then_category"]
PARAMS = dict(
    iterations=261,
    depth=6,
    learning_rate=0.06,
    l2_leaf_reg=5,
    loss_function="QuerySoftMax",
    eval_metric="NDCG:top=20",
    thread_count=4,
    counter_calc_method="SkipTest",
    allow_writing_files=False,
    verbose=False,
)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def save_npy(path, a):
    path = Path(path)
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("wb") as f:
        np.save(f, a)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    assert np.load(path, mmap_mode="r").shape == a.shape


def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    with np.load(path) as z:
        for k, v in arrays.items():
            assert z[k].shape == v.shape


def save_csv(path, t):
    with atomic_gzip(path) as f:
        t.to_csv(f, index=False)


def select_queries(q, n, prefix):
    q = q.copy()
    q["_hash"] = [
        hashlib.blake2b((prefix + x).encode(), digest_size=8).hexdigest()
        for x in q.query_id
    ]
    return (
        q.sort_values("_hash")
        .head(n)
        .sort_values(["cutoff_time", "cutoff_source_row"])
        .drop(columns="_hash")
        .reset_index(drop=True)
    )


def load_events():
    e = pd.read_csv(ROOT / "data/events_all.csv.gz", dtype=str, keep_default_na=False)
    e["seconds"] = pd.to_datetime(e.event_time, utc=True).astype("int64") / 1e9
    e["price"] = e.price.astype(float)
    e["source_row"] = e.source_row.astype("int64")
    return e.sort_values("source_row")


def load_catalog(day):
    from candidates import DailyCandidates
    from manual_features import Catalog

    p = ROOT / f"data/catalog_daily/{day}.csv.gz"
    f = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert f.last_seen.str[:10].lt(day).all()
    f["first_seconds"] = pd.to_datetime(f.first_seen, utc=True).astype("int64") / 1e9
    return DailyCandidates(p), Catalog(f)


def per_query(scores, y):
    order = np.argsort(-scores, axis=1, kind="stable")
    yy = np.take_along_axis(y, order, axis=1)
    return dict(
        ndcg=(yy[:, :20] / np.log2(np.arange(2, 22))).sum(axis=1),
        recall=yy[:, :20].any(axis=1).astype(float),
        rank=np.where(yy.any(axis=1), yy.argmax(axis=1) + 1, 0),
    )


def summarize(s):
    return {k: float(np.mean(s[k])) for k in ["ndcg", "recall"]}


def heuristic_scores(x, names):
    ix = {n: i for i, n in enumerate(names["numeric"])}
    pop = np.asarray(x[:, ix["prior_log_views"]], dtype="float64")
    seen = x[:, ix["item_view_count"]] > 0
    rec = np.asarray(x[:, ix["item_view_recency_log_seconds"]], dtype="float64")
    same = x[:, ix["same_last_category"]]
    return dict(
        global_popularity=pop,
        recent_then_category=np.where(seen, 3 + 1 / (1 + rec), 2 * same + pop / 100),
    )


def extra_parts(variant, selection):
    parts = []
    for family in ["windows", "combinations"]:
        if variant in ["auto_both", "random_matched"] or variant == "auto_" + family:
            parts.extend(
                (family, j)
                for j in selection[
                    "random_matched" if variant == "random_matched" else "selected"
                ][family]
            )
    return parts


def make_frame(x, c, wn, cn, names, variant, selection):
    d = pd.DataFrame(np.asarray(x, dtype="float32"), columns=names["numeric"])
    for family, j in extra_parts(variant, selection):
        d[selection["names"][family][j]] = (wn if family == "windows" else cn)[:, j]
    for j, n in enumerate(names["categorical"]):
        d[n] = c[:, j].astype("int64")
    return d


def bootstrap(t, left, right):
    delta = t[left].to_numpy() - t[right].to_numpy()
    uid, inv = np.unique(t.user_id.to_numpy(), return_inverse=True)
    count = np.bincount(inv)
    sums = np.bincount(inv, weights=delta)
    rng = np.random.default_rng(20261022)
    boot = np.empty(2000)
    for i in range(2000):
        ids = rng.integers(0, len(uid), len(uid))
        boot[i] = sums[ids].sum() / count[ids].sum()
    return dict(
        queries=len(t),
        users=len(uid),
        difference=float(delta.mean()),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        replicates=2000,
    )
