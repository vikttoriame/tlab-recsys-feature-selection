"""Generate -> train-only deduplication -> held-out VALIDATION residual selection."""

import time
from core import *


def normalize(z):
    z = np.asarray(z, dtype="float64").copy()
    for j in range(z.shape[1]):
        good = np.isfinite(z[:, j])
        med = np.median(z[good, j]) if good.any() else 0
        z[~good, j] = med
    z -= z.mean(axis=0)
    norm = np.sqrt((z * z).sum(axis=0))
    z /= np.where(norm > 1e-12, norm, 1)
    return z


def main():
    t = time.time()
    cache = ROOT / "cache"
    names = json.loads((cache / "library_names.json").read_text())
    base = np.load(cache / "train_X.npy", mmap_mode="r")
    q = pd.read_csv(cache / "validation_queries.csv.gz", dtype=str)
    y = np.load(cache / "validation_labels_candidates.npz")["labels"]
    covered = y.any(axis=1)
    labels = y[covered]
    scores = np.mean(
        [np.load(cache / f"validation_scores_manual_{s}.npy") for s in SEEDS], axis=0
    )[covered]
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    residual = labels - p
    late = q.loc[covered, "catalog_date"].ge("2019-10-23").to_numpy()
    assert late.any() and (~late).any()
    rng = np.random.default_rng(20261022)
    sample = np.sort(rng.choice(len(base), min(20000, len(base)), replace=False))
    base_corr = normalize(base[sample])
    selection = {}
    randoms = {}
    records = []
    for family in ["windows", "combinations"]:
        train = np.load(cache / f"train_{family}.npy", mmap_mode="r")
        val = np.load(cache / f"validation_{family}.npy", mmap_mode="r")
        corr = normalize(train[sample])
        constant = np.linalg.norm(corr, axis=0) < 0.5
        cc = np.abs(corr.T @ corr)
        cb = np.max(np.abs(corr.T @ base_corr), axis=1)
        recs = []
        for j, name in enumerate(names[family]):
            raw = np.asarray(train[:, j], dtype="float64")
            good = raw[np.isfinite(raw)]
            med = np.median(good) if len(good) else 0
            lo, hi = np.quantile(good, [0.01, 0.99]) if len(good) else [0, 0]
            v = np.asarray(val[:, j], dtype="float64").reshape(-1, 200)[covered]
            z = np.clip(np.nan_to_num(v, nan=med, posinf=hi, neginf=lo), lo, hi)
            z -= z.mean(axis=1, keepdims=True)
            z /= np.maximum(z.std(axis=1, keepdims=True), 1e-6)
            align = (residual * z).sum(axis=1)
            s0 = float(align[~late].mean())
            s1 = float(align[late].mean())
            score = min(abs(s0), abs(s1)) if s0 * s1 > 0 else 0
            recs.append(
                dict(
                    family=family,
                    index=j,
                    feature=name,
                    score=score,
                    alignment_validation_21_22=s0,
                    alignment_validation_23_24=s1,
                    train_median=float(med),
                    train_clip_low=float(lo),
                    train_clip_high=float(hi),
                    max_abs_train_correlation_with_manual=float(cb[j]),
                    selected=False,
                    reason="outside_budget",
                )
            )
        selected = []
        for r in sorted(recs, key=lambda x: (-x["score"], x["feature"])):
            j = r["index"]
            if constant[j]:
                r["reason"] = "constant"
                continue
            if cb[j] > 0.995:
                r["reason"] = "near_manual_duplicate"
                continue
            if r["score"] <= 1e-8:
                r["reason"] = "no_stable_validation_alignment"
                continue
            if selected and max(cc[j, k] for k in selected) > 0.995:
                r["reason"] = "near_selected_duplicate"
                continue
            if len(selected) >= 12:
                continue
            selected.append(j)
            r["selected"] = True
            r["reason"] = "selected"
        assert selected, f"No stable features: {family}"
        eligible = [j for j in range(len(recs)) if not constant[j] and cb[j] <= 0.995]
        order = np.random.default_rng(
            20261022 + (0 if family == "windows" else 1)
        ).permutation(eligible)
        rand = []
        for jj in order:
            j = int(jj)
            if rand and max(cc[j, k] for k in rand) > 0.995:
                continue
            rand.append(j)
            if len(rand) == len(selected):
                break
        assert len(rand) == len(selected)
        selection[family] = selected
        randoms[family] = sorted(rand)
        records.extend(recs)
        print("Selected", family, [names[family][j] for j in selected], flush=True)
    save_csv(ROOT / "results/feature_screening.csv.gz", pd.DataFrame(records))
    save_npz(
        ROOT / "results/selection_diagnostics.npz",
        validation_query_ids=q.loc[covered, "query_id"].to_numpy(dtype=str),
        manual_scores=scores,
        labels=labels,
        residuals=residual,
    )
    write_json(
        ROOT / "results/selection.json",
        dict(
            selected=selection,
            random_matched=randoms,
            names=names,
            selection_split="validation October21-24 only",
            validation_covered_queries=int(covered.sum()),
            validation_temporal_blocks=[int((~late).sum()), int(late.sum())],
            correlation_rows_train=len(sample),
            feature_budget_per_family=12,
            correlation_cutoff=0.995,
            selection_seconds=time.time() - t,
            primary_variant="auto_both",
            test_scored=False,
        ),
    )


if __name__ == "__main__":
    main()
