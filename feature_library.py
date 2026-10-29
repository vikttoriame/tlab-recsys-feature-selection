"""Finite, target-free automatic feature library. No arithmetic on identifiers."""

from collections import Counter, defaultdict
import numpy as np

WINDOWS = [
    ("e2", 2, None),
    ("e5", 5, None),
    ("e20", 20, None),
    ("m1", None, 60),
    ("m5", None, 300),
    ("m15", None, 900),
    ("m30", None, 1800),
]
WINDOW_FIELDS = [
    "log_events",
    "view_fraction",
    "cart_fraction",
    "purchase_fraction",
    "item_views",
    "category_view_share",
    "brand_view_share",
    "category_carts",
    "category_logprice_mean",
    "category_logprice_median",
    "category_logprice_std",
    "category_log_of_mean_price",
    "category_price_observations",
    "price_history_missing",
    "price_delta_category_median",
    "abs_price_delta_category_median",
]


def window_features(history, products, catalog, cutoff, windows=WINDOWS):
    n = len(products)
    meta = [catalog.rows[p] for p in products]
    cats = [r["category"] for r in meta]
    brands = [r["brand"] for r in meta]
    current = {r["product_id"]: r for r in history}
    prices = np.array(
        [
            current[p]["price"] if p in current else r["price"]
            for p, r in zip(products, meta)
        ]
    )
    lp = np.where(prices > 0, np.log(np.maximum(prices, 1e-12)), np.nan)
    cols = []
    names = []
    for scope, count, seconds in windows:
        h = (
            history[-count:]
            if count is not None
            else [r for r in history if r["seconds"] >= cutoff - seconds]
        )
        assert h
        v = [r for r in h if r["event_type"] == "view"]
        c = [r for r in h if r["event_type"] == "cart"]
        den = max(1, len(v))
        vc = Counter(r["category_id"] for r in v)
        vb = Counter(r["brand"] for r in v)
        vp = Counter(r["product_id"] for r in v)
        cc = Counter(r["category_id"] for r in c)
        groups = defaultdict(list)
        for r in v:
            if r["price"] > 0:
                groups[r["category_id"]].append(r["price"])
        stats = {}
        for cat, ps in groups.items():
            a = np.array(ps)
            logs = np.log(a)
            stats[cat] = (
                logs.mean(),
                np.median(logs),
                logs.std(),
                np.log(a.mean()),
                len(a),
            )
        values = [
            np.full(n, np.log1p(len(h))),
            np.full(n, len(v) / len(h)),
            np.full(n, len(c) / len(h)),
            np.full(n, sum(r["event_type"] == "purchase" for r in h) / len(h)),
            [vp[p] for p in products],
            [vc[x] / den for x in cats],
            [vb[x] / den if x else 0 for x in brands],
            [cc[x] for x in cats],
        ]
        for j in range(5):
            values.append([stats[x][j] if x in stats else np.nan for x in cats])
        values.append([x not in stats for x in cats])
        median = np.array([stats[x][1] if x in stats else np.nan for x in cats])
        values.extend([lp - median, np.abs(lp - median)])
        cols.extend(values)
        names.extend("auto_" + scope + "_" + f for f in WINDOW_FIELDS)
    return np.column_stack(cols).astype("float32"), names


def combination_features(base, base_names):
    b = {name: base[:, i] for i, name in enumerate(base_names)}
    cols = []
    names = []

    def add(name, v):
        names.append("auto_cross_" + name)
        cols.append(np.asarray(v, dtype="float32"))

    for scope in ["w3", "w10", "wall"]:
        p = scope + "_"
        delta = b[p + "price_delta_category_median"]
        dist = b[p + "abs_price_delta_category_median"]
        sd = b[p + "category_logprice_std"]
        cs = b[p + "category_view_share"]
        bs = b[p + "brand_view_share"]
        cc = b[p + "category_carts"]
        iv = b[p + "item_views"]
        close = np.exp(-np.minimum(dist, 30))
        scale = sd + 0.1
        add(p + "price_z_signed", delta / scale)
        add(p + "price_z_abs", dist / scale)
        add(p + "price_closeness", close)
        add(p + "closeness_x_category_share", close * cs)
        add(p + "closeness_x_brand_share", close * bs)
        add(p + "closeness_x_cart_presence", close * (cc > 0))
        add(p + "signed_price_x_category_share", delta * cs)
        add(p + "category_x_brand_share", cs * bs)
        add(p + "category_share_x_category_carts", cs * np.log1p(cc))
        add(p + "brand_share_x_item_carts", bs * np.log1p(b["item_cart_count"]))
        add(p + "item_repeat_x_category_share", np.log1p(iv) * cs)
        add(
            p + "cart_x_view_recency",
            np.log1p(b["item_cart_count"]) / (1 + b["item_view_recency_log_seconds"]),
        )
    for scope in ["w3", "w10"]:
        p = scope + "_"
        trend = b[p + "category_logprice_median"] - b["wall_category_logprice_median"]
        delta = b[p + "price_delta_category_median"]
        add(p + "price_context_trend", trend)
        add(p + "price_x_context_trend", delta * trend)
        add(
            p + "abs_price_distance_change",
            b[p + "abs_price_delta_category_median"]
            - b["wall_abs_price_delta_category_median"],
        )
        add(
            p + "category_share_change",
            b[p + "category_view_share"] - b["wall_category_view_share"],
        )
        add(
            p + "brand_share_change",
            b[p + "brand_view_share"] - b["wall_brand_view_share"],
        )
        add(
            p + "price_x_category_share_change",
            delta * (b[p + "category_view_share"] - b["wall_category_view_share"]),
        )
    out = np.column_stack(cols).astype("float32")
    assert not np.isinf(out).any()
    return out, names


def encode_auto(history, products, catalog, cutoff, base, base_names):
    w, wn = window_features(history, products, catalog, cutoff)
    c, cn = combination_features(base, base_names)
    assert not np.isinf(w).any()
    return w, c, wn, cn
