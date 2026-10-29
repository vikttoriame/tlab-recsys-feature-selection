"""Manual baseline features. Target-free API; all history already clipped to cutoff.
Category IDs and brands are categorical, never numerical measurements.
"""

import hashlib
from collections import Counter
import numpy as np

CAT_NAMES = ["candidate_category", "candidate_brand", "last_view_category"]


def brand_id(s):
    return int.from_bytes(
        hashlib.blake2b(s.encode(), digest_size=8).digest(), "big"
    ) & ((1 << 63) - 1)


class Catalog:
    def __init__(self, frame):
        self.rows = {}
        for row in frame.itertuples(index=False):
            self.rows[str(row.product_id)] = {
                "category": str(row.category_id),
                "brand": str(row.brand),
                "price": float(row.price),
                "views": int(row.views_before_day),
                "carts": int(row.carts_before_day),
                "purchases": int(row.purchases_before_day),
                "first": float(row.first_seconds),
            }


def encode(history, products, catalog, cutoff):
    # history is a list of dicts with event_type/product_id/category_id/brand/price/seconds.
    views = [r for r in history if r["event_type"] == "view"]
    last = views[-1]
    n = len(products)
    cols = {}

    def put(name, x):
        cols[name] = (
            np.full(n, x, dtype=np.float32)
            if np.isscalar(x)
            else np.asarray(x, dtype=np.float32)
        )

    meta = [catalog.rows[p] for p in products]
    cats = np.array([r["category"] for r in meta])
    brands = np.array([r["brand"] for r in meta])
    current = {r["product_id"]: r for r in history}
    price = np.array(
        [
            current[p]["price"] if p in current else r["price"]
            for p, r in zip(products, meta)
        ]
    )
    logprice = np.where(price > 0, np.log(np.maximum(price, 1e-12)), np.nan)
    pop = np.array([r["views"] for r in meta])
    cartpop = np.array([r["carts"] for r in meta])
    buyspop = np.array([r["purchases"] for r in meta])
    put("candidate_log_price", logprice)
    put("candidate_nonpositive_price", price <= 0)
    put("prior_log_views", np.log1p(pop))
    put("prior_log_carts", np.log1p(cartpop))
    put("prior_log_purchases", np.log1p(buyspop))
    put("prior_cart_rate", cartpop / (pop + 1))
    put("prior_purchase_rate", buyspop / (pop + 1))
    put(
        "item_age_log_seconds",
        np.log1p(np.maximum(0, cutoff - np.array([r["first"] for r in meta]))),
    )
    put("is_last_view", np.array(products) == last["product_id"])
    put("same_last_category", cats == last["category_id"])
    put("same_last_brand", (brands == last["brand"]) & (brands != ""))
    put("candidate_brand_missing", brands == "")
    lastlog = np.log(last["price"]) if last["price"] > 0 else np.nan
    put(
        "price_delta_from_last_in_category",
        np.where(cats == last["category_id"], logprice - lastlog, np.nan),
    )
    counts = {
        a: Counter(r["product_id"] for r in history if r["event_type"] == a)
        for a in ("view", "cart", "purchase")
    }
    lastview = {r["product_id"]: r["seconds"] for r in views}
    for a in counts:
        put(f"item_{a}_count", [counts[a][p] for p in products])
    put(
        "item_view_recency_log_seconds",
        [
            np.log1p(max(0, cutoff - lastview[p])) if p in lastview else np.nan
            for p in products
        ],
    )
    put("history_log_events", np.log1p(len(history)))
    put("history_log_views", np.log1p(len(views)))
    put(
        "history_log_duration_seconds", np.log1p(max(0, cutoff - history[0]["seconds"]))
    )
    put("history_distinct_categories", len(set(r["category_id"] for r in views)))
    put("history_distinct_products", len(set(r["product_id"] for r in views)))
    put(
        "history_repeat_fraction",
        1 - len(set(r["product_id"] for r in views)) / len(views),
    )
    put("hour_sin", np.sin((cutoff % 86400) / 86400 * 2 * np.pi))
    put("hour_cos", np.cos((cutoff % 86400) / 86400 * 2 * np.pi))
    for window in (3, 10, "all"):
        h = history if window == "all" else history[-window:]
        v = [r for r in h if r["event_type"] == "view"]
        carts = [r for r in h if r["event_type"] == "cart"]
        pref = f"w{window}_"
        den = max(1, len(v))
        put(pref + "log_events", np.log1p(len(h)))
        put(pref + "view_fraction", len(v) / len(h))
        put(pref + "cart_fraction", len(carts) / len(h))
        put(
            pref + "purchase_fraction",
            sum(r["event_type"] == "purchase" for r in h) / len(h),
        )
        cc = Counter(r["category_id"] for r in v)
        bc = Counter(r["brand"] for r in v)
        pc = Counter(r["product_id"] for r in v)
        ac = Counter(r["category_id"] for r in carts)
        put(pref + "item_views", [pc[p] for p in products])
        put(pref + "category_view_share", [cc[c] / den for c in cats])
        put(pref + "brand_view_share", [bc[b] / den if b else 0 for b in brands])
        put(pref + "category_carts", [ac[c] for c in cats])
        stats = {}
        for category in set(cats):
            prices = np.array(
                [
                    r["price"]
                    for r in v
                    if r["category_id"] == category and r["price"] > 0
                ]
            )
            if len(prices):
                logs = np.log(prices)
                stats[category] = (
                    logs.mean(),
                    np.median(logs),
                    logs.std(),
                    np.log(prices.mean()),
                    len(prices),
                )
        for j, name in enumerate(
            [
                "category_logprice_mean",
                "category_logprice_median",
                "category_logprice_std",
                "category_log_of_mean_price",
                "category_price_observations",
            ]
        ):
            put(pref + name, [stats[c][j] if c in stats else np.nan for c in cats])
        put(pref + "price_history_missing", [c not in stats for c in cats])
        median = np.array([stats[c][1] if c in stats else np.nan for c in cats])
        put(pref + "price_delta_category_median", logprice - median)
        put(pref + "abs_price_delta_category_median", np.abs(logprice - median))
    numeric_names = list(cols)
    matrix = np.column_stack(list(cols.values())).astype("float32")
    assert not np.isinf(matrix).any()
    categorical = np.column_stack(
        [
            np.array([int(c) for c in cats], dtype="int64"),
            np.array([brand_id(b) for b in brands], dtype="int64"),
            np.full(n, int(last["category_id"]), dtype="int64"),
        ]
    )
    # Fixed heuristic scores on exactly the same candidates.
    recent = {}
    rank = 0
    for r in reversed(views):
        if r["product_id"] not in recent:
            recent[r["product_id"]] = rank
            rank += 1
    heuristic = {
        "global_popularity": np.log1p(pop),
        "last_category_popularity": (cats == last["category_id"]) * 2
        + np.log1p(pop) / 100,
        "recent_then_category_popularity": np.array(
            [
                (
                    3 + 1 / (recent[p] + 1)
                    if p in recent
                    else 2 * (cats[j] == last["category_id"]) + np.log1p(pop[j]) / 100
                )
                for j, p in enumerate(products)
            ]
        ),
    }
    histcats = [r["category_id"] for r in views]
    sameprices = [
        r["price"]
        for r in views
        if r["category_id"] == last["category_id"] and r["price"] > 0
    ]
    shift = len(sameprices) >= 6 and abs(
        np.log(np.median(sameprices[-3:]) / np.median(sameprices[-6:-3]))
    ) >= np.log(1.2)
    flags = {
        "category_switch": len(histcats) > 1 and histcats[-1] != histcats[-2],
        "price_shift": bool(shift),
        "prefix_has_repeats": len(recent) < len(views),
    }
    return matrix, categorical, numeric_names, heuristic, flags
