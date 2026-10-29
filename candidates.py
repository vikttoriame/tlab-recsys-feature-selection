"""Label-independent, deterministic candidate retrieval from a past-only catalog.
No target parameter, no positive injection, no popularity from evaluation days.
"""

from collections import defaultdict
import pandas as pd


class DailyCandidates:
    def __init__(self, path):
        d = pd.read_csv(path, dtype=str, keep_default_na=False)
        d["views_before_day"] = d.views_before_day.astype("int64")
        d["pid_numeric"] = d.product_id.astype("int64")
        d = d.sort_values(["views_before_day", "pid_numeric"], ascending=[False, True])
        self.products = d.set_index("product_id").to_dict("index")
        self.global_order = d.product_id.tolist()
        self.categories = defaultdict(list)
        for row in d.itertuples():
            self.categories[row.category_id].append(row.product_id)

    def retrieve(self, history, k=200):
        # history must already be clipped to cutoff, including tied-time events.
        views = history[history.event_type.eq("view")]
        if views.empty:
            return self.global_order[:k]
        seen = []
        have = set()

        def add(pid):
            if pid in self.products and pid not in have and len(seen) < k:
                seen.append(pid)
                have.add(pid)

        # 40 distinct most recently viewed known products.
        for pid in reversed(views.product_id.tolist()):
            add(pid)
            if len(seen) >= min(40, k):
                break
        # 80 popular products in category of latest viewed item, as known in history.
        category = views.iloc[-1].category_id
        count = 0
        for pid in self.categories.get(category, []):
            before = len(seen)
            add(pid)
            count += int(len(seen) > before)
            if count >= 80 or len(seen) >= k:
                break
        # Fill remaining places by global popularity strictly before this day.
        for pid in self.global_order:
            add(pid)
            if len(seen) >= k:
                break
        return seen
