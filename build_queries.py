from core import *


def main():
    e = load_events()
    queries = []
    ambiguous = boundary = short = 0
    for uid, g in e.groupby("user_id", sort=False):
        r = g.to_dict("records")
        times = g.seconds.to_numpy()
        vpos = np.flatnonzero(g.event_type.to_numpy() == "view")
        if len(vpos) < 3:
            continue
        vt = times[vpos]
        begins = np.r_[0, np.flatnonzero(vt[1:] != vt[:-1]) + 1]
        ends = np.r_[begins[1:], len(vpos)]
        for j in range(len(begins) - 1):
            if ends[j] < 2:
                short += 1
                continue
            nextpos = vpos[begins[j + 1] : ends[j + 1]]
            targets = {r[k]["product_id"] for k in nextpos}
            if len(targets) != 1:
                ambiguous += 1
                continue
            cut = vt[begins[j]]
            hcount = int(np.searchsorted(times, cut, side="right"))
            last = r[hcount - 1]
            target = r[nextpos[0]]
            day = last["event_time"][:10]
            split = None
            for name, lo, hi in [
                ("train", "2019-10-02", "2019-10-21"),
                ("validation", "2019-10-21", "2019-10-25"),
                ("test", "2019-10-25", "2019-10-31"),
            ]:
                if lo <= day < hi:
                    if target["event_time"][:10] < hi:
                        split = name
                    else:
                        boundary += 1
                    break
            if split is None:
                continue
            queries.append(
                dict(
                    query_id="q_" + str(last["source_row"]),
                    user_id=uid,
                    cutoff_time=last["event_time"],
                    cutoff_source_row=last["source_row"],
                    history_events=hcount,
                    history_views=int(ends[j]),
                    target_time=target["event_time"],
                    target_product_id=target["product_id"],
                    target_source_row=target["source_row"],
                    target_seen_in_prefix=int(
                        target["product_id"]
                        in {r[k]["product_id"] for k in vpos[: ends[j]]}
                    ),
                    catalog_date=day,
                    split=split,
                )
            )
    q = pd.DataFrame(queries).sort_values(["cutoff_time", "cutoff_source_row"])
    assert q.query_id.is_unique
    assert (q.target_time > q.cutoff_time).all()
    summary = {}
    for split in ["train", "validation", "test"]:
        z = q[q.split.eq(split)].drop(columns="split")
        save_csv(ROOT / f"data/queries_{split}.csv.gz", z)
        summary[split] = dict(
            queries=len(z),
            users=int(z.user_id.nunique()),
            first=z.cutoff_time.min(),
            last=z.cutoff_time.max(),
        )
    m = json.loads((ROOT / "data/extraction_manifest.json").read_text())
    assert e.user_id.nunique() == m["selected_users"] and len(e) == m["selected_events"]
    assert set(e.user_id) == set(m["per_user"])
    write_json(
        ROOT / "results/query_manifest.json",
        dict(
            splits=summary,
            ambiguous_targets_excluded=ambiguous,
            cross_boundary_labels_excluded=boundary,
            short_prefixes_excluded=short,
            raw_events_preserved=len(e),
            raw_users_preserved=int(e.user_id.nunique()),
            history_scope="all known user events across all sessions",
            test_outcomes_scored=False,
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
