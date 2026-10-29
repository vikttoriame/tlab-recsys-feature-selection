"""Bottom-k USER sample. Retain every event of each final user; stream all October."""

import csv, gzip, hashlib, heapq, io, json, os, time
from pathlib import Path
from urllib.request import Request, urlopen
from atomic_gzip import atomic_gzip

ROOT = Path(__file__).resolve().parent
SOURCE = "https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz"
USER_COUNT = 5000
SCHEMA = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]


def main():
    out = ROOT / "data"
    cd = out / "catalog_daily"
    cd.mkdir(exist_ok=True)
    start = time.time()
    catalog = {}
    selected = {}
    heap = []
    hashes = {}
    counts = {}
    day = None
    last = ""
    events = 0
    source_users = set()
    with urlopen(
        Request(SOURCE, headers={"User-Agent": "T-Lab-reproducible-research/1.0"}),
        timeout=60,
    ) as response:
        headers = {
            k: response.headers.get(k)
            for k in ["ETag", "Last-Modified", "Content-Length"]
        }
        with gzip.GzipFile(fileobj=response) as gz, io.TextIOWrapper(
            gz, encoding="utf-8", newline=""
        ) as text:
            reader = csv.reader(text)
            assert next(reader) == SCHEMA
            for events, row in enumerate(reader, 1):
                assert len(row) == 9
                ts, action, pid, cid, code, brand, price, uid, sid = row
                newday = ts[:10]
                assert ts >= last
                last = ts
                assert newday.startswith("2019-10-")
                if newday != day:
                    if day:
                        print(
                            json.dumps(
                                dict(
                                    completed_day=day,
                                    rows=events - 1,
                                    elapsed_seconds=round(time.time() - start, 1),
                                )
                            ),
                            flush=True,
                        )
                    with atomic_gzip(cd / f"{newday}.csv.gz") as f:
                        w = csv.writer(f)
                        w.writerow(
                            [
                                "product_id",
                                "category_id",
                                "category_code",
                                "brand",
                                "price",
                                "first_seen",
                                "last_seen",
                                "views_before_day",
                                "carts_before_day",
                                "purchases_before_day",
                            ]
                        )
                        for product in sorted(catalog, key=int):
                            w.writerow([product] + catalog[product])
                    day = newday
                if pid not in catalog:
                    catalog[pid] = [cid, code, brand, price, ts, ts, 0, 0, 0]
                item = catalog[pid]
                if cid:
                    item[0] = cid
                if code:
                    item[1] = code
                if brand:
                    item[2] = brand
                item[3] = price
                item[5] = ts
                if action in ["view", "cart", "purchase"]:
                    item[6 + ["view", "cart", "purchase"].index(action)] += 1
                counts[action] = counts.get(action, 0) + 1
                if uid not in hashes:
                    hashes[uid] = int.from_bytes(
                        hashlib.blake2b(
                            ("tlab-users-v1|" + uid).encode(), digest_size=8
                        ).digest(),
                        "big",
                    )
                    h = hashes[uid]
                    if len(selected) < USER_COUNT:
                        selected[uid] = []
                        heapq.heappush(heap, (-h, uid))
                    elif h < -heap[0][0]:
                        _, evicted = heapq.heappop(heap)
                        del selected[evicted]
                        selected[uid] = []
                        heapq.heappush(heap, (-h, uid))
                if uid in selected:
                    selected[uid].append(row + [events])
    assert day == "2019-10-31" and len(selected) == USER_COUNT
    # Heap cutoff only decreases: a user rejected/evicted once cannot qualify later.
    expected = set(sorted(hashes, key=lambda u: (hashes[u], u))[:USER_COUNT])
    assert set(selected) == expected
    rows = sorted(
        (r for history in selected.values() for r in history), key=lambda r: r[-1]
    )
    sample = out / "events_all.csv.gz"
    with atomic_gzip(sample) as f:
        w = csv.writer(f)
        w.writerow(SCHEMA + ["source_row"])
        w.writerows(rows)
    users = sorted(selected, key=lambda u: (hashes[u], u))
    (out / "selected_users.json").write_text(json.dumps(users, indent=2) + "\n")
    per_user = {
        u: dict(
            events=len(selected[u]),
            source_sessions=len({r[8] for r in selected[u]}),
            first=selected[u][0][0],
            last=selected[u][-1][0],
        )
        for u in users
    }
    manifest = dict(
        source_url=SOURCE,
        source_headers=headers,
        source_events=events,
        source_users=len(hashes),
        source_event_counts=counts,
        selected_users=USER_COUNT,
        selected_events=len(rows),
        first=rows[0][0],
        last=rows[-1][0],
        sampling="5000 smallest blake2b64 big-endian(tlab-users-v1|user_id); fixed user cohort, every October event retained",
        history_preservation="Streaming bottom-k users; cutoff is monotone decreasing. All rows retained from first occurrence of each final user.",
        sample_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),
        per_user=per_user,
        elapsed_seconds=time.time() - start,
    )
    tmp = out / "extraction_manifest.json.writing"
    with tmp.open("w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(out / "extraction_manifest.json")
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "per_user"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
