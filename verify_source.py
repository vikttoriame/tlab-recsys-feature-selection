"""Independent second full-source pass: exact retained histories, including all sessions."""

import csv, gzip, hashlib, io, time
from urllib.request import Request, urlopen
from core import *
from extract import SOURCE, SCHEMA


def digest_row(h, row):
    h.update(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    )


def main():
    start = time.time()
    users = set(json.loads((ROOT / "data/selected_users.json").read_text()))
    local_hash = {u: hashlib.sha256() for u in users}
    source_hash = {u: hashlib.sha256() for u in users}
    local_count = {u: 0 for u in users}
    source_count = {u: 0 for u in users}
    with gzip.open(ROOT / "data/events_all.csv.gz", "rt", newline="") as f:
        r = csv.reader(f)
        assert next(r) == SCHEMA + ["source_row"]
        for row in r:
            local_count[row[7]] += 1
            digest_row(local_hash[row[7]], row)
    with urlopen(
        Request(
            SOURCE, headers={"User-Agent": "T-Lab-independent-history-verification/1.0"}
        ),
        timeout=60,
    ) as response:
        with gzip.GzipFile(fileobj=response) as gz, io.TextIOWrapper(
            gz, encoding="utf-8", newline=""
        ) as text:
            r = csv.reader(text)
            assert next(r) == SCHEMA
            day = ""
            for i, row in enumerate(r, 1):
                if row[0][:10] != day:
                    day = row[0][:10]
                    print(
                        "verify source",
                        day,
                        "elapsed",
                        round(time.time() - start),
                        flush=True,
                    )
                uid = row[7]
                if uid in users:
                    source_count[uid] += 1
                    digest_row(source_hash[uid], row + [str(i)])
    assert local_count == source_count
    for u in users:
        assert local_hash[u].hexdigest() == source_hash[u].hexdigest(), u
    write_json(
        ROOT / "results/full_history_verification.json",
        dict(
            passed=True,
            users=len(users),
            events=sum(source_count.values()),
            source_events_replayed=i,
            per_user_counts_and_all_row_hashes_match=True,
            scope="every event and original source row for every selected user, full October",
            elapsed_seconds=time.time() - start,
        ),
    )
    print("Complete-user-history replay passed", flush=True)


if __name__ == "__main__":
    main()
