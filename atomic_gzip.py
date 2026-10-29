"""Explicitly close text wrapper before gzip stream, fsync and rename."""

import contextlib, gzip, io, os
from pathlib import Path


@contextlib.contextmanager
def atomic_gzip(path):
    path = Path(path)
    tmp = path.with_name(path.name + ".writing")
    # CSV TextIOWrapper must flush while its gzip stream is still open.
    with tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
            try:
                yield text
            finally:
                text.flush()
                text.detach()
        raw.flush()
        os.fsync(raw.fileno())
    with gzip.open(tmp, "rb") as check:
        while check.read(1048576):
            pass
    tmp.replace(path)
