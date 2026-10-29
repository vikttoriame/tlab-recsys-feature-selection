import argparse, zipfile
from core import *


def main(checkpoint=False, code_only=False):
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(x in rel.parts for x in ["cache", "__pycache__", "tmp"]):
            continue
        if p.name == "SHA256SUMS" or p.name.endswith((".writing", ".tmp")):
            continue
        if checkpoint and p.suffix == ".log":
            continue
        if code_only and (
            rel.parts[0] in {"models", "logs"}
            or (rel.parts[0] == "data" and p.suffix != ".json")
        ):
            continue
        files.append(p)
    files.sort()
    expected = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    check = ROOT / "SHA256SUMS"
    check.write_text(
        "".join(
            expected[p.relative_to(ROOT).as_posix()]
            + "  "
            + p.relative_to(ROOT).as_posix()
            + "\n"
            for p in files
        )
    )
    files.append(check)
    path = ROOT.parent / (
        "tlab_submission_checkpoint.zip"
        if checkpoint
        else "tlab_submission_code.zip" if code_only else "tlab_submission.zip"
    )
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("wb") as raw:
        with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in files:
                z.write(p, "tlab_submission/" + p.relative_to(ROOT).as_posix())
        raw.flush()
        os.fsync(raw.fileno())
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            actual = hashlib.sha256(z.read("tlab_submission/" + rel)).hexdigest()
            assert actual == sha(p)
            if rel != "SHA256SUMS":
                assert actual == expected[rel], rel
    tmp.replace(path)
    print(
        json.dumps(
            dict(
                path=str(path),
                bytes=path.stat().st_size,
                files=len(files),
                verified=True,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="store_true")
    ap.add_argument("--code-only", action="store_true")
    a = ap.parse_args()
    assert not (a.checkpoint and a.code_only)
    main(a.checkpoint, a.code_only)
