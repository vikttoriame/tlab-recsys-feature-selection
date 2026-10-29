"""Reproduce in a fresh destination using bundled inputs or public source."""

import argparse, shutil, subprocess, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    dest = Path(a.output).resolve()
    assert not dest.exists(), "Use a NEW output directory"

    def ignore(path, names):
        return [
            n
            for n in names
            if n
            in [
                "cache",
                "models",
                "results",
                "logs",
                "tmp",
                "output",
                "__pycache__",
                "test_gate.json",
                "SHA256SUMS",
            ]
            or n.endswith((".zip", ".writing"))
            or (a.download and n == "data")
        ]

    shutil.copytree(root, dest, ignore=ignore)
    for n in ["data", "cache", "models", "results", "logs", "output/pdf"]:
        (dest / n).mkdir(parents=True, exist_ok=True)

    def run(name, *args):
        subprocess.run(
            [sys.executable, "-u", str(dest / name)] + list(args), check=True, cwd=dest
        )

    if a.download:
        run("extract.py")
        run("verify_source.py")
    else:
        import hashlib, json

        hashes = json.loads((dest / "data/dataset_hashes.json").read_text())
        for p, h in hashes.items():
            assert hashlib.sha256((dest / "data" / p).read_bytes()).hexdigest() == h, p
        shutil.copyfile(
            root / "results/full_history_verification.json",
            dest / "results/full_history_verification.json",
        )
    run("prepare_train.py")
    run("evaluate.py", "--freeze")
    run("evaluate.py")
    run("verify.py")
    run("make_report.py")
    run("package.py")


if __name__ == "__main__":
    main()
