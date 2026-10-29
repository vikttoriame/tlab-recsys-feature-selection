import subprocess, time
from core import *


def main():
    while (
        not (ROOT / "results/training_complete.json").exists()
        or not (ROOT / "results/full_history_verification.json").exists()
    ):
        time.sleep(2)
    for script, args in [
        ("evaluate.py", ["--freeze"]),
        ("evaluate.py", []),
        ("verify.py", []),
        ("make_report.py", []),
    ]:
        print("RUN", script, *args, flush=True)
        subprocess.run(
            [sys.executable, "-u", str(ROOT / script)] + args, check=True, cwd=ROOT
        )
    print("FINAL RESULTS READY", flush=True)


if __name__ == "__main__":
    main()
