import importlib.metadata, platform, subprocess, time
from core import *


def run(script, *args):
    print("RUN", script, *args, flush=True)
    subprocess.run(
        [sys.executable, "-u", str(ROOT / script)] + list(args), check=True, cwd=ROOT
    )


def main():
    while not (ROOT / "data/extraction_manifest.json").exists():
        time.sleep(2)
    write_json(
        ROOT / "results/environment.json",
        dict(
            python=platform.python_version(),
            packages={
                n: importlib.metadata.version(n)
                for n in [
                    "numpy",
                    "pandas",
                    "catboost",
                    "scikit-learn",
                    "joblib",
                    "reportlab",
                ]
            },
        ),
    )
    for script, args in [
        ("build_queries.py", []),
        ("build_features.py", []),
        ("checks.py", []),
        ("train.py", ["--stage", "manual"]),
        ("select_features.py", []),
        ("train.py", ["--stage", "auto"]),
    ]:
        run(script, *args)
    write_json(
        ROOT / "results/training_complete.json", dict(models=15, test_scored=False)
    )


if __name__ == "__main__":
    main()
