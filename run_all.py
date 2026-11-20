# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""One command to reproduce everything (works on Windows, macOS and Linux):

    python run_all.py            # Part A + Part B + inference + C1 experiment
    python run_all.py --skip-c1  # skip the C1 timing experiment

Each step is run as a separate process; the script stops at the first failure.
"""

import subprocess
import sys

STEPS = [
    ("Verify raw data", [sys.executable, "-m", "src.download_data"]),
    ("Part A pipeline", [sys.executable, "-m", "src.pipeline_a"]),
    ("Quality-gate demo: make corrupted copy", [sys.executable, "-m", "src.make_corrupted"]),
    ("Part B training", [sys.executable, "-m", "src.train"]),
    ("Score unseen rows", [sys.executable, "-m", "src.predict", "data/new/unseen_intakes.csv"]),
    ("Check unseen scores", [sys.executable, "-m", "src.evaluate_unseen"]),
    ("Score hand-made rows", [sys.executable, "-m", "src.predict", "data/new/handmade_intakes.csv"]),
    ("Tests", [sys.executable, "-m", "pytest", "-q"]),
    ("C1 feed experiment", [sys.executable, "exploration/c1_feed/feed_experiment.py"]),
]


def main():
    skip_c1 = "--skip-c1" in sys.argv
    for name, cmd in STEPS:
        if skip_c1 and name.startswith("C1"):
            continue
        print(f"\n=== {name} ===", flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"\nSTOPPED: step '{name}' failed.", file=sys.stderr)
            return 1
    # The gate demo is expected to FAIL, so it is checked separately.
    print("\n=== Quality-gate demo: pipeline on corrupted copy (expected to fail) ===", flush=True)
    demo = subprocess.run([sys.executable, "-m", "src.pipeline_a", "--input",
                           "data/corrupted/shelter_corrupted.csv", "--metrics-out",
                           "outputs/reports/gate_demo/metrics_corrupted.json"])
    if demo.returncode == 0:
        print("ERROR: the quality gate did not stop corrupted data!", file=sys.stderr)
        return 1
    print("Gate correctly stopped the corrupted data.\n\nAll steps finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
