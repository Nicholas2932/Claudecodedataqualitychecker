"""Run data_quality_checker against each test dataset and summarise."""
import re
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANSI = re.compile(r"\x1b\[[0-9;]*m")

DATASETS = ["test_clean.csv", "test_mild.csv", "test_dirty.csv", "test_filthy.csv"]


def run(csv):
    print(f"\n{'=' * 78}\n{csv}\n{'=' * 78}")
    res = subprocess.run(
        [sys.executable, "data_quality_checker.py", csv],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if res.returncode != 0:
        print("EXIT", res.returncode)
        print("STDERR:", res.stderr[:2000])

    text = ANSI.sub("", res.stdout)
    # Show only the status lines + summary
    keep = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[PASS]") or stripped.startswith("[WARN]") or stripped.startswith("[FAIL]"):
            print(line.rstrip())
        elif "SUMMARY" in line or "Overall:" in line or stripped.startswith("PASS:"):
            print(line.rstrip())

    # Print any tracebacks/check-fail messages
    for line in text.splitlines():
        if "could not complete" in line.lower():
            print(" >>", line.strip())


if __name__ == "__main__":
    for ds in DATASETS:
        run(ds)
