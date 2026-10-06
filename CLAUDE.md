# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (numpy, pandas, scipy, openpyxl, pyarrow, pyyaml)
python -m pip install -r requirements.txt

# Run the checker on a dataset (format auto-detected: csv, tsv/txt, xlsx/xls, parquet, json, jsonl)
python data_quality_checker.py path/to/dataset.csv
python data_quality_checker.py data.parquet --config config.example.yaml

# Regenerate the four synthetic test fixtures (test_clean/mild/dirty/filthy.csv)
python generate_test_datasets.py

# Run the full regression: runs the checker over all four fixtures, prints status lines + summary
python run_tests.py
```

There is no unit-test framework, linter, or build step. `run_tests.py` is the regression harness: it shells out to `data_quality_checker.py` for each fixture and is the canonical way to confirm a change hasn't altered check behavior. To "run a single test", invoke the checker directly on one fixture (e.g. `python data_quality_checker.py test_filthy.csv`).

## Architecture

This is a single-purpose CLI, not a package. The data flows: **`Config` → `load_dataframe` → `DataQualityChecker` → `CheckResult` list → renderers**, all inside [data_quality_checker.py](data_quality_checker.py).

- **`Config` dataclass** holds every tunable threshold, column/check exclusion, and report target. `Config.from_file` loads YAML or JSON (by extension) and *flattens* the nested `input.*`, `report.*` (→ `report_*` fields), and `thresholds.*` sections plus top-level keys onto the flat dataclass. **Defaults exactly reproduce the historical hardcoded constants**, so a no-`--config` run is unchanged — preserve this when touching defaults, since `run_tests.py` depends on it.

- **`load_dataframe(filepath, cfg)`** dispatches on file extension. Optional engines (openpyxl, pyarrow) surface as `ImportError`, which `main()` catches and turns into a "pip install -r requirements.txt" hint.

- **`DataQualityChecker`** runs 25 checks. Each check is a `_check_*(self) -> CheckResult` method reading thresholds from `self.cfg`; `run_all_checks` calls them from an explicit list, wraps each in try/except (a crashing check becomes a `WARN` rather than aborting the run), and drops any result whose `.name` is in `cfg.exclude_checks`. **To add a check: write a `_check_*` method and append it to the list in `run_all_checks`.** `exclude_columns` are dropped in `__init__` before `_num_cols`/`_obj_cols` are computed, so all checks see the filtered frame.

- **`CheckResult`** is the uniform output unit: `status` (PASS/WARN/FAIL), `severity`, `details`, `explanation` (the "WHY"), and `fix_code` (copy-paste remediation). Three renderers consume the same `results` list: `print_report` (ANSI terminal), `render_markdown`, and `render_html`. The manual-review catalogue (issues automation can't detect — leakage, sampling bias, etc.) lives in the module-level **`MANUAL_CHECKS`** constant shared by all three renderers; edit it there once.

## Conventions specific to this repo

- **Statistical baselines are intentional, not arbitrary.** Outlier checks only flag rates *well above* the expected normal-distribution baseline (~0.7% for 1.5×IQR, ~0.27% for |z|>3), and binary 0/1 columns are skipped. Missingness is classified MCAR/MAR/MNAR via Little's MCAR test and point-biserial/Cramér's V predictor detection. Read the comments before changing thresholds.

- **`generate_test_datasets.py` ordering is load-bearing.** Comments mark steps that must run last (exact-duplicate rows are appended after all column edits; duplicate *columns* are created before row duplication). Each injected issue is annotated with the check number it targets. Keep these invariants when editing fixtures, or `run_tests.py` results drift.
