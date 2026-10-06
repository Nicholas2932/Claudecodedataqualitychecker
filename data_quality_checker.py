#!/usr/bin/env python3
"""
data_quality_checker.py — Automated dataset quality checker.
Usage: python data_quality_checker.py path/to/dataset.csv
"""

import sys
import os
import re
import json
import html
import argparse
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy import stats # type: ignore

warnings.filterwarnings("ignore")

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

STATUS_COLOUR = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}


@dataclass
class CheckResult:
    name: str
    status: str        # "PASS" | "WARN" | "FAIL"
    severity: str      # "HIGH" | "MEDIUM" | "LOW"
    details: str = ""
    explanation: str = ""
    fix_code: str = ""


# ── configuration ───────────────────────────────────────────────────────────
@dataclass
class Config:
    """All tunable thresholds and behavior knobs.

    Defaults exactly reproduce the historical hardcoded values, so running
    without a config file leaves behavior unchanged.
    """
    # input
    delimiter: Optional[str] = None        # None = auto-detect for delimited text
    excel_sheet: Any = 0

    # report output targets (None = skip)
    report_html: Optional[str] = None
    report_markdown: Optional[str] = None

    # column / check exclusions
    exclude_columns: List[str] = field(default_factory=list)
    exclude_checks: List[str] = field(default_factory=list)
    non_negative_columns: List[str] = field(default_factory=list)

    # thresholds
    missing_fail_pct: float = 20.0
    near_constant_frac: float = 0.95
    correlation: float = 0.95
    outlier_iqr_pct: float = 2.0
    zscore_warn_pct: float = 0.5
    zscore_fail_pct: float = 2.0
    skewness: float = 2.0
    high_cardinality_ratio: float = 0.8
    low_cardinality_max: int = 10
    class_imbalance_frac: float = 0.90
    zero_concentration_frac: float = 0.50
    min_rows_for_outliers: int = 30
    date_sample_size: Optional[int] = 500   # None = scan all rows
    memory_warn_mb: float = 500.0

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load a YAML or JSON config, overlaying values onto the defaults.

        A partial config is valid — unspecified keys keep their default. The
        ``input``, ``report``, and ``thresholds`` sub-mappings are flattened
        onto the flat dataclass fields.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()

        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            import yaml  # type: ignore
            data = yaml.safe_load(raw) or {}
        elif ext == ".json":
            data = json.loads(raw) if raw.strip() else {}
        else:
            # Try YAML first (a superset of JSON), then fall back to JSON.
            try:
                import yaml  # type: ignore
                data = yaml.safe_load(raw) or {}
            except Exception:
                data = json.loads(raw) if raw.strip() else {}

        if not isinstance(data, dict):
            raise ValueError("Config root must be a mapping.")

        valid = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}

        # nested input.* / report.* sections
        for src_key, prefix in (("input", ""), ("report", "report_")):
            section = data.get(src_key) or {}
            if isinstance(section, dict):
                for k, v in section.items():
                    name = f"{prefix}{k}"
                    if name in valid:
                        kwargs[name] = v

        # nested thresholds.*
        thresholds = data.get("thresholds") or {}
        if isinstance(thresholds, dict):
            for k, v in thresholds.items():
                if k in valid:
                    kwargs[k] = v

        # top-level flat keys (exclude_columns, exclude_checks, ...)
        for k, v in data.items():
            if k in valid:
                kwargs[k] = v

        return cls(**kwargs)


# ── formatting helpers ────────────────────────────────────────────────────────
def _section(title: str) -> str:
    return f"\n{BOLD}{CYAN}{'─' * 4} {title} {'─' * max(0, 66 - len(title))}{RESET}"


def _fmt_code(code: str) -> str:
    return "\n".join(f"        {DIM}{line}{RESET}" for line in code.strip().splitlines())


# ── Little's MCAR test ────────────────────────────────────────────────────────
def _littles_mcar_test(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    num_df = df.select_dtypes(include=[np.number]).copy()
    num_df = num_df.dropna(axis=1, how="all")
    cols_with_miss = [c for c in num_df.columns if num_df[c].isnull().any()]
    if len(cols_with_miss) < 2:
        return None, None

    grand_means = num_df.mean()
    cov_matrix = num_df.cov()
    try:
        cov_inv = np.linalg.pinv(cov_matrix.values)
    except Exception:
        return None, None

    cols = list(num_df.columns)
    miss_pattern = num_df.isnull()
    unique_patterns = miss_pattern.drop_duplicates()

    d2, dof = 0.0, 0
    for _, pat in unique_patterns.iterrows():
        obs_cols = [c for c in cols if not pat[c]]
        if not obs_cols:
            continue
        mask = (miss_pattern[cols] == pat[cols]).all(axis=1)
        n_j = int(mask.sum())
        if n_j == 0:
            continue
        diff = (num_df.loc[mask, obs_cols].mean() - grand_means[obs_cols]).values
        obs_idx = [cols.index(c) for c in obs_cols]
        sub_inv = cov_inv[np.ix_(obs_idx, obs_idx)]
        d2 += n_j * float(diff @ sub_inv @ diff)
        dof += len(obs_cols)

    dof -= len(cols)
    if dof <= 0:
        return None, None
    return d2, float(1 - stats.chi2.cdf(d2, dof))


def _detect_mar_predictor(df: pd.DataFrame, col: str) -> Optional[str]:
    miss_ind = df[col].isnull().astype(int)
    if miss_ind.sum() == 0:
        return None
    best_col, best_strength = None, 0.0
    for other in df.columns:
        if other == col:
            continue
        ov = df[other].dropna()
        common = miss_ind.index.intersection(ov.index)
        if len(common) < 10:
            continue
        mi = miss_ind.loc[common]
        ov = df.loc[common, other]
        try:
            if pd.api.types.is_numeric_dtype(ov):
                r, _ = stats.pointbiserialr(mi, ov)
                strength = abs(r)
            else:
                ct = pd.crosstab(mi, ov)
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                chi2_val, _, _, _ = stats.chi2_contingency(ct)
                n = ct.values.sum()
                strength = float(np.sqrt(chi2_val / (n * (min(ct.shape) - 1))))
            if strength > 0.3 and strength > best_strength:
                best_strength, best_col = strength, other
        except Exception:
            continue
    return best_col


# ── DataQualityChecker ────────────────────────────────────────────────────────
class DataQualityChecker:
    def __init__(self, df: pd.DataFrame, filepath: str, config: Optional[Config] = None):
        self.cfg = config or Config()
        # Columns excluded by config are ignored by every check.
        excluded = set(self.cfg.exclude_columns)
        if excluded:
            df = df.drop(columns=[c for c in df.columns if c in excluded], errors="ignore")
        self.df = df
        self.filepath = filepath
        self.results: List[CheckResult] = []
        self._num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    # ── check 1 ───────────────────────────────────────────────────────────────
    def _check_missing_values(self) -> CheckResult:
        df = self.df
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return CheckResult("Missing Values", "PASS", "HIGH", "No missing values found.")

        pct = (missing / len(df) * 100).round(2)
        status = "FAIL" if (pct > self.cfg.missing_fail_pct).any() else "WARN"

        lil_stat, lil_p = _littles_mcar_test(df)
        global_mcar = lil_p is not None and lil_p >= 0.05

        mcar_cols, mar_cols, mnar_cols = [], [], []
        col_lines = []
        for col in missing.index:
            predictor = _detect_mar_predictor(df, col)
            if predictor:
                mech = f"likely MAR (correlates with '{predictor}')"
                mar_cols.append(col)
            elif global_mcar:
                mech = f"likely MCAR (Little's test p={lil_p:.3f})"
                mcar_cols.append(col)
            else:
                p_str = f"{lil_p:.3f}" if lil_p is not None else "N/A"
                mech = f"possibly MNAR (Little's test p={p_str}, no strong predictor found)"
                mnar_cols.append(col)
            col_lines.append(f"  {col}: {pct[col]:.1f}% missing — {mech}")

        little_note = ""
        if lil_p is not None:
            verdict = "fail to reject" if lil_p >= 0.05 else "reject"
            little_note = f"\n  Little's MCAR test: χ²={lil_stat:.2f}, p={lil_p:.4f} ({verdict} MCAR at α=0.05)"

        details = f"{len(missing)} column(s) with missing data:\n" + "\n".join(col_lines) + little_note

        explanation = (
            "Missing values cause ML training errors, biased statistics, and misleading visuals.\n"
            "  MCAR — missingness is purely random; mean/median imputation is unbiased.\n"
            "  MAR  — missingness depends on observed columns; use model-based / MICE imputation.\n"
            "  MNAR — missingness depends on the unobserved value itself; standard imputation\n"
            "          introduces bias. The mechanism must be modelled or the data recollected."
        )

        fix_parts = []
        if mcar_cols:
            lines = []
            for c in mcar_cols:
                if c in self._num_cols:
                    lines.append(f"df['{c}'].fillna(df['{c}'].median(), inplace=True)")
                else:
                    lines.append(f"df['{c}'].fillna(df['{c}'].mode()[0], inplace=True)")
            fix_parts.append("# MCAR — simple imputation\n" + "\n".join(lines))
        if mar_cols:
            fix_parts.append(
                f"# MAR — use MICE / IterativeImputer\n"
                f"from sklearn.experimental import enable_iterative_imputer  # noqa\n"
                f"from sklearn.impute import IterativeImputer\n"
                f"imp = IterativeImputer(random_state=42)\n"
                f"df[{mar_cols}] = imp.fit_transform(df[{mar_cols}])"
            )
        if mnar_cols:
            lines = [f"df['{c}_was_missing'] = df['{c}'].isnull().astype(int)" for c in mnar_cols]
            fix_parts.append(
                "# Possibly MNAR — add missingness indicator, then investigate\n"
                + "\n".join(lines)
                + "\n# Apply domain-appropriate imputation or remove column."
            )
        fix_parts.append(
            f"# Or drop rows with any missing value in affected columns\n"
            f"df.dropna(subset={list(missing.index)}, inplace=True)"
        )
        return CheckResult("Missing Values", status, "HIGH", details, explanation, "\n\n".join(fix_parts))

    # ── check 2 ───────────────────────────────────────────────────────────────
    def _check_duplicate_rows(self) -> CheckResult:
        n = self.df.duplicated().sum()
        if n == 0:
            return CheckResult("Duplicate Rows", "PASS", "HIGH", "No duplicate rows found.")
        return CheckResult(
            "Duplicate Rows", "FAIL", "HIGH",
            f"{n} exact duplicate row(s) ({n/len(self.df)*100:.1f}% of dataset).",
            "Duplicate rows inflate sample size, bias aggregations, and cause data leakage "
            "between train/test splits when the same row appears in both sets.",
            "df.drop_duplicates(inplace=True)\n"
            "# Inspect duplicates before dropping:\n"
            "df[df.duplicated(keep=False)].sort_values(df.columns.tolist())"
        )

    # ── check 3 ───────────────────────────────────────────────────────────────
    def _check_duplicate_columns(self) -> CheckResult:
        seen, pairs = {}, []
        for col in self.df.columns:
            key = tuple(self.df[col].values.tolist())
            if key in seen:
                pairs.append((seen[key], col))
            else:
                seen[key] = col
        if not pairs:
            return CheckResult("Duplicate Columns", "PASS", "HIGH", "No duplicate columns found.")
        details = ", ".join(f"'{a}' == '{b}'" for a, b in pairs)
        fix = "\n".join(f"df.drop(columns=['{b}'], inplace=True)  # identical to '{a}'" for a, b in pairs)
        return CheckResult(
            "Duplicate Columns", "FAIL", "HIGH",
            f"Identical-value column pair(s): {details}",
            "Duplicate columns waste memory, inflate ML feature importance, and introduce "
            "perfect multicollinearity in linear/logistic regression.",
            fix
        )

    # ── check 4 ───────────────────────────────────────────────────────────────
    def _check_unnamed_index(self) -> CheckResult:
        bad = [c for c in self.df.columns if re.match(r"^Unnamed:\s*\d+", str(c))]
        if not bad:
            return CheckResult("Unnamed Index Columns", "PASS", "MEDIUM", "No unnamed index columns.")
        return CheckResult(
            "Unnamed Index Columns", "WARN", "MEDIUM",
            f"Likely saved-index artifact column(s): {bad}",
            "pandas writes a column named 'Unnamed: 0' when df.to_csv() is called without "
            "index=False. These carry no analytical value.",
            f"df.drop(columns={bad}, inplace=True)\n"
            "# Prevent in future:\n"
            "df.to_csv('output.csv', index=False)"
        )

    # ── check 5 ───────────────────────────────────────────────────────────────
    def _check_mistyped_numeric(self) -> CheckResult:
        bad = []
        for col in self._obj_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            if pd.to_numeric(s, errors="coerce").notna().mean() > 0.9:
                bad.append(col)
        if not bad:
            return CheckResult("Mistyped Numeric Columns", "PASS", "HIGH",
                               "No object columns that should be numeric.")
        fix = "\n".join(f"df['{c}'] = pd.to_numeric(df['{c}'], errors='coerce')" for c in bad)
        return CheckResult(
            "Mistyped Numeric Columns", "WARN", "HIGH",
            f"Column(s) stored as object but contain numeric data: {bad}",
            "Numeric-as-string columns cannot be used in arithmetic, aggregations, or ML "
            "without conversion and consume significantly more memory.",
            fix
        )

    # ── check 6 ───────────────────────────────────────────────────────────────
    def _check_mistyped_datetime(self) -> CheckResult:
        numeric_obj = set()
        for col in self._obj_cols:
            s = self.df[col].dropna()
            if len(s) > 0 and pd.to_numeric(s, errors="coerce").notna().mean() > 0.9:
                numeric_obj.add(col)
        sample_n = self.cfg.date_sample_size
        bad = []
        for col in self._obj_cols:
            if col in numeric_obj:
                continue
            s = self.df[col].dropna().astype(str)
            s = s.head(sample_n) if sample_n else s
            if len(s) < 5:
                continue
            try:
                if pd.to_datetime(s, infer_datetime_format=True, errors="coerce").notna().mean() > 0.8:
                    bad.append(col)
            except Exception:
                continue
        if not bad:
            return CheckResult("Mistyped Datetime Columns", "PASS", "MEDIUM",
                               "No string columns appear to contain datetime data.")
        fix = "\n".join(
            f"df['{c}'] = pd.to_datetime(df['{c}'], infer_datetime_format=True, errors='coerce')"
            for c in bad
        )
        return CheckResult(
            "Mistyped Datetime Columns", "WARN", "MEDIUM",
            f"Column(s) stored as object but contain date/time data: {bad}",
            "Datetime-as-string columns prevent time-series operations, date arithmetic, "
            "chronological sorting, and temporal feature extraction.",
            fix
        )

    # ── check 7 ───────────────────────────────────────────────────────────────
    def _check_mixed_types(self) -> CheckResult:
        PLACEHOLDERS = {"n/a", "na", "none", "null", "nan", "unknown", "-", "--", "?", "missing"}
        bad = []
        for col in self._obj_cols:
            s = self.df[col].dropna().astype(str)
            if len(s) == 0:
                continue
            num_mask = pd.to_numeric(s, errors="coerce").notna()
            dirty_mask = s.str.strip().str.lower().isin(PLACEHOLDERS)
            if num_mask.sum() > 0 and dirty_mask.sum() > 0:
                bad.append((col, int(num_mask.sum()), int(dirty_mask.sum())))
        if not bad:
            return CheckResult("Mixed Types in Columns", "PASS", "HIGH",
                               "No columns mix numeric values with placeholder strings.")
        details = "; ".join(f"'{c}' ({n} numeric, {d} placeholder strings)" for c, n, d in bad)
        fix_lines = []
        for col, _, _ in bad:
            fix_lines.append(
                f"df['{col}'] = pd.to_numeric(\n"
                f"    df['{col}'].replace({{'N/A': None, 'unknown': None, '-': None,\n"
                f"                         'null': None, 'none': None, '?': None}},\n"
                f"                        regex=False).str.strip(),\n"
                f"    errors='coerce'\n"
                f")"
            )
        return CheckResult(
            "Mixed Types in Columns", "FAIL", "HIGH",
            f"Column(s) mixing real numbers with non-numeric placeholder strings: {details}",
            "Columns with both numeric values and strings like 'N/A' or 'unknown' are "
            "parsed as object dtype. Arithmetic is impossible and true NaNs are hidden.",
            "\n\n".join(fix_lines)
        )

    # ── check 8 ───────────────────────────────────────────────────────────────
    def _check_constant_columns(self) -> CheckResult:
        bad = [c for c in self.df.columns if self.df[c].dropna().nunique() <= 1]
        if not bad:
            return CheckResult("Constant Columns", "PASS", "HIGH", "No zero-variance columns found.")
        return CheckResult(
            "Constant Columns", "FAIL", "HIGH",
            f"Column(s) with only one unique non-null value: {bad}",
            "Constant columns carry zero information. They cause division-by-zero in "
            "normalisation, break correlation computations, and waste memory.",
            f"df.drop(columns={bad}, inplace=True)"
        )

    # ── check 9 ───────────────────────────────────────────────────────────────
    def _check_near_constant(self) -> CheckResult:
        bad = []
        for col in self.df.columns:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            vc = s.value_counts(normalize=True)
            if vc.iloc[0] >= self.cfg.near_constant_frac:
                bad.append((col, vc.index[0], f"{vc.iloc[0]*100:.1f}%"))
        pct_label = f"{self.cfg.near_constant_frac*100:.0f}%"
        if not bad:
            return CheckResult("Near-Constant Columns", "PASS", "MEDIUM",
                               f"No columns with a single value dominating ≥{pct_label} of rows.")
        details = "; ".join(f"'{c}' ('{v}' = {p})" for c, v, p in bad)
        cols = [c for c, _, _ in bad]
        return CheckResult(
            "Near-Constant Columns", "WARN", "MEDIUM",
            f"Column(s) where one value dominates ≥{pct_label} of non-null rows: {details}",
            "Near-constant columns have near-zero variance. They provide minimal predictive "
            "signal, can degrade tree splits, and may indicate a data collection error.",
            f"df.drop(columns={cols}, inplace=True)\n"
            "# Inspect before dropping:\n"
            + "\n".join(f"print(df['{c}'].value_counts())" for c, _, _ in bad)
        )

    # ── check 10 ──────────────────────────────────────────────────────────────
    def _check_all_unique(self) -> CheckResult:
        n = len(self.df)
        bad = [c for c in self.df.columns if self.df[c].nunique() == n]
        if not bad:
            return CheckResult("All-Unique Columns (Potential IDs)", "PASS", "MEDIUM",
                               "No columns where every row has a unique value.")
        return CheckResult(
            "All-Unique Columns (Potential IDs)", "WARN", "MEDIUM",
            f"Column(s) with 100% unique values (likely identifiers): {bad}",
            "All-unique columns are almost certainly IDs or free-text. As ML features they "
            "cause severe overfitting; in aggregations they produce meaningless counts of 1.",
            "\n".join(f"df.set_index('{c}', inplace=True)  # if true ID" for c in bad)
            + f"\n# Or drop:\ndf.drop(columns={bad}, inplace=True)"
        )

    # ── check 11 ──────────────────────────────────────────────────────────────
    def _check_whitespace(self) -> CheckResult:
        bad = []
        for col in self._obj_cols:
            s = self.df[col].dropna().astype(str)
            n = (s != s.str.strip()).sum()
            if n > 0:
                bad.append((col, int(n)))
        if not bad:
            return CheckResult("Whitespace in String Columns", "PASS", "MEDIUM",
                               "No leading/trailing whitespace found.")
        details = "; ".join(f"'{c}' ({n} rows)" for c, n in bad)
        fix = "\n".join(f"df['{c}'] = df['{c}'].str.strip()" for c, _ in bad)
        return CheckResult(
            "Whitespace in String Columns", "FAIL", "MEDIUM",
            f"Leading/trailing whitespace in: {details}",
            "Whitespace causes silent mismatches in groupby, merges, and categorical "
            "comparisons. 'Male' and ' Male' are treated as distinct categories.",
            fix
        )

    # ── check 12 ──────────────────────────────────────────────────────────────
    def _check_empty_strings(self) -> CheckResult:
        bad = []
        for col in self._obj_cols:
            n = (self.df[col].astype(str).str.strip() == "").sum()
            if n > 0:
                bad.append((col, int(n)))
        if not bad:
            return CheckResult("Empty Strings", "PASS", "MEDIUM", "No empty string values found.")
        details = "; ".join(f"'{c}' ({n} rows)" for c, n in bad)
        fix = "import numpy as np\n" + "\n".join(
            f"df['{c}'].replace('', np.nan, inplace=True)" for c, _ in bad
        )
        return CheckResult(
            "Empty Strings", "FAIL", "MEDIUM",
            f"Empty string ('') values in: {details}",
            "Empty strings pass isnull() checks and pollute aggregations, length "
            "calculations, and model encodings as invisible placeholder values.",
            fix
        )

    # ── check 13 ──────────────────────────────────────────────────────────────
    def _check_inconsistent_casing(self) -> CheckResult:
        bad = []
        for col in self._obj_cols:
            s = self.df[col].dropna().astype(str).str.strip()
            uniq = s.unique()
            if len(uniq) > 50:
                continue
            if len({v.lower() for v in uniq}) < len(uniq):
                bad.append(col)
        if not bad:
            return CheckResult("Inconsistent Casing", "PASS", "LOW",
                               "No mixed-case inconsistency in categorical columns.")
        fix = "\n".join(f"df['{c}'] = df['{c}'].str.strip().str.lower()" for c in bad)
        return CheckResult(
            "Inconsistent Casing", "WARN", "LOW",
            f"Column(s) with mixed-case values (e.g. 'Male' vs 'male'): {bad}",
            "Inconsistent casing inflates cardinality: 'Male', 'male', and 'MALE' become "
            "three separate categories in groupby, one-hot encoding, and value counts.",
            fix
        )

    # ── check 14 ──────────────────────────────────────────────────────────────
    def _check_outliers_iqr(self) -> CheckResult:
        # Normal data already produces ~0.7% of values outside the 1.5×IQR fence,
        # so we only flag rates substantially above that baseline.
        min_rows = self.cfg.min_rows_for_outliers
        bad = []
        skipped = []
        for col in self._num_cols:
            s = self.df[col].dropna()
            if len(s) < min_rows:
                skipped.append(col)
                continue
            # Skip binary columns — IQR is meaningless for them.
            if set(s.unique()).issubset({0, 1, 0.0, 1.0}):
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((s < lo) | (s > hi)).sum())
            pct = n / len(s) * 100
            if pct > self.cfg.outlier_iqr_pct:                               # well above the normal baseline
                bad.append((col, n, round(float(lo), 4), round(float(hi), 4), round(pct, 2)))
        skip_note = (
            f"\nNote: {len(skipped)} column(s) skipped (fewer than {min_rows} "
            f"non-null values, too few for a reliable IQR fence): {skipped}"
            if skipped else ""
        )
        if not bad:
            return CheckResult("Outliers (IQR Method)", "PASS", "MEDIUM",
                               f"No columns with anomalous IQR-outlier rates "
                               f"(>{self.cfg.outlier_iqr_pct}% of values).{skip_note}")
        details = "; ".join(f"'{c}' ({n} = {p}%, fence [{lo}, {hi}])" for c, n, lo, hi, p in bad) + skip_note
        fix_lines = []
        for col, _, lo, hi, _ in bad:
            fix_lines.append(
                f"# '{col}' — Winsorise (cap at fence):\n"
                f"df['{col}'] = df['{col}'].clip(lower={lo}, upper={hi})\n"
                f"# Or remove outlier rows:\n"
                f"df = df[(df['{col}'] >= {lo}) & (df['{col}'] <= {hi})]"
            )
        return CheckResult(
            "Outliers (IQR Method)", "WARN", "MEDIUM",
            f"{len(bad)} column(s) with >{self.cfg.outlier_iqr_pct}% of values "
            f"beyond the 1.5×IQR fence: {details}",
            "A high rate of IQR outliers skews means, inflates variance, distorts "
            "regression coefficients, and dominates distance-based algorithms "
            "(k-NN, k-means, PCA). Normal data produces ~0.7% outliers — anything "
            "much higher signals heavy tails, mixed populations, or errors.",
            "\n\n".join(fix_lines)
        )

    # ── check 15 ──────────────────────────────────────────────────────────────
    def _check_outliers_zscore(self) -> CheckResult:
        # Under a normal distribution, ~0.27% of values exceed |z|=3.
        # Only flag columns whose rate is well above this baseline.
        min_rows = self.cfg.min_rows_for_outliers
        bad = []
        skipped = []
        for col in self._num_cols:
            s = self.df[col].dropna()
            if len(s) < min_rows or s.std() == 0:
                if len(s) < min_rows:
                    skipped.append(col)
                continue
            if set(s.unique()).issubset({0, 1, 0.0, 1.0}):
                continue
            n = int((np.abs(stats.zscore(s)) > 3).sum())
            if n == 0:
                continue
            pct = n / len(s) * 100
            if pct > self.cfg.zscore_fail_pct:
                bad.append((col, n, round(pct, 2), "FAIL"))
            elif pct > self.cfg.zscore_warn_pct:
                bad.append((col, n, round(pct, 2), "WARN"))
            # else: within the expected ~0.27% baseline, ignore.
        skip_note = (
            f"\nNote: {len(skipped)} column(s) skipped (fewer than {min_rows} "
            f"non-null values, too few for a reliable z-score): {skipped}"
            if skipped else ""
        )
        if not bad:
            return CheckResult("Extreme Outliers (Z-Score |z|>3)", "PASS", "HIGH",
                               f"No columns with an anomalous rate of |z|>3 values.{skip_note}")
        has_fail = any(tag == "FAIL" for *_, tag in bad)
        status = "FAIL" if has_fail else "WARN"
        details = "; ".join(f"'{c}' ({n} rows = {p}%)" for c, n, p, _ in bad) + skip_note
        fix_lines = []
        for col, *_ in bad:
            fix_lines.append(
                f"from scipy import stats\n"
                f"mask = np.abs(stats.zscore(df['{col}'].fillna(df['{col}'].median()))) <= 3\n"
                f"df = df[mask]"
            )
        return CheckResult(
            "Extreme Outliers (Z-Score |z|>3)", status, "HIGH",
            f"{len(bad)} column(s) with an excessive |z|>3 rate "
            f"(>{self.cfg.zscore_warn_pct}%): {details}",
            "Under a normal distribution only ~0.27% of values exceed |z|=3. Rates above\n"
            "this expected baseline indicate heavy tails, mixed populations, measurement\n"
            "errors, or data entry mistakes. Extreme outliers dominate squared-error\n"
            "loss and distort Pearson correlations.",
            "import numpy as np\n\n" + "\n\n".join(fix_lines)
        )

    # ── check 16 ──────────────────────────────────────────────────────────────
    def _check_skewness(self) -> CheckResult:
        bad = []
        for col in self._num_cols:
            s = self.df[col].dropna()
            if len(s) < 10:
                continue
            skew = float(s.skew())
            if abs(skew) > self.cfg.skewness:
                bad.append((col, round(skew, 3)))
        if not bad:
            return CheckResult("High Skewness", "PASS", "MEDIUM",
                               f"No numeric columns with |skewness| > {self.cfg.skewness}.")
        details = "; ".join(f"'{c}' (skew={sk})" for c, sk in bad)
        fix_lines = []
        for col, skew in bad:
            if skew > 0:
                fix_lines.append(
                    f"# '{col}' right-skewed — log1p or Box-Cox transform:\n"
                    f"df['{col}_log'] = np.log1p(df['{col}'].clip(lower=0))"
                )
            else:
                fix_lines.append(
                    f"# '{col}' left-skewed — square transform:\n"
                    f"df['{col}_sq'] = df['{col}'] ** 2"
                )
        return CheckResult(
            "High Skewness", "WARN", "MEDIUM",
            f"{len(bad)} column(s) with |skewness| > {self.cfg.skewness}: {details}",
            "High skewness violates normality assumptions in linear models, t-tests, "
            "ANOVA, and Pearson correlation. It also slows gradient descent convergence.",
            "import numpy as np\n\n" + "\n\n".join(fix_lines)
        )

    # ── check 17 ──────────────────────────────────────────────────────────────
    def _check_high_correlation(self) -> CheckResult:
        if len(self._num_cols) < 2:
            return CheckResult("High Feature Correlation", "PASS", "MEDIUM",
                               "Fewer than 2 numeric columns — skipped.")
        corr = self.df[self._num_cols].corr().abs()
        pairs = []
        for i in range(len(self._num_cols)):
            for j in range(i + 1, len(self._num_cols)):
                r = corr.iloc[i, j]
                if r >= self.cfg.correlation:
                    pairs.append((self._num_cols[i], self._num_cols[j], round(float(r), 4)))
        if not pairs:
            return CheckResult("High Feature Correlation", "PASS", "MEDIUM",
                               f"No numeric column pairs with |r| ≥ {self.cfg.correlation}.")
        details = "; ".join(f"('{a}', '{b}') r={r}" for a, b, r in pairs)
        fix_lines = ["# Drop one column from each highly correlated pair:"]
        for a, b, _ in pairs:
            fix_lines.append(f"df.drop(columns=['{b}'], inplace=True)  # correlated with '{a}'")
        fix_lines.append(
            "\n# Alternatively, use VIF to identify which columns to drop:\n"
            "from statsmodels.stats.outliers_influence import variance_inflation_factor\n"
            "vif_data = pd.DataFrame({\n"
            "    'feature': df[numeric_cols].columns,\n"
            "    'VIF': [variance_inflation_factor(df[numeric_cols].values, i)\n"
            "            for i in range(len(numeric_cols))]\n"
            "})\n"
            "print(vif_data[vif_data['VIF'] > 10])  # VIF > 10 = multicollinearity"
        )
        return CheckResult(
            "High Feature Correlation", "WARN", "MEDIUM",
            f"{len(pairs)} column pair(s) with |r| ≥ {self.cfg.correlation}: {details}",
            "Highly correlated features cause multicollinearity in regression (unstable "
            "coefficients, inflated SEs). They also double-count information and waste "
            "model capacity.",
            "\n".join(fix_lines)
        )

    # ── check 18 ──────────────────────────────────────────────────────────────
    def _check_high_cardinality_categorical(self) -> CheckResult:
        if len(self.df) < 50:
            return CheckResult("High-Cardinality Categorical Columns", "PASS", "LOW",
                               "Dataset too small (<50 rows) — cardinality check skipped.")
        bad = []
        for col in self._obj_cols:
            ratio = self.df[col].nunique() / len(self.df)
            if ratio >= self.cfg.high_cardinality_ratio:
                bad.append((col, self.df[col].nunique(), f"{ratio*100:.1f}%"))
        card_label = f"{self.cfg.high_cardinality_ratio*100:.0f}%"
        if not bad:
            return CheckResult("High-Cardinality Categorical Columns", "PASS", "LOW",
                               f"No object columns with ≥{card_label} unique values.")
        details = "; ".join(f"'{c}' ({n} unique, {p})" for c, n, p in bad)
        fix_lines = []
        for col, _, _ in bad:
            fix_lines.append(
                f"# Drop if '{col}' is an ID:\ndf.drop(columns=['{col}'], inplace=True)\n\n"
                f"# Or target-encode for ML (fit on train only):\n"
                f"means = df_train.groupby('{col}')['target'].mean()\n"
                f"df['{col}_enc'] = df['{col}'].map(means).fillna(means.mean())"
            )
        return CheckResult(
            "High-Cardinality Categorical Columns", "WARN", "LOW",
            f"Column(s) with ≥{card_label} unique values (likely IDs or free text): {details}",
            "One-hot encoding high-cardinality columns produces thousands of sparse columns "
            "and causes severe overfitting. They are usually identifiers with no predictive value.",
            "\n\n".join(fix_lines)
        )

    # ── check 19 ──────────────────────────────────────────────────────────────
    def _check_low_cardinality_numeric(self) -> CheckResult:
        bad = [
            (c, self.df[c].nunique())
            for c in self._num_cols
            if 1 < self.df[c].nunique() <= self.cfg.low_cardinality_max and len(self.df) > 20
        ]
        if not bad:
            return CheckResult("Low-Cardinality Numeric Columns", "PASS", "LOW",
                               f"No numeric columns with ≤{self.cfg.low_cardinality_max} unique values.")
        details = "; ".join(f"'{c}' ({n} unique)" for c, n in bad)
        fix_lines = []
        for col, _ in bad:
            fix_lines.append(
                f"# If '{col}' is categorical, convert:\n"
                f"df['{col}'] = df['{col}'].astype('category')\n"
                f"# Or one-hot encode:\n"
                f"df = pd.get_dummies(df, columns=['{col}'], prefix='{col}')"
            )
        return CheckResult(
            "Low-Cardinality Numeric Columns", "WARN", "LOW",
            f"Numeric column(s) with very few unique values: {details}",
            "Numeric columns with ≤10 unique values are likely ordinal categories "
            "(e.g. ratings 1–5, boolean flags). Treating them as continuous in "
            "distance or regression models can misrepresent their structure.",
            "\n\n".join(fix_lines)
        )

    # ── check 20 ──────────────────────────────────────────────────────────────
    def _check_class_imbalance(self) -> CheckResult:
        bad = []
        for col in self.df.columns:
            nu = self.df[col].nunique()
            if nu < 2 or nu > 20:
                continue
            vc = self.df[col].value_counts(normalize=True)
            if vc.iloc[0] >= self.cfg.class_imbalance_frac:
                bad.append((col, vc.index[0], f"{vc.iloc[0]*100:.1f}%", nu))
        imb_label = f"{self.cfg.class_imbalance_frac*100:.0f}%"
        if not bad:
            return CheckResult("Class Imbalance", "PASS", "MEDIUM",
                               f"No low-cardinality columns with dominant class ≥{imb_label}.")
        details = "; ".join(f"'{c}' ('{v}' = {p}, {k} classes)" for c, v, p, k in bad)
        fix_lines = []
        for col, _, _, _ in bad:
            fix_lines.append(
                f"# If '{col}' is the target — oversample minority with SMOTE:\n"
                f"from imblearn.over_sampling import SMOTE\n"
                f"X_res, y_res = SMOTE(random_state=42).fit_resample(X, df['{col}'])\n\n"
                f"# Or use class_weight='balanced' in your model:\n"
                f"from sklearn.ensemble import RandomForestClassifier\n"
                f"model = RandomForestClassifier(class_weight='balanced')\n\n"
                f"# Always use stratified splits:\n"
                f"from sklearn.model_selection import train_test_split\n"
                f"X_tr, X_te, y_tr, y_te = train_test_split(\n"
                f"    X, y, stratify=df['{col}'], test_size=0.2, random_state=42)"
            )
        return CheckResult(
            "Class Imbalance", "WARN", "MEDIUM",
            f"Column(s) with dominant class ≥{imb_label}: {details}",
            "Severe class imbalance causes classifiers to predict the majority class "
            "exclusively. Accuracy becomes meaningless; use F1/AUC. Models fail to "
            "learn the minority class without resampling or cost-sensitive learning.",
            "\n\n".join(fix_lines)
        )

    # ── check 21 ──────────────────────────────────────────────────────────────
    def _check_suspicious_zeros(self) -> CheckResult:
        bad = []
        for col in self._num_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            # Skip binary 0/1 indicator columns — high zero fraction is by design.
            if set(s.unique()).issubset({0, 1, 0.0, 1.0}):
                continue
            frac = (s == 0).sum() / len(s)
            if frac >= self.cfg.zero_concentration_frac:
                bad.append((col, f"{frac*100:.1f}%"))
        zero_label = f"{self.cfg.zero_concentration_frac*100:.0f}%"
        if not bad:
            return CheckResult("Suspicious Zero Concentration", "PASS", "LOW",
                               f"No numeric columns with ≥{zero_label} zero values.")
        details = "; ".join(f"'{c}' ({p} zeros)" for c, p in bad)
        fix_lines = []
        for col, _ in bad:
            fix_lines.append(
                f"# If zeros represent missing data:\n"
                f"df['{col}'].replace(0, np.nan, inplace=True)\n\n"
                f"# If intentional sparse column, log1p-transform:\n"
                f"df['{col}_log'] = np.log1p(df['{col}'])\n\n"
                f"# Or add a binary presence indicator:\n"
                f"df['{col}_nonzero'] = (df['{col}'] != 0).astype(int)"
            )
        return CheckResult(
            "Suspicious Zero Concentration", "WARN", "LOW",
            f"Column(s) with ≥{zero_label} zero values: {details}",
            "High zero concentration may indicate zeros substituted for missing values, "
            "a sparse measurement process, or a data collection artefact. Zeros distort "
            "means, correlations, and ML feature distributions.",
            "import numpy as np\n\n" + "\n\n".join(fix_lines)
        )

    # ── check 22 ──────────────────────────────────────────────────────────────
    def _check_negative_non_negative(self) -> CheckResult:
        KEYWORDS = ["age", "price", "cost", "count", "amount", "qty", "quantity",
                    "salary", "revenue", "sales", "distance", "duration", "weight",
                    "height", "size", "length", "score", "rating"]
        explicit = set(self.cfg.non_negative_columns)
        bad = []
        for col in self._num_cols:
            col_l = col.lower()
            if col in explicit or any(kw in col_l for kw in KEYWORDS):
                n = int((self.df[col] < 0).sum())
                if n > 0:
                    bad.append((col, n))
        disclaimer = (
            "\n  Note: detection is heuristic — columns are matched by name keywords "
            "(age, price, count, ...).\n  A semantically non-negative column with an "
            "unrecognized name (e.g. 'rev', 'amt') is NOT auto-checked;\n  list it under "
            "'non_negative_columns' in your config to force the check."
        )
        if not bad:
            return CheckResult("Negative Values in Non-Negative Columns", "PASS", "HIGH",
                               "No suspicious negatives in semantically non-negative columns."
                               + disclaimer)
        details = "; ".join(f"'{c}' ({n} negative rows)" for c, n in bad) + disclaimer
        fix_lines = []
        for col, _ in bad:
            fix_lines.append(
                f"# Inspect negative rows:\nprint(df[df['{col}'] < 0][['{col}']])\n\n"
                f"# Replace with NaN:\ndf.loc[df['{col}'] < 0, '{col}'] = np.nan\n\n"
                f"# Or take absolute value if sign is a data entry error:\n"
                f"df['{col}'] = df['{col}'].abs()"
            )
        return CheckResult(
            "Negative Values in Non-Negative Columns", "WARN", "HIGH",
            f"Column(s) expected to be ≥0 contain negatives: {details}",
            "Columns like 'age', 'price', or 'count' should never be negative. "
            "Negative values almost always represent data entry errors, unit sign "
            "conventions, or encoding artefacts.",
            "import numpy as np\n\n" + "\n\n".join(fix_lines)
        )

    # ── check 23 ──────────────────────────────────────────────────────────────
    def _check_column_name_quality(self) -> CheckResult:
        issues = []
        seen = {}
        for col in self.df.columns:
            s = str(col)
            key = s.lower()
            if key in seen:
                issues.append(f"Duplicate header (case-insensitive): '{seen[key]}' and '{s}'")
            seen[key] = s
            if s != s.strip():
                issues.append(f"Whitespace in name: '{s}'")
            if re.search(r"[^a-zA-Z0-9_]", s):
                issues.append(f"Special characters in name: '{s}'")
        if not issues:
            return CheckResult("Column Name Quality", "PASS", "MEDIUM",
                               "All column names are clean.")
        fix = (
            "df.columns = (\n"
            "    df.columns\n"
            "    .str.strip()\n"
            "    .str.lower()\n"
            "    .str.replace(r'[^a-z0-9_]', '_', regex=True)\n"
            "    .str.replace(r'_+', '_', regex=True)\n"
            "    .str.strip('_')\n"
            ")"
        )
        return CheckResult(
            "Column Name Quality", "FAIL", "MEDIUM",
            f"Column name issues found: {'; '.join(issues)}",
            "Column names with spaces, special characters, or duplicates cause attribute "
            "access errors (df.my col), ambiguous references, and issues with SQL, "
            "scikit-learn pipelines, and feature stores.",
            fix
        )

    # ── check 24 ──────────────────────────────────────────────────────────────
    def _check_date_format_inconsistency(self) -> CheckResult:
        PATTERNS = [
            r"^\d{4}-\d{2}-\d{2}",
            r"^\d{2}/\d{2}/\d{4}",
            r"^\d{2}-\d{2}-\d{4}",
            r"^\d{2}\.\d{2}\.\d{4}",
            r"^\d{1,2}/\d{1,2}/\d{2}$",
            r"^\d{4}/\d{2}/\d{2}",
        ]
        sample_n = self.cfg.date_sample_size
        bad = []
        for col in self._obj_cols:
            s = self.df[col].dropna().astype(str)
            s = s.head(sample_n) if sample_n else s
            if len(s) < 5:
                continue
            if pd.to_datetime(s, infer_datetime_format=True, errors="coerce").notna().mean() < 0.5:
                continue
            matched = sum(1 for pat in PATTERNS if s.str.match(pat).any())
            if matched > 1:
                bad.append(col)
        if not bad:
            return CheckResult("Date Format Inconsistency", "PASS", "MEDIUM",
                               "No date columns with mixed format patterns.")
        fix = "\n".join(
            f"df['{c}'] = pd.to_datetime(df['{c}'], infer_datetime_format=True, errors='coerce')"
            for c in bad
        ) + "\n# Verify:\n" + "\n".join(
            f"print(df['{c}'].isna().sum(), 'values could not be parsed in {c}')"
            for c in bad
        )
        return CheckResult(
            "Date Format Inconsistency", "WARN", "MEDIUM",
            f"Column(s) with mixed date format patterns: {bad}",
            "Mixed date formats cause pd.to_datetime() to silently misparse values or "
            "return NaT. This breaks chronological ordering, time-delta calculations, "
            "and temporal feature extraction.",
            fix
        )

    # ── check 25 ──────────────────────────────────────────────────────────────
    def _check_memory_usage(self) -> CheckResult:
        mem_mb = self.df.memory_usage(deep=True).sum() / 1024 ** 2
        if mem_mb <= self.cfg.memory_warn_mb:
            return CheckResult("Memory Usage", "PASS", "LOW",
                               f"Dataset uses {mem_mb:.1f} MB — within acceptable range.")
        return CheckResult(
            "Memory Usage", "WARN", "LOW",
            f"Dataset uses {mem_mb:.1f} MB in memory.",
            "Large in-memory datasets can exhaust RAM and make repeated operations "
            "very slow. Downcasting dtypes is a fast, lossless win.",
            "# Downcast numeric columns:\n"
            "for col in df.select_dtypes('float64').columns:\n"
            "    df[col] = pd.to_numeric(df[col], downcast='float')\n"
            "for col in df.select_dtypes('int64').columns:\n"
            "    df[col] = pd.to_numeric(df[col], downcast='integer')\n\n"
            "# Convert low-cardinality object columns to category:\n"
            "for col in df.select_dtypes('object').columns:\n"
            "    if df[col].nunique() / len(df) < 0.5:\n"
            "        df[col] = df[col].astype('category')\n\n"
            "print(f'New usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')"
        )

    # ── orchestration ─────────────────────────────────────────────────────────
    def run_all_checks(self):
        checks = [
            self._check_missing_values,
            self._check_duplicate_rows,
            self._check_duplicate_columns,
            self._check_unnamed_index,
            self._check_mistyped_numeric,
            self._check_mistyped_datetime,
            self._check_mixed_types,
            self._check_constant_columns,
            self._check_near_constant,
            self._check_all_unique,
            self._check_whitespace,
            self._check_empty_strings,
            self._check_inconsistent_casing,
            self._check_outliers_iqr,
            self._check_outliers_zscore,
            self._check_skewness,
            self._check_high_correlation,
            self._check_high_cardinality_categorical,
            self._check_low_cardinality_numeric,
            self._check_class_imbalance,
            self._check_suspicious_zeros,
            self._check_negative_non_negative,
            self._check_column_name_quality,
            self._check_date_format_inconsistency,
            self._check_memory_usage,
        ]
        excluded = set(self.cfg.exclude_checks)
        for fn in checks:
            try:
                result = fn()
            except Exception as exc:
                name = fn.__name__.replace("_check_", "").replace("_", " ").title()
                result = CheckResult(name, "WARN", "LOW",
                                     f"Check could not complete: {exc}")
            if result.name in excluded:
                continue
            self.results.append(result)

    # ── report ────────────────────────────────────────────────────────────────
    def print_report(self):
        filename = os.path.basename(self.filepath)
        border = "═" * 68

        print(f"\n{BOLD}╔{border}╗{RESET}")
        title = f"DATA QUALITY REPORT — {filename}"
        print(f"{BOLD}║  {title:<66}║{RESET}")
        print(f"{BOLD}╚{border}╝{RESET}")

        mem_mb = self.df.memory_usage(deep=True).sum() / 1024 ** 2
        print(f"\n  Rows: {BOLD}{len(self.df):,}{RESET}   "
              f"Columns: {BOLD}{len(self.df.columns)}{RESET}   "
              f"Memory: {BOLD}{mem_mb:.2f} MB{RESET}\n")

        print(_section("AUTOMATED CHECKS"))

        for r in self.results:
            clr = STATUS_COLOUR.get(r.status, RESET)
            sev = f"  {DIM}[{r.severity}]{RESET}" if r.status != "PASS" else ""
            print(f"\n  {clr}{BOLD}[{r.status}]{RESET}  {BOLD}{r.name}{RESET}{sev}")
            if r.details:
                for line in r.details.splitlines():
                    print(f"        {line}")
            if r.explanation:
                print(f"\n        {BOLD}WHY:{RESET}")
                for line in r.explanation.splitlines():
                    print(f"        {line}")
            if r.fix_code:
                print(f"\n        {BOLD}FIX:{RESET}")
                print(_fmt_code(r.fix_code))

        n_pass = sum(1 for r in self.results if r.status == "PASS")
        n_warn = sum(1 for r in self.results if r.status == "WARN")
        n_fail = sum(1 for r in self.results if r.status == "FAIL")

        print(_section("SUMMARY"))
        print(f"\n  {GREEN}{BOLD}PASS: {n_pass}{RESET}   "
              f"{YELLOW}{BOLD}WARN: {n_warn}{RESET}   "
              f"{RED}{BOLD}FAIL: {n_fail}{RESET}")

        if n_fail > 0:
            print(f"\n  {RED}{BOLD}Overall: FAIL — resolve FAIL issues before analysis or modelling.{RESET}")
        elif n_warn > 0:
            print(f"\n  {YELLOW}{BOLD}Overall: WARN — review warnings; some may be acceptable for your use case.{RESET}")
        else:
            print(f"\n  {GREEN}{BOLD}Overall: PASS — no automated issues detected.{RESET}")

        self._print_manual_suggestions()
        print()

    # ── manual suggestions ────────────────────────────────────────────────────
    def _print_manual_suggestions(self):
        print(_section("MANUAL CHECKS — automated tests cannot detect these"))
        print(f"\n  {DIM}Review each item using domain knowledge. These issues are real "
              f"but invisible to code.{RESET}\n")

        for i, (title, problem, fix) in enumerate(MANUAL_CHECKS, 1):
            print(f"  {BOLD}{i:2}. {title}{RESET}")
            print(f"\n      {DIM}Problem:{RESET}")
            for line in problem.splitlines():
                print(f"        {line}")
            print(f"\n      {DIM}How to check / fix:{RESET}")
            for line in fix.splitlines():
                print(f"        {DIM}{line}{RESET}")
            print()


# ── manual-check catalogue (shared by terminal + Markdown + HTML renderers) ───
MANUAL_CHECKS = [
            (
                "Data Leakage",
                "Features may encode information about the target that would not be available "
                "at prediction time (e.g. a 'final_diagnosis' column when predicting early symptoms). "
                "This inflates model accuracy during evaluation but fails silently in production.",
                "Audit the timeline of every feature relative to the target event.\n"
                "Use time-aware splits:\n"
                "    from sklearn.model_selection import TimeSeriesSplit\n"
                "    tscv = TimeSeriesSplit(n_splits=5)"
            ),
            (
                "Temporal Ordering",
                "If data represents events over time, rows may not be sorted chronologically. "
                "Random train/test splits on unordered time data leak future information into training.",
                "Sort by timestamp before any split:\n"
                "    df.sort_values('date_col', inplace=True)\n"
                "    df.reset_index(drop=True, inplace=True)"
            ),
            (
                "Train/Test Preprocessing Contamination",
                "Fitting scalers, imputers, or encoders on the full dataset before splitting "
                "leaks test-set statistics into training. This produces optimistically biased "
                "evaluation scores that do not reflect real-world performance.",
                "Always fit preprocessing inside a pipeline, on training data only:\n"
                "    from sklearn.pipeline import Pipeline\n"
                "    from sklearn.preprocessing import StandardScaler\n"
                "    pipe = Pipeline([('scaler', StandardScaler()), ('model', your_model)])\n"
                "    pipe.fit(X_train, y_train)  # scaler sees only X_train"
            ),
            (
                "Unit Inconsistencies",
                "A numeric column may mix units (kg and lbs, USD and EUR, cm and inches) "
                "with no indicator column. Values look individually plausible but comparisons "
                "and aggregations are meaningless.",
                "Inspect value distributions across subgroups (country, source system, time period).\n"
                "Standardise to one unit:\n"
                "    # Example: rows from US source use lbs; convert to kg\n"
                "    df.loc[df['source'] == 'US', 'weight_kg'] = df['weight_lbs'] * 0.453592"
            ),
            (
                "Business Rule Violations",
                "Domain-specific logical constraints may be violated: end_date < start_date, "
                "age > 130, discount_rate > 1.0, delivered_qty > ordered_qty. These cannot be "
                "detected without knowing your business rules.",
                "Define and assert constraints explicitly:\n"
                "    assert (df['end_date'] >= df['start_date']).all(), 'Date order violated'\n"
                "    bad_rows = df[df['discount_rate'] > 1.0]\n"
                "    print(f'{len(bad_rows)} rows violate the discount constraint')"
            ),
            (
                "Survivorship Bias",
                "The dataset may only contain entities that survived a selection or observation "
                "process (e.g. only companies still operating, only patients who completed "
                "treatment). Conclusions drawn may not apply to the full population.",
                "Identify what is NOT in the dataset. If possible, obtain data on non-survivors "
                "and compare key distributions. State the selection criteria clearly in any "
                "analysis report or model card."
            ),
            (
                "Sampling Bias / Representativeness",
                "The sample may not represent the target population due to demographic skews, "
                "geographic concentration, or convenience sampling. A model that performs well "
                "in-sample can fail badly in production on underrepresented groups.",
                "Compare key variable distributions against known population benchmarks:\n"
                "    print(df['region'].value_counts(normalize=True))\n"
                "If biased, apply inverse-probability weighting:\n"
                "    df['weight'] = population_freq / sample_freq"
            ),
            (
                "Definition Drift / Concept Drift",
                "A column's definition may have changed over time — e.g. a 'revenue' KPI "
                "computed differently before/after a system migration. Models trained on "
                "mixed-definition data produce unreliable results.",
                "Plot feature distributions split by time period and look for discontinuities:\n"
                "    df.groupby(df['date'].dt.year)['revenue'].median().plot()\n"
                "    # Look for sudden step changes that indicate definition changes."
            ),
            (
                "Imputation Mechanism Mismatch",
                "The automated check uses Little's MCAR test and correlation analysis, but "
                "statistical tests have limited power on small datasets and CANNOT prove MNAR. "
                "MNAR missingness (e.g. high earners omitting income) looks identical to MCAR "
                "in the observed data. Always confirm the mechanism with a domain expert.",
                "MCAR → mean/median/mode or KNNImputer.\n"
                "MAR  → IterativeImputer (MICE) — fit on train fold only:\n"
                "    from sklearn.experimental import enable_iterative_imputer  # noqa\n"
                "    from sklearn.impute import IterativeImputer\n"
                "    imp = IterativeImputer(random_state=42)\n"
                "    X_train_imp = imp.fit_transform(X_train)\n"
                "    X_test_imp  = imp.transform(X_test)\n"
                "MNAR → model the selection/censoring process or collect more data."
            ),
            (
                "Proxy / Fairness Variables",
                "Columns that appear neutral may act as proxies for protected attributes "
                "(race, gender, age, religion, disability status). Including them in a model "
                "can lead to discriminatory outcomes that are ethically wrong and often illegal.",
                "Check correlations between candidate proxy columns and sensitive attributes:\n"
                "    # e.g. postcode → ethnicity, surname → national origin\n"
                "    for col in ['postcode', 'school_name', 'surname']:\n"
                "        if col in df.columns:\n"
                "            print(df.groupby(col)['outcome'].mean())\n"
                "Consult a legal/ethics team before deploying high-stakes models "
                "(credit, hiring, healthcare, criminal justice)."
            ),
]


# ── report renderers (shared summary helpers) ────────────────────────────────
def _status_counts(results: List[CheckResult]) -> Tuple[int, int, int]:
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    return n_pass, n_warn, n_fail


def _overall_verdict(n_fail: int, n_warn: int) -> Tuple[str, str]:
    if n_fail:
        return "FAIL", "resolve FAIL issues before analysis or modelling."
    if n_warn:
        return "WARN", "review warnings; some may be acceptable for your use case."
    return "PASS", "no automated issues detected."


def render_markdown(checker: "DataQualityChecker") -> str:
    df, results = checker.df, checker.results
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    n_pass, n_warn, n_fail = _status_counts(results)
    verdict, msg = _overall_verdict(n_fail, n_warn)
    badge = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL"}

    out: List[str] = [
        f"# Data Quality Report — {os.path.basename(checker.filepath)}",
        "",
        f"- **Rows:** {len(df):,}",
        f"- **Columns:** {len(df.columns)}",
        f"- **Memory:** {mem_mb:.2f} MB",
        "",
        "## Automated Checks",
    ]
    for r in results:
        sev = f"  _({r.severity})_" if r.status != "PASS" else ""
        out += ["", f"### {badge.get(r.status, r.status)} — {r.name}{sev}"]
        if r.details:
            out += ["", r.details.strip()]
        if r.explanation:
            out += ["", f"**Why:** {r.explanation.strip()}"]
        if r.fix_code:
            out += ["", "**Fix:**", "```python", r.fix_code.strip(), "```"]

    out += [
        "", "## Summary", "",
        "| Status | Count |", "|--------|-------|",
        f"| PASS | {n_pass} |", f"| WARN | {n_warn} |", f"| FAIL | {n_fail} |",
        "", f"**Overall: {verdict}** — {msg}",
        "", "## Manual Checks — automated tests cannot detect these", "",
        "_Review each item using domain knowledge. These issues are real but invisible to code._",
    ]
    for i, (title, problem, fix) in enumerate(MANUAL_CHECKS, 1):
        out += ["", f"### {i}. {title}", "", f"**Problem:** {problem}",
                "", "**How to check / fix:**", "```", fix, "```"]
    out.append("")
    return "\n".join(out)


def render_html(checker: "DataQualityChecker") -> str:
    df, results = checker.df, checker.results
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    n_pass, n_warn, n_fail = _status_counts(results)
    verdict, msg = _overall_verdict(n_fail, n_warn)
    esc = html.escape
    colors = {"PASS": "#1a7f37", "WARN": "#9a6700", "FAIL": "#cf222e"}
    title = esc(os.path.basename(checker.filepath))

    def pill(status: str) -> str:
        return (f'<span class="pill" style="background:{colors.get(status, "#555")}">'
                f'{esc(status)}</span>')

    p: List[str] = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Data Quality Report — {title}</title>",
        "<style>"
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "margin:0;background:#f6f8fa;color:#1f2328;}"
        ".wrap{max-width:960px;margin:0 auto;padding:32px 24px;}"
        "h1{font-size:1.6rem;}h2{margin-top:2rem;border-bottom:1px solid #d0d7de;padding-bottom:.3rem;}"
        ".meta{color:#57606a;margin:.5rem 0 1.5rem;}"
        ".check,.manual{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:14px 20px;margin:12px 0;}"
        ".manual{border-style:dashed;}"
        ".pill{color:#fff;font-size:.75rem;font-weight:700;border-radius:999px;padding:2px 10px;margin-right:8px;}"
        ".sev{color:#57606a;font-size:.8rem;}.label{font-weight:700;}.why{margin:.6rem 0;}"
        "pre{background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px;overflow:auto;"
        "font-family:SFMono-Regular,Consolas,monospace;font-size:.85rem;white-space:pre-wrap;}"
        "table{border-collapse:collapse;}td,th{border:1px solid #d0d7de;padding:6px 14px;text-align:left;}"
        ".verdict{font-weight:700;font-size:1.1rem;margin:1rem 0;}"
        "</style></head><body><div class='wrap'>",
        f"<h1>Data Quality Report — {title}</h1>",
        f"<div class='meta'>Rows: <b>{len(df):,}</b> &nbsp; Columns: <b>{len(df.columns)}</b>"
        f" &nbsp; Memory: <b>{mem_mb:.2f} MB</b></div>",
        "<h2>Automated Checks</h2>",
    ]
    for r in results:
        sev = f"<span class='sev'>[{esc(r.severity)}]</span>" if r.status != "PASS" else ""
        p.append("<div class='check'>")
        p.append(f"<div>{pill(r.status)}<b>{esc(r.name)}</b> {sev}</div>")
        if r.details:
            p.append(f"<pre>{esc(r.details.strip())}</pre>")
        if r.explanation:
            p.append(f"<div class='why'><span class='label'>Why:</span>"
                     f"<pre>{esc(r.explanation.strip())}</pre></div>")
        if r.fix_code:
            p.append(f"<div class='label'>Fix:</div><pre>{esc(r.fix_code.strip())}</pre>")
        p.append("</div>")

    p.append("<h2>Summary</h2>")
    p.append("<table><tr><th>PASS</th><th>WARN</th><th>FAIL</th></tr>"
             f"<tr><td>{n_pass}</td><td>{n_warn}</td><td>{n_fail}</td></tr></table>")
    p.append(f"<div class='verdict' style='color:{colors.get(verdict)}'>"
             f"Overall: {esc(verdict)} — {esc(msg)}</div>")
    p.append("<h2>Manual Checks — automated tests cannot detect these</h2>")
    p.append("<p class='meta'>Review each item using domain knowledge. "
             "These issues are real but invisible to code.</p>")
    for i, (mtitle, problem, fix) in enumerate(MANUAL_CHECKS, 1):
        p.append("<div class='manual'>")
        p.append(f"<div class='label'>{i}. {esc(mtitle)}</div>")
        p.append(f"<div class='why'><span class='label'>Problem:</span> {esc(problem)}</div>")
        p.append(f"<div class='label'>How to check / fix:</div><pre>{esc(fix)}</pre>")
        p.append("</div>")
    p.append("</div></body></html>")
    return "\n".join(p)


# ── multi-format loader ───────────────────────────────────────────────────────
def load_dataframe(filepath: str, cfg: Config) -> pd.DataFrame:
    """Load a dataset, dispatching on file extension. Optional engines
    (openpyxl, pyarrow) raise ImportError if not installed — handled by main()."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        return pd.read_csv(filepath, sep=cfg.delimiter) if cfg.delimiter else pd.read_csv(filepath)
    if ext in (".tsv", ".txt"):
        if cfg.delimiter:
            return pd.read_csv(filepath, sep=cfg.delimiter)
        # auto-detect delimiter
        return pd.read_csv(filepath, sep=None, engine="python")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath, sheet_name=cfg.excel_sheet)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(filepath)
    if ext == ".json":
        return pd.read_json(filepath)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(filepath, lines=True)

    print(f"{YELLOW}Warning: unrecognized extension '{ext}'. Attempting CSV read...{RESET}")
    return pd.read_csv(filepath)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    # Ensure UTF-8 output on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Automated dataset quality checker — 25 checks + manual-review guide.")
    parser.add_argument(
        "datafile", nargs="?",
        help="Path to the dataset (.csv, .tsv/.txt, .xlsx/.xls, .parquet, .json, .jsonl).")
    parser.add_argument(
        "-c", "--config", default=None,
        help="Path to a YAML/JSON config of thresholds, exclusions, and report outputs.")
    args = parser.parse_args()

    if not args.datafile:
        parser.print_help()
        sys.exit(1)

    filepath = args.datafile
    if not os.path.exists(filepath):
        print(f"\n{RED}Error: File not found — '{filepath}'{RESET}\n")
        sys.exit(1)

    # ── load configuration (defaults reproduce historical behaviour) ──────────
    cfg = Config()
    if args.config:
        if not os.path.exists(args.config):
            print(f"\n{RED}Error: Config file not found — '{args.config}'{RESET}\n")
            sys.exit(1)
        try:
            cfg = Config.from_file(args.config)
        except Exception as exc:
            print(f"\n{RED}Error reading config '{args.config}': {exc}{RESET}\n")
            sys.exit(1)

    # ── load dataset (format auto-detected by extension) ──────────────────────
    print(f"\n{DIM}Loading '{filepath}'...{RESET}")
    try:
        df = load_dataframe(filepath, cfg)
    except ImportError as exc:
        print(f"\n{RED}Missing dependency for this file type: {exc}{RESET}")
        print(f"{DIM}Install all engines with: pip install -r requirements.txt{RESET}\n")
        sys.exit(1)
    except Exception as exc:
        print(f"\n{RED}Error reading data file: {exc}{RESET}\n")
        sys.exit(1)

    if isinstance(df, dict):  # multi-sheet Excel
        first = next(iter(df))
        print(f"{YELLOW}Workbook has multiple sheets; using '{first}'. "
              f"Set input.excel_sheet in config to choose another.{RESET}")
        df = df[first]

    if df.empty:
        print(f"\n{RED}Error: The dataset is empty.{RESET}\n")
        sys.exit(1)

    print(f"{DIM}Running 25 quality checks...{RESET}")
    checker = DataQualityChecker(df, filepath, cfg)
    checker.run_all_checks()
    checker.print_report()

    # ── optional report exports ───────────────────────────────────────────────
    if cfg.report_markdown:
        try:
            with open(cfg.report_markdown, "w", encoding="utf-8") as fh:
                fh.write(render_markdown(checker))
            print(f"  {GREEN}Markdown report written to '{cfg.report_markdown}'{RESET}")
        except Exception as exc:
            print(f"  {RED}Could not write Markdown report: {exc}{RESET}")
    if cfg.report_html:
        try:
            with open(cfg.report_html, "w", encoding="utf-8") as fh:
                fh.write(render_html(checker))
            print(f"  {GREEN}HTML report written to '{cfg.report_html}'{RESET}")
        except Exception as exc:
            print(f"  {RED}Could not write HTML report: {exc}{RESET}")
    if cfg.report_markdown or cfg.report_html:
        print()


if __name__ == "__main__":
    main()
