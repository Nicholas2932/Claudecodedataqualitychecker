"""
generate_test_datasets.py — Produces 4 synthetic CSV files to stress-test
data_quality_checker.py:
  1. test_clean.csv   — fully clean dataset (all checks should PASS)
  2. test_mild.csv    — a handful of common, low-severity issues
  3. test_dirty.csv   — many issues (rebuilt, broader coverage)
  4. test_filthy.csv  — every detectable issue triggered at least once
"""

import numpy as np
import pandas as pd

np.random.seed(0)


# ── 1. CLEAN ──────────────────────────────────────────────────────────────────
def make_clean(n: int = 500) -> pd.DataFrame:
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age_years": np.random.randint(18, 75, n),
        "income_usd": np.random.normal(60000, 15000, n).round(2),
        "region": np.random.choice(["north", "south", "east", "west"], n),
        "is_member": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "satisfaction": np.random.choice(["low", "medium", "high"], n, p=[0.3, 0.4, 0.3]),
    })
    return df


# ── 2. MILD ───────────────────────────────────────────────────────────────────
def make_mild(n: int = 500) -> pd.DataFrame:
    df = make_clean(n)

    # add 2% missing in income (MCAR-like)
    mask = np.random.choice(n, int(0.02 * n), replace=False)
    df.loc[mask, "income_usd"] = np.nan

    # whitespace in a few region values
    idx = np.random.choice(n, 5, replace=False)
    df.loc[idx, "region"] = df.loc[idx, "region"].astype(str) + "  "

    # mild casing inconsistency
    idx = np.random.choice(n, 3, replace=False)
    df.loc[idx, "satisfaction"] = df.loc[idx, "satisfaction"].str.upper()

    # one mild outlier in income
    df.loc[0, "income_usd"] = 1_000_000

    return df


# ── 3. DIRTY ──────────────────────────────────────────────────────────────────
def make_dirty(n: int = 500) -> pd.DataFrame:
    df = make_clean(n)

    # MAR-like: income missing more often for younger customers
    young = df["age_years"] < 30
    young_idx = df[young].index
    drop = np.random.choice(young_idx, min(40, len(young_idx)), replace=False)
    df.loc[drop, "income_usd"] = np.nan

    # column that should be numeric but is stored as string; a stray
    # non-numeric value keeps pandas from auto-converting on CSV read.
    df["bonus_pct"] = (np.random.rand(len(df)) * 10).round(2).astype(str)
    df.loc[0, "bonus_pct"] = "pending"

    # constant column
    df["data_source"] = "system_A"

    # negative ages
    df.loc[df.sample(5, random_state=1).index, "age_years"] = -1

    # whitespace + inconsistent casing in region
    idx = df.sample(20, random_state=2).index
    df.loc[idx, "region"] = " NORTH "

    # outliers in income (substantial — should trip both IQR and z-score)
    out_idx = df.sample(15, random_state=3).index
    df.loc[out_idx, "income_usd"] = 5_000_000

    # duplicates MUST be last so later column additions don't break them
    df = pd.concat([df, df.iloc[:8]], ignore_index=True)

    return df


# ── 4. FILTHY — every issue triggered ─────────────────────────────────────────
def make_filthy(n: int = 300) -> pd.DataFrame:
    rng = np.random.RandomState(42)

    df = pd.DataFrame({
        "Unnamed: 0": range(n),                                              # check 4
        "age": rng.randint(20, 70, n).astype(float),                         # check 22 (negs added)
        "Age": rng.randint(20, 70, n).astype(float),                         # check 23 (dup name diff case)
        "price": rng.exponential(100, n).round(2),                           # check 16 (skew), 22 (negs)
        "salary ": rng.normal(50000, 10000, n),                              # check 23 (trailing space)
        "income_usd": rng.normal(60000, 15000, n),                           # missing val target
        "gender": rng.choice(["Male", "male", "FEMALE", "Female"], n),       # check 13 inconsistent casing
        "notes": ["  text  " if i % 3 == 0 else "ok" for i in range(n)],     # check 11 whitespace
        # Note: pandas auto-converts "" → NaN on CSV read. Use a non-empty
        # placeholder that survives roundtrip but is semantically empty.
        "label": ["   " if i % 25 == 0 else "yes" for i in range(n)],        # check 11 whitespace, near-constant
        "constant_col": "X",                                                 # check 8 constant
        "near_const": ["A"] * (n - 3) + ["B", "C", "D"],                     # check 9 near-constant
        "id_string": [f"user_{i}" for i in range(n)],                        # check 10/18 all-unique
        "rating_1_5": rng.randint(1, 6, n),                                  # check 19 low-card numeric
        "mostly_zero": [0] * (n - 30) + list(rng.randn(30)),                 # check 21 suspicious zeros
        # Use "unknown" (not "N/A") so pandas does NOT auto-convert it to NaN
        # on read — this is what makes the mixed-types check fire.
        "mixed_col": [str(i) if i % 5 != 0 else "unknown" for i in range(n)],  # check 7 mixed types
        # Add a stray non-numeric value so pandas keeps the column as object,
        # otherwise read_csv will auto-convert it back to float64.
        "numeric_as_str": [str(rng.randn()) if i != 0 else "n_a"
                           for i in range(n)],                                # check 5 mistyped numeric
        "date_str_mixed": (["2023-01-15"] * (n // 2)
                           + ["15/01/2023"] * (n - n // 2)),                 # check 24 date format mix
        "date_iso": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),  # check 6 mistyped datetime
        "highly_corr_1": rng.randn(n),                                       # check 17 high correlation
        "imbalanced": ["A"] * (n - 10) + ["B"] * 10,                         # check 20 class imbalance
        "bad name!": rng.randn(n),                                           # check 23 special chars
    })
    # make a near-duplicate column for check 17
    df["highly_corr_2"] = df["highly_corr_1"] * 1.0001 + 1e-6 * rng.randn(n)

    # ── inject issues ────────────────────────────────────────────────────────
    # Missing values (some > 20% to trigger FAIL)
    df.loc[rng.choice(n, 80, replace=False), "income_usd"] = np.nan          # 26.6% missing
    df.loc[rng.choice(n, 15, replace=False), "age"] = np.nan                 # 5% missing

    # negative values in non-negative columns
    df.loc[rng.choice(n, 4, replace=False), "age"] = -5
    df.loc[rng.choice(n, 3, replace=False), "price"] = -50

    # extreme outliers (>2% rate to trip the FAIL threshold)
    df.loc[rng.choice(n, 12, replace=False), "price"] = 100_000              # ~4% of values
    df.loc[rng.choice(n, 12, replace=False), "income_usd"] = 10_000_000

    # duplicate column (must come BEFORE row-duplication so the row
    # duplicates also share the value)
    df["salary_copy"] = df["salary "]                                        # check 3 duplicate columns

    # exact duplicate rows — must be the LAST step
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    return df


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = {
        "test_clean.csv":  make_clean(),
        "test_mild.csv":   make_mild(),
        "test_dirty.csv":  make_dirty(),
        "test_filthy.csv": make_filthy(),
    }
    for fname, df in datasets.items():
        df.to_csv(fname, index=False)
        print(f"{fname:20s}  rows={len(df):4d}  cols={df.shape[1]:2d}")
