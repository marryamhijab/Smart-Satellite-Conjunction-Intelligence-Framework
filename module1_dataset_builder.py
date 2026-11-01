#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smart Satellite Conjunction Intelligence Framework
Module 1 : Early Collision Risk Prediction  --  dataset preparation

What this script does (train_data.csv ONLY; test files are never touched):
  1. Inspects the local train_data.csv (columns, dtypes, events, time_to_tca, risk, NaNs).
  2. Applies the Module 1 eligibility rules per event.
  3. Writes   module1_eligible_dataset.csv   (original rows + role/leakage-safe metadata).
  4. Splits ELIGIBLE EVENTS 90/10 (stratified by target-risk category, fixed seed).
  5. Writes   module1_train_90.csv , module1_validation_10.csv
  6. Writes reports + runs assertions.  The original file is never modified (SHA-256 checked).

Usage (Windows):
    python module1_dataset_builder.py
    python module1_dataset_builder.py --input "C:\\path\\to\\train_data.csv"
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_INPUT = (
    r"C:\Users\User\Downloads\Satellite Conjunction Framework"
    r"\Collision Avoidance Challenge - Dataset"
    r"\Collision Avoidance Challenge - Dataset"
    r"\kelvins_competition_data\train_data"
)
OUTPUT_DIR_NAME = "module1_dataset_outputs"
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.10
MIN_EARLY_CDMS = 3          # need >= 3 CDMs with time_to_tca >= EARLY_THRESHOLD
EARLY_THRESHOLD = 2.0       # days: only CDMs at/above this may be model input
TARGET_THRESHOLD = 1.0      # days: target CDM must be strictly below this
MIN_EVENTS_PER_CATEGORY = 2  # needed for a stratified split

REQUIRED_COLS = ["event_id", "time_to_tca", "risk"]

# Risk categories. 'risk' in the Kelvins data is expected to be log10(Pc);
# -6 is the usual operational threshold (Pc = 1e-6). This is VERIFIED by the
# script only as far as the data allows (see report); docs must confirm.
RISK_CATEGORY_ORDER = [
    "C0_risk_eq_-30",
    "C1_risk_lt_-10",
    "C2_-10_to_-6",
    "C3_-6_to_-5",
    "C4_ge_-5",
]

# Columns that, in the Kelvins dataset, are suspected to carry whole-event
# (possibly future) information. The script cannot verify this -> it WARNS.
SUSPECT_LEAKAGE_COLS = ["max_risk_estimate", "max_risk_scaling"]


def log(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Path handling / documentation discovery
# --------------------------------------------------------------------------- #
def clean_path(raw: str) -> Path:
    """Strip stray quotes / brackets / spaces that come from copy-pasting paths."""
    s = str(raw).strip()
    while s and s[0] in "[]\"' ":
        s = s[1:]
    while s and s[-1] in "[]\"' ":
        s = s[:-1]
    return Path(s)


def locate_csv(path: Path) -> Path:
    """Accept a file path OR a folder (e.g. .../train_data) containing train_data.csv."""
    if path.is_file():
        return path
    if not path.exists():
        alt = path.with_name(path.name + ".csv")
        if alt.is_file():
            return alt
        raise FileNotFoundError(f"Path does not exist: {path}")
    candidates = [p for p in path.rglob("*.csv") if p.name.lower() == "train_data.csv"]
    if not candidates:
        candidates = list(path.rglob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No CSV file found inside: {path}")
    names = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Several CSV files found; pass the exact file with --input:\n  " + names
    )


def find_local_documentation(csv_path: Path):
    """List doc-like files near the CSV and grep txt/md files for relevant lines."""
    exts = {".pdf", ".txt", ".md", ".docx", ".html", ".htm"}
    seen, docs = set(), []
    folders = [csv_path.parent] + list(csv_path.parents)[1:4]
    for folder in folders:
        try:
            for p in folder.iterdir():
                if p.is_file() and p.suffix.lower() in exts and p not in seen:
                    seen.add(p)
                    docs.append(p)
        except OSError:
            continue
    hits = []
    keywords = ("risk", "time_to_tca", "-30", "event_id", "tca")
    for p in docs:
        if p.suffix.lower() in {".txt", ".md"}:
            try:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if any(k in line.lower() for k in keywords):
                        hits.append(f"[{p.name}] {line.strip()[:200]}")
                        if len(hits) >= 40:
                            break
            except OSError:
                pass
    return docs, hits


def sha256_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Loading + inspection
# --------------------------------------------------------------------------- #
def load_dataset(csv_path: Path) -> pd.DataFrame:
    log(f"[1/7] Reading: {csv_path}")
    df = pd.read_csv(csv_path)  # default parsing: values kept as in the file
    log(f"      -> {df.shape[0]:,} rows x {df.shape[1]} columns")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            "CRITICAL: required column(s) not found: "
            f"{missing}\nColumns present: {list(df.columns)}\n"
            "Not guessing replacements. Fix the column names or update REQUIRED_COLS."
        )
    for c in ("time_to_tca", "risk"):
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise SystemExit(f"CRITICAL: column '{c}' is not numeric (dtype={df[c].dtype}).")
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Collect inspection facts (nothing is assumed; all numbers come from the file)."""
    info = {}
    info["n_rows"], info["n_cols"] = df.shape
    info["n_events_raw"] = int(df["event_id"].nunique(dropna=True))
    info["n_nan_event_id"] = int(df["event_id"].isna().sum())

    t = df["time_to_tca"]
    info["t_min"], info["t_max"] = float(t.min()), float(t.max())
    info["t_nan"] = int(t.isna().sum())
    info["t_negative"] = int((t < 0).sum())
    # Unit sanity check: Kelvins time_to_tca is in DAYS (max ~ 7). Refuse otherwise.
    if info["t_max"] > 30:
        raise SystemExit(
            f"CRITICAL: time_to_tca max = {info['t_max']:.2f}. This does not look like DAYS "
            "(expected roughly 0-7). Refusing to apply 2-day / 1-day thresholds on an "
            "unverified unit. Check the dataset documentation."
        )

    r = df["risk"]
    info["risk_min"], info["risk_max"] = float(r.min()), float(r.max())
    info["risk_nan"] = int(r.isna().sum())
    info["risk_eq_m30_rows"] = int(np.isclose(r, -30.0).sum())
    info["risk_ge_m6_rows"] = int((r >= -6).sum())
    info["risk_gt_0_rows"] = int((r > 0).sum())

    # Is there any CDM-id / creation-timestamp column that could define order?
    order_like = [c for c in df.columns
                  if any(k in c.lower() for k in ("cdm_id", "creation", "created", "date", "timestamp"))]
    info["order_like_cols"] = order_like

    # Is the file already in chronological order inside each event?
    diff = df.groupby("event_id", sort=False)["time_to_tca"].diff()
    info["events_file_order_not_desc"] = int(
        df.loc[diff > 0, "event_id"].nunique()
    )
    info["rows_equal_time_to_prev"] = int((diff == 0).sum())

    info["suspect_cols_present"] = [c for c in SUSPECT_LEAKAGE_COLS if c in df.columns]
    return info


# --------------------------------------------------------------------------- #
# Event-level eligibility
# --------------------------------------------------------------------------- #
def order_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chronological order inside an event.

    VERIFIED RULE: time_to_tca = time remaining until TCA, so a LARGER value is an
    EARLIER CDM. Sort by time_to_tca DESCENDING; equal values keep original file order
    (stable). No CDM-id/creation-date column is used unless the report shows one exists.
    """
    df = df.copy()
    df["_orig_row"] = np.arange(len(df))
    df = df.sort_values(
        ["event_id", "time_to_tca", "_orig_row"],
        ascending=[True, False, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    df["cdm_seq"] = df.groupby("event_id").cumcount() + 1
    return df


def build_event_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per event with eligibility facts + exclusion reasons."""
    t = df["time_to_tca"]
    df["_is_early"] = t >= EARLY_THRESHOLD
    df["_is_mid"] = (t >= TARGET_THRESHOLD) & (t < EARLY_THRESHOLD)
    df["_t_nan"] = t.isna()
    g = df.groupby("event_id", sort=True)
    last = g.tail(1).set_index("event_id")  # df is sorted -> last row = final chronological CDM

    ev = pd.DataFrame(index=last.index)
    ev["n_cdms"] = g.size()
    ev["n_early_cdms"] = g["_is_early"].sum().astype(int)
    ev["n_mid_cdms_1to2d"] = g["_is_mid"].sum().astype(int)
    ev["has_nan_time"] = g["_t_nan"].any()
    ev["target_time_to_tca"] = last["time_to_tca"]
    ev["target_risk"] = last["risk"]
    ev["target_orig_row"] = last["_orig_row"]
    min_t = g["time_to_tca"].transform("min")
    ev["n_rows_tied_at_min_time"] = (df["time_to_tca"] == min_t).groupby(df["event_id"]).sum().astype(int)

    reasons = []
    for row in ev.itertuples():
        rs = []
        if row.has_nan_time:
            rs.append("missing_time_to_tca")
        if row.n_early_cdms < MIN_EARLY_CDMS:
            rs.append(f"fewer_than_{MIN_EARLY_CDMS}_early_cdms(>=2d)")
        if not (row.target_time_to_tca < TARGET_THRESHOLD):
            rs.append("final_cdm_time_to_tca_not_below_1d")
        if pd.isna(row.target_risk):
            rs.append("target_risk_missing")
        reasons.append(";".join(rs))
    ev["exclusion_reasons"] = reasons
    ev["eligible"] = ev["exclusion_reasons"] == ""
    return ev


def assign_roles(df: pd.DataFrame, eligible_ids) -> pd.DataFrame:
    """Keep eligible events' rows only and label every CDM row with a role."""
    out = df[df["event_id"].isin(eligible_ids)].copy()
    t = out["time_to_tca"]
    role = np.select(
        [t >= EARLY_THRESHOLD, t >= TARGET_THRESHOLD],
        ["INPUT_EARLY", "HISTORICAL_NOT_INPUT"],
        default="LATE_NON_TARGET",
    ).astype(object)
    is_last = out.groupby("event_id").cumcount(ascending=False).to_numpy() == 0
    role[is_last] = "TARGET"
    out["cdm_role"] = role
    out["is_model_input"] = out["cdm_role"] == "INPUT_EARLY"
    return out


# --------------------------------------------------------------------------- #
# Risk categories + stratified split
# --------------------------------------------------------------------------- #
def risk_category(risk: pd.Series) -> pd.Series:
    conds = [
        np.isclose(risk, -30.0),
        risk < -10,
        (risk >= -10) & (risk < -6),
        (risk >= -6) & (risk < -5),
        risk >= -5,
    ]
    return pd.Series(np.select(conds, RISK_CATEGORY_ORDER, default="UNASSIGNED"),
                     index=risk.index)


def merge_rare_categories(cats: pd.Series, notes: list) -> pd.Series:
    """Merge neighbouring categories until each has >= MIN_EVENTS_PER_CATEGORY events."""
    groups = [[c] for c in RISK_CATEGORY_ORDER if (cats == c).any()]

    def count(grp):
        return int(cats.isin(grp).sum())

    while len(groups) > 1:
        counts = [count(g) for g in groups]
        i = int(np.argmin(counts))
        if counts[i] >= MIN_EVENTS_PER_CATEGORY:
            break
        nb = i - 1 if i == len(groups) - 1 else (i + 1 if i == 0 else
                                                  (i - 1 if counts[i - 1] <= counts[i + 1] else i + 1))
        lo, hi = sorted((i, nb))
        notes.append(f"Merged rare category {groups[i]} (n={counts[i]}) with {groups[nb]} for stratification.")
        groups[lo] = groups[lo] + groups[hi]
        del groups[hi]
    mapping = {c: "+".join(g) for g in groups for c in g}
    return cats.map(mapping)


def split_events(events: pd.DataFrame, notes: list):
    ids = events.index.to_numpy()
    strat = events["strat_label"].to_numpy()
    train_ids, val_ids = train_test_split(
        ids, test_size=VALIDATION_FRACTION, random_state=RANDOM_SEED,
        shuffle=True, stratify=strat,
    )
    return set(train_ids.tolist()), set(val_ids.tolist())


# --------------------------------------------------------------------------- #
# Validation (assertions)
# --------------------------------------------------------------------------- #
def run_validations(csv_path, original_hash_before, eligible, train, val, events_elig, results: list):
    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        assert cond, f"VALIDATION FAILED: {name} {detail}"

    # original file untouched
    check("Original file unchanged (SHA-256)", sha256_of(csv_path) == original_hash_before)

    inp = eligible[eligible["cdm_role"] == "INPUT_EARLY"]
    tgt = eligible[eligible["cdm_role"] == "TARGET"]
    ids = set(events_elig.index)

    check("Every eligible event has exactly one TARGET row",
          tgt["event_id"].nunique() == len(ids) and len(tgt) == len(ids))
    check("Every TARGET has time_to_tca < 1 day", (tgt["time_to_tca"] < TARGET_THRESHOLD).all())
    check("Every INPUT row has time_to_tca >= 2 days", (inp["time_to_tca"] >= EARLY_THRESHOLD).all())
    n_in = inp.groupby("event_id").size()
    check("Every eligible event has >= 3 INPUT CDMs",
          set(n_in.index) == ids and (n_in >= MIN_EARLY_CDMS).all())
    check("is_model_input is True only for INPUT_EARLY",
          (eligible["is_model_input"] == (eligible["cdm_role"] == "INPUT_EARLY")).all())
    check("TARGET / mid-window / late rows are never model input",
          (~eligible.loc[eligible["cdm_role"] != "INPUT_EARLY", "is_model_input"]).all())
    last_in = inp.groupby("event_id")["cdm_seq"].max()
    tgt_seq = tgt.set_index("event_id")["cdm_seq"]
    check("Every INPUT CDM is chronologically before its TARGET",
          (last_in.reindex(tgt_seq.index) < tgt_seq).all())
    check("TARGET is the chronologically last CDM of each event",
          (tgt_seq == eligible.groupby("event_id")["cdm_seq"].max().reindex(tgt_seq.index)).all())
    check("risk_category present only on TARGET rows",
          eligible.loc[eligible["cdm_role"] != "TARGET", "risk_category"].isna().all()
          and eligible.loc[eligible["cdm_role"] == "TARGET", "risk_category"].notna().all())

    # original values preserved
    orig_risk = pd.read_csv(csv_path, usecols=["risk"])["risk"].to_numpy()
    same = np.array_equal(orig_risk[eligible["orig_row_number"].to_numpy()],
                          eligible["risk"].to_numpy(), equal_nan=True)
    check("risk values identical to original file (via orig_row_number)", same)

    # split checks
    tr_ev, va_ev = set(train["event_id"]), set(val["event_id"])
    check("No event appears in both train and validation", tr_ev.isdisjoint(va_ev))
    check("Train + validation events == all eligible events", (tr_ev | va_ev) == ids)
    check("Train + validation rows == eligible rows", len(train) + len(val) == len(eligible))
    frac = len(va_ev) / len(ids)
    check("Validation share of events ~ 10% (+-1pp)", abs(frac - VALIDATION_FRACTION) <= 0.01,
          f"(actual {frac:.4%})")
    ttr = train[train["cdm_role"] == "TARGET"]["risk_category"].value_counts(normalize=True)
    tva = val[val["cdm_role"] == "TARGET"]["risk_category"].value_counts(normalize=True)
    cats = sorted(set(ttr.index) | set(tva.index))
    worst = max(abs(ttr.get(c, 0) - tva.get(c, 0)) for c in cats)
    check("Risk-category proportions train vs validation within 5pp", worst <= 0.05,
          f"(max diff {worst:.3%})")


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def pct(a, b):
    return f"{(100.0 * a / b):.2f}%" if b else "n/a"


def write_inspection_report(path, csv_path, df_raw_cols, dtypes, info, ev, elig_ev, docs, doc_hits,
                            missing_tbl, notes, hash_before, early_dist, all_rows_count):
    L = []
    A = L.append
    A("=" * 78)
    A("MODULE 1 - DATASET INSPECTION & ELIGIBILITY REPORT")
    A("=" * 78)
    A(f"Source file : {csv_path}")
    A(f"SHA-256     : {hash_before}  (verified unchanged at the end)")
    A("")
    A("1. ORIGINAL DATASET")
    A(f"   Rows (CDM records) : {info['n_rows']:,}")
    A(f"   Columns            : {info['n_cols']}")
    A(f"   Unique event_id    : {info['n_events_raw']:,}  (rows with NaN event_id: {info['n_nan_event_id']})")
    A("")
    A("   Column / dtype / non-null / missing")
    for c, d, nn, ms in dtypes:
        A(f"   - {c:<40} {str(d):<10} non-null={nn:<8} missing={ms}")
    A("")
    A("2. TIME_TO_TCA VERIFICATION")
    A(f"   min={info['t_min']:.6f}  max={info['t_max']:.6f}  NaN={info['t_nan']}  negative={info['t_negative']}")
    A("   Unit check: max is within a plausible day-range -> treated as DAYS (documentation should still be confirmed).")
    A(f"   Events whose file order is NOT descending time_to_tca: {info['events_file_order_not_desc']}")
    A(f"   Rows with time_to_tca equal to previous row in same event: {info['rows_equal_time_to_prev']}")
    A(f"   CDM-id / creation-date like columns found: {info['order_like_cols'] if info['order_like_cols'] else 'NONE'}")
    A("   Ordering rule used: sort by time_to_tca DESCENDING (larger = earlier CDM); ties keep file order.")
    A("   !! If a real CDM creation timestamp/id existed it would be preferable; none was detected above"
      if not info["order_like_cols"] else
      "   !! Candidate ordering columns exist (listed above) - review them manually.")
    A("")
    A("3. RISK COLUMN")
    A(f"   min={info['risk_min']:.4f}  max={info['risk_max']:.4f}  NaN={info['risk_nan']}")
    A(f"   rows with risk == -30 : {info['risk_eq_m30_rows']:,} ({pct(info['risk_eq_m30_rows'], info['n_rows'])})")
    A(f"   rows with risk >= -6  : {info['risk_ge_m6_rows']:,}")
    A(f"   rows with risk > 0    : {info['risk_gt_0_rows']:,}  (log10 scale would make these impossible - check)")
    A("   Interpretation: 'risk' is treated as log10(collision probability); -30 looks like a floor/sentinel")
    A("   for ~zero probability. It is NOT dropped and NOT treated as missing (no documentation proves otherwise).")
    A("   It is placed in its own category (C0) so the split handles it explicitly.")
    A("")
    A("4. LOCAL DOCUMENTATION")
    if docs:
        for d in docs:
            A(f"   - {d}")
        A("   (PDF/DOCX files are NOT parsed by this script: please confirm units and the meaning of")
        A("    risk = -30 manually.)")
    else:
        A("   No documentation file found near the CSV -> definitions above are UNVERIFIED by docs.")
    for h in doc_hits:
        A(f"   > {h}")
    A("")
    A("5. LEAKAGE WARNING (columns that may hold whole-event information)")
    if info["suspect_cols_present"]:
        for c in info["suspect_cols_present"]:
            A(f"   - '{c}' exists. Its exact definition cannot be verified here; if it is computed from")
            A("     the full event (incl. future CDMs) it MUST NOT be used as a model input.")
            A("     It was kept in the files unchanged; exclude it at training time unless docs prove it is safe.")
    else:
        A("   None of the known suspect columns are present.")
    A("")
    A("6. ELIGIBILITY RULES APPLIED")
    A(f"   - >= {MIN_EARLY_CDMS} CDMs with time_to_tca >= {EARLY_THRESHOLD} day  (model input: INPUT_EARLY)")
    A(f"   - Target = chronologically LAST CDM of the event; its time_to_tca must be < {TARGET_THRESHOLD} day")
    A("   - CDMs with 1 <= time_to_tca < 2 are kept as HISTORICAL_NOT_INPUT (never model input)")
    A("   - Rows with time_to_tca < 1 that are not the final CDM -> LATE_NON_TARGET (never model input)")
    A("   - Events with missing time_to_tca or missing target risk are excluded")
    A("")
    n_ev = len(ev)
    n_el = int(ev["eligible"].sum())
    A("7. RESULTS")
    A(f"   Total events analysed : {n_ev:,}")
    A(f"   Eligible events       : {n_el:,} ({pct(n_el, n_ev)})")
    A(f"   Excluded events       : {n_ev - n_el:,} ({pct(n_ev - n_el, n_ev)})")
    A(f"   CDM rows in eligible dataset : {all_rows_count:,}")
    A("")
    A("   Exclusion reasons (an event may have more than one):")
    excl = ev[~ev["eligible"]]
    single = {}
    for s in excl["exclusion_reasons"]:
        for r in s.split(";"):
            single[r] = single.get(r, 0) + 1
    for r, c in sorted(single.items(), key=lambda x: -x[1]):
        A(f"   - {r:<45} {c:,}")
    A("   Exclusion reason combinations:")
    for r, c in excl["exclusion_reasons"].value_counts().items():
        A(f"   - {r:<70} {c:,}")
    A("")
    A("8. EARLY CDMs (time_to_tca >= 2d) PER ELIGIBLE EVENT")
    A(f"   min={int(elig_ev['n_early_cdms'].min())}  median={elig_ev['n_early_cdms'].median():.1f}  "
      f"mean={elig_ev['n_early_cdms'].mean():.2f}  max={int(elig_ev['n_early_cdms'].max())}")
    for k, v in early_dist.items():
        A(f"   {int(k):>3} early CDMs : {v:,} events")
    A("")
    A("9. TARGET CDM SELECTION")
    A("   Target = last CDM by time_to_tca (smallest value) within the event.")
    A(f"   Eligible events whose smallest time_to_tca is tied by >1 row: {int((elig_ev['n_rows_tied_at_min_time'] > 1).sum())}"
      "  (the LAST of the tied rows in the file is used)")
    A(f"   Target time_to_tca (eligible): min={elig_ev['target_time_to_tca'].min():.4f} "
      f"max={elig_ev['target_time_to_tca'].max():.4f}")
    A("")
    A("10. MISSING VALUES (columns with at least one missing value)")
    if missing_tbl:
        for c, n, p in missing_tbl:
            A(f"   - {c:<40} {n:,} ({p:.3f}%)")
    else:
        A("   No missing values in the original dataset.")
    A("")
    A("11. TARGET-RISK DISTRIBUTION (eligible events)")
    tr = elig_ev["target_risk"]
    A(f"   min={tr.min():.3f} median={tr.median():.3f} max={tr.max():.3f}")
    n_m30 = int(np.isclose(tr, -30.0).sum())
    A(f"   Targets with risk = -30 : {n_m30:,} ({pct(n_m30, len(tr))})")
    A(f"   Targets with risk >= -6 (Pc >= 1e-6, 'high risk'): {int((tr >= -6).sum()):,} ({pct(int((tr >= -6).sum()), len(tr))})")
    for c, n in elig_ev["risk_category"].value_counts().reindex(RISK_CATEGORY_ORDER).dropna().items():
        A(f"   {c:<20} {int(n):,} ({pct(n, len(tr))})")
    A("")
    A("12. NOTES / ASSUMPTIONS / LIMITATIONS")
    for n in notes:
        A(f"   - {n}")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def write_split_report(path, events, train_ids, val_ids, results, notes):
    L = []
    A = L.append
    n = len(events)
    A("=" * 78)
    A("MODULE 1 - 90/10 EVENT-LEVEL STRATIFIED SPLIT REPORT")
    A("=" * 78)
    A(f"Random seed : {RANDOM_SEED}")
    A("Method      : sklearn.train_test_split on UNIQUE EVENTS, stratified by target-risk category")
    A("Categories  : C0 risk==-30 | C1 < -10 | C2 [-10,-6) | C3 [-6,-5) | C4 >= -5  (rare ones merged if n<2)")
    A("")
    A(f"Eligible events : {n:,}")
    A(f"Training        : {len(train_ids):,} events ({pct(len(train_ids), n)})")
    A(f"Validation      : {len(val_ids):,} events ({pct(len(val_ids), n)})")
    A("")
    tr = events.loc[sorted(train_ids)]
    va = events.loc[sorted(val_ids)]
    A("Stratification label distribution (event counts and shares)")
    A(f"{'label':<36}{'all':>8}{'train':>9}{'train%':>9}{'val':>7}{'val%':>9}{'|diff|':>9}")
    for lab in sorted(events["strat_label"].unique()):
        a = int((events["strat_label"] == lab).sum())
        t_ = int((tr["strat_label"] == lab).sum())
        v_ = int((va["strat_label"] == lab).sum())
        A(f"{lab:<36}{a:>8}{t_:>9}{100*t_/len(tr):>8.2f}%{v_:>7}{100*v_/max(len(va),1):>8.2f}%"
          f"{abs(100*t_/len(tr)-100*v_/max(len(va),1)):>8.2f}%")
    A("")
    A("Original (un-merged) risk categories")
    A(f"{'category':<20}{'train':>8}{'train%':>9}{'val':>7}{'val%':>9}")
    for lab in RISK_CATEGORY_ORDER:
        t_ = int((tr["risk_category"] == lab).sum())
        v_ = int((va["risk_category"] == lab).sum())
        A(f"{lab:<20}{t_:>8}{100*t_/len(tr):>8.2f}%{v_:>7}{100*v_/max(len(va),1):>8.2f}%")
    A("")
    A(f"Target risk == -30 : train {int(np.isclose(tr['target_risk'], -30).sum())}, "
      f"validation {int(np.isclose(va['target_risk'], -30).sum())}")
    A(f"Target risk >= -6  : train {int((tr['target_risk'] >= -6).sum())}, "
      f"validation {int((va['target_risk'] >= -6).sum())}")
    A("")
    A("Validation checks (all executed as assertions)")
    for name, ok, detail in results:
        A(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    A("")
    A("Limitations / notes")
    for note in notes:
        A(f"  - {note}")
    A("  - The validation (10%) partition must NOT be used for training or tuning.")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Module 1 dataset preparation")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Path to train_data.csv or its folder")
    ap.add_argument("--out", default=None, help="Output folder (default: next to this script)")
    args = ap.parse_args()

    try:
        csv_path = locate_csv(clean_path(args.input))
    except FileNotFoundError as e:
        raise SystemExit(f"ERROR: {e}")

    base = Path(args.out) if args.out else Path(__file__).resolve().parent
    out_dir = base / OUTPUT_DIR_NAME if args.out is None or base.name != OUTPUT_DIR_NAME else base
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output folder: {out_dir}")

    notes = []
    hash_before = sha256_of(csv_path)
    docs, doc_hits = find_local_documentation(csv_path)

    df = load_dataset(csv_path)
    dtypes = [(c, df[c].dtype, int(df[c].notna().sum()), int(df[c].isna().sum())) for c in df.columns]
    missing_tbl = [(c, int(df[c].isna().sum()), 100.0 * df[c].isna().mean())
                   for c in df.columns if df[c].isna().any()]
    original_cols = list(df.columns)

    log("[2/7] Inspecting dataset ...")
    info = inspect_dataset(df)
    if info["n_nan_event_id"]:
        notes.append(f"{info['n_nan_event_id']} rows with NaN event_id were ignored in processing.")
        df = df[df["event_id"].notna()]
    if not info["order_like_cols"]:
        notes.append("No CDM-id / creation-timestamp column exists; chronology is derived from time_to_tca "
                     "(descending). Ties keep original file order.")
    if info["events_file_order_not_desc"]:
        notes.append(f"{info['events_file_order_not_desc']} events were not already in descending time_to_tca "
                     "order in the file; they were re-ordered in memory only.")
    if info["suspect_cols_present"]:
        notes.append(f"Possible leakage columns present and kept unchanged: {info['suspect_cols_present']} "
                     "- verify their definition before using them as features.")
    notes.append("risk = -30 was retained as a valid (floor) value, never treated as missing.")
    notes.append("Original CSV is never modified; all outputs are new files.")
    if not docs:
        notes.append("No local documentation was found: units of time_to_tca and meaning of risk are inferred "
                     "from data ranges only.")

    log("[3/7] Ordering CDMs chronologically and evaluating eligibility ...")
    df = order_chronologically(df)
    ev = build_event_table(df)
    n_ev, n_el = len(ev), int(ev["eligible"].sum())
    log(f"      events: {n_ev:,} | eligible: {n_el:,} | excluded: {n_ev - n_el:,}")
    if n_el < 20:
        raise SystemExit("Too few eligible events to continue; check the inspection facts / rules.")

    excluded = ev[~ev["eligible"]].reset_index()[
        ["event_id", "n_cdms", "n_early_cdms", "n_mid_cdms_1to2d", "target_time_to_tca",
         "target_risk", "exclusion_reasons"]]
    excluded.to_csv(out_dir / "module1_excluded_events.csv", index=False)

    log("[4/7] Building eligible dataset ...")
    elig_ev = ev[ev["eligible"]].copy()
    elig_ev["risk_category"] = risk_category(elig_ev["target_risk"])
    if (elig_ev["risk_category"] == "UNASSIGNED").any():
        raise SystemExit("Some target risks could not be categorised (unexpected values).")
    elig_ev["strat_label"] = merge_rare_categories(elig_ev["risk_category"], notes)
    for lab, cnt in elig_ev["strat_label"].value_counts().items():
        if cnt < 10:
            notes.append(f"Rare stratum '{lab}' has only {cnt} eligible events; its 90/10 representation is coarse.")

    eligible = assign_roles(df, set(elig_ev.index))
    eligible["orig_row_number"] = eligible["_orig_row"]  # 0-based row index in the original file
    eligible["n_early_cdms"] = eligible["event_id"].map(elig_ev["n_early_cdms"]).astype(int)
    tgt_mask = eligible["cdm_role"] == "TARGET"
    eligible["risk_category"] = pd.Series(pd.NA, index=eligible.index, dtype="object")
    eligible.loc[tgt_mask, "risk_category"] = eligible.loc[tgt_mask, "event_id"].map(elig_ev["risk_category"])
    meta_cols = ["orig_row_number", "cdm_seq", "cdm_role", "is_model_input", "n_early_cdms", "risk_category"]
    eligible = eligible[original_cols + meta_cols]
    del df  # free memory

    log("[5/7] Stratified 90/10 split by event ...")
    train_ids, val_ids = split_events(elig_ev, notes)
    train = eligible[eligible["event_id"].isin(train_ids)]
    val = eligible[eligible["event_id"].isin(val_ids)]

    log("[6/7] Validating ...")
    results = []
    run_validations(csv_path, hash_before, eligible, train, val, elig_ev, results)
    for name, ok, detail in results:
        log(f"      [PASS] {name} {detail}")

    log("[7/7] Writing files ...")
    eligible.to_csv(out_dir / "module1_eligible_dataset.csv", index=False)
    train.to_csv(out_dir / "module1_train_90.csv", index=False)
    val.to_csv(out_dir / "module1_validation_10.csv", index=False)

    ev_out = elig_ev.copy()
    ev_out["split"] = np.where(ev_out.index.isin(val_ids), "validation_10", "train_90")
    ev_out.reset_index().to_csv(out_dir / "module1_event_summary.csv", index=False)

    early_dist = elig_ev["n_early_cdms"].value_counts().sort_index().head(30)
    write_inspection_report(out_dir / "module1_inspection_report.txt", csv_path, original_cols, dtypes, info,
                            ev, elig_ev, docs, doc_hits, missing_tbl, notes, hash_before, early_dist,
                            len(eligible))
    write_split_report(out_dir / "module1_split_report.txt", elig_ev, train_ids, val_ids, results, notes)

    log("")
    log("DONE. Files in: " + str(out_dir))
    for p in sorted(out_dir.iterdir()):
        log(f"   - {p.name}  ({p.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as err:
        print(f"\nSTOPPED - {err}", file=sys.stderr)
        sys.exit(2)
