"""
filter_dataset.py - builds the two module datasets from the Kelvins CDM data and validates them.

    python filter_dataset.py --source "C:\\path\\Collision Avoidance Challenge - Dataset.zip" --out-dir outputs

Eligibility (identical event population for both modules - they differ only in the INPUT window):
  * final CDM of the event (smallest time_to_tca) must satisfy time_to_tca < 1        [Kelvins protocol]
  * event must have >= 3 CDMs with time_to_tca >= 2                                    [PROJECT requirement]
  * Module 1 inputs: CDMs with time_to_tca >= 2        Module 2 inputs: CDMs with time_to_tca >= 1
  * target = risk of the final CDM; it is never part of the inputs; risk values are never altered.
Official test files (test_data.csv + test_data_private.csv) are processed for Module 1 only, with the
private labels written to a SEPARATE file that must only be used for final evaluation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd



N_FOLDS = 5


# ------------------------------------------------------------------------------------------------
def assign_splits(elig: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, event-level, stratified (final high-risk vs not) split + 5 CV folds."""
    rng = np.random.default_rng(C.SPLIT_SEED)
    split = pd.Series(index=elig.index, dtype=object)
    fold = pd.Series(index=elig.index, dtype=int)
    strata = (elig["final_risk"] >= C.HIGH_RISK_LOG10).astype(int)
    for s in (0, 1):
        ids = np.sort(elig.index[strata == s].to_numpy())
        rng.shuffle(ids)
        n = len(ids)
        n_val, n_te = int(round(n * C.SPLIT_FRACTIONS["val"])), int(round(n * C.SPLIT_FRACTIONS["internal_test"]))
        split.loc[ids[:n_val]] = "val"
        split.loc[ids[n_val:n_val + n_te]] = "internal_test"
        split.loc[ids[n_val + n_te:]] = "train"
        perm = rng.permutation(ids)
        fold.loc[perm] = np.arange(n) % N_FOLDS
    return pd.DataFrame({"split": split, "cv_fold": fold})


def risk_bins(s: pd.Series) -> dict:
    edges = [("< -30", s < -30), ("== -30", s == -30), ("(-30, -10]", (s > -30) & (s <= -10)),
             ("(-10, -6)", (s > -10) & (s < -6)), ("[-6, -5)", (s >= -6) & (s < -5)),
             ("[-5, -4)", (s >= -5) & (s < -4)), (">= -4", s >= -4), ("NaN", s.isna())]
    return {k: int(m.sum()) for k, m in edges}


def make_module(raw: pd.DataFrame, ev: pd.DataFrame, meta: pd.DataFrame, module: int):
    """Return (inputs_df, targets_df) for module 1 or 2 from eligible events."""
    min_tto = C.EARLY_MIN_DAYS if module == 1 else C.TARGET_MAX_DAYS
    ids = ev.index
    d = raw[raw["event_id"].isin(ids) & (raw["time_to_tca"] >= min_tto)].copy()
    d["source_row_index"] = d.index
    d = d.sort_values(["event_id", "time_to_tca", "source_row_index"], ascending=[True, False, True], kind="mergesort")
    d["cdm_order"] = d.groupby("event_id").cumcount()
    d["input_role"] = np.where(d["time_to_tca"] >= C.EARLY_MIN_DAYS, "early_input", "intermediate_input")
    d["source"] = "train"
    d["event_uid"] = "train:" + d["event_id"].astype(int).astype(str)
    d["split"] = d["event_id"].map(meta["split"])
    d["cv_fold"] = d["event_id"].map(meta["cv_fold"]).astype(int)
    d["module"] = module
    lead = ["event_uid", "source", "split", "cv_fold", "module", "cdm_order", "input_role", "source_row_index"]
    inputs = d[lead + list(raw.columns)].reset_index(drop=True)

    t = ev.copy()
    n_used_inter = t["n_intermediate"] if module == 2 else 0
    targets = pd.DataFrame({
        "event_uid": "train:" + t.index.astype(int).astype(str), "source": "train", "event_id": t.index.astype(int),
        "split": meta["split"].reindex(t.index).to_numpy(), "cv_fold": meta["cv_fold"].reindex(t.index).to_numpy(),
        "module": module, "target_source_row_index": t["final_row_index"].to_numpy(),
        "target_time_to_tca": t["final_time_to_tca"].to_numpy(), "target_risk": t["final_risk"].to_numpy(),
        "target_is_high_risk": (t["final_risk"] >= C.HIGH_RISK_LOG10).to_numpy(),
        "n_input_cdms": (t["n_early"] + n_used_inter).to_numpy(), "n_early_cdms": t["n_early"].to_numpy(),
        "n_intermediate_cdms_used": (t["n_intermediate"] if module == 2 else 0 * t["n_intermediate"]).to_numpy(),
        "n_intermediate_cdms_available": t["n_intermediate"].to_numpy(),
        "n_unused_target_window_cdms": (t["n_target_window"] - 1).to_numpy(), "eligibility_status": "eligible",
    })
    return inputs, targets


def write_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    log.info("wrote %s (%d rows, %.1f MB)", path, len(df), path.stat().st_size / 1e6)


# ------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--work-dir", type=Path, default=Path("work"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    a = ap.parse_args()
    out = a.out_dir
    C.setup_logging(out / "reports" / "filter_dataset.log")

    files = C.locate_dataset(a.source, a.work_dir)
    hashes_before = {str(p): C.sha256_file(p) for p in files.raw_sources}
    train, test, priv = C.load_csv(files.train), C.load_csv(files.test), C.load_csv(files.private)
    rep: dict = {"constants": {"early_min_days": C.EARLY_MIN_DAYS, "target_max_days": C.TARGET_MAX_DAYS,
                               "min_early_cdms": C.MIN_EARLY_CDMS, "high_risk_log10": C.HIGH_RISK_LOG10,
                               "split_seed": C.SPLIT_SEED, "split_fractions": C.SPLIT_FRACTIONS, "cv_folds": N_FOLDS}}

    # ---- invalid rows (reported, never silently dropped) ---------------------------------------------
    bad = train["event_id"].isna() | train["time_to_tca"].isna()
    rep["train_rows_removed_invalid_event_id_or_time"] = int(bad.sum())
    if bad.any():
        log.warning("%d train rows with invalid event_id/time_to_tca are excluded", int(bad.sum()))
        train = train[~bad]
    train["event_id"] = train["event_id"].astype(int)

    # ---- event table + sequential funnel ------------------------------------------------------------
    ev = C.build_event_table(train, "risk")
    ev["exclusion_reason"] = ev.apply(C.exclusion_reason, axis=1)
    funnel, remaining = [], ev.copy()
    funnel.append({"step": "all events in train_data.csv", "events_removed": 0, "cdms_removed": 0,
                   "events_left": len(remaining), "cdms_left": int(remaining.n_cdms.sum())})
    for reason, label in [("no_final_cdm_with_time_to_tca_lt_1", "final CDM must have time_to_tca < 1"),
                          ("final_target_risk_missing_or_invalid", "final target risk must be valid (not NaN)"),
                          ("fewer_than_3_early_cdms_(time_to_tca_ge_2)", "need >= 3 CDMs with time_to_tca >= 2")]:
        rm = remaining[remaining.exclusion_reason == reason]
        remaining = remaining[remaining.exclusion_reason != reason]
        funnel.append({"step": label, "events_removed": len(rm), "cdms_removed": int(rm.n_cdms.sum()),
                       "events_left": len(remaining), "cdms_left": int(remaining.n_cdms.sum())})
        log.info("funnel | %-48s -%5d events | left %5d", label, len(rm), len(remaining))
    elig = remaining
    rep["train_funnel"] = funnel
    rep["train_exclusion_reasons_sequential"] = ev["exclusion_reason"].replace("", "ELIGIBLE").value_counts().to_dict()
    rep["train_independent_flags"] = {
        "final_cdm_tto_lt_1": int(ev.has_cdm_in_target_window.sum()),
        "has_>=3_early_cdms": int(ev.has_min_early_cdms.sum()),
        "target_risk_valid": int(ev.target_risk_valid.sum()),
        "fails_only_target_window": int((~ev.has_cdm_in_target_window & ev.has_min_early_cdms).sum()),
        "fails_only_early_count": int((ev.has_cdm_in_target_window & ~ev.has_min_early_cdms).sum()),
        "fails_both": int((~ev.has_cdm_in_target_window & ~ev.has_min_early_cdms).sum()),
    }

    # ---- splits (event level; official test labels are NOT used anywhere here) ------------------------
    meta = assign_splits(elig)
    rep["split_events"] = meta["split"].value_counts().to_dict()
    rep["split_high_risk_events"] = elig.final_risk.ge(C.HIGH_RISK_LOG10).groupby(meta["split"]).sum().astype(int).to_dict()
    rep["cv_fold_sizes"] = meta["cv_fold"].value_counts().sort_index().to_dict()
    status = ev.join(meta)
    status["eligible"] = status["exclusion_reason"] == ""
    status.insert(0, "event_uid", "train:" + status.index.astype(str))
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    status.reset_index().to_csv(out / "metadata" / "event_status_train.csv", index=False)

    # ---- module outputs ---------------------------------------------------------------------------------
    m1_in, m1_tg = make_module(train, elig, meta, 1)
    m2_in, m2_tg = make_module(train, elig, meta, 2)
    write_gz(m1_in, out / "module1_early_prediction" / "inputs.csv.gz")
    write_gz(m1_tg, out / "module1_early_prediction" / "targets.csv.gz")
    write_gz(m2_in, out / "module2_updated_prediction" / "inputs.csv.gz")
    write_gz(m2_tg, out / "module2_updated_prediction" / "targets.csv.gz")

    # ---- official test (Module 1 only) ---------------------------------------------------------------------
    ev_te = C.build_event_table(test, "risk")
    pv = priv.set_index("event_id")
    ev_te["private_row_present"] = ev_te.index.isin(pv.index)
    ev_te["label_tto"] = pv["time_to_tca"].reindex(ev_te.index)
    ev_te["label_risk"] = pv["true_risk"].reindex(ev_te.index)
    ev_te["exclusion_reason"] = np.select(
        [~ev_te.private_row_present, ~(ev_te.label_tto < C.TARGET_MAX_DAYS), ev_te.label_risk.isna(), ev_te.n_early < C.MIN_EARLY_CDMS],
        ["no_private_label_row", "label_cdm_not_lt_1_day", "label_risk_missing", "fewer_than_3_early_cdms_(time_to_tca_ge_2)"], "")
    te_ok = ev_te[ev_te.exclusion_reason == ""]
    rep["official_test"] = {"events": len(ev_te), "eligible_module1": len(te_ok),
                            "exclusion_reasons": ev_te.exclusion_reason.replace("", "ELIGIBLE").value_counts().to_dict(),
                            "eligible_high_risk": int((te_ok.label_risk >= C.HIGH_RISK_LOG10).sum()),
                            "eligible_label_eq_minus30": int((te_ok.label_risk == -30).sum()),
                            "module2_possible": False,
                            "module2_reason": "intermediate window (1-2 days to TCA) was removed from test_data.csv by the competition"}
    t_in = test[test["event_id"].isin(te_ok.index)].copy()
    t_in["source_row_index"] = t_in.index
    t_in = t_in.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort")
    t_in["cdm_order"] = t_in.groupby("event_id").cumcount()
    t_in["event_uid"], t_in["source"], t_in["split"] = "test:" + t_in.event_id.astype(str), "test", "official_test"
    t_in["module"], t_in["input_role"] = 1, "early_input"
    t_in = t_in[["event_uid", "source", "split", "module", "cdm_order", "input_role", "source_row_index"] + list(test.columns)]
    labels = pd.DataFrame({"event_uid": "test:" + te_ok.index.astype(str), "source": "test", "event_id": te_ok.index,
                           "label_time_to_tca": te_ok.label_tto.to_numpy(), "true_risk": te_ok.label_risk.to_numpy(),
                           "is_high_risk": (te_ok.label_risk >= C.HIGH_RISK_LOG10).to_numpy(),
                           "n_input_cdms": te_ok.n_early.to_numpy()})
    write_gz(t_in.reset_index(drop=True), out / "module1_early_prediction" / "official_test" / "inputs.csv.gz")
    write_gz(labels, out / "module1_early_prediction" / "official_test" / "labels_private_FINAL_EVAL_ONLY.csv.gz")
    ev_te.reset_index().to_csv(out / "metadata" / "event_status_official_test.csv", index=False)

    # ---- summary statistics -------------------------------------------------------------------------------------
    rep["modules"] = {}
    for m, (i, t) in {1: (m1_in, m1_tg), 2: (m2_in, m2_tg)}.items():
        rep["modules"][f"module{m}"] = {
            "eligible_events": len(t), "input_cdms": len(i), "early_input_cdms": int((i.input_role == "early_input").sum()),
            "intermediate_input_cdms": int((i.input_role == "intermediate_input").sum()), "target_records": len(t),
            "events_with_>=1_intermediate_cdm": int((t.n_intermediate_cdms_available > 0).sum()),
            "events_with_0_intermediate_cdm": int((t.n_intermediate_cdms_available == 0).sum()),
            "input_cdms_per_event": {"min": int(t.n_input_cdms.min()), "median": float(t.n_input_cdms.median()),
                                     "max": int(t.n_input_cdms.max())},
            "target_risk_distribution": risk_bins(t.target_risk),
            "target_risk_eq_minus30": int((t.target_risk == -30).sum()),
            "target_risk_lt_minus30": int((t.target_risk < -30).sum()),
            "target_high_risk_events": int(t.target_is_high_risk.sum()),
            "input_rows_risk_eq_minus30": int((i.risk == -30).sum()), "input_rows_risk_lt_minus30": int((i.risk < -30).sum()),
            "target_time_to_tca_negative_events": int((t.target_time_to_tca < 0).sum()),
        }
    rep["target_cdms_unused_in_any_input"] = int(m1_tg.n_unused_target_window_cdms.sum())
    rep["raw_sha256_before"] = hashes_before
    rep["raw_sha256_after"] = {str(p): C.sha256_file(p) for p in files.raw_sources}
    rep["raw_data_unchanged"] = rep["raw_sha256_before"] == rep["raw_sha256_after"]
    C.write_json(rep, out / "reports" / "filtration_report.json")

    from validate_outputs import run_validation
    ok = run_validation(out, files.train, rep)
    from report_writer import write_filtration_md
    write_filtration_md(rep, out / "reports" / "filtration_report.md")
    log.info("ALL VALIDATION CHECKS PASSED" if ok else "VALIDATION FAILED - see reports/validation_results.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
