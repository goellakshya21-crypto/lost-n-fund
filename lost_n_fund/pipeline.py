"""Stage functions. Each stage reads the previous stage's files, so any stage can be re-run on its own
(e.g. from a daily Airflow job) without redoing the others.

  generate      -> data/                                   (demo only)
  train         -> models/registry/<version>/              (weekly; promoted only if not worse)
  train-uplift  -> models/uplift_model.joblib              (once enough measured campaigns exist)
  campaign      -> outputs/campaign_<date>/  01 at-risk pairs + diagnosis, 02 candidates, 03 offers,
                   04 assignments, 05 merchant alerts + action queues           (daily / weekly)
  decide        -> 06 merchant decisions, 07 customer notifications            (after merchants respond)
  measure       -> 08 ledger + settlement lines, 09 results + outcomes, 10 merchant summaries, report
                                                                                (90 days later)
  purge         -> deletes customer-level files older than compliance.retention_days    (daily)
  erase         -> removes named customers everywhere (DPDP right to erasure)            (on request)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import synthetic
from .data import compliance_counts, default_snapshot, eligible_customer_ids, load_data, pseudonym
from .diagnosis import diagnose, truth_check
from .features import FEATURE_COLUMNS
from .ledger import OfferLedger
from .matching import match_pairs, select_at_risk
from .measure import measure_campaign, pooled_readout
from .messaging import (alerts_preview, build_customer_notifications, build_merchant_alerts, expand_decisions,
                        load_contact_history, simulate_merchant_decisions, weekly_merchant_summaries)
from .model import (load_model, psi, register_model, score_pairs, shadow_compare, shadow_version, train_model)
from .monitoring import check_guardrails, update_health
from .offers import (apply_budgets, apply_min_group, assign_holdout, committed_this_month, design_offers,
                     eligibility, in_global_holdout, load_assumptions, one_per_customer)
from .uplift import load_uplift, predict_uplift, train_uplift

# files inside a run folder that contain customer-level (personal) data
PERSONAL_FILES = [
    "01_at_risk_pairs.csv", "02_candidates.csv", "03_offers.csv", "03_offers_dropped.csv",
    "04_campaign_assignments.csv", "05_merchant_alerts.json", "05_merchant_alerts_preview.txt",
    "05_queue_paytm_fix.csv", "05_queue_wait.csv", "06_merchant_decisions.csv", "06_demo_merchant_decisions.csv",
    "07_customer_notifications.csv", "08_post_period_transactions_SIMULATED.csv", "08_events_SIMULATED.csv",
    "09_campaign_outcomes.csv", "demo_state.json",
]
# kept after purge with customer ids replaced by salted hashes (financial records must be auditable)
PSEUDONYMISE_FILES = ["08_ledger.csv"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_dir_for(cfg: dict, snapshot: pd.Timestamp) -> Path:
    return Path(cfg["paths"]["output_dir"]) / f"campaign_{snapshot:%Y-%m-%d}"


def _restrict_to_eligible(data, cfg: dict) -> set:
    """Customers who did not consent (and minors) are never trained on, scored or messaged."""
    eligible = eligible_customer_ids(data.customers, cfg)
    if cfg["compliance"]["only_process_consented"]:
        data.txns = data.txns[data.txns["customer_id"].isin(eligible)]
        if data.activity is not None:
            data.activity = data.activity[data.activity["customer_id"].isin(eligible)]
    return eligible


def _state(cfg: dict) -> Path:
    p = Path(cfg["paths"]["state_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------
def stage_generate(cfg: dict) -> None:
    _log("[generate] building synthetic UPI data ...")
    info = synthetic.generate_dataset(cfg, cfg["paths"]["data_dir"])
    _log(f"[generate] {info['transactions']:,} payments, {info['customers']:,} customers, "
         f"{info['merchants']} merchants, {info['start']} -> {info['end']}")
    _log(f"[generate] hidden causes: {info['types']} | travellers {info['travellers']:,} | "
         f"failing shops {info['declining_merchants']}")


def stage_train(cfg: dict) -> None:
    _log("[train] loading data and building point-in-time (customer x shop) snapshots ...")
    data = load_data(cfg["paths"]["data_dir"], cfg)
    eligible = _restrict_to_eligible(data, cfg)
    _log(f"[train] using {len(eligible):,} of {len(data.customers):,} customers (consented to offers, not minors)")
    bundle = train_model(data, cfg)
    decision = register_model(bundle, cfg["paths"]["model_dir"], cfg)
    c = bundle["card"]
    v, raw, b = c["validation"], c["validation_uncalibrated"], c["baseline_logistic_validation"]
    _log(f"[train] out-of-time validation ({c['validation_snapshot']}, {v['n']:,} pairs, lapse rate "
         f"{v['churn_rate']:.1%}): AUC {v['roc_auc']:.3f} | PR-AUC {v['pr_auc']:.3f} | top-10% lift "
         f"{v['lift_top10pct']}x | logistic baseline AUC {b['roc_auc']:.3f}")
    _log(f"[train] calibration: worst decile gap {raw['max_calibration_gap']:.3f} raw -> "
         f"{v['max_calibration_gap']:.3f} calibrated (cross-fitted)")
    _log(f"[train] {c['shop_lapsers_still_active_on_app']:.0%} of shop lapsers were still active on the app: "
         "a customer-level 'left the app' model would have called them safe")
    _log(f"[train] top signals: {', '.join(f['feature'] for f in c['feature_importance'][:6])}")
    _log(f"[train] registered {c['version']}: {'PROMOTED' if decision['promoted'] else 'kept as shadow'} "
         f"({decision['reason']})")


def stage_train_uplift(cfg: dict) -> None:
    card = train_uplift(cfg)
    if card.get("trained"):
        _log(f"[train-uplift] trained on {card['treated']} treated / {card['control']} control from "
             f"{len(card['runs'])} runs | held-out Qini {card['qini_heldout']} vs risk ranking {card['qini_risk_ranking']}")
    else:
        _log(f"[train-uplift] not trained: {card['reason']}. Offers keep using P(lapse) x save rate.")


def _load_carryover(cfg: dict, snap: pd.Timestamp) -> pd.DataFrame:
    p = _state(cfg) / "carryover.csv"
    if not p.exists():
        return pd.DataFrame(columns=["customer_id", "merchant_id", "first_queued"])
    q = pd.read_csv(p, parse_dates=["first_queued"])
    return q[(snap - q["first_queued"]).dt.days <= cfg["matching"]["carryover_days"]]


def _save_carryover(cfg: dict, old: pd.DataFrame, held: pd.DataFrame, alerted: pd.DataFrame,
                    snap: pd.Timestamp) -> pd.DataFrame:
    keys = ["customer_id", "merchant_id"]
    parts = [x for x in (old, held[keys].assign(first_queued=snap) if len(held) else None) if x is not None and len(x)]
    q = (pd.concat(parts, ignore_index=True).drop_duplicates(keys, keep="first") if parts
         else pd.DataFrame(columns=keys + ["first_queued"]))
    if len(alerted):
        q = q[~q.set_index(keys).index.isin(alerted.set_index(keys).index)]
    q.to_csv(_state(cfg) / "carryover.csv", index=False)
    return q


def stage_campaign(cfg: dict, snapshot: str | None = None, force: bool = False) -> Path:
    data = load_data(cfg["paths"]["data_dir"], cfg)
    snap = pd.Timestamp(snapshot) if snapshot else default_snapshot(data.txns)
    run_dir = run_dir_for(cfg, snap)
    run_id = run_dir.name
    comp = cfg["compliance"]

    hard, warn = check_guardrails(cfg, exclude_run=run_id)
    for w in warn:
        _log(f"[campaign] WARNING: {w}")
    if hard:
        msg = "; ".join(hard)
        if not force:
            raise SystemExit(f"[campaign] PAUSED by guardrails: {msg}. Investigate, then re-run with --force.")
        _log(f"[campaign] guardrail breach overridden with --force: {msg}")
    for issue in data.quality["issues"]:
        _log(f"[campaign] data check: {issue}")

    eligible = _restrict_to_eligible(data, cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_model(cfg["paths"]["model_dir"])
    _log(f"[campaign] scoring regulars (customer x shop) as of {snap.date()} with model {bundle['version']} -> {run_dir}")
    scores = score_pairs(data, bundle, snap, cfg)

    # ---- model health: drift + shadow model ----
    drift = psi(bundle["drift_ref"], scores[FEATURE_COLUMNS])
    top_drift = dict(list(drift.items())[:5])
    max_psi = max(drift.values()) if drift else 0.0
    if max_psi > cfg["training"]["psi_warn"]:
        _log(f"[campaign] WARNING: input drift (PSI {max_psi:.2f} on {next(iter(drift))}). Retrain soon.")
    shadow = None
    sv = shadow_version(cfg["paths"]["model_dir"])
    if sv:
        sb = load_model(cfg["paths"]["model_dir"], sv)
        shadow = {"version": sv, **shadow_compare(scores, sb["model"].predict_proba(scores[sb["features"]])[:, 1])}

    # ---- who is at risk, and why ----
    carry = _load_carryover(cfg, snap)
    at_risk = diagnose(select_at_risk(scores, cfg, carry), cfg)
    at_risk.to_csv(run_dir / "01_at_risk_pairs.csv", index=False)
    check = truth_check(at_risk, cfg["paths"]["data_dir"], snap)

    cands, funnel, queues = match_pairs(at_risk, data.merchants, data.customers, cfg)
    cands.to_csv(run_dir / "02_candidates.csv", index=False)
    fix_cols = ["customer_id", "merchant_id", "route", "diagnosis_evidence", "lapse_prob"]
    queues["paytm_fix"][fix_cols].to_csv(run_dir / "05_queue_paytm_fix.csv", index=False)
    queues["wait"][fix_cols].assign(recheck_after=str((snap + pd.Timedelta(days=14)).date())).to_csv(
        run_dir / "05_queue_wait.csv", index=False)
    ops = (queues["ops_alert"].groupby(["merchant_id", "name"])
           .agg(regulars_affected=("customer_id", "nunique"), evidence=("diagnosis_evidence", "first"))
           .reset_index().rename(columns={"name": "merchant_name"}))
    ops.to_csv(run_dir / "05_ops_merchant_alerts.csv", index=False)   # shop-level only, no customer ids

    # ---- eligibility (before randomisation), offers, holdouts, budgets, anonymity ----
    history = load_contact_history(cfg["paths"]["state_dir"])
    elig, ineligible = eligibility(cands, history, cfg, snap, run_id)
    assume = load_assumptions(cfg)
    ub = load_uplift(cfg)
    upl = predict_uplift(ub, elig) if (ub is not None and len(elig)) else None
    designed, not_profitable = design_offers(elig, cfg, assume, run_id, upl)
    best, dup = one_per_customer(designed)
    gh_mask = in_global_holdout(best["customer_id"], cfg, snap) if len(best) else []
    gh = best[gh_mask] if len(best) else best
    main = best[~gh_mask] if len(best) else best
    budgeted, over = apply_budgets(main, data.merchants, cfg, committed_this_month(history, snap, run_id))
    final, small = apply_min_group(budgeted, cfg)
    carry_after = _save_carryover(cfg, carry, small, final, snap)
    final.to_csv(run_dir / "03_offers.csv", index=False)
    dropped = pd.concat([ineligible, not_profitable, dup, over, small], ignore_index=True)
    dropped.to_csv(run_dir / "03_offers_dropped.csv", index=False)

    assign = assign_holdout(final, cfg, run_id)
    if len(gh):
        gh_rows = assign_holdout(gh, {**cfg, "campaign": {**cfg["campaign"], "holdout_share": 0.0}}, run_id)
        assign = pd.concat([assign, gh_rows.assign(group="global_holdout")], ignore_index=True)
    assign.to_csv(run_dir / "04_campaign_assignments.csv", index=False)

    alerts = build_merchant_alerts(assign, data.merchants, data.customers, cfg, snap)
    _write_json(run_dir / "05_merchant_alerts.json", alerts)
    (run_dir / "05_merchant_alerts_preview.txt").write_text(alerts_preview(alerts), encoding="utf-8")

    # ---- summary ----
    t = assign[assign["group"] == "treatment"]
    shop_t = t[t["needs_merchant_approval"].astype(bool)]
    smallest = int(shop_t.groupby("merchant_id").size().min()) if len(shop_t) else 0
    funnel = {
        "customers_total": int(len(data.customers)),
        "eligible_consented_adults": int(len(eligible)),
        "regular_pairs_scored": int(len(scores)),
        **funnel,
        "eligible_after_caps": int(len(elig)),
        "offer_pays_for_itself": int(designed["customer_id"].nunique()) if len(designed) else 0,
        "global_holdout": int(len(gh)),
        "within_budgets": int(len(budgeted)),
        "held_for_anonymity": int(len(small)),
        "final_offers": int(len(final)),
        "treatment": int(len(t)),
        "control_holdout": int((assign["group"] == "control").sum()),
        "shops_alerted": len(alerts),
        "paytm_funded_offers": int((~t["needs_merchant_approval"].astype(bool)).sum()),
    }
    sensitive = set(comp.get("sensitive_categories", []))
    summary = {
        "run_id": run_id, "snapshot": str(snap.date()),
        "model": {"version": bundle["version"], "max_psi": round(max_psi, 3), "top_drift": top_drift,
                  "shadow": shadow, "risk_tiers": scores["risk_tier"].value_counts().to_dict()},
        "data_quality": data.quality,
        "assumptions": assume, "uplift_model_used": upl is not None,
        "funnel": funnel,
        "routes": at_risk["route"].value_counts().to_dict(),
        "actions": {"paytm_fix": int(len(queues["paytm_fix"])), "ops_alert_shops": int(len(ops)),
                    "wait": int(len(queues["wait"])), "skip": int(len(queues["skip"]))},
        "diagnosis_vs_truth": check,
        "carryover_queue": int(len(carry_after)),
        "compliance": {
            **compliance_counts(data.customers),
            "processed_only_consented": bool(comp["only_process_consented"]),
            "minors_excluded": bool(comp["exclude_minors"]),
            "sensitive_categories_excluded": sorted(sensitive),
            "at_risk_pairs_skipped_sensitive": funnel.get("sensitive_pairs_skipped", 0),
            "merchant_view": comp["merchant_view"],
            "min_customers_per_alert": comp["min_customers_per_alert"],
            "count_bands": comp["count_bands"],
            "smallest_alert": smallest,
            "customers_held_for_anonymity": int(len(small)),
            "names_shown_to_merchants": int(sum(len(a["named_customers"]) for a in alerts)),
            "exact_counts_shown_to_merchants": 0,
            "phone_numbers_in_files": 0,
            "settlement_netting": comp["settlement_netting"],
            "retention_days": comp["retention_days"],
            "delete_after": str((snap + pd.Timedelta(days=comp["retention_days"])).date()),
            "data_residency": comp["data_residency"],
            "sms_fallback": bool(comp.get("sms_fallback")),
            "cross_app_linking": bool(comp.get("cross_app_linking")),
        },
        "drop_reasons": dropped["drop_reason"].value_counts().to_dict() if len(dropped) else {},
        "offer_mix": t.groupby(["offer_type", "assigned_by"]).size().rename("n").reset_index().to_dict(orient="records"),
        "by_funder": t.groupby("funder").agg(offers=("offer_id", "size"), expected_cost=("expected_cost", "sum"),
                                             merchant_ev=("merchant_ev", "sum"), paytm_ev=("paytm_ev", "sum"))
                      .round(1).reset_index().to_dict(orient="records"),
        "expected": {
            "recovered_gmv": round(float(t["expected_recovered_gmv"].sum()), 2),
            "merchant_ev": round(float(t["merchant_ev"].sum()), 2),
            "paytm_ev": round(float(t["paytm_ev"].sum()), 2),
            "cashback_cost": round(float(t["expected_cost"].sum()), 2),
        },
    }
    if comp["retention_days"] < cfg["campaign"]["measurement_days"]:
        _log("[campaign] WARNING: retention_days is shorter than the measurement window; results can't be measured.")
    _write_json(run_dir / "run_summary.json", summary)
    update_health(cfg, run_id, "campaign", {"max_psi": round(max_psi, 3), "offers": funnel["final_offers"],
                                            "shops_alerted": len(alerts)})

    f = funnel
    _log(f"[campaign] {f['regular_pairs_scored']:,} regular pairs -> {f['at_risk_pairs']:,} at risk "
         f"({f['at_risk_customers']:,} customers)")
    _log(f"[campaign] why: " + ", ".join(f"{k} {v}" for k, v in summary["routes"].items()))
    _log(f"[campaign] actions: {summary['actions']['paytm_fix']} payment fixes, {len(ops)} shops flagged to ops, "
         f"{summary['actions']['wait']} waiting, {summary['actions']['skip']} gone")
    if check:
        _log(f"[campaign] diagnosis vs hidden truth (demo): {check['accuracy_on_leaving']:.0%} right on truly-leaving "
             f"pairs; shop-funded precision {check['shop_funded_precision']:.0%}")
    _log(f"[campaign] {f['offer_candidates_pairs']} offer candidates -> {f['eligible_after_caps']} after caps -> "
         f"{f['final_offers']} offers ({f['treatment']} treatment / {f['control_holdout']} holdout, "
         f"{f['global_holdout']} global holdout) | {f['shops_alerted']} shops alerted, "
         f"{f['paytm_funded_offers']} Paytm-funded | {f['held_for_anonymity']} held for anonymity "
         f"(queue now {len(carry_after)})")
    _log(f"[campaign] assumptions: {assume['source']} | uplift model: {'yes' if upl is not None else 'not yet'} | "
         f"max drift PSI {max_psi:.2f}" + (f" | shadow {sv} top-10% overlap {shadow['top10pct_overlap']:.0%}" if shadow else ""))
    return run_dir


def stage_decide(cfg: dict, run_dir: Path, decisions_path: str | None = None) -> None:
    run_dir = Path(run_dir)
    data = load_data(cfg["paths"]["data_dir"], cfg)
    summary = _read_json(run_dir / "run_summary.json")
    snap = pd.Timestamp(summary["snapshot"])
    assign = pd.read_csv(run_dir / "04_campaign_assignments.csv")
    if decisions_path:
        raw = pd.read_csv(decisions_path)          # merchant_id,decision  or  offer_id,decision
        _log(f"[decide] using merchant decisions from {decisions_path}")
    else:
        raw = simulate_merchant_decisions(assign, data.merchants, cfg)
        _log("[decide] DEMO: simulating merchant approvals (autopilot shops approve automatically)")
    decisions = expand_decisions(raw, assign)
    decisions.to_csv(run_dir / "06_merchant_decisions.csv", index=False)

    history = load_contact_history(cfg["paths"]["state_dir"])
    notes, assign, history = build_customer_notifications(assign, decisions, data.customers, cfg, snap,
                                                          summary["run_id"], history)
    notes.to_csv(run_dir / "07_customer_notifications.csv", index=False)
    assign.to_csv(run_dir / "04_campaign_assignments.csv", index=False)
    history.to_csv(_state(cfg) / "contact_history.csv", index=False)

    t = assign[assign["group"] == "treatment"]
    shop = raw.drop_duplicates("merchant_id") if "merchant_id" in raw.columns else pd.DataFrame()
    approval = float((shop["decision"].astype(str).str.lower().str.startswith("approv")).mean()) if len(shop) else None
    summary["funnel"]["merchant_approved"] = int((t["merchant_decision"] == "approved").sum())
    summary["funnel"]["delivered_to_customer"] = int(t["delivered"].sum())
    summary["funnel"]["suppressed"] = t.loc[t["suppressed_reason"].fillna("") != "", "suppressed_reason"].value_counts().to_dict()
    summary["shop_approval_rate"] = approval
    _write_json(run_dir / "run_summary.json", summary)
    update_health(cfg, summary["run_id"], "decide", {"approval_rate": approval,
                                                     "delivered": summary["funnel"]["delivered_to_customer"]})
    rate = f"shop approval rate {approval:.0%}" if approval is not None else "no shop approvals needed"
    _log(f"[decide] {summary['funnel']['merchant_approved']}/{len(t)} offers approved ({rate}); "
         f"{summary['funnel']['delivered_to_customer']} sent to customers in their language")


def _simulator_truth(cfg: dict, assign: pd.DataFrame, merchants: pd.DataFrame, snap: pd.Timestamp,
                     reps: int = 5) -> dict:
    """DEMO ONLY: the effect the simulator actually builds in, found by simulating the same treated
    customers with and without their offer. Shows what the measurement is trying to recover."""
    from .measure import customer_outcomes
    out = {}
    for label, deliv in (("offer", True), ("no_offer", False)):
        a = assign.copy()
        a.loc[a["group"] == "treatment", "delivered"] = deliv
        vals = []
        for i in range(reps):
            post, _ = synthetic.simulate_post_period(cfg["paths"]["data_dir"], a, merchants, snap,
                                                     cfg["campaign"]["measurement_days"], {**cfg, "seed": 1000 + i})
            o = customer_outcomes(a, post, snap, cfg)
            o = o[o["group"] == "treatment"]
            vals.append((o["returned_30"].mean(), o["returned_post_offer"].mean()))
        out[label] = np.mean(vals, axis=0)
    d = 100 * (out["offer"] - out["no_offer"])
    return {"returned_30_pts": round(float(d[0]), 1), "returned_post_offer_pts": round(float(d[1]), 1),
            "note": "built-in effect of the simulator on these treated customers (averaged over re-simulations)"}


def _ledger_for(assign: pd.DataFrame, merchants: pd.DataFrame, cfg: dict, events: pd.DataFrame | None) -> OfferLedger:
    live = assign[(assign["group"] == "treatment") & assign["delivered"].fillna(False).astype(bool)]
    opt = {}
    if events is not None and len(events):
        oo = events[events["event"] == "opt_out"]
        opt = dict(zip(oo["customer_id"], pd.to_datetime(oo["ts"])))
    return OfferLedger(live, merchants, cfg, opt)


def stage_measure(cfg: dict, run_dir: Path, post_txns_path: str | None = None, events_path: str | None = None) -> None:
    from .report import build_report

    run_dir = Path(run_dir)
    data = load_data(cfg["paths"]["data_dir"], cfg)
    summary = _read_json(run_dir / "run_summary.json")
    snap = pd.Timestamp(summary["snapshot"])
    assign = pd.read_csv(run_dir / "04_campaign_assignments.csv")
    if "delivered" not in assign.columns:
        raise RuntimeError("Run the `decide` stage before `measure`.")

    if post_txns_path:
        post = pd.read_csv(post_txns_path, parse_dates=["ts"])
        events = pd.read_csv(events_path, parse_dates=["ts"]) if events_path else None
        simulated = False
        _log(f"[measure] using real post-campaign payments from {post_txns_path}")
    else:
        post, events = synthetic.simulate_post_period(cfg["paths"]["data_dir"], assign, data.merchants, snap,
                                                      cfg["campaign"]["measurement_days"], cfg)
        post.to_csv(run_dir / "08_post_period_transactions_SIMULATED.csv", index=False)
        events.to_csv(run_dir / "08_events_SIMULATED.csv", index=False)
        simulated = True
        _log(f"[measure] DEMO: simulated {cfg['campaign']['measurement_days']} days of payments after the campaign")

    ledger = _ledger_for(assign, data.merchants, cfg, events)
    ledger.replay(post)
    ledger.frame().to_csv(run_dir / "08_ledger.csv", index=False)
    lines = ledger.settlement_lines(cfg["compliance"]["settlement_netting"])
    lines.to_csv(run_dir / "08_settlement_lines.csv", index=False)

    results, outcomes = measure_campaign(assign, post, events, ledger, data.merchants, cfg, snap)
    results["simulated_outcomes"] = simulated
    if simulated:
        results["simulator_truth"] = _simulator_truth(cfg, assign, data.merchants, snap)
    _write_json(run_dir / "09_campaign_results.json", results)
    # training data for the uplift model: features at send time + treatment + outcome
    # (personal data: deleted by `purge` after compliance.retention_days)
    pairs = pd.read_csv(run_dir / "01_at_risk_pairs.csv")
    keep = ["customer_id", "merchant_id"] + [c for c in FEATURE_COLUMNS if c in pairs.columns]
    outcomes.drop(columns=[c for c in FEATURE_COLUMNS if c in outcomes.columns], errors="ignore") \
        .merge(pairs[keep], on=["customer_id", "merchant_id"], how="left") \
        .to_csv(run_dir / "09_campaign_outcomes.csv", index=False)

    results["pooled"] = pooled_readout(cfg)
    _write_json(run_dir / "09_campaign_results.json", results)
    learned = {**results.get("learned", {}), "source_run": summary["run_id"], "simulated": simulated}
    _write_json(_state(cfg) / "learned_assumptions.json", learned)
    board = pd.DataFrame(results.get("merchant_scoreboard", []))
    if len(board):
        _write_json(run_dir / "10_merchant_weekly_summaries.json", weekly_merchant_summaries(board, data.merchants, cfg))

    summary["funnel"]["redeemed"] = results.get("economics", {}).get("customers_paid_cashback")
    _write_json(run_dir / "run_summary.json", summary)
    update_health(cfg, summary["run_id"], "measure", results.get("safety", {}))
    report = build_report(run_dir, cfg)

    it = results.get("itt", {})
    if it.get("returned_30"):
        r30, rpo, e = it["returned_30"], it["returned_post_offer"], results["economics"]
        _log(f"[measure] came back to the shop within 30d: {r30['treated_mean']:.1f}% vs {r30['control_mean']:.1f}% "
             f"-> {r30['effect']:+.1f} pts (95% CI {r30['ci95'][0]:+.1f} to {r30['ci95'][1]:+.1f}, "
             f"{r30['clusters']} shop clusters)")
        if rpo:
            _log(f"[measure] still coming AFTER the offer ended: {rpo['effect']:+.1f} pts "
                 f"(95% CI {rpo['ci95'][0]:+.1f} to {rpo['ci95'][1]:+.1f})")
        p = results["power"]
        _log(f"[measure] power: this sample detects effects >= {p['mde_pts']} pts; a {p['target_pts']:.0f}-pt effect "
             f"needs ~{p['needed_control']:,} control / {p['needed_treated']:,} treated (~{p['runs_like_this_needed']} runs like this)")
        if results.get("simulator_truth"):
            st = results["simulator_truth"]
            _log(f"[measure] DEMO check: the simulator's built-in effect is {st['returned_30_pts']:+.1f} pts "
                 f"({st['returned_post_offer_pts']:+.1f} after the offer); the gap to the estimate above is sampling noise")
        _log(f"[measure] cashback Rs {e['cashback_total']:,.0f} (shops Rs {e['merchant_funded_cashback']:,.0f}, "
             f"Paytm Rs {e['paytm_funded_cashback']:,.0f}, Rs {e['reversed_on_refund']:,.0f} clawed back on refunds) | "
             f"shop profit after cashback Rs {e['merchant_profit_after_cashback']:,.0f} | Paytm value Rs {e['paytm_value']:,.0f}")
        s = results["safety"]
        _log(f"[measure] opt-outs {s['opt_out_rate']:.1%}, complaints {s['complaint_rate']:.2%} | "
             f"learned save rate {learned.get('save_rate')} ({'simulated: not used' if simulated else 'used next run'})")
    _log(f"[measure] report -> {report}")


def stage_purge(cfg: dict, as_of: str | None = None) -> None:
    """Delete customer-level files from campaign runs older than the retention period.

    Kept: run_summary.json, 09_campaign_results.json, report.html, settlement lines (shop-level),
    and the cashback ledger with customer ids replaced by salted hashes (financial audit trail).
    """
    today = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
    keep = cfg["compliance"]["retention_days"]
    n_runs = n_files = 0
    for run in sorted(Path(cfg["paths"]["output_dir"]).glob("campaign_*")):
        try:
            snap = pd.Timestamp(run.name.replace("campaign_", ""))
        except ValueError:
            continue
        if (today - snap).days <= keep:
            continue
        removed = [f for f in PERSONAL_FILES if (run / f).exists()]
        for f in removed:
            (run / f).unlink()
        for f in PSEUDONYMISE_FILES:
            p = run / f
            if p.exists():
                df = pd.read_csv(p)
                if "customer_id" in df.columns and not df["customer_id"].astype(str).str.startswith("h:").all():
                    df["customer_id"] = "h:" + df["customer_id"].astype(str).map(lambda c: pseudonym(c, cfg))
                    df.to_csv(p, index=False)
                    removed.append(f + " (pseudonymised)")
        if removed:
            n_runs += 1
            n_files += len(removed)
            summary_path = run / "run_summary.json"
            if summary_path.exists():
                s = _read_json(summary_path)
                s.setdefault("compliance", {})["purged_on"] = str(today.date())
                s["compliance"]["purged_files"] = removed
                _write_json(summary_path, s)
            _log(f"[purge] {run.name}: removed {len(removed)} files with personal data")
    # contact history is only needed for the frequency cap and the per-customer offer limit
    hist_path = _state(cfg) / "contact_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path, parse_dates=["sent_on"])
        cutoff = today - pd.Timedelta(days=max(cfg["campaign"]["frequency_cap_days"], cfg["compliance"]["abuse_history_days"]))
        kept = hist[hist["sent_on"] >= cutoff]
        if len(kept) < len(hist):
            kept.to_csv(hist_path, index=False)
            _log(f"[purge] contact history: removed {len(hist) - len(kept)} rows older than {cutoff.date()}")
    carry = _state(cfg) / "carryover.csv"
    if carry.exists():
        q = pd.read_csv(carry, parse_dates=["first_queued"])
        q[(today - q["first_queued"]).dt.days <= cfg["matching"]["carryover_days"]].to_csv(carry, index=False)
    _log(f"[purge] done as of {today.date()}: {n_files} files handled across {n_runs} runs (retention {keep} days)")


def stage_erase(cfg: dict, customers_file: str) -> None:
    """DPDP right to erasure: remove these customers from every file we hold, and make sure they never
    come back in. Only a salted hash is kept, on the suppression list and in the cashback ledger."""
    ids = set(pd.read_csv(customers_file)["customer_id"].astype(str))
    hashes = {pseudonym(c, cfg) for c in ids}
    sup = _state(cfg) / "erased_customers.txt"
    old = set(sup.read_text(encoding="utf-8").split()) if sup.exists() else set()
    sup.write_text("\n".join(sorted(old | hashes)) + "\n", encoding="utf-8")
    touched = 0
    paths = list(Path(cfg["paths"]["output_dir"]).glob("campaign_*/*.csv")) + \
        [_state(cfg) / "contact_history.csv", _state(cfg) / "carryover.csv"]
    for p in paths:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "customer_id" not in df.columns:
            continue
        hit = df["customer_id"].astype(str).isin(ids)
        if not hit.any():
            continue
        if p.name in PSEUDONYMISE_FILES:
            df.loc[hit, "customer_id"] = "h:" + df.loc[hit, "customer_id"].astype(str).map(lambda c: pseudonym(c, cfg))
        else:
            df = df[~hit]
        df.to_csv(p, index=False)
        touched += 1
    for p in Path(cfg["paths"]["output_dir"]).glob("campaign_*/demo_state.json"):
        st = _read_json(p)
        st["payments"] = [x for x in st.get("payments", []) if x.get("customer_id") not in ids]
        st["opt_outs"] = [c for c in st.get("opt_outs", []) if c not in ids]
        _write_json(p, st)
    _log(f"[erase] {len(ids)} customers erased from {touched} files; suppression list now {len(old | hashes)} hashes")


DEMO_STATE_FILES = ["contact_history.csv", "carryover.csv", "program_health.json", "learned_assumptions.json",
                    "erased_customers.txt"]


def stage_all(cfg: dict, snapshot: str | None = None) -> None:
    """Demo: regenerate data, so start from a clean programme state too."""
    for f in DEMO_STATE_FILES:
        p = Path(cfg["paths"]["state_dir"]) / f
        if p.exists():
            p.unlink()
    stage_generate(cfg)
    stage_train(cfg)
    run_dir = stage_campaign(cfg, snapshot)
    stage_decide(cfg, run_dir)
    stage_measure(cfg, run_dir)
