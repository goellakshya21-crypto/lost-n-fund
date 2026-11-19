"""Export the demo as a static site (for Vercel or any static host).

    python run_pipeline.py export-site          ->  site/

The live demo is a FastAPI server; static hosts can't run it. This writes:
  site/index.html, site/static/*   the same screens
  site/static/api-static.js        the demo API + offer ledger, in the browser (state in localStorage)
  site/data/demo.json              a snapshot of the latest campaign run (synthetic data only)
  site/report.html                 the campaign report
Every viewer gets their own demo state; "Reset demo" clears it.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lost_n_fund.config import PROJECT_ROOT
from lost_n_fund.diagnosis import ACTION_LABELS, REASON_LABELS
from lost_n_fund.messaging import WHY_TEXT, offer_terms

from .server import STATIC, Store, latest_run


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else round(f, 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return str(v)
    return v


def build_snapshot(st: Store) -> dict:
    cfg = st.cfg
    ov = st.overview()                       # static parts of the Ops screen
    a = st.assign
    offers = []
    for r in a.itertuples(index=False):
        lang = r.cust_lang if r.cust_lang in ("hi", "en") else "en"
        offers.append({
            "offer_id": r.offer_id, "customer_id": r.customer_id, "merchant_id": r.merchant_id,
            "merchant_name": r.merchant_name, "merchant_locality": r.merchant_locality, "category": r.category,
            "group": r.group, "needs_merchant_approval": bool(r.needs_merchant_approval), "funder": r.funder,
            "paytm_share": r.paytm_share, "offer_type": r.offer_type, "cashback": r.cashback, "pct": r.pct,
            "cap": r.cap, "uses": int(r.uses), "streak_visits": None if pd.isna(r.streak_visits) else int(r.streak_visits),
            "min_bill": float(r.min_bill), "valid_from": r.valid_from, "valid_till": r.valid_till,
            "validity_days": int(r.validity_days), "avg_bill": float(r.avg_ticket_at_merchant),
            "masked": r.masked, "first_name": r.first_name, "customer_locality": r.customer_locality,
            "language": lang,
            "title": f"{r.merchant_name} par aapke liye offer" if lang == "hi" else f"An offer at {r.merchant_name}",
            "terms": offer_terms(r, lang, r.valid_till),
        })
    merchants = {}
    alerts = {}
    for mid in st.alerts:
        d = st.merchant_detail(mid)
        app, pres = d["app"], d["presenter"]
        merchants[mid] = {**app["merchant"], "auto_approve": bool(st.merchants.loc[mid, "auto_approve"]),
                          "monthly_offer_budget": float(st.merchants.loc[mid, "monthly_offer_budget"])}
        alerts[mid] = {k: app[k] for k in ("texts", "band", "budget_cap", "expected_cost", "extra_sales_range",
                                           "cofunded", "autopilot", "segment", "offer_mix", "named")}
        alerts[mid]["presenter"] = pres
    # budgets for every shop that has a treatment offer (the ledger checks them at payment time)
    budgets = {m: float(st.merchants.loc[m, "monthly_offer_budget"]) for m in a["merchant_id"].unique()}
    ar = st.at_risk
    cols = ["customer_id", "name", "locality", "lapse_prob", "risk_tier", "route", "route_label", "action_label",
            "diagnosis_evidence", "offer_id", "merchant_id", "merchant_name", "no_offer_reason"]
    rows = ar[cols].astype(object).where(ar[cols].notna(), None).values.tolist()
    risk = {r.customer_id: {"lapse_prob": float(r.lapse_prob), "risk_tier": r.risk_tier, "reasons": r.reasons,
                            "route": r.route, "route_label": r.route_label, "action_label": r.action_label,
                            "evidence": r.diagnosis_evidence}
            for r in ar[ar["customer_id"].isin(set(a["customer_id"]))].itertuples(index=False)}
    return _clean({
        "snapshot": str(st.snapshot.date()),
        "overview": {k: ov[k] for k in ("run", "funnel", "routes", "actions", "diagnosis_check", "model", "measured",
                                        "compliance")},
        "routing": {k: v["action"] for k, v in cfg["routing"].items()},
        "reason_labels": REASON_LABELS, "action_labels": ACTION_LABELS,
        "rules": {"one_redemption_per_day": cfg["abuse"]["one_redemption_per_day"],
                  "clawback_on_refund": cfg["abuse"]["clawback_on_refund"]},
        "offers": offers, "merchants": merchants, "alerts": alerts, "budgets": budgets,
        "at_risk": {"cols": cols, "rows": rows}, "risk": risk, "recent": st.recent, "why_text": WHY_TEXT,
    })


def export_site(cfg: dict, run_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    run_dir = Path(run_dir) if run_dir else latest_run(cfg)
    st = Store(cfg, run_dir)
    st.state = st._empty_state()           # export a clean demo, not whatever was clicked locally
    st._rebuild_ledger()
    out = Path(out_dir) if out_dir else PROJECT_ROOT / "site"
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("static", "data"):          # keep out/.vercel (the project link) between exports
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir()
    for f in ("app.js", "styles.css", "api-static.js"):
        shutil.copyfile(STATIC / f, out / "static" / f)
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace('<script src="/static/app.js',
                        '<script>window.LNF_STATIC = true;</script>\n  <script src="/static/api-static.js?v=1"></script>\n'
                        '  <script src="/static/app.js')
    (out / "index.html").write_text(html, encoding="utf-8")
    report = run_dir / "report.html"
    if report.exists():
        shutil.copyfile(report, out / "report.html")
    snap = build_snapshot(st)
    (out / "data" / "demo.json").write_text(json.dumps(snap, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    (out / "vercel.json").write_text(json.dumps({
        "cleanUrls": True,
        "headers": [{"source": "/(.*)", "headers": [{"key": "X-Robots-Tag", "value": "noindex"}]}],
    }, indent=2), encoding="utf-8")
    return out
