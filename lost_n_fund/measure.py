"""Measure the campaign against the holdouts, learn the real assumptions, and check the guardrails.

What is measured, and why:
  * Primary read: intent-to-treat (everyone assigned an offer vs everyone held out), with
    regression adjustment for pre-period spend (CUPED/ANCOVA) to cut noise, and standard errors
    clustered by shop (customers at one shop are not independent).
  * Approved shops only: customers were randomised WITHIN each shop and approval is a shop-level
    decision, so treatment vs control inside approving shops is also unbiased - and closer to the
    effect of an offer that actually goes out.
  * After the offer ends: "came back within 30 days" counts the discounted visit itself. What pays
    the bills is whether they are still coming once the cashback is gone (days after expiry to 90).
  * Global holdout: ~5% of customers get no offer all quarter. Comparing them with this run's
    control is a sanity check now, and over a quarter it measures the programme's total effect,
    including whether people learn to go quiet to get offers.
  * Power: the smallest effect this sample can reliably detect, and the sample a target effect needs.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ statistics
def _ols_cluster(y: np.ndarray, X: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with cluster-robust (CR1) standard errors."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    groups = pd.Series(np.arange(len(y))).groupby(clusters).indices
    for idx in groups.values():
        s = X[idx].T @ u[idx]
        meat += np.outer(s, s)
    n, k, g = len(y), X.shape[1], max(len(groups), 2)
    V = XtX_inv @ meat @ XtX_inv * (g / (g - 1)) * ((n - 1) / max(n - k, 1))
    return beta, np.sqrt(np.clip(np.diag(V), 0, None))


COVARIATE = {   # pre-period version of each outcome, for CUPED / ANCOVA adjustment
    "returned_30": "visits_per_month", "returned_in_offer": "visits_per_month",
    "returned_post_offer": "visits_per_month", "active_m1": "visits_per_month", "active_m2": "visits_per_month",
    "active_m3": "visits_per_month", "gmv_at_merchant_90": "pre_gmv_at_merchant",
    "active_platform_30": "txn_90", "gmv_platform_90": "customer_monthly_gmv",
}


def effect(df: pd.DataFrame, outcome: str, covariate: str | None = "auto",
           treat_col: str = "is_treat") -> dict | None:
    """Treatment effect on `outcome`, CUPED-adjusted and shop-clustered. Points for rates, Rs for money."""
    if covariate == "auto":
        covariate = COVARIATE.get(outcome)
    if covariate is not None and covariate not in df.columns:
        covariate = None
    d = df.dropna(subset=[outcome])
    t = d[treat_col].to_numpy(dtype=float)
    if t.sum() < 2 or (1 - t).sum() < 2:
        return None
    y = d[outcome].to_numpy(dtype=float)
    cols = [np.ones(len(d)), t]
    if covariate:
        x = d[covariate].to_numpy(dtype=float)
        cols.append(x - x.mean())
    beta, se = _ols_cluster(y, np.column_stack(cols), d["merchant_id"].to_numpy())
    _, se_raw = _ols_cluster(y, np.column_stack(cols[:2]), d["merchant_id"].to_numpy())
    rate = set(np.unique(y)) <= {0.0, 1.0}
    k = 100.0 if rate else 1.0
    return {
        "treated": int(t.sum()), "control": int((1 - t).sum()),
        "treated_mean": round(float(y[t == 1].mean()) * (100 if rate else 1), 2),
        "control_mean": round(float(y[t == 0].mean()) * (100 if rate else 1), 2),
        "effect": round(float(beta[1]) * k, 2),
        "ci95": [round(float(beta[1] - 1.96 * se[1]) * k, 2), round(float(beta[1] + 1.96 * se[1]) * k, 2)],
        "significant": bool(beta[1] - 1.96 * se[1] > 0 or beta[1] + 1.96 * se[1] < 0),
        "unit": "pts" if rate else "Rs",
        "variance_cut_by_cuped": round(1 - (se[1] / se_raw[1]) ** 2, 3) if covariate and se_raw[1] > 0 else 0.0,
        "clusters": int(d["merchant_id"].nunique()),
    }


def power(p0: float, n_t: int, n_c: int, target_pts: float, alpha_z: float = 1.96, power_z: float = 0.84) -> dict:
    """Minimum detectable effect now, and the sample a target effect needs (same treated:control ratio)."""
    p0 = min(max(p0, 0.01), 0.99)
    var = p0 * (1 - p0)
    mde = (alpha_z + power_z) * math.sqrt(var / max(n_t, 1) + var / max(n_c, 1))
    ratio = n_t / max(n_c, 1) if n_c else 4.0
    d = target_pts / 100.0
    need_c = (alpha_z + power_z) ** 2 * (var / ratio + var) / d ** 2
    return {"mde_pts": round(100 * mde, 1), "target_pts": target_pts,
            "needed_control": int(math.ceil(need_c)), "needed_treated": int(math.ceil(need_c * ratio)),
            "runs_like_this_needed": int(math.ceil(need_c / max(n_c, 1)))}


# ------------------------------------------------------------------ outcomes
def customer_outcomes(assign: pd.DataFrame, post: pd.DataFrame, snapshot: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    days = cfg["campaign"]["measurement_days"]
    short = cfg["campaign"]["short_window_days"]
    end = snapshot + pd.Timedelta(days=days)
    post = post[(post["ts"] >= snapshot) & (post["ts"] < end)].copy()
    post["day"] = (post["ts"] - snapshot).dt.days
    ok = post[post["status"].str.upper() == "SUCCESS"]
    ref = post[post["status"].str.upper() == "REFUNDED"]

    cust = assign.sort_values("total_ev", ascending=False).drop_duplicates("customer_id").copy()
    cust["validity_days"] = cust["validity_days"].fillna(10).astype(int)
    cust["pre_gmv_at_merchant"] = cust["monthly_gmv_at_merchant"] * cfg["features"]["lookback_days"] / 30.0
    keys = ["customer_id", "merchant_id"]
    at = ok.merge(cust[keys + ["validity_days"]], on=keys)
    at["returned_30"] = at["day"] < short
    at["returned_in_offer"] = at["day"] < at["validity_days"]
    at["returned_post_offer"] = at["day"] >= at["validity_days"]   # still coming once the cashback is gone
    for k in (1, 2, 3):
        at[f"active_m{k}"] = (at["day"] >= 30 * (k - 1)) & (at["day"] < 30 * k)
    flag_cols = ["returned_30", "returned_in_offer", "returned_post_offer", "active_m1", "active_m2", "active_m3"]
    flags = at.groupby("customer_id")[flag_cols].any()
    cust = cust.set_index("customer_id")
    cust[flag_cols] = flags.reindex(cust.index, fill_value=False).astype(bool)
    ref_at = ref.merge(cust.reset_index()[keys], on=keys)
    cust["gmv_at_merchant_90"] = (at.groupby("customer_id")["amount"].sum().reindex(cust.index).fillna(0.0)
                                  - ref_at.groupby("customer_id")["amount"].sum().reindex(cust.index).fillna(0.0))
    cust["active_platform_30"] = cust.index.isin(set(ok.loc[ok["day"] < short, "customer_id"]))
    cust["gmv_platform_90"] = (ok.groupby("customer_id")["amount"].sum().reindex(cust.index).fillna(0.0)
                               - ref.groupby("customer_id")["amount"].sum().reindex(cust.index).fillna(0.0))
    for c in ("returned_30", "returned_in_offer", "returned_post_offer", "active_m1", "active_m2", "active_m3",
              "active_platform_30"):
        cust[c] = cust[c].astype(int)
    cust["is_treat"] = (cust["group"] == "treatment").astype(int)
    return cust.reset_index()


def measure_campaign(assign: pd.DataFrame, post: pd.DataFrame, events: pd.DataFrame, ledger,
                     merchants: pd.DataFrame, cfg: dict, snapshot: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    days = cfg["campaign"]["measurement_days"]
    cust = customer_outcomes(assign, post, snapshot, cfg)
    tc = cust[cust["group"].isin(["treatment", "control"])]
    res: dict = {"measurement_window": [str(snapshot.date()), str((snapshot + pd.Timedelta(days=days - 1)).date())],
                 "groups": cust["group"].value_counts().to_dict()}
    if tc["is_treat"].sum() == 0 or (1 - tc["is_treat"]).sum() == 0:
        res["warning"] = "Need both treatment and control customers to measure uplift."
        return res, cust

    res["itt"] = {
        "returned_30": effect(tc, "returned_30"),
        "returned_post_offer": effect(tc, "returned_post_offer"),
        "gmv_at_merchant_90": effect(tc, "gmv_at_merchant_90"),
        "active_platform_30": effect(tc, "active_platform_30"),
        "gmv_platform_90": effect(tc, "gmv_platform_90"),
    }
    # approved shops only (randomised within shop; approval is a shop-level decision)
    shop_rows = assign[(assign["group"] == "treatment") & assign["needs_merchant_approval"].astype(bool)]
    shop_dec = shop_rows.groupby("merchant_id")["merchant_decision"].agg(lambda s: (s == "approved").any())
    approved_shops = set(shop_dec[shop_dec].index)
    shop_tc_all = tc[tc["needs_merchant_approval"].astype(bool)]
    ap = shop_tc_all[shop_tc_all["merchant_id"].isin(approved_shops)]
    res["approved_shops"] = {"shops": len(approved_shops), "shops_asked": int(shop_dec.size),
                             "returned_30": effect(ap, "returned_30"),
                             "returned_post_offer": effect(ap, "returned_post_offer"),
                             "gmv_at_merchant_90": effect(ap, "gmv_at_merchant_90")}
    res["by_route"] = {r: effect(g, "returned_30") for r, g in tc.groupby("route")}
    res["by_risk_tier"] = {r: effect(g, "returned_30") for r, g in tc.groupby("risk_tier")}
    ctrl = tc[tc["is_treat"] == 0]
    res["by_offer_type"] = {}
    for t, g in tc[tc["is_treat"] == 1].groupby("offer_type"):
        res["by_offer_type"][t] = effect(pd.concat([g, ctrl]), "returned_post_offer")
    gh = cust[cust["group"] == "global_holdout"]
    if len(gh):
        sanity = pd.concat([ctrl, gh.assign(is_treat=1)])   # control vs global holdout: should be ~0
        res["global_holdout"] = {"customers": int(len(gh)),
                                 "control_vs_global_holdout_returned_30": effect(sanity, "returned_30"),
                                 "note": "both got no offer; a big gap would mean the randomisation is broken"}

    T = tc[tc["is_treat"] == 1]
    base = float(ctrl["returned_30"].mean())
    res["power"] = power(base, len(T), len(ctrl), cfg["campaign"]["target_uplift_pts"])

    # ---------------- money (90 days) ----------------
    st = ledger.stats()
    e = ledger.frame()
    shop_offers = set(assign.loc[assign["needs_merchant_approval"].astype(bool), "offer_id"])
    m_cash = float(e.loc[e["offer_id"].isin(shop_offers), "merchant_share"].sum())
    p_cash = float(e["paytm_share"].sum())
    gm = res["itt"]["gmv_at_merchant_90"]
    gp = res["itt"]["gmv_platform_90"]
    shop_T = T[T["needs_merchant_approval"].astype(bool)]
    shop_tc = tc[tc["needs_merchant_approval"].astype(bool)]
    gm_shop = effect(shop_tc, "gmv_at_merchant_90") if len(shop_tc) else None
    margins = cfg["offers"]["category_rules"]
    w = shop_T.assign(margin=shop_T["category"].map(lambda c: margins.get(c, {"margin": 0.15})["margin"]))
    blended = float((w["pre_gmv_at_merchant"] * w["margin"]).sum() / max(w["pre_gmv_at_merchant"].sum(), 1))
    incr_shop_gmv = (gm_shop["effect"] if gm_shop else 0.0) * len(shop_T)
    act = res["itt"]["active_platform_30"]
    users_kept = (act["effect"] / 100.0 if act else 0.0) * len(T)
    months = days / 30.0
    econ = cfg["economics"]
    res["economics"] = {
        "offers_delivered": int(T["delivered"].fillna(False).astype(bool).sum()),
        "customers_paid_cashback": st["customers_paid_cashback"],
        "redemption_rate_of_delivered": round(st["customers_paid_cashback"] / max(int(T["delivered"].fillna(False).astype(bool).sum()), 1), 4),
        "cashback_total": st["cashback_total"],
        "merchant_funded_cashback": round(m_cash, 2),
        "paytm_funded_cashback": round(p_cash, 2),
        "reversed_on_refund": st["reversed_amount"],
        "blocked_at_checkout": st["blocked"],
        "incremental_merchant_gmv_90d": round(incr_shop_gmv, 2),
        "incremental_gmv_per_treated_customer": gm["effect"] if gm else None,
        "incremental_platform_gmv_90d": round((gp["effect"] if gp else 0.0) * len(T), 2),
        "merchant_blended_margin": round(blended, 4),
        "merchant_profit_after_cashback": round(incr_shop_gmv * blended - m_cash, 2),
        "merchant_roi": round((incr_shop_gmv * blended - m_cash) / m_cash, 2) if m_cash else None,
        "paytm_value": round((gp["effect"] if gp else 0.0) * len(T) * econ["paytm_value_per_gmv"]
                             + users_kept * econ["paytm_value_per_user_month"] * months - p_cash, 2),
        "users_kept_active": round(users_kept, 1),
        "customer_cashback_received": st["cashback_total"],
    }

    # ---------------- guardrail inputs ----------------
    delivered = max(res["economics"]["offers_delivered"], 1)
    ev = events if events is not None else pd.DataFrame(columns=["customer_id", "event", "ts"])
    res["safety"] = {
        "opt_outs": int((ev["event"] == "opt_out").sum()),
        "complaints": int((ev["event"] == "complaint").sum()),
        "opt_out_rate": round(float((ev["event"] == "opt_out").sum()) / delivered, 4),
        "complaint_rate": round(float((ev["event"] == "complaint").sum()) / delivered, 4),
        "approval_rate": round(len(approved_shops) / max(shop_dec.size, 1), 4) if shop_dec.size else None,
        "merchant_budget_burn": round(m_cash / max(float(assign.loc[(assign["group"] == "treatment")
                                                                    & assign["delivered"].fillna(False).astype(bool),
                                                                    "max_merchant_liability"].sum()), 1), 4),
        "paytm_budget_burn": round(p_cash / cfg["campaign"]["paytm_monthly_budget"], 4),
    }

    # ---------------- learned assumptions (fed back into the next campaign) ----------------
    res["learned"] = learn_assumptions(tc, res, cfg)

    # ---------------- merchant scoreboard + pooled proof ----------------
    netted = e.groupby("merchant_id")["merchant_share"].sum()
    board = (shop_T.groupby(["merchant_id", "merchant_name", "category"])
             .agg(offers=("offer_id", "size"), delivered=("delivered", "sum"), came_back=("returned_30", "sum"),
                  sales_from_these_customers=("gmv_at_merchant_90", "sum"))
             .reset_index())
    board["cashback_netted"] = board["merchant_id"].map(netted).fillna(0.0).round(2)
    board = board.sort_values("sales_from_these_customers", ascending=False)
    res["merchant_scoreboard"] = board.round(2).to_dict(orient="records")
    res["pooled_by_category"] = {c: effect(g, "returned_30") for c, g in shop_tc.groupby("category")}
    return res, cust


def learn_assumptions(tc: pd.DataFrame, res: dict, cfg: dict) -> dict:
    """Turn measured effects into the numbers the offer engine assumes.

    save rate  = effect per delivered offer (CACE) / average P(lapse)
    curve      = uplift in month k / uplift in month 1 (how much of the win sticks)

    A small pilot gives a noisy number, so the measured value is shrunk toward the prior:
        used = w x measured + (1 - w) x prior,   w = control size / (control size + control size needed)
    One small run barely moves the assumption; a properly powered pilot replaces it."""
    econ = cfg["economics"]
    T = tc[tc["is_treat"] == 1]
    n_c = int((tc["is_treat"] == 0).sum())
    need = max(res.get("power", {}).get("needed_control", 1000), 1)
    w = n_c / (n_c + need)
    deliv = float(T["delivered"].fillna(False).astype(bool).mean()) if len(T) else 0.0
    itt = res["itt"]["returned_30"]
    mean_p = float(T["lapse_prob"].mean()) if len(T) else 0.0
    out = {"delivered_share": round(deliv, 4), "mean_lapse_prob": round(mean_p, 4), "evidence_weight": round(w, 3)}
    prior = econ["assumed_save_rate"]
    if itt and deliv > 0 and mean_p > 0:
        measured = float(np.clip(itt["effect"] / 100.0 / deliv / mean_p, 0, 1))
        out["save_rate_measured"] = round(measured, 4)
        out["save_rate"] = round(w * measured + (1 - w) * prior, 4)
    by_type = {}
    for t, e in res.get("by_offer_type", {}).items():
        g = T[T["offer_type"] == t]
        if e and len(g) >= 50 and g["lapse_prob"].mean() > 0:
            m = float(np.clip(e["effect"] / 100.0 / max(deliv, 1e-9) / g["lapse_prob"].mean(), 0, 1))
            wt = len(g) / (len(g) + need)
            by_type[t] = round(wt * m + (1 - wt) * out.get("save_rate", prior), 4)
    out["save_rate_by_type"] = by_type
    m = [effect(tc, f"active_m{k}", covariate=None) for k in (1, 2, 3)]
    prior_curve = econ["assumed_retention_curve"]
    if all(m) and m[0]["effect"] > 0:
        meas = [1.0] + [float(np.clip(x["effect"] / m[0]["effect"], 0, 1.5)) for x in m[1:]]
        out["retention_curve_measured"] = [round(x, 3) for x in meas]
        out["retention_curve"] = [round(w * a + (1 - w) * b, 3) for a, b in zip(meas, prior_curve)]
    return out


def pooled_readout(cfg: dict) -> dict | None:
    """The pilot so far: every measured run still inside the retention window, pooled.
    This is how a pilot accumulates enough customers to see a real effect."""
    from pathlib import Path
    frames = []
    for run in sorted(Path(cfg["paths"]["output_dir"]).glob("campaign_*")):
        f = run / "09_campaign_outcomes.csv"
        if f.exists():
            frames.append(pd.read_csv(f).assign(run=run.name))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df[df["group"].isin(["treatment", "control"])]
    if df.empty:
        return None
    base = float(df.loc[df["is_treat"] == 0, "returned_30"].mean())
    return {"runs": int(df["run"].nunique()), "returned_30": effect(df, "returned_30"),
            "returned_post_offer": effect(df, "returned_post_offer"),
            "power": power(base, int(df["is_treat"].sum()), int((df["is_treat"] == 0).sum()),
                           cfg["campaign"]["target_uplift_pts"])}
