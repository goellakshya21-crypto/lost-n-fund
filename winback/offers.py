"""Offer engine: who is eligible, what to offer, who pays, and whether it pays for itself.

Offers are CASHBACK, not a price cut. The customer pays the full bill at the counter (the Soundbox
announces the full amount, nothing changes for the shopkeeper), Paytm credits the cashback, and the
shop's share is netted from its settlement as one weekly line.

Three offer types:
  flat    "Rs 20 cashback on your next payment of Rs 200+"
  streak  "pay on 3 different days within 14 days, get Rs 45 cashback"   (rebuilds the habit;
          only for customers who used to come at least 3 times a month)
  pct     "5% cashback (up to Rs 40) on your next 2 payments"           (kept as an exploration arm)

Expected cost uses each customer's own rhythm at the shop: a loyal customer who visits twice a
week will complete a 3-visit streak anyway, and that cashback is pure cost. Expected gain uses the
calibrated P(lapse at this shop) x save rate (or the uplift model once there is enough data) x the
months of spend a saved customer keeps.

An offer is only made if it pays for itself for whoever funds it:
  shop-funded:  merchant EV > 0
  co-funded:    merchant EV > 0, total EV > 0, and Paytm's expected cost <= its per-offer subsidy cap
  Paytm-funded: Paytm EV > 0 (value of keeping a user on the app + GMV value)
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import poisson


def _u(*parts) -> float:
    """Deterministic uniform in [0, 1) from a key: reproducible randomisation, no stored seed needed."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:12]
    return int(h, 16) / float(16 ** 12)


def _round5(x: float) -> float:
    return float(max(5, 5 * round(x / 5)))


# ------------------------------------------------------------------ assumptions (config or learned)
def load_assumptions(cfg: dict) -> dict:
    econ = cfg["economics"]
    out = {"save_rate": econ["assumed_save_rate"], "save_rate_by_type": {},
           "retention_curve": list(econ["assumed_retention_curve"]), "source": "config assumptions"}
    p = Path(cfg["paths"]["state_dir"]) / "learned_assumptions.json"
    if econ.get("use_learned_assumptions") and p.exists():
        learned = json.loads(p.read_text(encoding="utf-8"))
        if learned.get("simulated") and not econ.get("allow_simulated_learning"):
            out["source"] = "config assumptions (learned values exist but come from a simulated run)"
            return out
        if learned.get("save_rate") is not None:
            out["save_rate"] = float(np.clip(learned["save_rate"], 0.0, 1.0))
        out["save_rate_by_type"] = learned.get("save_rate_by_type", {})
        if learned.get("retention_curve"):
            out["retention_curve"] = learned["retention_curve"]
        out["source"] = f"measured in {learned.get('source_run')}"
    return out


# ------------------------------------------------------------------ eligibility (anti-abuse, caps)
def eligibility(cands: pd.DataFrame, history: pd.DataFrame, cfg: dict, snapshot: pd.Timestamp,
                run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applied BEFORE randomisation, so the holdout and the shop counts only contain people who
    could actually receive an offer."""
    ab, camp = cfg["abuse"], cfg["campaign"]
    df = cands.copy()
    prior = history[history["run_id"] != run_id] if len(history) else history
    win = prior[prior["sent_on"] >= snapshot - pd.Timedelta(days=cfg["compliance"]["abuse_history_days"])]
    n_prior = win.groupby("customer_id").size()
    recent = set(prior.loc[prior["sent_on"] >= snapshot - pd.Timedelta(days=camp["frequency_cap_days"]), "customer_id"])
    df["prior_offers"] = df["customer_id"].map(n_prior).fillna(0).astype(int)
    reason = np.select(
        [df["tenure_days"] < ab["min_platform_tenure_days"],
         df["first_visit_days"] < ab["min_pair_tenure_days"],
         df["customer_id"].isin(recent),
         df["prior_offers"] >= ab["max_offers_per_window"]],
        ["new account (anti-abuse)", "new at this shop (anti-abuse)",
         f"got an offer in the last {camp['frequency_cap_days']} days",
         f"already had {ab['max_offers_per_window']} offers in {cfg['compliance']['abuse_history_days']} days"],
        "")
    df["value_mult"] = np.where(df["prior_offers"] >= 1, ab["repeat_offer_value_mult"], 1.0)
    dropped = df[reason != ""].assign(drop_reason=reason[reason != ""])
    return df[reason == ""].reset_index(drop=True), dropped


# ------------------------------------------------------------------ design
def _q_bill(mu: float, sd: float, min_bill: float, topup: float) -> float:
    """Share of their payments at this shop that would qualify (bill >= min), incl. near-miss top-ups."""
    z = (math.log(max(min_bill, 1)) - mu) / sd
    z_near = (math.log(max(0.7 * min_bill, 1)) - mu) / sd
    above = 1 - ndtr(z)
    near = ndtr(z) - ndtr(z_near)
    return float(np.clip(above + topup * near, 0.0, 1.0))


def _economics(r, otype: str, level: float, rules: dict, cfg: dict, assume: dict, recovered_prob: float) -> dict:
    oc, econ = cfg["offers"], cfg["economics"]
    lookback = cfg["features"]["lookback_days"]
    pct = rules["max_pct"] * level / 100.0
    ref_bill = max(r.avg_ticket_at_merchant, r.min_bill)
    rate = r.p_visit_days / lookback                       # visit days per day at this shop
    q = _q_bill(r.log_amt_mu, r.log_amt_sd, r.min_bill, econ["bill_topup_rate"])
    lam = rate * r.validity_days * q                       # qualifying visits expected if they behave as usual
    p = r.lapse_prob
    if otype == "flat":
        reward = _round5(min(max(pct * ref_bill, 5), rules["cap"]) * r.value_mult)
        need, max_liab = 1, reward
        redeem_if_stay = poisson.sf(0, lam)
        exp_cost = reward * ((1 - p) * redeem_if_stay + recovered_prob)
        terms = {"cashback": reward, "pct": None, "cap": None, "uses": 1, "streak_visits": None}
    elif otype == "streak":
        need = oc["streak_visits"]
        reward = _round5(min(max(oc["streak_reward_mult"] * pct * ref_bill, 10), 2 * rules["cap"]) * r.value_mult)
        max_liab = reward
        redeem_if_stay = poisson.sf(need - 1, lam)
        exp_cost = reward * ((1 - p) * redeem_if_stay + recovered_prob)
        terms = {"cashback": reward, "pct": None, "cap": None, "uses": 1, "streak_visits": need}
    else:  # pct
        uses = oc["pct_uses"]
        per_use = min(pct * ref_bill, rules["cap"]) * r.value_mult
        exp_uses = sum(poisson.sf(j - 1, lam) for j in range(1, uses + 1))
        exp_cost = per_use * ((1 - p) * exp_uses + recovered_prob * uses)
        max_liab = rules["cap"] * r.value_mult * uses
        redeem_if_stay = poisson.sf(0, lam)
        terms = {"cashback": None, "pct": round(100 * pct, 1), "cap": round(rules["cap"] * r.value_mult, 0),
                 "uses": uses, "streak_visits": None}
    curve = float(sum(assume["retention_curve"]))
    recovered_gmv = recovered_prob * r.monthly_gmv_at_merchant * curve
    merchant_gain = recovered_gmv * rules["margin"]
    share = 0.0 if pd.isna(r.route_paytm_share) else float(r.route_paytm_share)
    if r.route == "app_switch":
        platform_value = recovered_prob * (r.customer_monthly_gmv * econ["paytm_value_per_gmv"]
                                           + econ["paytm_value_per_user_month"]) * curve
    else:
        platform_value = recovered_gmv * econ["paytm_value_per_gmv"]
    merchant_cost, paytm_cost = exp_cost * (1 - share), exp_cost * share
    merchant_ev = merchant_gain - merchant_cost
    paytm_ev = platform_value - paytm_cost
    total_ev = merchant_gain + platform_value - exp_cost
    if r.route_action == "paytm_offer":
        ok = paytm_ev > 0
    elif share > 0:
        ok = merchant_ev > 0 and total_ev > 0 and paytm_cost <= cfg["offers"]["paytm_max_subsidy_per_offer"]
    else:
        ok = merchant_ev > 0
    return {
        "offer_type": otype, "level": level, **terms,
        "p_redeem_if_stays": round(float(redeem_if_stay), 4),
        "expected_cost": round(float(exp_cost), 2),
        "max_liability": round(float(max_liab), 2),
        "max_merchant_liability": round(float(max_liab * (1 - share)), 2),
        "max_paytm_liability": round(float(max_liab * share), 2),
        "expected_recovered_gmv": round(float(recovered_gmv), 2),
        "expected_merchant_gain": round(float(merchant_gain), 2),
        "merchant_ev": round(float(merchant_ev), 2),
        "paytm_ev": round(float(paytm_ev), 2),
        "total_ev": round(float(total_ev), 2),
        "recovered_prob": round(float(recovered_prob), 4),
        "funder": "paytm" if share >= 1 else ("cofunded" if share > 0 else "merchant"),
        "paytm_share": share,
        "ok": bool(ok),
    }


def design_offers(cands: pd.DataFrame, cfg: dict, assume: dict, run_id: str,
                  uplift: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    oc = cfg["offers"]
    if cands.empty:
        return cands.assign(drop_reason=pd.Series(dtype=str)), cands.assign(drop_reason=pd.Series(dtype=str))
    df = cands.copy()
    rules_all = oc["category_rules"]
    q = df.groupby("category")["monthly_gmv_at_merchant"].transform(lambda s: s.quantile(oc["high_value_quantile"]))
    df["value_tier"] = np.where(df["monthly_gmv_at_merchant"] >= q, "high_value", "low_value")
    min_cat = df["category"].map(lambda c: rules_all.get(c, {}).get("min_bill", np.nan))
    df["min_bill"] = np.maximum(min_cat, (oc["min_bill_ticket_share"] * df["avg_ticket_at_merchant"] / 10).round() * 10)
    df["validity_days"] = np.clip((oc["validity_gap_mult"] * df["p_avg_gap"]).round(),
                                  oc["validity_min_days"], oc["validity_max_days"]).astype(int)

    kept, dropped = [], []
    for i, r in enumerate(df.itertuples(index=False)):
        rules = rules_all.get(r.category)
        base = r._asdict()
        if rules is None:
            dropped.append({**base, "drop_reason": f"no offer rules for category '{r.category}'"})
            continue
        start = oc["level_matrix"].get(r.risk_tier, oc["level_matrix"]["medium"])[r.value_tier]
        policy_type = "streak" if r.visits_per_month >= oc["streak_min_visits_per_month"] else "flat"
        types = [policy_type] + [t for t in oc["types"] if t != policy_type]
        if r.visits_per_month < oc["streak_min_visits_per_month"]:
            types = [t for t in types if t != "streak"]
        explore = _u(run_id, r.customer_id, "explore") < oc["exploration_share"]
        cand = []
        for t in types:
            # persuadable share: the uplift model's estimate if trained, else P(lapse) x save rate
            if uplift is not None:
                rp = float(np.clip(uplift[i], 0, 1))
            else:
                rp = r.lapse_prob * float(assume["save_rate_by_type"].get(t, assume["save_rate"]))
            levels = oc["level_ladder"] if explore else [lv for lv in oc["level_ladder"] if lv <= start + 1e-9]
            if t == "pct":   # "1.5% cashback" is too small to notice; % offers start at half the category max
                levels = [lv for lv in levels if lv >= 0.5]
            for lv in levels:
                cand.append(_economics(r, t, lv, rules, cfg, assume, rp))
        good = [c for c in cand if c["ok"]]
        if not good:
            dropped.append({**base, "drop_reason": "no offer pays for itself for whoever funds it"})
            continue
        if explore:
            chosen = good[int(_u(run_id, r.customer_id, "arm") * len(good))]
            chosen["assigned_by"] = "explore"
        else:
            chosen = good[0]   # policy type first, most generous profitable level first
            chosen["assigned_by"] = "policy" if chosen["offer_type"] == policy_type else "policy_fallback"
        kept.append({**base, **chosen})
    off = pd.DataFrame(kept)
    if len(off):
        off = off.drop(columns="ok")
    return off, pd.DataFrame(dropped)


def one_per_customer(offers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if offers.empty:
        return offers, offers
    o = offers.sort_values(["customer_id", "total_ev"], ascending=[True, False])
    dup = o.duplicated("customer_id")
    return o[~dup].reset_index(drop=True), o[dup].assign(drop_reason="better offer at another shop")


# ------------------------------------------------------------------ holdouts, budgets, anonymity
def quarter_salt(snapshot: pd.Timestamp) -> str:
    return f"{snapshot.year}Q{(snapshot.month - 1) // 3 + 1}"


def in_global_holdout(customer_ids: pd.Series, cfg: dict, snapshot: pd.Timestamp) -> np.ndarray:
    """~5% of customers get no WinBack offer for the whole quarter (programme-level control).
    Hash-based, so the same customers stay out on every run in the quarter."""
    salt = quarter_salt(snapshot)
    share = cfg["campaign"]["global_holdout_share"]
    return np.array([_u("global", salt, c) < share for c in customer_ids], dtype=bool)


def committed_this_month(history: pd.DataFrame, snapshot: pd.Timestamp, run_id: str) -> tuple[pd.Series, float]:
    if history.empty or "max_merchant_liability" not in history.columns:
        return pd.Series(dtype=float), 0.0
    h = history[(history["run_id"] != run_id) & (history["sent_on"].dt.to_period("M") == snapshot.to_period("M"))]
    return h.groupby("merchant_id")["max_merchant_liability"].sum(), float(h["max_paytm_liability"].sum())


def apply_budgets(offers: pd.DataFrame, merchants: pd.DataFrame, cfg: dict, committed: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Worst-case liability must fit each shop's monthly budget (its own share only) and Paytm's
    programme budget (Paytm's share). Best offers first."""
    if offers.empty:
        return offers, offers
    m_committed, p_committed = committed
    df = offers.merge(merchants[["merchant_id", "monthly_offer_budget"]], on="merchant_id", how="left")
    df = df.sort_values("total_ev", ascending=False)
    left_m = (df["monthly_offer_budget"].fillna(cfg["campaign"]["merchant_monthly_budget_default"])
              - df["merchant_id"].map(m_committed).fillna(0))
    df["cum_m"] = df.groupby("merchant_id")["max_merchant_liability"].cumsum()
    df["n_at_m"] = df.groupby("merchant_id").cumcount() + 1
    df["cum_p"] = df["max_paytm_liability"].cumsum()
    ok_m = (df["max_merchant_liability"] == 0) | (df["cum_m"] <= left_m)
    ok_n = ~df["needs_merchant_approval"] | (df["n_at_m"] <= cfg["matching"]["max_offers_per_merchant"])
    ok_p = df["cum_p"] <= cfg["campaign"]["paytm_monthly_budget"] - p_committed
    ok = ok_m & ok_n & ok_p
    reason = np.select([~ok_m, ~ok_n, ~ok_p], ["shop's monthly budget reached", "max offers per shop reached",
                                              "Paytm programme budget reached"], "")
    over = df[~ok].assign(drop_reason=reason[~ok.to_numpy()])
    return df[ok].drop(columns=["cum_m", "n_at_m", "cum_p"]).reset_index(drop=True), over


def apply_min_group(offers: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A shop is only alerted about enough customers that it can't guess who they are. Only applies
    to offers the shop approves; Paytm-funded offers never reach the shop."""
    k = cfg["compliance"]["min_customers_per_alert"]
    if offers.empty or k <= 1:
        return offers, offers.iloc[0:0]
    need = math.ceil(k / (1 - cfg["campaign"]["holdout_share"]))
    shop = offers["needs_merchant_approval"].astype(bool)
    n = offers[shop].groupby("merchant_id")["customer_id"].transform("nunique").reindex(offers.index).fillna(need)
    ok = ~shop | (n >= need)
    small = offers[~ok].assign(drop_reason=f"fewer than {need} customers at this shop (held for a later run)")
    return offers[ok].reset_index(drop=True), small


def assign_holdout(offers: pd.DataFrame, cfg: dict, run_id: str) -> pd.DataFrame:
    """Randomly keep some customers as this run's control group, stratified by shop (and by who
    funds the offer), so every alerted shop stays above the anonymity threshold. Hash-based order:
    reproducible and auditable."""
    share = cfg["campaign"]["holdout_share"]
    df = offers.copy()
    df["_u"] = [_u(run_id, c, "holdout") for c in df["customer_id"]]
    df["_rank"] = df.groupby(["merchant_id", "needs_merchant_approval"])["_u"].rank(method="first") - 1
    n = df.groupby(["merchant_id", "needs_merchant_approval"])["customer_id"].transform("size")
    df["group"] = np.where(df["_rank"] < (share * n).round(), "control", "treatment")
    df = df.drop(columns=["_u", "_rank"])
    df.insert(0, "offer_id", ["WB-" + hashlib.sha1(f"{run_id}|{c}|{m}".encode()).hexdigest()[:10].upper()
                              for c, m in zip(df["customer_id"], df["merchant_id"])])
    return df
