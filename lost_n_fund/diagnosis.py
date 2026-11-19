"""Why is this customer fading at this shop, and who should pay to fix it?

A merchant-funded discount is the right tool for exactly one cause: the customer moved to
another shop. For every other cause it wastes the shop's money:

  cause              signal                                               action           who pays
  merchant_decline   the shop's own footfall down / Soundbox offline     ops alert        nobody (no offer)
  seasonal           paying in places they never used / shops in their   wait             nobody (no offer)
                     locality quiet (festival travel)
  payment_friction   recent payments failing                             fix routing      Paytm (no offer)
  normal_gap         visits barely down, gap within their normal range   wait             nobody (no offer)
  gone               quiet everywhere, bills / P2P gone too              skip             nobody
  app_switch         scan-and-pay down everywhere, bills/P2P steady      Paytm cashback   Paytm (shop not asked)
  shop_switch        the spend lost here shows up at another shop of     win-back offer   the shop
                     the same kind (or quick commerce, for kiranas)
  unclear            only this shop dropped, no replacement seen         win-back offer   shop + Paytm co-fund
                     (often cash: the shop may lose nothing)

Rules are deliberately simple and auditable; thresholds live in config["diagnosis"].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REASON_LABELS = {
    "merchant_decline": "Shop itself is declining",
    "seasonal": "Travelling / whole area quiet",
    "payment_friction": "Payments failing",
    "normal_gap": "Normal gap for them",
    "gone": "Gone quiet everywhere",
    "app_switch": "Paying with another app",
    "shop_switch": "Moved to another shop",
    "unclear": "Dropped at this shop only (cash? offline rival?)",
}
ACTION_LABELS = {
    "merchant_offer": "Win-back offer, shop pays",
    "paytm_offer": "Cashback, Paytm pays (shop not asked)",
    "paytm_fix": "Fix payments (no offer)",
    "ops_alert": "Alert merchant ops (no offer)",
    "wait": "Wait and recheck (no offer)",
    "skip": "No action",
}


def _pct(x: float) -> str:
    return f"{100 * (x - 1):+.0f}%"


def diagnose(pairs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Adds route, route_action, route_paytm_share and diagnosis_evidence to each pair."""
    dc = cfg["diagnosis"]
    df = pairs.copy()
    has_md = df["m_all_trend"].notna()
    has_act = df["act_ratio"].notna()
    shop_trend = np.where(has_md, df["m_all_trend"], df["m_trend"])

    decline = (shop_trend < 1 - dc["merchant_decline"]) | (df["m_offline_14"].fillna(0) >= dc["soundbox_offline_days"])
    seasonal = (df["c_away_share_14"] >= dc["away_share"]) | (df["loc_ratio"] < 1 - dc["locality_dip"])
    friction = (df["fail_rate_30"] >= dc["friction_fail_rate"]) & (df["failures_30"] >= dc["friction_min_failures"])
    normal_gap = (((df["p_trend"] >= dc["normal_gap_trend"]) & (df["p_overdue"] < dc["normal_gap_overdue"]))
                  | (df["visits_per_month"] < dc["habit_visits_per_month"]))
    elsewhere_low = df["c_other_trend"] < dc["elsewhere_drop"]
    app_low = np.fmin(df["act_ratio"], df["opens_ratio"]) < dc["app_activity_gone"]
    # without app-activity data "gone" and "another app" look the same; only call it gone when it is stark
    gone = np.where(has_act, elsewhere_low & app_low, (df["c_other_trend"] < 0.4) & (df["recency_days"] > 14))
    app_switch = has_act & elsewhere_low & ~app_low
    shop_switch = (df["c_same_cat_shift"] >= dc["same_cat_shift"]) | (df["c_new_same_cat_30"] > 0)

    df["route"] = np.select(
        [decline, seasonal, friction, normal_gap, gone, app_switch, shop_switch],
        ["merchant_decline", "seasonal", "payment_friction", "normal_gap", "gone", "app_switch", "shop_switch"],
        "unclear")

    ev = []
    for r, st in zip(df.itertuples(index=False), shop_trend):
        e = []
        if r.route == "merchant_decline":
            e.append(f"shop's total payments {_pct(st)} vs usual")
            if r.m_offline_14 == r.m_offline_14 and r.m_offline_14 > 0:
                e.append(f"Soundbox offline {r.m_offline_14:.0f} of last 14 days")
        elif r.route == "seasonal":
            if r.c_away_share_14 >= 0.5:
                e.append(f"{r.c_away_share_14:.0%} of recent payments in places they never used before")
            e.append(f"shops in {r.customer_locality} {_pct(r.loc_ratio)} in the last 2 weeks")
        elif r.route == "payment_friction":
            e.append(f"{r.fail_rate_30:.0%} of payments failed in 30 days ({r.failures_30:.0f} failures)")
        elif r.route == "normal_gap":
            if r.visits_per_month < 2:
                e.append(f"occasional visitor (~{r.visits_per_month:.1f}/month): a month away is normal for them")
            else:
                e.append(f"visits here {_pct(r.p_trend)}, {r.p_overdue:.1f}x their usual gap: still within normal")
        else:
            e.append(f"payments at other shops {_pct(r.c_other_trend)}")
            if r.act_ratio == r.act_ratio:
                e.append(f"bills/P2P on the app {_pct(r.act_ratio)}")
            if r.route == "app_switch" and r.m_share_trend == r.m_share_trend and r.m_share_trend < 0.9:
                e.append(f"our app's share of this shop's payments {_pct(r.m_share_trend)}")
            if r.c_same_cat_shift >= 0.3:
                e.append(f"~{min(r.c_same_cat_shift, 1):.0%} of their usual spend here now goes to another "
                         f"{'kirana or quick-commerce app' if r.category == 'kirana' else r.category}")
            elif r.c_new_same_cat_30 > 0:
                e.append(f"started at {r.c_new_same_cat_30:.0f} new {r.category} shop(s)")
            if r.c_qc_trend > 1.5 and r.c_qc_share_30 >= 0.05:
                e.append(f"quick-commerce now {r.c_qc_share_30:.0%} of spend")
        ev.append("; ".join(e))
    df["diagnosis_evidence"] = ev
    routing = cfg["routing"]
    df["route_action"] = df["route"].map(lambda x: routing[x]["action"])
    df["route_paytm_share"] = df["route"].map(lambda x: routing[x].get("paytm_share", np.nan))
    return df


# ------------------------------------------------------------------ demo only: check against ground truth
def truth_check(at_risk: pd.DataFrame, data_dir: Path, snapshot: pd.Timestamp) -> dict | None:
    """Compare the diagnosis with the simulator's hidden truth. Only possible on synthetic data."""
    data_dir = Path(data_dir)
    if not (data_dir / "_ground_truth.csv").exists() or at_risk.empty:
        return None
    meta = json.loads((data_dir / "_meta.json").read_text(encoding="utf-8"))
    day = (snapshot.normalize() - pd.Timestamp(meta["start_date"])).days
    truth = pd.read_csv(data_dir / "_ground_truth.csv").set_index("customer_id")
    mt = pd.read_csv(data_dir / "_merchant_truth.csv").set_index("merchant_id")
    t = truth.loc[at_risk["customer_id"]]
    prim = t["primary_merchant"].to_numpy() == at_risk["merchant_id"].to_numpy()
    typ = t["type"].to_numpy()
    started = t["onset_day"].to_numpy() <= day + 10
    away = (t["travel_start"].to_numpy() <= day) & (t["travel_end"].to_numpy() > day - 3)
    declining = mt["decline_onset_day"].reindex(at_risk["merchant_id"]).to_numpy() <= day
    affected = np.where(np.isin(typ, ["shop_switch", "cash_shift"]), prim,
                        np.isin(typ, ["app_switch", "gone", "friction"]))
    mapping = {"shop_switch": "shop_switch", "cash_shift": "unclear", "app_switch": "app_switch",
               "gone": "gone", "friction": "payment_friction"}
    expected = np.select(
        [declining & prim, away, affected & started],
        ["merchant_decline", "seasonal", pd.Series(typ).map(mapping).fillna("not_leaving").to_numpy()],
        "not_leaving")
    got = at_risk["route"].to_numpy()
    leaving = expected != "not_leaving"
    conf = pd.crosstab(pd.Series(expected, name="truth"), pd.Series(got, name="diagnosed"))
    per = {k: round(float((got[expected == k] == k).mean()), 3) for k in np.unique(expected) if k != "not_leaving"}
    # the decision that matters: does the shop end up paying for the right people?
    shop_pays = np.isin(got, ["shop_switch", "unclear"])
    should_pay = np.isin(expected, ["shop_switch", "unclear"])
    return {
        "pairs_checked": int(len(at_risk)),
        "truly_leaving_share": round(float(leaving.mean()), 3),
        "accuracy_on_leaving": round(float((got[leaving] == expected[leaving]).mean()), 3) if leaving.any() else None,
        "recall_by_cause": per,
        "shop_funded_precision": round(float(should_pay[shop_pays].mean()), 3) if shop_pays.any() else None,
        "shop_funded_recall": round(float(shop_pays[should_pay].mean()), 3) if should_pay.any() else None,
        "confusion": {r: {c: int(v) for c, v in row.items()} for r, row in conf.iterrows()},
    }
