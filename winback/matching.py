"""Turn diagnosed at-risk pairs into offer candidates and action queues.

Only two routes lead to an offer:
  merchant_offer  the shop funds it (fully, or co-funded with Paytm). The shop must be enrolled and
                  use our QR/Soundbox, because its share is netted from its settlement.
  paytm_offer     Paytm funds it (customer is paying with another app). The shop is never asked or
                  told, so any regular, non-sensitive shop works.
Everything else goes to a queue for someone to act on, with no offer: payment fixes, merchant
ops alerts, and "wait and recheck" (travel, normal gaps).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_at_risk(scores: pd.DataFrame, cfg: dict, carryover: pd.DataFrame | None = None) -> pd.DataFrame:
    """High/medium risk pairs that have actually missed their usual visit. Pairs waiting in the
    anonymity carry-over queue stay in with a slightly relaxed threshold, so a small shop can
    collect enough customers across runs."""
    due = scores["p_overdue"] >= cfg["matching"]["min_overdue_ratio"]
    at_risk = scores["risk_tier"].isin(["high", "medium"]) & due
    if carryover is not None and len(carryover):
        queued = scores.set_index(["customer_id", "merchant_id"]).index.isin(
            carryover.set_index(["customer_id", "merchant_id"]).index)
        relaxed = scores["lapse_prob"] >= 0.8 * cfg["risk_tiers"]["medium"]
        at_risk |= queued & relaxed & due
        scores = scores.assign(carried_over=queued & relaxed & due & ~scores["risk_tier"].isin(["high", "medium"]))
    else:
        scores = scores.assign(carried_over=False)
    return scores[at_risk].reset_index(drop=True)


def match_pairs(at_risk: pd.DataFrame, merchants: pd.DataFrame, customers: pd.DataFrame,
                cfg: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Returns (offer candidates, funnel counts, action queues)."""
    comp = cfg["compliance"]
    w = cfg["matching"]["affinity_weights"]
    funnel = {"at_risk_pairs": int(len(at_risk)), "at_risk_customers": int(at_risk["customer_id"].nunique())}

    df = at_risk.merge(merchants[["merchant_id", "name", "locality", "enrolled", "paytm_acquired"]],
                       on="merchant_id", how="left")
    df = df.merge(customers[["customer_id", "offers_consent", "is_minor", "share_with_merchants", "language"]],
                  on="customer_id", how="left")
    # defence in depth: non-consenting customers and minors are normally removed before scoring
    df = df[df["offers_consent"].fillna(False).astype(bool)]
    if comp["exclude_minors"]:
        df = df[~df["is_minor"].fillna(True).astype(bool)]
    sensitive = df["category"].isin(set(comp.get("sensitive_categories", [])))
    funnel["sensitive_pairs_skipped"] = int(sensitive.sum())
    df = df[~sensitive]

    queues = {
        "paytm_fix": df[df["route_action"] == "paytm_fix"],
        "ops_alert": df[df["route_action"] == "ops_alert"],
        "wait": df[df["route_action"] == "wait"],
        "skip": df[df["route_action"] == "skip"],
    }
    funnel["routed"] = df["route"].value_counts().to_dict()

    offer = df[df["route_action"].isin(["merchant_offer", "paytm_offer"])].copy()
    shop_ok = offer["enrolled"].fillna(False).astype(bool) & offer["paytm_acquired"].fillna(False).astype(bool)
    needs_shop = offer["route_action"] == "merchant_offer"
    funnel["shop_not_enrolled"] = int((needs_shop & ~shop_ok).sum())
    offer = offer[~needs_shop | shop_ok]

    offer["affinity"] = (w["visit_share"] * offer["p_visit_share"]
                         + w["recency"] * np.exp(-offer["p_days_since"] / 30.0)
                         + w["gmv_share"] * offer["p_gmv_share"]).round(4)
    offer = offer.rename(columns={"name": "merchant_name", "p_days_since": "last_visit_days",
                                  "p_avg_ticket": "avg_ticket_at_merchant", "p_visit_share": "visit_share",
                                  "p_first_days": "first_visit_days"})
    offer["needs_merchant_approval"] = offer["route_action"] == "merchant_offer"
    funnel["offer_candidates_pairs"] = int(len(offer))
    funnel["offer_candidates_customers"] = int(offer["customer_id"].nunique())
    return offer.reset_index(drop=True), funnel, queues
