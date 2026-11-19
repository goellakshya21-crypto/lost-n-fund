"""Pipeline configuration.

Every number the pipeline uses lives here. Override any subset with a JSON file:
    python run_pipeline.py all --config my_config.json
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG: dict = {
    "seed": 42,
    "platform": {
        "name": "Paytm",
        # Which side of UPI the data comes from:
        #   "payer"    = our app users' payments at any shop (transactions.csv)
        #   "acquirer" = payments into shops that use our QR/Soundbox, from ANY app (merchant_daily.csv)
        #   "both"     = both (Paytm and PhonePe can have both; BHIM is mostly payer-side)
        "data_vantage": "both",
    },
    "paths": {
        "data_dir": "data",        # transactions.csv, merchants.csv, customers.csv (+ optional tables)
        "model_dir": "models",     # model registry + the promoted model
        "output_dir": "outputs",   # one folder per campaign run
        "state_dir": "state",      # contact history, carry-over queue, learned assumptions, program health
    },
    # Only used by `generate` (demo data). Real data replaces this.
    "synthetic": {
        # ~400 app-paying regulars per kirana, so the 5-customer anonymity rule is realistic to meet
        "n_customers": 40000,
        "n_merchants": 180,
        "days": 210,
        "end_date": "2026-11-19",
        "disengage_share": 0.5,
        # why disengaging customers fade (hidden ground truth the pipeline has to diagnose)
        "disengage_types": {"shop_switch": 0.32, "cash_shift": 0.18, "app_switch": 0.18, "gone": 0.18,
                            "friction": 0.14},
        "festive_window": ["2026-10-25", "2026-11-09"],
        # migrant-heavy localities where many people travel home for Chhath
        "travel_localities": ["Mayur Vihar", "Rohini Sec 7", "Noida Sec 18"],
        "travel_window": ["2026-11-03", "2026-11-24"],
        "travel_share": 0.35,
        "declining_merchant_share": 0.04,
    },
    # Privacy and regulatory guardrails. See README "Compliance".
    "compliance": {
        "only_process_consented": True,        # customers without consent are not trained on, scored or messaged
        "exclude_minors": True,                # under-18s are never scored or targeted
        "sensitive_categories": ["pharmacy"],  # never used for WinBack: purchases there can reveal health
        "merchant_view": "anonymous",          # "anonymous" = counts only (recommended); "named" = first name +
                                               # initial ONLY for customers who agreed to be visible to shops
        "min_customers_per_alert": 5,          # never alert a shop about fewer customers than this
        "min_customers_for_ranges": 10,        # bill / visit ranges only when the group is at least this big
        "count_bands": [5, 10, 20, 50],        # shops see "5-9 regulars", never an exact count
        "cost_rounding": 50,                   # costs shown to shops are rounded so the count can't be backed out
        "settlement_netting": "weekly",        # merchant's cashback share is netted from settlement as ONE weekly line
        "retention_days": 120,                 # customer-level campaign files deleted after this (>= measurement)
        "abuse_history_days": 180,             # contact history kept this long, only to enforce the offer caps
        "data_residency": "IN",
        "sms_fallback": False,                 # needs TRAI DLT-registered templates + DND scrubbing first
        "cross_app_linking": False,            # linking a user to their UPI IDs on other apps: off until legal review
        "pseudonym_salt": "change-me-in-production",  # salts the hashes kept in the ledger and erasure list
    },
    "features": {"lookback_days": 90},
    "labels": {
        "active_window_days": 30,   # customer-level features use customers who paid in the last N days
        "churn_window_days": 30,    # outcome window
        "pair_lapse_share": 0.30,   # a customer "lapses at shop A" if their spend there in the next window
                                    # falls below this share of their usual monthly spend there
    },
    "training": {
        "n_train_snapshots": 4,
        "snapshot_spacing_days": 15,
        "promote_max_auc_drop": 0.01,   # a new model replaces the current one unless it is worse by more than this
        "psi_warn": 0.25,               # feature drift (population stability index) that triggers a warning
    },
    # Calibrated P(lapse at this shop) bands. Only high + medium are candidates.
    "risk_tiers": {"high": 0.60, "medium": 0.35},
    "matching": {
        "min_visits": 3,                 # a "regular": paid at A at least this often in the lookback
        "max_days_since_visit": 45,      # ...and recently enough that A still counts them as a regular
        "min_overdue_ratio": 1.0,        # only act once they have actually missed their usual visit
        "top_k_merchants": 1,            # one offer per customer per run
        "max_offers_per_merchant": 25,   # don't flood a kirana owner
        "affinity_weights": {"visit_share": 0.5, "recency": 0.3, "gmv_share": 0.2},
        "carryover_days": 21,            # customers held back for anonymity wait this long for more at the same shop
    },
    # Why is this customer fading? Thresholds for the root-cause diagnosis.
    "diagnosis": {
        "merchant_decline": 0.35,        # shop's total footfall down >= 35% -> the shop is the problem
        "soundbox_offline_days": 3,      # Soundbox offline this many of the last 14 days -> shop problem
        "locality_dip": 0.25,            # shops in their locality down >= 25% vs usual -> travel / festival, wait
        "away_share": 0.5,               # >= half of recent payments in places they never used -> travelling
        "friction_fail_rate": 0.10,      # >= this share of recent payments failed ...
        "friction_min_failures": 2,      # ... and at least this many failures
        "normal_gap_trend": 0.6,         # visits here still >= 60% of usual ...
        "normal_gap_overdue": 2.0,       # ... and < 2 usual gaps since the last visit -> just a normal gap, wait
        "habit_visits_per_month": 2.0,   # occasional visitors (fewer visits): a month away is normal for them
        "elsewhere_drop": 0.75,          # payments at other shops below 75% of usual -> dropping everywhere
        "app_activity_gone": 0.6,        # ... and bills/P2P or app opens below 60% of usual -> gone; else another app
        "same_cat_shift": 0.3,           # >= 30% of their usual spend here now goes to a shop of the same kind
    },
    # Who pays for each root cause.
    #   merchant_offer: shop funds it (paytm_share = Paytm's share of the cashback)
    #   paytm_offer:    Paytm funds it; the shop is never asked or told
    #   paytm_fix / ops_alert / wait / skip: no offer
    "routing": {
        "shop_switch":      {"action": "merchant_offer", "paytm_share": 0.0},
        "unclear":          {"action": "merchant_offer", "paytm_share": 0.5},
        "app_switch":       {"action": "paytm_offer",    "paytm_share": 1.0},
        "payment_friction": {"action": "paytm_fix"},
        "merchant_decline": {"action": "ops_alert"},
        "seasonal":         {"action": "wait"},
        "normal_gap":       {"action": "wait"},
        "gone":             {"action": "skip"},
    },
    "offers": {
        # Offers are CASHBACK: the customer pays the full bill, gets cashback from Paytm, and the shop's share is
        # netted from its settlement. The Soundbox announces the full amount, so nothing changes at the counter.
        #   flat:   "Rs 20 cashback on your next payment of Rs 200+"
        #   streak: "pay on 3 different days within the validity, get Rs 45 cashback"  (rebuilds the habit)
        #   pct:    "5% cashback (up to Rs 40) on your next 2 payments"                  (exploration arm)
        "types": ["flat", "streak", "pct"],
        "streak_visits": 3,
        "streak_min_visits_per_month": 3,   # streaks only for customers who used to come at least this often
        "streak_reward_mult": 1.5,          # streak reward vs a single flat cashback
        "pct_uses": 2,
        # starting generosity (share of the category's max %) by risk tier x customer value
        "level_matrix": {"high": {"high_value": 1.0, "low_value": 0.5}, "medium": {"high_value": 0.5, "low_value": 0.3}},
        "level_ladder": [1.0, 0.75, 0.5, 0.3],   # step down until the offer pays for itself
        "high_value_quantile": 0.6,
        "validity_gap_mult": 1.75,          # validity = 1.75 x the customer's usual gap at this shop ...
        "validity_min_days": 7,             # ... clipped to this range
        "validity_max_days": 21,
        "min_bill_ticket_share": 0.7,       # min bill = max(category min, 70% of their usual bill there)
        "exploration_share": 0.15,          # random offer type/level for some customers, to learn what works
        "paytm_max_subsidy_per_offer": 15,  # Paytm's expected cost per co-funded offer may not exceed this (Rs)
        "use_uplift_model": True,           # use models/uplift_model.joblib when it exists
        # min_bill (Rs), max_pct = most generous cashback as % of their usual bill, cap per payment (Rs), margin
        "category_rules": {
            "kirana":      {"min_bill": 150,  "max_pct": 5,  "cap": 40,  "margin": 0.12},
            "pharmacy":    {"min_bill": 200,  "max_pct": 5,  "cap": 60,  "margin": 0.18},
            "restaurant":  {"min_bill": 300,  "max_pct": 10, "cap": 100, "margin": 0.35},
            "cafe":        {"min_bill": 120,  "max_pct": 10, "cap": 50,  "margin": 0.40},
            "salon":       {"min_bill": 300,  "max_pct": 10, "cap": 100, "margin": 0.45},
            "fuel":        {"min_bill": 500,  "max_pct": 2,  "cap": 30,  "margin": 0.04},
            "apparel":     {"min_bill": 1000, "max_pct": 10, "cap": 200, "margin": 0.40},
            "electronics": {"min_bill": 2000, "max_pct": 5,  "cap": 250, "margin": 0.15},
        },
    },
    # Anti-abuse rules (checked when choosing customers, and again at every payment).
    "abuse": {
        "min_platform_tenure_days": 60,     # new accounts can't farm win-back offers
        "min_pair_tenure_days": 30,         # must have been a regular at this shop for a while
        "max_offers_per_window": 2,         # at most 2 win-back offers per customer per abuse_history_days
        "repeat_offer_value_mult": 0.5,     # the second offer in that window is worth half
        "one_redemption_per_day": True,     # stops splitting one bill into several payments
        "clawback_on_refund": True,         # refund or chargeback -> the cashback is reversed
    },
    # Assumptions used to *decide* offers before campaign data exists. `measure` writes measured
    # values to state/learned_assumptions.json; set use_learned_assumptions to use them.
    "economics": {
        "assumed_save_rate": 0.20,                  # share of would-be lapsers an offer brings back
        "assumed_retention_curve": [1.0, 0.7, 0.5],  # share of a saved customer's usual spend kept in months 1..3
        "bill_topup_rate": 0.3,                     # chance a near-miss bill is topped up to the minimum
        "paytm_value_per_gmv": 0.004,               # blended value per Rs of GMV (wallet/card MDR, Soundbox...)
        "paytm_value_per_user_month": 15.0,         # value of keeping a user active on the app (lending, bills...)
        "use_learned_assumptions": True,
        "allow_simulated_learning": False,          # never learn from simulated outcomes unless asked
    },
    "campaign": {
        "holdout_share": 0.20,          # at-risk customers who get NO offer this run (per-campaign control)
        "global_holdout_share": 0.05,   # customers who get NO WinBack offer all quarter (programme-level control)
        "frequency_cap_days": 21,
        "measurement_days": 90,         # 90 days: in-offer window + what sticks after the offer ends
        "short_window_days": 30,
        "target_uplift_pts": 5.0,       # the effect the pilot must be able to detect (for the power calculation)
        "merchant_monthly_budget_default": 1500,
        "paytm_monthly_budget": 60000,  # Paytm's own budget for Paytm-funded and co-funded cashback
    },
    # Kill switches: `campaign` refuses to run while a hard limit is breached (override with --force).
    "guardrails": {
        "max_opt_out_rate": 0.03,          # opt-outs / offers delivered (hard)
        "max_complaint_rate": 0.005,       # complaints / offers delivered (hard)
        "min_merchant_approval_rate": 0.30,  # warning
        "max_budget_burn": 1.0,            # cashback paid / budget (hard)
    },
    # Demo only: how simulated merchants respond to alerts.
    "merchant_simulation": {"approve_prob": 0.75},
}


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, encoding="utf-8") as fh:
            _deep_merge(cfg, json.load(fh))
    resolved = {}
    for key, value in cfg["paths"].items():
        p = Path(value)
        resolved[key] = p if p.is_absolute() else PROJECT_ROOT / p
    cfg["paths"] = resolved
    return cfg
