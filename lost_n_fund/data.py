"""Load and validate the input tables.

This is the contract with production. Point `paths.data_dir` at a folder with these CSVs
(exported from the warehouse) and every later stage works unchanged.

Required (payer side: our app users' payments):
  transactions.csv   txn_id, customer_id, merchant_id, amount, ts, status, payment_mode
  merchants.csv      merchant_id, name, owner_name, category, locality, enrolled, auto_approve,
                     monthly_offer_budget, language   (+ optional paytm_acquired, whatsapp_opt_in)
  customers.csv      customer_id, name, locality, offers_consent, is_minor
                     (+ optional share_with_merchants, language)
Optional (better diagnosis when present):
  app_activity.csv   customer_id, week_start, app_opens, bill_payments, p2p_payments
  merchant_daily.csv merchant_id, date, txns_all_apps, txns_paytm_app, gmv_all_apps, soundbox_online
                     (acquirer side: payments into our shops from ANY UPI app, aggregated per day)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SCHEMA = {
    "transactions": ["txn_id", "customer_id", "merchant_id", "amount", "ts", "status", "payment_mode"],
    "merchants": ["merchant_id", "name", "owner_name", "category", "locality",
                  "enrolled", "auto_approve", "monthly_offer_budget", "language"],
    # no phone numbers needed: offers are delivered in-app using the platform's own records
    "customers": ["customer_id", "name", "locality", "offers_consent", "is_minor"],
    "app_activity": ["customer_id", "week_start", "app_opens", "bill_payments", "p2p_payments"],
    "merchant_daily": ["merchant_id", "date", "txns_all_apps", "txns_paytm_app", "gmv_all_apps", "soundbox_online"],
}
OPTIONAL_COLUMNS = {
    "merchants": {"paytm_acquired": True, "whatsapp_opt_in": False},
    "customers": {"share_with_merchants": False, "language": "hi"},
}
VALID_STATUS = {"SUCCESS", "FAILED"}


@dataclass
class Data:
    txns: pd.DataFrame
    merchants: pd.DataFrame
    customers: pd.DataFrame
    activity: pd.DataFrame | None = None
    merchant_daily: pd.DataFrame | None = None
    quality: dict = field(default_factory=dict)


def pseudonym(customer_id: str, cfg: dict) -> str:
    """Salted hash used where a record must outlive the personal data (ledger, erasure list)."""
    salt = cfg["compliance"].get("pseudonym_salt", "lost_n_fund")
    return hashlib.sha256(f"{salt}|{customer_id}".encode()).hexdigest()[:20]


def erased_hashes(cfg: dict) -> set:
    p = Path(cfg["paths"]["state_dir"]) / "erased_customers.txt"
    return set(p.read_text(encoding="utf-8").split()) if p.exists() else set()


def _check(df: pd.DataFrame, table: str) -> pd.DataFrame:
    missing = [c for c in SCHEMA[table] if c not in df.columns]
    if missing:
        raise ValueError(f"{table}.csv is missing columns: {missing}")
    return df


def _drop(df: pd.DataFrame, bad: pd.Series, what: str, issues: list) -> pd.DataFrame:
    n = int(bad.sum())
    if n:
        issues.append(f"{what}: {n:,} rows dropped")
    return df[~bad]


def load_data(data_dir: Path, cfg: dict | None = None, today: pd.Timestamp | None = None) -> Data:
    data_dir = Path(data_dir)
    if not (data_dir / "transactions.csv").exists():
        raise FileNotFoundError(
            f"No transactions.csv in {data_dir}. Run `python run_pipeline.py generate` for demo data.")
    vantage = (cfg or {}).get("platform", {}).get("data_vantage", "both")
    if vantage not in ("payer", "both"):
        raise ValueError("Lost & Fund needs payer-side data (your app users' payments) to reach customers. "
                         "Set platform.data_vantage to 'payer' or 'both'.")
    issues: list[str] = []

    txns = _check(pd.read_csv(data_dir / "transactions.csv"), "transactions")
    merchants = _check(pd.read_csv(data_dir / "merchants.csv"), "merchants")
    customers = _check(pd.read_csv(data_dir / "customers.csv"), "customers")

    # ---- merchants / customers: keys must be unique ----
    for name, df, key in (("merchants", merchants, "merchant_id"), ("customers", customers, "customer_id")):
        if df[key].isna().any() or df[key].duplicated().any():
            raise ValueError(f"{name}.csv has missing or duplicate {key}s")
    for table, defaults in OPTIONAL_COLUMNS.items():
        df = merchants if table == "merchants" else customers
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
    for col in ("enrolled", "auto_approve", "paytm_acquired", "whatsapp_opt_in"):
        merchants[col] = merchants[col].fillna(False).astype(bool)
    for col in ("offers_consent", "is_minor", "share_with_merchants"):
        if customers[col].isna().any():
            issues.append(f"customers.{col}: {int(customers[col].isna().sum())} blanks treated as False")
        customers[col] = customers[col].fillna(False).astype(bool)
    # sharing with merchants only counts if the customer also consented to offers
    customers["share_with_merchants"] &= customers["offers_consent"]
    customers["language"] = customers["language"].fillna("hi")
    merchants["monthly_offer_budget"] = pd.to_numeric(merchants["monthly_offer_budget"], errors="coerce").fillna(0)

    # ---- right to erasure: customers who asked to be deleted never come back in ----
    if cfg is not None:
        gone = erased_hashes(cfg)
        if gone:
            h = customers["customer_id"].map(lambda c: pseudonym(c, cfg))
            erased = set(customers.loc[h.isin(gone), "customer_id"])
            customers = customers[~customers["customer_id"].isin(erased)]
            txns = txns[~txns["customer_id"].isin(erased)]
            if erased:
                issues.append(f"erasure list: {len(erased)} customers excluded")

    # ---- transactions ----
    txns["ts"] = pd.to_datetime(txns["ts"], errors="coerce")
    txns["status"] = txns["status"].astype(str).str.upper()
    txns["amount"] = pd.to_numeric(txns["amount"], errors="coerce")
    txns = _drop(txns, txns["txn_id"].duplicated(), "duplicate txn_id", issues)
    txns = _drop(txns, txns[["customer_id", "merchant_id"]].isna().any(axis=1), "missing customer/merchant id", issues)
    txns = _drop(txns, txns["ts"].isna(), "unreadable timestamp", issues)
    txns = _drop(txns, ~(txns["amount"] > 0), "zero, negative or missing amount", issues)
    txns = _drop(txns, ~txns["status"].isin(VALID_STATUS), "unknown status", issues)
    txns = _drop(txns, ~txns["merchant_id"].isin(set(merchants["merchant_id"])), "unknown merchant", issues)
    txns = _drop(txns, ~txns["customer_id"].isin(set(customers["customer_id"])), "unknown customer", issues)
    if txns.empty:
        raise ValueError("No usable transactions after validation: " + "; ".join(issues))
    # the shop's category and locality on every transaction are used by several features
    txns = txns.merge(merchants[["merchant_id", "category", "locality"]].rename(columns={"locality": "m_locality"}),
                      on="merchant_id", how="left")

    # ---- optional tables ----
    activity = md = None
    if (data_dir / "app_activity.csv").exists():
        activity = _check(pd.read_csv(data_dir / "app_activity.csv", parse_dates=["week_start"]), "app_activity")
        activity = activity[activity["customer_id"].isin(set(customers["customer_id"]))]
    if vantage == "both" and (data_dir / "merchant_daily.csv").exists():
        md = _check(pd.read_csv(data_dir / "merchant_daily.csv", parse_dates=["date"]), "merchant_daily")
        bad = md["txns_paytm_app"] > md["txns_all_apps"]
        md = _drop(md, bad, "merchant_daily rows with more app payments than total", issues)

    # ---- freshness: stale data means stale risk scores ----
    last = txns["ts"].max()
    if today is not None and (today - last).days > 3:
        issues.append(f"data is {(today - last).days} days old (last payment {last.date()})")

    quality = {
        "transactions": int(len(txns)),
        "first_payment": str(txns["ts"].min().date()),
        "last_payment": str(last.date()),
        "failed_share": round(float((txns["status"] == "FAILED").mean()), 4),
        "has_app_activity": activity is not None,
        "has_merchant_daily": md is not None,
        "data_vantage": vantage,
        "issues": issues,
    }
    return Data(txns.reset_index(drop=True), merchants, customers.reset_index(drop=True), activity, md, quality)


def eligible_customer_ids(customers: pd.DataFrame, cfg: dict) -> set:
    """Customers Lost & Fund may process at all: consented to offers and not a minor."""
    ok = customers["offers_consent"].copy()
    if cfg["compliance"]["exclude_minors"]:
        ok &= ~customers["is_minor"]
    return set(customers.loc[ok, "customer_id"])


def compliance_counts(customers: pd.DataFrame) -> dict:
    return {
        "customers_total": int(len(customers)),
        "no_offers_consent": int((~customers["offers_consent"]).sum()),
        "minors": int(customers["is_minor"].sum()),
        "share_with_merchants": int(customers["share_with_merchants"].sum()),
    }


def default_snapshot(txns: pd.DataFrame) -> pd.Timestamp:
    """The day after the last transaction = 'today' for scoring."""
    return txns["ts"].max().normalize() + pd.Timedelta(days=1)
