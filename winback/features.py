"""Point-in-time features and labels for (customer, shop) pairs.

The question is "will customer B stop coming to shop A?", not "will B stop using the app".
Someone who moved from Sharma Kirana to a quick-commerce app still pays on our app every day,
so a customer-level model calls them safe; the pair-level model sees the drop at Sharma Kirana.

Everything is computed "as of" a snapshot date using only data before that date, so training
never peeks into the future. Optional tables (app activity, acquirer-side shop data) become
NaN features when missing; the gradient-boosting model handles NaN natively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln

CUSTOMER_FEATURES = [
    "recency_days", "txn_30", "txn_90", "txn_14", "txn_14_28", "gmv_30", "gmv_90",
    "freq_trend", "gmv_trend", "short_trend", "avg_ticket",
    "active_days_30", "active_days_90", "avg_gap_days", "gap_std", "recency_over_gap",
    "n_merchants", "n_categories", "top_merchant_share",
    "fail_rate_30", "fail_rate_90", "wallet_share", "card_share",
    "tenure_days", "weeks_active_12",
]
PAIR_FEATURES = [
    "p_visits_90", "p_visits_30", "p_visits_14", "p_gmv_90", "p_gmv_30", "p_days_since", "p_first_days",
    "p_avg_gap", "p_gap_std", "p_overdue", "p_trend", "p_short_trend", "p_visit_share", "p_gmv_share",
    "p_avg_ticket", "p_dow_conc", "p_alive", "p_silence",
]
CONTEXT_FEATURES = [
    "c_other_trend",        # payments at OTHER shops, last 30d vs usual
    "c_new_same_cat_30",    # new shops of the same kind started in the last 30d
    "c_same_cat_shift",     # extra spend at other shops of the same kind, as a share of usual spend here
    "c_away_share_14",      # share of the last 14 days' payments made in places they never used before
    "c_qc_share_30",        # quick-commerce share of spend, last 30d
    "c_qc_trend",           # quick-commerce payments, last 30d vs usual
    "act_ratio",            # bill payments + P2P, last 4 weeks vs the 8 before   (app_activity)
    "opens_ratio",          # app opens, same comparison                            (app_activity)
    "m_trend",              # the shop's payments from our users, last 30d vs usual
    "m_all_trend",          # the shop's payments from ALL apps, last 30d vs usual  (merchant_daily)
    "m_share_trend",        # our app's share of the shop's payments, last 30d vs usual (merchant_daily)
    "m_offline_14",         # days the Soundbox was offline in the last 14            (merchant_daily)
    "loc_ratio",            # the customer's whole locality, last 14d vs usual
    "category_code",
]
FEATURE_COLUMNS = CUSTOMER_FEATURES + PAIR_FEATURES + CONTEXT_FEATURES
CATEGORY_CODES = {c: i for i, c in enumerate(
    ["kirana", "pharmacy", "restaurant", "cafe", "salon", "fuel", "apparel", "electronics", "quick_commerce"])}


def _count(df: pd.DataFrame, mask, keys, col=None) -> pd.Series:
    g = df[mask].groupby(keys)
    return g.size() if col is None else g[col].sum()


def _ratio(recent, earlier, recent_days, earlier_days, k=1.0):
    """(recent rate) / (earlier rate) with smoothing, both scaled to per-30-days."""
    return (recent * 30 / recent_days + k) / (earlier * 30 / earlier_days + k)


# ------------------------------------------------------------------ customer level
def customer_features(win: pd.DataFrame, hist: pd.DataFrame, snapshot: pd.Timestamp, lookback: int) -> pd.DataFrame:
    """Customer-wide behaviour (all shops). One row per customer with a successful payment in the window."""
    ok = win["status"].eq("SUCCESS")
    succ, fail = win[ok], win[~ok]
    f = pd.DataFrame(index=pd.Index(np.sort(succ["customer_id"].unique()), name="customer_id"))
    g = succ.groupby("customer_id")
    f["recency_days"] = g["age"].min()
    f["txn_90"] = g.size()
    f["gmv_90"] = g["amount"].sum()
    f["txn_30"] = _count(succ, succ["age"] < 30, "customer_id")
    f["gmv_30"] = _count(succ, succ["age"] < 30, "customer_id", "amount")
    f["txn_14"] = _count(succ, succ["age"] < 14, "customer_id")
    f["txn_14_28"] = _count(succ, (succ["age"] >= 14) & (succ["age"] < 28), "customer_id")
    f = f.fillna(0.0)
    prev_months = (lookback - 30) / 30.0
    f["freq_trend"] = (f["txn_30"] + 1) / ((f["txn_90"] - f["txn_30"]) / prev_months + 1)
    f["gmv_trend"] = (f["gmv_30"] + 100) / ((f["gmv_90"] - f["gmv_30"]) / prev_months + 100)
    f["short_trend"] = (f["txn_14"] + 1) / (f["txn_14_28"] + 1)
    f["avg_ticket"] = f["gmv_90"] / f["txn_90"].clip(lower=1)

    days = succ.assign(day=succ["ts"].dt.normalize())[["customer_id", "day", "age"]]
    days = days.sort_values(["customer_id", "day"]).drop_duplicates(["customer_id", "day"])
    f["active_days_90"] = days.groupby("customer_id").size()
    f["active_days_30"] = _count(days, days["age"] < 30, "customer_id")
    gs = days.groupby("customer_id")["day"].diff().dt.days.groupby(days["customer_id"])
    f["avg_gap_days"] = gs.mean().reindex(f.index).fillna(float(lookback))
    f["gap_std"] = gs.std().reindex(f.index).fillna(0.0)
    f["active_days_30"] = f["active_days_30"].fillna(0.0)
    f["recency_over_gap"] = f["recency_days"] / (f["avg_gap_days"] + 1)

    f["n_merchants"] = g["merchant_id"].nunique()
    f["n_categories"] = g["category"].nunique()
    per_m = succ.groupby(["customer_id", "merchant_id"]).size()
    f["top_merchant_share"] = per_m.groupby(level=0).max() / f["txn_90"]

    f30 = _count(fail, fail["age"] < 30, "customer_id").reindex(f.index, fill_value=0)
    f90 = fail.groupby("customer_id").size().reindex(f.index, fill_value=0)
    f["fail_rate_30"] = f30 / (f30 + f["txn_30"]).clip(lower=1)
    f["fail_rate_90"] = f90 / (f90 + f["txn_90"]).clip(lower=1)
    f["failures_30"] = f30
    f["wallet_share"] = _count(succ, succ["payment_mode"].eq("WALLET"), "customer_id").reindex(f.index, fill_value=0) / f["txn_90"]
    f["card_share"] = _count(succ, succ["payment_mode"].eq("CARD"), "customer_id").reindex(f.index, fill_value=0) / f["txn_90"]

    first_seen = hist[hist["customer_id"].isin(f.index)].groupby("customer_id")["ts"].min()
    f["tenure_days"] = (snapshot - first_seen).dt.days
    recent = succ[succ["age"] < 84]
    f["weeks_active_12"] = recent.assign(w=(recent["age"] // 7)).groupby("customer_id")["w"].nunique()
    return f.fillna(0.0)


# ------------------------------------------------------------------ BG/NBD "still a regular?"
def fit_bgnbd(x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> tuple[float, float, float, float]:
    """Fit the BG/NBD model (Fader, Hardie & Lee 2005) for non-contractual repeat visits.

    x = repeat visit days, t_x = day of last visit since first, T = days observed since first visit.
    """
    x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))

    def nll(logp):
        r, alpha, a, b = np.exp(logp)
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        a3 = -(r + x) * np.log(alpha + T)
        a4 = np.where(x > 0, np.log(a) - np.log(b + np.maximum(x, 1) - 1) - (r + x) * np.log(alpha + t_x), -np.inf)
        m = np.maximum(a3, a4)
        ll = a1 + a2 + m + np.log(np.exp(a3 - m) + np.exp(a4 - m))
        return -ll.mean() + 1e-4 * np.sum(logp ** 2)

    res = minimize(nll, x0=np.log([1.0, 5.0, 1.0, 3.0]), method="L-BFGS-B", bounds=[(-6, 6)] * 4)
    return tuple(float(v) for v in np.exp(res.x))


def bgnbd_p_alive(params, x, t_x, T) -> np.ndarray:
    r, alpha, a, b = params
    x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))
    log_ratio = np.log(a) - np.log(b + np.maximum(x, 1) - 1) + (r + x) * (np.log(alpha + T) - np.log(alpha + t_x))
    return np.where(x > 0, expit(-log_ratio), 1.0)


# ------------------------------------------------------------------ pair level
def pair_stats(succ: pd.DataFrame, lookback: int) -> pd.DataFrame:
    keys = ["customer_id", "merchant_id"]
    g = succ.groupby(keys)
    p = g.agg(p_visits_90=("txn_id", "size"), p_gmv_90=("amount", "sum"),
              p_days_since=("age", "min"), p_first_days=("age", "max"), category=("category", "first"))
    p["p_visits_30"] = _count(succ, succ["age"] < 30, keys).reindex(p.index, fill_value=0)
    p["p_visits_14"] = _count(succ, succ["age"] < 14, keys).reindex(p.index, fill_value=0)
    p["p_visits_14_28"] = _count(succ, (succ["age"] >= 14) & (succ["age"] < 28), keys).reindex(p.index, fill_value=0)
    p["p_gmv_30"] = _count(succ, succ["age"] < 30, keys, "amount").reindex(p.index, fill_value=0.0)
    la = np.log(succ["amount"].clip(lower=1))
    p["log_amt_mu"] = la.groupby([succ["customer_id"], succ["merchant_id"]]).mean()
    p["log_amt_sd"] = la.groupby([succ["customer_id"], succ["merchant_id"]]).std().fillna(0.45).clip(lower=0.15)

    d = succ.assign(day=succ["ts"].dt.normalize())[keys + ["day", "age"]].drop_duplicates(keys + ["day"])
    d = d.sort_values(keys + ["day"])
    gd = d.groupby(keys)
    p["p_visit_days"] = gd.size()
    gaps = gd["day"].diff().dt.days
    gg = gaps.groupby([d["customer_id"], d["merchant_id"]])
    p["p_avg_gap"] = gg.mean().reindex(p.index).fillna(float(lookback))
    p["p_gap_std"] = gg.std().reindex(p.index).fillna(0.0)
    dow = d.assign(dow=d["day"].dt.weekday).groupby(keys + ["dow"]).size()
    p["p_dow_conc"] = dow.groupby(level=[0, 1]).max() / p["p_visit_days"]
    p["bg_x"] = p["p_visit_days"] - 1
    p["bg_T"] = gd["age"].max().round()                      # days since first visit in the window
    p["bg_tx"] = p["bg_T"] - gd["age"].min().round()         # last visit, days after the first
    return p.reset_index()


def _context(win: pd.DataFrame, pairs: pd.DataFrame, cust: pd.DataFrame, customers: pd.DataFrame,
             activity: pd.DataFrame | None, md: pd.DataFrame | None, snapshot: pd.Timestamp,
             lookback: int) -> pd.DataFrame:
    succ = win[win["status"].eq("SUCCESS")]
    out = pairs

    # elsewhere: the customer's payments at other shops
    c30 = cust["txn_30"].reindex(out["customer_id"]).to_numpy()
    c90 = cust["txn_90"].reindex(out["customer_id"]).to_numpy()
    other_30 = c30 - out["p_visits_30"].to_numpy()
    other_90 = c90 - out["p_visits_90"].to_numpy()
    out["c_other_trend"] = _ratio(other_30, other_90 - other_30, 30, lookback - 30)

    # new shops of the same kind (they found a replacement)
    first = succ.groupby(["customer_id", "merchant_id"]).agg(first=("age", "max"), category=("category", "first"))
    new = first[first["first"] < 30].groupby(["customer_id", "category"]).size().rename("c_new_same_cat_30")
    out = out.merge(new.reset_index(), on=["customer_id", "category"], how="left")
    out["c_new_same_cat_30"] = out["c_new_same_cat_30"].fillna(0)

    # did the spend lost here show up at other shops of the same kind? (quick commerce counts as kirana)
    kind = succ["category"].replace({"quick_commerce": "kirana"})
    s2 = succ.assign(kind=kind, recent=succ["age"] < 30)
    by = s2.groupby(["customer_id", "kind", "merchant_id", "recent"])["amount"].sum().unstack("recent", fill_value=0.0)
    by = by.rename(columns={True: "r", False: "e"}).reset_index()
    for c in ("r", "e"):
        if c not in by.columns:
            by[c] = 0.0
    kind_tot = by.groupby(["customer_id", "kind"])[["r", "e"]].sum()
    key = pd.MultiIndex.from_arrays([out["customer_id"], out["category"].replace({"quick_commerce": "kirana"})])
    own = by.set_index(["customer_id", "merchant_id"])[["r", "e"]].reindex(
        pd.MultiIndex.from_frame(out[["customer_id", "merchant_id"]]), fill_value=0.0).to_numpy()
    tot = kind_tot.reindex(key, fill_value=0.0).to_numpy()
    other_r, other_e = tot[:, 0] - own[:, 0], tot[:, 1] - own[:, 1]
    gain = other_r - other_e * 30 / (lookback - 30)          # extra monthly spend at same-kind shops
    out["c_same_cat_shift"] = np.clip(gain / np.clip(out["p_gmv_90"].to_numpy() / (lookback / 30), 50, None), -2, 3)

    # paying in places they never used before (travelling home, moved city)
    seen = set(zip(succ.loc[succ["age"] >= 14, "customer_id"], succ.loc[succ["age"] >= 14, "m_locality"]))
    rec = succ[(succ["age"] < 14) & (succ["m_locality"] != "Online")]
    new_place = np.array([(c, m) not in seen for c, m in zip(rec["customer_id"], rec["m_locality"])], dtype=bool)
    away = pd.Series(new_place, index=rec.index).groupby(rec["customer_id"]).mean()
    out["c_away_share_14"] = out["customer_id"].map(away).fillna(0.0).to_numpy()

    qc = succ[succ["category"].eq("quick_commerce")]
    qc30 = _count(qc, qc["age"] < 30, "customer_id")
    qc90 = qc.groupby("customer_id").size()
    qcg30 = _count(qc, qc["age"] < 30, "customer_id", "amount")
    cq = pd.DataFrame({"qc30": qc30, "qc90": qc90, "qcg30": qcg30}).reindex(cust.index).fillna(0)
    out["c_qc_share_30"] = (cq["qcg30"] / cust["gmv_30"].clip(lower=1)).reindex(out["customer_id"]).to_numpy()
    out["c_qc_trend"] = _ratio(cq["qc30"], cq["qc90"] - cq["qc30"], 30, lookback - 30).reindex(out["customer_id"]).to_numpy()

    # app activity (bills, P2P, opens) - optional table
    if activity is not None and len(activity):
        a = activity[(activity["week_start"] < snapshot) & (activity["week_start"] >= snapshot - pd.Timedelta(weeks=12))]
        a = a.assign(recent=a["week_start"] >= snapshot - pd.Timedelta(weeks=4),
                     use=a["bill_payments"] + a["p2p_payments"])
        r = a[a["recent"]].groupby("customer_id")[["use", "app_opens"]].sum()
        e = a[~a["recent"]].groupby("customer_id")[["use", "app_opens"]].sum()
        both = r.join(e, lsuffix="_r", rsuffix="_e", how="outer").fillna(0)
        both["act_ratio"] = (both["use_r"] / 4 + 0.25) / (both["use_e"] / 8 + 0.25)
        both["opens_ratio"] = (both["app_opens_r"] / 4 + 0.5) / (both["app_opens_e"] / 8 + 0.5)
        out = out.merge(both[["act_ratio", "opens_ratio"]].reset_index(), on="customer_id", how="left")
    else:
        out["act_ratio"] = np.nan
        out["opens_ratio"] = np.nan

    # the shop's own trend, from our users' payments
    m30 = _count(succ, succ["age"] < 30, "merchant_id")
    m90 = succ.groupby("merchant_id").size()
    mt = _ratio(m30.reindex(m90.index, fill_value=0), m90 - m30.reindex(m90.index, fill_value=0), 30, lookback - 30)
    out["m_trend"] = out["merchant_id"].map(mt)

    # the shop as the acquirer sees it (all apps) - optional table
    if md is not None and len(md):
        w = md[(md["date"] < snapshot) & (md["date"] >= snapshot - pd.Timedelta(days=lookback))]
        rec = w["date"] >= snapshot - pd.Timedelta(days=30)
        agg = pd.DataFrame({
            "all_r": w[rec].groupby("merchant_id")["txns_all_apps"].sum(),
            "all_e": w[~rec].groupby("merchant_id")["txns_all_apps"].sum(),
            "our_r": w[rec].groupby("merchant_id")["txns_paytm_app"].sum(),
            "our_e": w[~rec].groupby("merchant_id")["txns_paytm_app"].sum(),
            "off14": (w[w["date"] >= snapshot - pd.Timedelta(days=14)]
                      .assign(off=lambda x: 1 - x["soundbox_online"]).groupby("merchant_id")["off"].sum()),
        }).fillna(0).rename_axis("merchant_id")
        agg["m_all_trend"] = _ratio(agg["all_r"], agg["all_e"], 30, lookback - 30)
        share_r = agg["our_r"] / agg["all_r"].clip(lower=1)
        share_e = agg["our_e"] / agg["all_e"].clip(lower=1)
        agg["m_share_trend"] = (share_r + 0.01) / (share_e + 0.01)
        agg["m_offline_14"] = agg["off14"]
        out = out.merge(agg[["m_all_trend", "m_share_trend", "m_offline_14"]].reset_index(), on="merchant_id", how="left")
    else:
        out["m_all_trend"] = np.nan
        out["m_share_trend"] = np.nan
        out["m_offline_14"] = np.nan

    # the customer's locality: are the shops THERE quieter (festival travel, roadworks)?
    loc = customers.set_index("customer_id")["locality"]
    lr = succ[succ["age"] < 14].groupby("m_locality").size() / 14
    le = succ[succ["age"] >= 14].groupby("m_locality").size() / (lookback - 14)
    loc_ratio = (lr / le.clip(lower=1e-9)).fillna(1.0)
    out["loc_ratio"] = out["customer_id"].map(loc).map(loc_ratio).fillna(1.0).to_numpy()
    out["customer_locality"] = out["customer_id"].map(loc).to_numpy()
    out["category_code"] = out["category"].map(CATEGORY_CODES)   # unknown category -> NaN (missing)
    return out


def build_features(txns: pd.DataFrame, customers: pd.DataFrame, snapshot, cfg: dict,
                   activity: pd.DataFrame | None = None, md: pd.DataFrame | None = None,
                   bg_params: tuple | None = None) -> tuple[pd.DataFrame, tuple]:
    """One row per (customer, shop) pair where the customer is a regular at the shop.

    Returns (pairs with FEATURE_COLUMNS plus helper columns, fitted BG/NBD parameters).
    """
    snapshot = pd.Timestamp(snapshot)
    lookback = cfg["features"]["lookback_days"]
    mc = cfg["matching"]
    hist = txns[txns["ts"] < snapshot]
    win = hist[hist["ts"] >= snapshot - pd.Timedelta(days=lookback)].copy()
    win["age"] = (snapshot - win["ts"]).dt.total_seconds() / 86400.0
    succ = win[win["status"].eq("SUCCESS")]

    cust = customer_features(win, hist, snapshot, lookback)
    pairs = pair_stats(succ, lookback)
    pairs = pairs[(pairs["p_visits_90"] >= mc["min_visits"]) & (pairs["p_days_since"] <= mc["max_days_since_visit"])]
    pairs = pairs[pairs["category"] != "quick_commerce"].reset_index(drop=True)

    if bg_params is None:
        bg_params = fit_bgnbd(pairs["bg_x"], pairs["bg_tx"], pairs["bg_T"])
    pairs["p_alive"] = bgnbd_p_alive(bg_params, pairs["bg_x"], pairs["bg_tx"], pairs["bg_T"])
    pairs["p_silence"] = np.exp(-pairs["p_visit_days"] / lookback * pairs["p_days_since"])
    pairs["p_overdue"] = pairs["p_days_since"] / (pairs["p_avg_gap"] + 1)
    pairs["p_trend"] = _ratio(pairs["p_visits_30"], pairs["p_visits_90"] - pairs["p_visits_30"], 30, lookback - 30)
    pairs["p_short_trend"] = (pairs["p_visits_14"] + 1) / (pairs["p_visits_14_28"] + 1)
    tot = cust[["txn_90", "gmv_90"]].reindex(pairs["customer_id"]).to_numpy()
    pairs["p_visit_share"] = pairs["p_visits_90"] / np.clip(tot[:, 0], 1, None)
    pairs["p_gmv_share"] = pairs["p_gmv_90"] / np.clip(tot[:, 1], 1, None)
    pairs["p_avg_ticket"] = (pairs["p_gmv_90"] / pairs["p_visits_90"]).round(0)

    pairs = _context(win, pairs, cust, customers, activity, md, snapshot, lookback)
    pairs = pairs.merge(cust[CUSTOMER_FEATURES + ["failures_30"]].reset_index(), on="customer_id", how="left")

    # helper columns used by matching / offers / messaging
    months = lookback / 30.0
    pairs["visits_per_month"] = (pairs["p_visits_90"] / months).round(1)
    pairs["monthly_gmv_at_merchant"] = (pairs["p_gmv_90"] / months).round(0)
    pairs["customer_monthly_gmv"] = (pairs["gmv_90"] / months).round(0)
    pairs["snapshot"] = snapshot
    return pairs, bg_params


def build_labels(txns: pd.DataFrame, snapshot, pairs: pd.DataFrame, cfg: dict) -> np.ndarray:
    """1 = lapsed at the shop: spend there in the next window below `pair_lapse_share` of usual."""
    snapshot = pd.Timestamp(snapshot)
    window = cfg["labels"]["churn_window_days"]
    fut = txns[(txns["ts"] >= snapshot) & (txns["ts"] < snapshot + pd.Timedelta(days=window))
               & txns["status"].eq("SUCCESS")]
    spent = fut.groupby(["customer_id", "merchant_id"])["amount"].sum()
    idx = pd.MultiIndex.from_frame(pairs[["customer_id", "merchant_id"]])
    future = spent.reindex(idx, fill_value=0.0).to_numpy()
    usual = pairs["monthly_gmv_at_merchant"].to_numpy() * window / 30.0
    return (future < cfg["labels"]["pair_lapse_share"] * usual).astype(int)


def reason_codes(f: pd.DataFrame, merchant_names: pd.Series | None = None) -> pd.Series:
    """Plain-language reasons for ops staff (never shown to the merchant)."""
    out = []
    names = merchant_names if merchant_names is not None else pd.Series(dtype=str)
    for r in f.itertuples(index=False):
        shop = names.get(r.merchant_id, "this shop")
        reasons = []
        if r.p_trend < 0.7:
            reasons.append(f"visits to {shop} down {100 * (1 - min(r.p_trend, 1)):.0f}%")
        if r.p_overdue > 1.3 and r.p_days_since >= 4:
            reasons.append(f"usually every {r.p_avg_gap:.0f}d there, last visit {r.p_days_since:.0f}d ago")
        if r.fail_rate_30 >= 0.08:
            reasons.append(f"{r.fail_rate_30:.0%} of recent payments failed")
        if r.c_new_same_cat_30 > 0:
            reasons.append(f"started paying at a new {r.category}")
        if r.c_qc_trend > 1.5 and r.c_qc_share_30 >= 0.1:
            reasons.append("buying more on quick-commerce apps")
        if r.c_other_trend < 0.5:
            reasons.append("quieter on the app overall")
        out.append("; ".join(reasons[:3]) if reasons else "pattern similar to past lapsers")
    return pd.Series(out, index=f.index)
