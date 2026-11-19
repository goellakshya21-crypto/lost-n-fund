"""Synthetic UPI data so the pipeline runs end to end without real data.

`generate_dataset` writes the files production would provide:
  transactions.csv     payer side: our app users' payments at any shop
  merchants.csv        shops (incl. whether they use our QR/Soundbox = `paytm_acquired`)
  customers.csv        consent flags, language
  app_activity.csv     weekly app opens, bill payments and P2P transfers per user (optional table)
  merchant_daily.csv   acquirer side: daily payments into our shops from ANY UPI app (optional table)
plus hidden ground truth (files starting with "_") that ONLY the simulators and the
diagnosis accuracy check read. The model never sees the ground truth.

Why customers fade (the thing the pipeline has to diagnose):
  shop_switch  stops going to their main shop, starts at another kirana or a quick-commerce app
  cash_shift   stops paying their main shop by UPI (cash), nothing else changes
  app_switch   keeps shopping everywhere but pays with another UPI app; still pays bills on ours
  friction     payments start failing, then they move to another app
  gone         stops everything (moved city, lost phone)
plus two things that look like churn but aren't the customer's choice:
  travel       migrants go home for Chhath for 2-3 weeks, then come back
  decline      their shop is failing (footfall down, Soundbox offline)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LOCALITIES = [
    "Lajpat Nagar", "Karol Bagh", "Rohini Sec 7", "Dwarka Sec 10", "Saket", "Mayur Vihar",
    "Janakpuri", "Pitampura", "Noida Sec 18", "Gurugram Sec 14", "Indirapuram", "Vasant Kunj",
]
OUT_OF_TOWN = ["Patna", "Gaya", "Varanasi", "Gorakhpur"]

# share of merchants, median bill (Rs), how often a customer goes there, chance a customer uses one
CATEGORIES = {
    "kirana":      dict(share=0.42, ticket=220,  propensity=1.00, adopt=1.00, budget=1500),
    "pharmacy":    dict(share=0.12, ticket=380,  propensity=0.22, adopt=0.65, budget=2000),
    "restaurant":  dict(share=0.12, ticket=550,  propensity=0.25, adopt=0.60, budget=4000),
    "cafe":        dict(share=0.09, ticket=160,  propensity=0.35, adopt=0.50, budget=1500),
    "salon":       dict(share=0.07, ticket=450,  propensity=0.08, adopt=0.35, budget=3000),
    "fuel":        dict(share=0.05, ticket=700,  propensity=0.20, adopt=0.40, budget=2000),
    "apparel":     dict(share=0.08, ticket=1400, propensity=0.05, adopt=0.30, budget=6000),
    "electronics": dict(share=0.05, ticket=2800, propensity=0.02, adopt=0.15, budget=8000),
}
TICKET = {**{c: v["ticket"] for c, v in CATEGORIES.items()}, "quick_commerce": 450}

NAME_PARTS = {
    "kirana": (["Shree Ganesh", "Gupta", "Sharma", "Balaji", "Maa Durga", "Agarwal", "Bansal",
                "Krishna", "Laxmi", "Jain", "Mittal", "Om Sai"],
               ["Kirana Store", "General Store", "Provision Store", "Super Mart", "Kirana & Dairy"]),
    "pharmacy": (["Jeevan", "Sanjeevani", "City", "Arogya", "Shiv", "New Life", "Apna"],
                 ["Medicos", "Medical Store", "Chemist"]),
    "restaurant": (["Annapurna", "Punjabi", "Sagar", "Haveli", "Dilli", "Swad"],
                   ["Dhaba", "Restaurant", "Bhojnalaya", "Rasoi"]),
    "cafe": (["Chai", "Masala", "Kulhad", "Cutting", "Brew"], ["Point", "Corner", "Adda", "Junction"]),
    "salon": (["Looks", "Style", "Glamour", "Royal", "Smart"], ["Salon", "Hair Studio", "Parlour"]),
    "fuel": (["Hari Om", "Highway", "Ganga", "Sai Kripa"], ["Fuel Station", "Filling Station"]),
    "apparel": (["Fashion", "Trends", "Libas", "Kapda"], ["Hub", "House", "Emporium"]),
    "electronics": (["Digital", "Mobile", "Tech", "Star"], ["Electronics", "World", "Zone"]),
}
QC_NAMES = ["QuickBasket (online)", "TenMin Mart (online)", "InstaGrocer (online)"]
FIRST_NAMES = ["Rahul", "Priya", "Amit", "Neha", "Vikas", "Pooja", "Suresh", "Anjali", "Rohit", "Kavita",
               "Manoj", "Sunita", "Arjun", "Deepika", "Sanjay", "Meera", "Karan", "Ritu", "Ajay", "Swati",
               "Nitin", "Geeta", "Ravi", "Shalini", "Vivek", "Asha", "Harish", "Nisha", "Gaurav", "Rekha"]
SURNAMES = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Agarwal", "Mehta", "Jain", "Yadav", "Chauhan",
            "Malhotra", "Arora", "Bhatia", "Kapoor", "Saxena", "Mishra", "Pandey", "Joshi", "Rawat", "Sethi"]

PAYMENT_MODES = (["UPI", "WALLET", "CARD"], [0.80, 0.11, 0.09])
HOURS = np.arange(7, 23)
HOUR_W = np.array([2, 4, 6, 7, 7, 6, 5, 5, 6, 7, 9, 10, 10, 8, 5, 3], dtype=float)
HOUR_W /= HOUR_W.sum()

FADING = ("shop_switch", "cash_shift", "app_switch", "gone", "friction")
# how much an offer can do for each kind of fade (multiplies the customer's own responsiveness)
TYPE_RESPONSE = {"none": 0.0, "shop_switch": 1.0, "cash_shift": 0.85, "app_switch": 0.85, "friction": 0.2, "gone": 0.1}
DECLINE_RESPONSE = 0.15
CHUNK = 4000


def _season(start: pd.Timestamp, day_idx: np.ndarray, festive: list[str] | None) -> np.ndarray:
    dates = start + pd.to_timedelta(day_idx, unit="D")
    mult = np.where(dates.weekday >= 5, 1.15, 1.0)
    if festive:
        lo, hi = pd.Timestamp(festive[0]), pd.Timestamp(festive[1])
        mult = mult * np.where((dates >= lo) & (dates <= hi), 1.3, 1.0)
    return np.asarray(mult, dtype=float)


def _decay(t: np.ndarray, onset: np.ndarray, tau) -> np.ndarray:
    tau = np.broadcast_to(np.asarray(tau, dtype=float), onset.shape)
    d = np.exp(-np.maximum(t[None, :] - onset[:, None], 0) / tau[:, None])
    d[d < 0.03] = 0.0
    return d


def _factors(tr: dict, t: np.ndarray, travel: bool = True) -> dict:
    """Activity multipliers per customer x day for each payment stream.

    P = main shop, R = their other regular shops, N = the new shop they switch to,
    T = out-of-town spending while travelling, S = visits to the main shop paid with ANOTHER app
    (only visible on the acquirer side).
    """
    typ = tr["type"]
    n = len(typ)
    d = _decay(t, tr["onset"], tr["tau"])

    def is_(*names):
        return np.isin(typ, names)[:, None]

    P = np.where(is_(*FADING), d, 1.0)
    R = np.where(is_("app_switch", "gone", "friction"), d, 1.0)
    N = np.where(is_("shop_switch"), (1.0 - d) * 0.9, 0.0)
    S = np.where(is_("app_switch", "friction"), 1.0 - d, 0.0)
    # the main shop itself is failing: footfall falls to 30%, its customers drift to another kirana
    md = 0.3 + 0.7 * _decay(t, tr["decl_onset"], 15.0)
    still_here = np.where(is_("none", "shop_switch", "cash_shift"), 1.0, d)
    P = P * md
    N = np.clip(N + (1.0 - md) * 0.9 * still_here, 0.0, 1.0)
    T = np.zeros_like(P)
    if travel:
        away = ((t[None, :] >= tr["travel_start"][:, None]) & (t[None, :] < tr["travel_end"][:, None])).astype(float)
        P, R, N, S = P * (1 - away), R * (1 - away), N * (1 - away), S * (1 - away)
        T = 0.7 * away
    return {"P": P, "R": R, "N": N, "T": T, "S": S, "n": n}


def _fail_p(typ: np.ndarray, onset: np.ndarray, abs_day: np.ndarray) -> np.ndarray:
    near = (abs_day >= onset - 25) & (abs_day < onset + 60)
    return np.where(near & (typ == "friction"), 0.25, np.where(near & np.isin(typ, FADING), 0.035, 0.012))


def _padded(groups: list[np.ndarray], weights: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(max(len(g) for g in groups), 1)
    ids = np.full((len(groups), width), -1, dtype=np.int64)
    cum = np.ones((len(groups), width))
    sizes = np.array([len(g) for g in groups])
    for i, (g, w) in enumerate(zip(groups, weights)):
        if len(g):
            ids[i, : len(g)] = g
            cum[i, : len(g)] = np.cumsum(w / w.sum())
    return ids, cum, sizes


def _rows(counts: np.ndarray, t0: int) -> tuple[np.ndarray, np.ndarray]:
    r, d = np.nonzero(counts)
    reps = counts[r, d]
    return np.repeat(r, reps), np.repeat(d, reps) + t0


def _stream_payments(rng, F: dict, base: np.ndarray, w_p: np.ndarray, t0: int, prim: np.ndarray,
                     new_m: np.ndarray, oot_m: np.ndarray, rest_pad: tuple) -> list[pd.DataFrame]:
    """Draw payments for one chunk of customers. Returns (row, merchant_idx, abs_day) frames per stream."""
    ids, cum, sizes = rest_pad
    out = []
    for key, rate, merchant_of in (
        ("P", base * w_p[:, None] * F["P"], lambda r: prim[r]),
        ("N", base * w_p[:, None] * F["N"], lambda r: new_m[r]),
        ("T", base * F["T"], lambda r: oot_m[r]),
    ):
        r, day = _rows(rng.poisson(rate), t0)
        keep = merchant_of(r) >= 0
        out.append(pd.DataFrame({"row": r[keep], "m": merchant_of(r)[keep], "abs_day": day[keep]}))
    r, day = _rows(rng.poisson(base * (1 - w_p)[:, None] * F["R"]), t0)
    has = sizes[r] > 0
    r, day = r[has], day[has]
    u = rng.random(len(r))
    pick = np.minimum((cum[r] < u[:, None]).sum(axis=1), sizes[r] - 1)
    out.append(pd.DataFrame({"row": r, "m": ids[r, pick], "abs_day": day}))
    return out


def _finish_payments(rng, parts: pd.DataFrame, start: pd.Timestamp, cust_ids: np.ndarray, m_ids: np.ndarray,
                     m_cat: np.ndarray, spend_mult: np.ndarray, typ: np.ndarray, onset: np.ndarray) -> pd.DataFrame:
    n = len(parts)
    rows = parts["row"].to_numpy()
    m = parts["m"].to_numpy()
    abs_day = parts["abs_day"].to_numpy()
    medians = np.array([TICKET[c] for c in m_cat[m]], dtype=float)
    amount = np.round(medians * spend_mult[rows] * rng.lognormal(0, 0.45, n), 0).clip(10)
    status = np.where(rng.random(n) < _fail_p(typ[rows], onset[rows], abs_day), "FAILED", "SUCCESS")
    ts = (start + pd.to_timedelta(abs_day, unit="D") + pd.to_timedelta(rng.choice(HOURS, n, p=HOUR_W), unit="h")
          + pd.to_timedelta(rng.integers(0, 3600, n), unit="s"))
    return pd.DataFrame({"customer_id": cust_ids[rows], "merchant_id": m_ids[m], "amount": amount, "ts": ts,
                         "status": status, "payment_mode": rng.choice(PAYMENT_MODES[0], n, p=PAYMENT_MODES[1])})


def _truth_dict(truth: pd.DataFrame, m_index: dict) -> dict:
    return {
        "type": truth["type"].to_numpy(),
        "onset": truth["onset_day"].to_numpy(dtype=float),
        "tau": truth["tau"].to_numpy(dtype=float),
        "decl_onset": truth["primary_decline_onset"].to_numpy(dtype=float),
        "travel_start": truth["travel_start"].to_numpy(dtype=float),
        "travel_end": truth["travel_end"].to_numpy(dtype=float),
        "lam0": truth["lambda0"].to_numpy(dtype=float),
        "w_p": truth["w_primary"].to_numpy(dtype=float),
        "join": truth["join_day"].to_numpy(dtype=float),
        "spend_mult": truth["spend_mult"].to_numpy(dtype=float),
        "prim": np.array([m_index[m] for m in truth["primary_merchant"]]),
        "new_m": np.array([m_index[m] for m in truth["new_merchant"]]),
        "oot_m": np.array([m_index.get(m, -1) if isinstance(m, str) else -1 for m in truth["out_of_town_merchant"]]),
    }


# ======================================================================================
def generate_dataset(cfg: dict, out_dir: Path) -> dict:
    sc = cfg["synthetic"]
    rng = np.random.default_rng(cfg["seed"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_days = sc["days"]
    end = pd.Timestamp(sc["end_date"])
    start = end - pd.Timedelta(days=n_days)
    day_of = lambda s: (pd.Timestamp(s) - start).days  # noqa: E731

    # ---------------- local merchants ----------------
    n_m = sc["n_merchants"]
    cats = list(CATEGORIES)
    shares = np.array([CATEGORIES[c]["share"] for c in cats])
    m_cat = ["kirana"] * (2 * len(LOCALITIES))
    m_loc = [loc for loc in LOCALITIES for _ in range(2)]
    rest = n_m - len(m_cat)
    m_cat += list(rng.choice(cats, rest, p=shares / shares.sum()))
    m_loc += list(rng.choice(LOCALITIES, rest))
    names, owners = [], []
    for c in m_cat:
        pre, suf = NAME_PARTS[c]
        names.append(f"{rng.choice(pre)} {rng.choice(suf)}")
        owners.append(f"{rng.choice(FIRST_NAMES)} {rng.choice(SURNAMES)}")
    acquired = rng.random(n_m) < 0.85          # shop uses our QR / Soundbox (needed for settlement netting)
    local = pd.DataFrame({
        "merchant_id": [f"M{i:04d}" for i in range(n_m)],
        "name": names, "owner_name": owners, "category": m_cat, "locality": m_loc,
        "enrolled": acquired & (rng.random(n_m) < 0.85),
        "auto_approve": rng.random(n_m) < 0.30,   # "autopilot": approved a monthly budget once
        "monthly_offer_budget": [CATEGORIES[c]["budget"] for c in m_cat],
        "language": rng.choice(["hi", "en"], n_m, p=[0.65, 0.35]),
        "paytm_acquired": acquired,
        "whatsapp_opt_in": rng.random(n_m) < 0.6,
    })
    # quick-commerce apps (online) and out-of-town shops (where migrants spend while home)
    qc = pd.DataFrame({"merchant_id": [f"Q{i:03d}" for i in range(len(QC_NAMES))], "name": QC_NAMES,
                       "owner_name": "-", "category": "quick_commerce", "locality": "Online"})
    oot = pd.DataFrame({"merchant_id": [f"X{i:03d}" for i in range(8)],
                        "name": [f"{rng.choice(NAME_PARTS['kirana'][0])} Store" for _ in range(8)],
                        "owner_name": "-", "category": "kirana",
                        "locality": [OUT_OF_TOWN[i % len(OUT_OF_TOWN)] for i in range(8)]})
    for df in (qc, oot):
        df["enrolled"] = False
        df["auto_approve"] = False
        df["monthly_offer_budget"] = 0
        df["language"] = "en"
        df["paytm_acquired"] = False
        df["whatsapp_opt_in"] = False
    merchants = pd.concat([local, qc, oot], ignore_index=True)
    m_ids = merchants["merchant_id"].to_numpy()
    m_cat_all = merchants["category"].to_numpy()
    m_index = {m: i for i, m in enumerate(m_ids)}

    # some kiranas are failing
    kir = np.where((local["category"] == "kirana").to_numpy())[0]
    n_decl = max(1, int(round(sc.get("declining_merchant_share", 0.04) * len(kir))))
    decl = rng.choice(kir, n_decl, replace=False)
    m_decl_onset = np.full(len(merchants), 10**6, dtype=float)
    m_decl_onset[decl] = rng.integers(n_days - 50, n_days - 20, n_decl)

    # ---------------- customers + their regular shops ----------------
    n_c = sc["n_customers"]
    c_loc = rng.choice(LOCALITIES, n_c)
    by_loc_cat = {k: np.asarray(v) for k, v in local.groupby(["locality", "category"]).groups.items()}
    by_cat = {k: np.asarray(v) for k, v in local.groupby("category").groups.items()}
    qc_idx = np.array([m_index[m] for m in qc["merchant_id"]])
    oot_idx = np.array([m_index[m] for m in oot["merchant_id"]])

    prim = np.zeros(n_c, dtype=np.int64)
    new_m = np.zeros(n_c, dtype=np.int64)
    rest_groups, rest_weights, w_p = [], [], np.zeros(n_c)
    for i in range(n_c):
        local_k = by_loc_cat[(c_loc[i], "kirana")]
        prim[i] = int(rng.choice(local_k))
        others_k = local_k[local_k != prim[i]]
        ms, ws = [], []
        if rng.random() < 0.45:
            ms.append(int(rng.choice(others_k)))
            ws.append(0.5)
        for cat, spec in CATEGORIES.items():
            if cat == "kirana" or rng.random() >= spec["adopt"]:
                continue
            pool = by_loc_cat.get((c_loc[i], cat))
            if pool is None or rng.random() > 0.8:
                pool = by_cat[cat]
            m = int(rng.choice(pool))
            if m not in ms:
                ms.append(m)
                ws.append(spec["propensity"] * rng.lognormal(0, 0.4))
        w_p[i] = 1.6 / (1.6 + sum(ws))
        rest_groups.append(np.array(ms, dtype=np.int64))
        rest_weights.append(np.array(ws, dtype=float))
        # where they would go if they left their main shop: another local kirana or a quick-commerce app
        unused = np.setdiff1d(others_k, ms)
        new_m[i] = int(rng.choice(qc_idx)) if (rng.random() < 0.4 or len(unused) == 0) else int(rng.choice(unused))

    cust_ids = np.array([f"C{i:05d}" for i in range(n_c)])
    consent = rng.random(n_c) < 0.80
    customers = pd.DataFrame({
        "customer_id": cust_ids,
        "name": [f"{rng.choice(FIRST_NAMES)} {rng.choice(SURNAMES)}" for _ in range(n_c)],
        "locality": c_loc,
        # explicit, purpose-specific consent: "use my payments to send me offers from shops I visit"
        "offers_consent": consent,
        # separate optional consent: "shops I visit may see that I am one of their regulars"
        "share_with_merchants": consent & (rng.random(n_c) < 0.35),
        # from KYC; only the flag is needed, not the date of birth (data minimisation)
        "is_minor": rng.random(n_c) < 0.03,
        "language": rng.choice(["hi", "en"], n_c, p=[0.55, 0.45]),
    })

    # ---------------- hidden behaviour (ground truth) ----------------
    types = list(sc["disengage_types"])
    tp = np.array([sc["disengage_types"][k] for k in types], dtype=float)
    is_dis = rng.random(n_c) < sc["disengage_share"]
    typ = np.where(is_dis, rng.choice(types, n_c, p=tp / tp.sum()), "none")
    travels = np.isin(c_loc, sc.get("travel_localities", [])) & (rng.random(n_c) < sc.get("travel_share", 0.0))
    tw = sc.get("travel_window") or [str(end.date()), str(end.date())]
    t_start = np.where(travels, day_of(tw[0]) + rng.integers(-3, 5, n_c), 10**6).astype(float)
    t_end = np.where(travels, t_start + rng.integers(12, 23, n_c), 10**6).astype(float)
    truth = pd.DataFrame({
        "customer_id": cust_ids,
        "type": typ,
        "lambda0": np.clip(rng.lognormal(np.log(0.22), 0.7, n_c), 0.02, 2.5),   # payments per day
        "w_primary": w_p,
        "primary_merchant": m_ids[prim],
        "new_merchant": m_ids[new_m],
        "out_of_town_merchant": np.where(travels, m_ids[rng.choice(oot_idx, n_c)], None),
        "join_day": np.where(rng.random(n_c) < 0.7, 0, rng.integers(0, n_days - 40, n_c)),
        "onset_day": np.where(is_dis, rng.integers(20, n_days + 100, n_c), 10**6),
        "tau": rng.uniform(6, 30, n_c),
        "responsiveness": rng.beta(2, 5, n_c),        # how winnable they are, before the type multiplier
        "spend_mult": rng.lognormal(0, 0.3, n_c),
        "travels": travels, "travel_start": t_start, "travel_end": t_end,
        "primary_decline_onset": m_decl_onset[prim],
        "bills_per_week": rng.lognormal(np.log(0.5), 0.5, n_c),
        "p2p_per_week": rng.lognormal(np.log(0.8), 0.6, n_c),
        "opens_per_week": rng.lognormal(np.log(6), 0.5, n_c),
    })
    tr = _truth_dict(truth, m_index)
    rest_pad = _padded(rest_groups, rest_weights)

    # ---------------- payments, in chunks of customers ----------------
    t = np.arange(n_days)
    season = _season(start, t, sc.get("festive_window"))
    parts, shadow = [], np.zeros((len(merchants), n_days))
    for lo in range(0, n_c, CHUNK):
        sl = slice(lo, min(lo + CHUNK, n_c))
        sub = {k: v[sl] for k, v in tr.items()}
        F = _factors(sub, t)
        base = sub["lam0"][:, None] * season[None, :] * (t[None, :] >= sub["join"][:, None])
        pad = tuple(a[sl] for a in rest_pad)
        for p in _stream_payments(rng, F, base, sub["w_p"], 0, sub["prim"], sub["new_m"], sub["oot_m"], pad):
            p["row"] += lo
            parts.append(p)
        # visits to the main shop paid with another app (acquirer side only)
        s_counts = rng.poisson(base * sub["w_p"][:, None] * F["S"])
        np.add.at(shadow, sub["prim"], s_counts)
    parts = pd.concat(parts, ignore_index=True)
    tx = _finish_payments(rng, parts, start, cust_ids, m_ids, m_cat_all, tr["spend_mult"], tr["type"], tr["onset"])
    tx = tx.sort_values("ts", kind="stable").reset_index(drop=True)
    tx.insert(0, "txn_id", [f"T{i:08d}" for i in range(len(tx))])

    # ---------------- weekly app activity (bills, P2P, app opens) ----------------
    n_weeks = n_days // 7
    tw_mid = np.arange(n_weeks) * 7 + 3
    act_rows = []
    for lo in range(0, n_c, CHUNK):
        sl = slice(lo, min(lo + CHUNK, n_c))
        sub = {k: v[sl] for k, v in tr.items()}
        d = _decay(tw_mid, sub["onset"], sub["tau"])
        ty = sub["type"][:, None]
        bills_f = np.select([ty == "gone", ty == "friction"], [d, 0.5 + 0.5 * d], 1.0)
        p2p_f = np.select([ty == "gone", ty == "friction", ty == "app_switch"], [d, 0.3 + 0.7 * d, 0.5 + 0.5 * d], 1.0)
        open_f = np.select([ty == "gone", ty == "friction", ty == "app_switch"], [d, 0.5 + 0.5 * d, 0.6 + 0.4 * d], 1.0)
        alive = tw_mid[None, :] >= sub["join"][:, None]
        tb = truth.iloc[sl]
        bills = rng.poisson(tb["bills_per_week"].to_numpy()[:, None] * bills_f * alive)
        p2p = rng.poisson(tb["p2p_per_week"].to_numpy()[:, None] * p2p_f * alive)
        opens = rng.poisson(tb["opens_per_week"].to_numpy()[:, None] * open_f * alive)
        r, w = np.nonzero(alive)
        act_rows.append(pd.DataFrame({
            "customer_id": cust_ids[sl][r],
            "week_start": (start + pd.to_timedelta(w * 7, unit="D")).strftime("%Y-%m-%d"),
            "app_opens": opens[r, w], "bill_payments": bills[r, w], "p2p_payments": p2p[r, w]}))
    activity = pd.concat(act_rows, ignore_index=True)

    # ---------------- acquirer side: daily payments into our shops from any app ----------------
    md = merchant_daily(tx, merchants, start, n_days, shadow, m_decl_onset, season, rng)

    tx.to_csv(out_dir / "transactions.csv", index=False)
    merchants.to_csv(out_dir / "merchants.csv", index=False)
    customers.to_csv(out_dir / "customers.csv", index=False)
    activity.to_csv(out_dir / "app_activity.csv", index=False)
    md.to_csv(out_dir / "merchant_daily.csv", index=False)
    truth.to_csv(out_dir / "_ground_truth.csv", index=False)
    rest_df = pd.DataFrame({
        "customer_id": np.repeat(cust_ids, [len(g) for g in rest_groups]),
        "merchant_id": m_ids[np.concatenate(rest_groups)] if any(len(g) for g in rest_groups) else [],
        "weight": np.concatenate([w / w.sum() if len(w) else w for w in rest_weights]),
    })
    rest_df.to_csv(out_dir / "_customer_merchants.csv", index=False)
    pd.DataFrame({"merchant_id": m_ids, "decline_onset_day": m_decl_onset,
                  "declining": m_decl_onset < 10**5}).to_csv(out_dir / "_merchant_truth.csv", index=False)
    meta = {"start_date": str(start.date()), "end_date": str(end.date()), "days": n_days,
            "festive_window": sc.get("festive_window"), "travel_window": sc.get("travel_window"),
            "travel_localities": sc.get("travel_localities")}
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"customers": n_c, "merchants": len(merchants), "transactions": len(tx),
            "start": str(start.date()), "end": str(end.date()),
            "types": pd.Series(typ).value_counts().to_dict(), "travellers": int(travels.sum()),
            "declining_merchants": int(n_decl)}


def merchant_daily(tx: pd.DataFrame, merchants: pd.DataFrame, start: pd.Timestamp, n_days: int,
                   shadow: np.ndarray, m_decl_onset: np.ndarray, season: np.ndarray, rng) -> pd.DataFrame:
    """What the acquirer sees: every UPI payment into the shop, whichever app the payer used."""
    acq = np.where(merchants["paytm_acquired"].to_numpy())[0]
    m_ids = merchants["merchant_id"].to_numpy()
    m_index = {m: i for i, m in enumerate(m_ids)}
    ok = tx[tx["status"] == "SUCCESS"]
    day = (ok["ts"].dt.normalize() - start).dt.days.to_numpy()
    mi = ok["merchant_id"].map(m_index).to_numpy()
    paytm_cnt = np.zeros((len(merchants), n_days))
    paytm_gmv = np.zeros((len(merchants), n_days))
    np.add.at(paytm_cnt, (mi, day), 1)
    np.add.at(paytm_gmv, (mi, day), ok["amount"].to_numpy())
    t = np.arange(n_days)
    decline = 0.3 + 0.7 * _decay(t, m_decl_onset, 15.0)
    # most payers at a Paytm-QR shop use PhonePe / GPay: 2.5-5x the Paytm-app volume
    base = paytm_cnt[:, :60].mean(axis=1).clip(0.5) * rng.uniform(2.5, 5.0, len(merchants))
    other = rng.poisson(base[:, None] * season[None, :] * decline) + shadow
    ticket = np.array([TICKET[c] for c in merchants["category"]], dtype=float)
    p_off = 0.02 + 0.45 * (1 - decline)
    online = rng.random(p_off.shape) >= p_off
    rows = []
    for m in acq:
        rows.append(pd.DataFrame({
            "merchant_id": m_ids[m],
            "date": (start + pd.to_timedelta(t, unit="D")).strftime("%Y-%m-%d"),
            "txns_all_apps": (paytm_cnt[m] + other[m]).astype(int),
            "txns_paytm_app": paytm_cnt[m].astype(int),
            "gmv_all_apps": np.round(paytm_gmv[m] + other[m] * ticket[m] * rng.lognormal(0, 0.15, n_days), 0),
            "soundbox_online": online[m].astype(int),
        }))
    return pd.concat(rows, ignore_index=True)


# ======================================================================================
def simulate_post_period(data_dir: Path, assign: pd.DataFrame, merchants: pd.DataFrame, snapshot: pd.Timestamp,
                         horizon_days: int, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DEMO ONLY: simulate the days after the campaign for everyone in the campaign (treatment, control,
    global holdout). Returns (payments, customer events: opt-outs and complaints).

    In production this is replaced by the real payments after the campaign (`measure --post-txns`)
    and the real opt-out / complaint log (`--events`). The measurement code is identical either way.
    """
    data_dir = Path(data_dir)
    if not (data_dir / "_ground_truth.csv").exists():
        raise FileNotFoundError("Outcome simulation needs synthetic ground truth. "
                                "With real data pass --post-txns instead.")
    rng = np.random.default_rng(cfg["seed"] + 7)
    meta = json.loads((data_dir / "_meta.json").read_text(encoding="utf-8"))
    start = pd.Timestamp(meta["start_date"])
    truth_all = pd.read_csv(data_dir / "_ground_truth.csv").set_index("customer_id")
    rest = pd.read_csv(data_dir / "_customer_merchants.csv")
    all_m = pd.read_csv(data_dir / "merchants.csv")
    m_ids = all_m["merchant_id"].to_numpy()
    m_index = {m: i for i, m in enumerate(m_ids)}
    m_cat = all_m["category"].to_numpy()

    per = assign.sort_values("total_ev", ascending=False).drop_duplicates("customer_id").reset_index(drop=True)
    cust_ids = per["customer_id"].to_numpy()
    truth = truth_all.loc[cust_ids].reset_index()
    tr = _truth_dict(truth, m_index)
    n = len(cust_ids)
    rg = rest[rest["customer_id"].isin(set(cust_ids))].groupby("customer_id")
    groups, weights = [], []
    for c in cust_ids:
        if c in rg.groups:
            g = rg.get_group(c)
            groups.append(np.array([m_index[m] for m in g["merchant_id"]], dtype=np.int64))
            weights.append(g["weight"].to_numpy(dtype=float))
        else:
            groups.append(np.array([], dtype=np.int64))
            weights.append(np.array([], dtype=float))
    rest_pad = _padded(groups, weights)

    t0 = (snapshot.normalize() - start).days
    t = t0 + np.arange(horizon_days)
    F = _factors(tr, t, travel=False)

    delivered = per["delivered"].fillna(False).to_numpy().astype(bool)
    V = per["validity_days"].fillna(10).to_numpy().astype(int)
    off_m = np.array([m_index[m] for m in per["merchant_id"]])
    is_primary = off_m == tr["prim"]
    level = per["level"].fillna(0.5).to_numpy(dtype=float)
    otype = per["offer_type"].fillna("flat").to_numpy()
    in_window = np.arange(horizon_days)[None, :] < V[:, None]
    fac_at_m = np.where(is_primary[:, None], F["P"], F["R"])
    leaving = (fac_at_m * in_window).sum(axis=1) / in_window.sum(axis=1).clip(1) < 0.8
    mult = np.array([TYPE_RESPONSE.get(x, 0.0) for x in tr["type"]])
    declining_main = (tr["decl_onset"] <= t[-1]) & is_primary & (tr["type"] == "none")
    mult = np.where(declining_main, DECLINE_RESPONSE, mult)
    p_save = truth["responsiveness"].to_numpy() * mult * (0.7 + 0.3 * level)
    saved = delivered & leaving & (rng.random(n) < p_save)
    stick = rng.random(n) < (0.55 + 0.10 * (otype == "streak") - 0.05 * (otype == "pct"))
    resp_day = (rng.random(n) * 0.6 * V).astype(int)

    k = np.arange(horizon_days)[None, :]
    after = np.where(stick[:, None], 0.8, 0.85 * np.exp(-np.maximum(k - V[:, None], 0) / 15.0))
    restore = np.where(k < resp_day[:, None], 0.0, np.where(k < V[:, None], 0.85, after)) * saved[:, None]
    whole_app = np.isin(tr["type"], ["app_switch", "friction", "gone"])[:, None]
    F["P"] = np.where(is_primary[:, None] | whole_app, np.maximum(F["P"], restore), F["P"])
    F["N"] = F["N"] * (1 - restore * is_primary[:, None])
    F["R"] = np.where(whole_app, np.maximum(F["R"], restore), F["R"])
    # travel happens regardless of offers
    away = ((t[None, :] >= tr["travel_start"][:, None]) & (t[None, :] < tr["travel_end"][:, None])).astype(float)
    for key in ("P", "R", "N"):
        F[key] = F[key] * (1 - away)
    F["T"] = 0.7 * away

    season = _season(start, t, None)
    base = tr["lam0"][:, None] * season[None, :]
    parts = pd.concat(_stream_payments(rng, F, base, tr["w_p"], t0, tr["prim"], tr["new_m"], tr["oot_m"], rest_pad),
                      ignore_index=True)
    tx = _finish_payments(rng, parts, start, cust_ids, m_ids, m_cat, tr["spend_mult"], tr["type"], tr["onset"])

    # customers who respond to the offer make the qualifying payments it asks for
    oc = cfg["offers"]
    need = np.select([otype == "streak", otype == "pct"], [oc["streak_visits"], oc["pct_uses"]], 1)
    min_bill = per["min_bill"].fillna(0).to_numpy(dtype=float)
    extra = []
    for i in np.where(saved)[0]:
        days = np.arange(resp_day[i], max(V[i], resp_day[i] + 1))
        pick = rng.choice(days, size=min(need[i], len(days)), replace=False)
        amt = np.maximum(min_bill[i], np.round(TICKET[m_cat[off_m[i]]] * tr["spend_mult"][i]
                                               * rng.lognormal(0, 0.3, len(pick)), 0))
        extra.append(pd.DataFrame({
            "customer_id": cust_ids[i], "merchant_id": m_ids[off_m[i]], "amount": amt,
            "ts": start + pd.to_timedelta(t0 + pick, unit="D") + pd.to_timedelta(rng.integers(9, 21, len(pick)), unit="h"),
            "status": "SUCCESS", "payment_mode": "UPI"}))
    if extra:
        tx = pd.concat([tx] + extra, ignore_index=True)

    # people who would have come anyway sometimes top a near-miss bill up to the minimum
    live = per[delivered][["customer_id", "merchant_id", "min_bill", "validity_days"]]
    j = tx.reset_index().merge(live, on=["customer_id", "merchant_id"])
    in_val = (j["ts"] < snapshot + pd.to_timedelta(j["validity_days"], unit="D"))
    near = in_val & (j["amount"] < j["min_bill"]) & (j["amount"] >= 0.7 * j["min_bill"])
    topup = near & (rng.random(len(j)) < cfg["economics"]["bill_topup_rate"])
    tx.loc[j.loc[topup, "index"].to_numpy(), "amount"] = j.loc[topup, "min_bill"].to_numpy()

    tx = tx.sort_values("ts", kind="stable").reset_index(drop=True)
    tx.insert(0, "txn_id", [f"P{i:08d}" for i in range(len(tx))])
    tx["refund_of"] = pd.NA

    # ~1% of successful payments at the offer shops are refunded 0-3 days later
    ok = tx[(tx["status"] == "SUCCESS") & tx.set_index(["customer_id", "merchant_id"]).index.isin(
        per.set_index(["customer_id", "merchant_id"]).index)]
    ref = ok[rng.random(len(ok)) < 0.01]
    if len(ref):
        refunds = pd.DataFrame({
            "txn_id": [f"R{i:08d}" for i in range(len(ref))], "customer_id": ref["customer_id"].to_numpy(),
            "merchant_id": ref["merchant_id"].to_numpy(), "amount": ref["amount"].to_numpy(),
            "ts": ref["ts"].to_numpy() + pd.to_timedelta(rng.integers(1, 4, len(ref)), unit="D"),
            "status": "REFUNDED", "payment_mode": ref["payment_mode"].to_numpy(), "refund_of": ref["txn_id"].to_numpy()})
        tx = pd.concat([tx, refunds], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)

    # opt-outs (~1.5% of delivered offers) and complaints (~0.2%)
    ev = []
    for kind, rate in (("opt_out", 0.015), ("complaint", 0.002)):
        who = np.where(delivered & (rng.random(n) < rate))[0]
        ev.append(pd.DataFrame({"customer_id": cust_ids[who], "event": kind,
                                "ts": snapshot + pd.to_timedelta(rng.integers(0, 10, len(who)), unit="D")
                                + pd.to_timedelta(rng.integers(8, 22, len(who)), unit="h")}))
    events = pd.concat(ev, ignore_index=True).sort_values("ts").reset_index(drop=True)
    return tx, events
