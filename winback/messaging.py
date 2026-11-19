"""P -> A: merchant alerts.  A -> P: approvals.  P -> B: customer offers.

Privacy by design (NPCI UPI rules + DPDP Act):
  * The lapse score uses B's payments at OTHER shops too, so telling shop A "Pooja is slipping
    away" would share Pooja's payment data with a third party. Merchants see counts and ranges only.
  * Counts are shown as bands ("5-9 regulars"), never exact, so week-to-week changes can't single
    anyone out. Bill / visit ranges only appear for groups of 10+. Costs are rounded up to Rs 50 so
    the count can't be backed out of the cost.
  * The shop approves a budget for the whole group. Paytm picks the customers and sends the offer.
  * Cashback is settled as one weekly netting line, so a redemption doesn't reveal who was picked.
  * Merchant numbers are cautious (25th percentile), not averages: over-promising loses merchants.
  * Customer messages: in the customer's language, real dates only (no fake urgency), every message
    says how to stop. No phone numbers anywhere; delivery is in-app (push + shown when they scan).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

OPT_OUT_TEXT = {
    "en": "Don't want these? Tap 'Stop these offers' or go to Settings > Offers.",
    "hi": "Ye offers nahi chahiye? 'Stop these offers' dabayein ya Settings > Offers mein jaayein.",
}
WHY_TEXT = {
    "en": "You pay at this shop regularly and agreed to get offers based on your Paytm payments. "
          "Paytm chose you; the shop does not know who received this offer.",
    "hi": "Aap is dukaan par regular payment karte hain aur aapne Paytm payments ke basis par offers ke liye haan kaha tha. "
          "Paytm ne aapko chuna hai; dukaandaar ko nahi pata ki ye offer kise mila.",
}


def mask_name(full_name: str) -> str:
    parts = str(full_name).split()
    return parts[0] if len(parts) == 1 else f"{parts[0]} {parts[-1][0]}."


def _rs(x: float) -> str:
    return f"Rs {x:,.0f}"


def count_band(n: int, bands: list[int]) -> str:
    """5 -> '5-9', 12 -> '10-19', 70 -> '50+'. Below the first band: 'a few'."""
    if n < bands[0]:
        return "a few"
    for lo, hi in zip(bands, bands[1:]):
        if lo <= n < hi:
            return f"{lo}-{hi - 1}"
    return f"{bands[-1]}+"


def round_up(x: float, step: int) -> int:
    return int(step * math.ceil(max(x, 0) / step)) if step > 1 else int(round(x))


def round_down(x: float, step: int) -> int:
    return int(step * math.floor(max(x, 0) / step)) if step > 1 else int(round(x))


def offer_terms(r, lang: str = "en", valid_till=None) -> str:
    """One customer's offer in plain words."""
    till = f"{pd.Timestamp(valid_till):%d %b}" if valid_till is not None else ""
    shop = r.merchant_name
    if r.offer_type == "streak":
        if lang == "hi":
            return (f"{till} tak {shop} par {int(r.streak_visits)} alag dino Paytm se pay karein "
                    f"(har baar {_rs(r.min_bill)}+), aur {_rs(r.cashback)} cashback paayein.")
        return (f"Pay at {shop} on {int(r.streak_visits)} different days by {till} "
                f"({_rs(r.min_bill)}+ each) and get {_rs(r.cashback)} cashback.")
    if r.offer_type == "pct":
        if lang == "hi":
            return (f"{till} tak {shop} par agle {int(r.uses)} payments ({_rs(r.min_bill)}+) par "
                    f"{r.pct:g}% cashback ({_rs(r.cap)} tak).")
        return (f"Get {r.pct:g}% cashback (up to {_rs(r.cap)}) on your next {int(r.uses)} payments of "
                f"{_rs(r.min_bill)}+ at {shop}, until {till}.")
    if lang == "hi":
        return (f"{shop} par {_rs(r.min_bill)} ya zyada ke agle payment par {_rs(r.cashback)} cashback. "
                f"{till} tak Paytm se pay karein, cashback apne aap mil jayega.")
    return (f"Get {_rs(r.cashback)} cashback on your next payment of {_rs(r.min_bill)} or more at {shop}. "
            f"Pay with Paytm by {till}; cashback is credited automatically.")


def offer_mix_text(g: pd.DataFrame, lang: str = "en") -> str:
    """What the shop is approving, without per-customer detail."""
    parts = []
    for t, label_en, label_hi in (("streak", "after {k} visits", "{k} visits ke baad"),
                                  ("flat", "on the next bill", "agle bill par"),
                                  ("pct", "% on the next {u} bills", "% agle {u} bills par")):
        s = g[g["offer_type"] == t]
        if s.empty:
            continue
        if t == "pct":
            lo, hi = s["pct"].min(), s["pct"].max()
            amt = f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"
            lab = (label_hi if lang == "hi" else label_en).format(u=int(s["uses"].iloc[0]))
            parts.append(f"{amt}{lab}")
        else:
            lo, hi = s["cashback"].min(), s["cashback"].max()
            amt = _rs(lo) if lo == hi else f"{_rs(lo)}-{hi:,.0f}"
            lab = (label_hi if lang == "hi" else label_en).format(k=int(s["streak_visits"].max() if t == "streak" else 1))
            parts.append(f"{amt} cashback {lab}")
    return "; ".join(parts)


def merchant_texts(owner_first: str, band: str, mix: str, exp_cost: int, budget_cap: int, sales: list[int],
                   cofunded: bool, autopilot: bool) -> dict:
    """The P -> A alert in Hinglish and English, plus a short Soundbox voice line. Bands only, never names."""
    co_en = " (Paytm pays half the cashback for some of them)" if cofunded else ""
    co_hi = " (kuch grahakon ka aadha cashback Paytm dega)" if cofunded else ""
    ap_en = " Autopilot is on: this runs within your monthly budget unless you pause it." if autopilot else ""
    ap_hi = " Autopilot on hai: ye aapke monthly budget ke andar apne aap chalega, chahein to rok sakte hain." if autopilot else ""
    return {
        "hi": {
            # one line for the app card (budget, cost and sales sit in tiles right below it)
            "short": (f"Aapke {band} regular grahak kam aa rahe hain. Unhe Lost & Fund cashback bhejein? "
                      f"Paytm grahak chunega, naam aapko nahi dikhenge."),
            "message": (f"Namaste {owner_first} ji! Aapke {band} regular grahak kuch dinon se kam aa rahe hain. "
                        f"Unhe Lost & Fund cashback bhejein ({mix})? Paytm khud grahak chunega aur message karega "
                        f"(aapko naam ya number nahi dikhenge). Budget zyada se zyada {_rs(budget_cap)}; "
                        f"anumanit kharcha lagbhag {_rs(exp_cost)}{co_hi}. Extra bikri ka andaaza: {_rs(sales[0])}-"
                        f"{sales[1]:,}/mahina. Cashback hafte mein ek baar settlement se katega.{ap_hi} "
                        f"Approve karein: Paytm for Business > Lost & Fund."),
            "voice": "Paytm alert. Aapke kuch purane grahakon ke liye Lost and Fund offer taiyaar hai. App mein dekhein.",
        },
        "en": {
            "short": f"{band} of your regulars are coming less. Send them Lost & Fund cashback? Paytm picks them; you won't see names.",
            "message": (f"Hi {owner_first}! {band} of your regular customers have been coming less. "
                        f"Send them Lost & Fund cashback ({mix})? Paytm picks and notifies them "
                        f"(you won't see names or numbers). Budget up to {_rs(budget_cap)}; expected cost about "
                        f"{_rs(exp_cost)}{co_en}. Likely extra sales: {_rs(sales[0])}-{sales[1]:,}/month. "
                        f"Cashback is netted from your settlement once a week.{ap_en} "
                        f"Approve in Paytm for Business > Lost & Fund."),
            "voice": "Paytm alert. A Lost and Fund offer for some of your regular customers is ready. Please check the app.",
        },
    }


def sales_range(g: pd.DataFrame, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Likely extra sales in the first month: the 25th-75th percentile of a simulation, not a single
    average. For a small group the low end is often Rs 0, and the shop should hear that."""
    rng = np.random.default_rng(seed)
    p = g["recovered_prob"].to_numpy()
    v = g["monthly_gmv_at_merchant"].to_numpy()
    sims = (rng.random((draws, len(p))) < p) @ v
    return float(np.quantile(sims, 0.25)), float(np.quantile(sims, 0.75))


def segment_summary(g: pd.DataFrame, cfg: dict) -> dict:
    """What a merchant may see about the group: a band, and ranges only for big enough groups."""
    comp = cfg["compliance"]
    n = len(g)
    out = {"customers_band": count_band(n, comp["count_bands"])}
    if n >= comp["min_customers_for_ranges"]:
        bill = g["avg_ticket_at_merchant"]
        out.update({
            "usual_visits_per_month": round(float(g["visits_per_month"].median())),
            "typical_days_since_last_visit": int(round(float(g["last_visit_days"].median()), -1) or 10),
            "typical_bill_range": [int(round(bill.quantile(0.25), -1)), int(round(bill.quantile(0.75), -1))],
        })
    return out


def build_merchant_alerts(assign: pd.DataFrame, merchants: pd.DataFrame, customers: pd.DataFrame,
                          cfg: dict, snapshot: pd.Timestamp) -> list[dict]:
    """One alert per shop about its at-risk regulars (treatment group of shop-approved offers only)."""
    comp = cfg["compliance"]
    mode = comp["merchant_view"]
    step = comp["cost_rounding"]
    treat =assign[(assign["group"] == "treatment") & assign["needs_merchant_approval"].astype(bool)]
    treat = treat.merge(customers[["customer_id", "name"]], on="customer_id", how="left")
    m = merchants.set_index("merchant_id")
    respond_by = (snapshot + pd.Timedelta(days=2)).date()
    alerts = []
    for mid, g in treat.groupby("merchant_id"):
        mm = m.loc[mid]
        owner_first = str(mm["owner_name"]).split()[0]
        lang = mm["language"] if mm["language"] in ("hi", "en") else "en"
        band = count_band(len(g), comp["count_bands"])
        exp_cost = round_up(float((g["expected_cost"] * (1 - g["paytm_share"])).sum()), step)
        budget_cap = round_up(float(g["max_merchant_liability"].sum()), step)
        lo, hi = sales_range(g, seed=int(hashlib.md5(mid.encode()).hexdigest()[:8], 16))
        sales = [round_down(lo, step), max(round_up(hi, step), round_down(lo, step) + step)]
        cofunded = bool((g["paytm_share"] > 0).any())
        autopilot = bool(mm["auto_approve"])
        texts = merchant_texts(owner_first, band, offer_mix_text(g, lang), exp_cost, budget_cap, sales,
                               cofunded, autopilot)[lang]
        named = g[g["share_with_merchants"].astype(bool)] if mode == "named" else g.iloc[0:0]
        channels = ["business_app_push", "soundbox_voice"] + (["whatsapp_template"] if mm["whatsapp_opt_in"] else [])
        alerts.append({
            "merchant_id": mid, "merchant_name": mm["name"], "category": mm["category"], "locality": mm["locality"],
            "language": mm["language"], "autopilot": autopilot, "channels": channels, "merchant_view": mode,
            "message": texts["message"], "soundbox_voice": texts["voice"], "respond_by": str(respond_by),
            "decision": "approve or decline a budget for the whole group; Paytm chooses and notifies the customers",
            "customers_band": band, "offer_mix": offer_mix_text(g, lang),
            "expected_cost": exp_cost, "budget_cap": budget_cap, "extra_sales_range_per_month": sales,
            "cofunded_by_paytm": cofunded, "settlement": "one netting line per week",
            "segment": segment_summary(g, cfg),
            "named_customers": [
                {"customer": mask_name(r.name), "usual_visits_per_month": float(r.visits_per_month),
                 "offer": offer_terms(r, "en", snapshot + pd.Timedelta(days=int(r.validity_days) - 1))}
                for r in named.sort_values("total_ev", ascending=False).itertuples()],
        })
    return sorted(alerts, key=lambda a: a["merchant_name"])


def alerts_preview(alerts: list[dict], limit: int = 5) -> str:
    lines = []
    for a in alerts[:limit]:
        s = a["segment"]
        lines.append(f"=== {a['merchant_name']} ({a['category']}, {a['locality']}) -> {a['merchant_id']}  "
                     f"[{a['merchant_view']}{', autopilot' if a['autopilot'] else ''}]")
        lines.append(a["message"])
        lines.append(f"[Soundbox] {a['soundbox_voice']}")
        detail = f"   {a['customers_band']} customers | offer: {a['offer_mix']}"
        if "typical_bill_range" in s:
            detail += (f" | ~{s['usual_visits_per_month']} visits/month | bill Rs {s['typical_bill_range'][0]}-"
                       f"{s['typical_bill_range'][1]}")
        lines.append(detail)
        for c in a["named_customers"]:
            lines.append(f"   - {c['customer']} (agreed to be visible): {c['offer']}")
        lines.append("")
    return "\n".join(lines)


def simulate_merchant_decisions(assign: pd.DataFrame, merchants: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """DEMO ONLY: stand-in for merchants tapping approve / decline (one decision per shop).
    Autopilot shops approved a monthly budget once, so their offers go out automatically."""
    rng = np.random.default_rng(cfg["seed"] + 202)
    sim = cfg["merchant_simulation"]
    need = assign[(assign["group"] == "treatment") & assign["needs_merchant_approval"].astype(bool)]
    m = merchants.set_index("merchant_id")
    rows = []
    for mid in sorted(need["merchant_id"].unique()):
        auto = bool(m.loc[mid, "auto_approve"])
        ok = auto or rng.random() < sim["approve_prob"]
        rows.append({"merchant_id": mid, "decision": "approved" if ok else "rejected",
                     "decided_via": "autopilot" if auto else "merchant_app"})
    return pd.DataFrame(rows, columns=["merchant_id", "decision", "decided_via"])


def expand_decisions(decisions: pd.DataFrame, assign: pd.DataFrame) -> pd.DataFrame:
    """Merchant-level (merchant_id, decision) or offer-level (offer_id, decision) -> one row per offer.
    Paytm-funded offers need no shop approval."""
    decisions = decisions.copy()
    decisions["decision"] = decisions["decision"].astype(str).str.lower().replace(
        {"approve": "approved", "yes": "approved", "reject": "rejected", "decline": "rejected",
         "declined": "rejected", "no": "rejected"})
    if "decided_via" not in decisions.columns:
        decisions["decided_via"] = "merchant_app"
    treat = assign[assign["group"] == "treatment"]
    shop = treat[treat["needs_merchant_approval"].astype(bool)][["offer_id", "merchant_id"]]
    if "offer_id" in decisions.columns:
        out = shop.merge(decisions[["offer_id", "decision", "decided_via"]], on="offer_id", how="inner")
    elif "merchant_id" in decisions.columns:
        out = shop.merge(decisions[["merchant_id", "decision", "decided_via"]], on="merchant_id", how="inner")
    else:
        raise ValueError("decisions CSV needs a merchant_id or offer_id column, plus decision")
    paytm = treat[~treat["needs_merchant_approval"].astype(bool)][["offer_id"]].assign(
        decision="approved", decided_via="paytm_programme")
    return pd.concat([out[["offer_id", "decision", "decided_via"]], paytm], ignore_index=True)


HISTORY_COLUMNS = ["customer_id", "sent_on", "run_id", "offer_id", "merchant_id",
                   "max_merchant_liability", "max_paytm_liability", "route"]


def load_contact_history(state_dir: Path) -> pd.DataFrame:
    p = Path(state_dir) / "contact_history.csv"
    if p.exists():
        h = pd.read_csv(p, parse_dates=["sent_on"])
        for c in HISTORY_COLUMNS:
            if c not in h.columns:
                h[c] = 0.0 if "liability" in c else ""
        return h[HISTORY_COLUMNS]
    return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "sent_on" else (float if "liability" in c else str))
                         for c in HISTORY_COLUMNS})


def build_customer_notifications(assign: pd.DataFrame, decisions: pd.DataFrame, customers: pd.DataFrame,
                                 cfg: dict, snapshot: pd.Timestamp, run_id: str,
                                 history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (notifications, updated assignments, updated contact history)."""
    a = assign.drop(columns=[c for c in ("merchant_decision", "decided_via", "delivered", "suppressed_reason",
                                         "valid_from", "valid_till") if c in assign.columns])
    a = a.merge(decisions[["offer_id", "decision", "decided_via"]], on="offer_id", how="left")
    a["merchant_decision"] = np.where(a["group"] != "treatment", "not_sent_" + a["group"],
                                      a["decision"].fillna("no_response"))
    a = a.drop(columns="decision")
    approved = a["merchant_decision"] == "approved"

    # consent can be withdrawn between scoring and sending: re-check right before sending
    still_ok = set(customers.loc[customers["offers_consent"], "customer_id"])
    a["suppressed_reason"] = ""
    a.loc[approved & ~a["customer_id"].isin(still_ok), "suppressed_reason"] = "consent_withdrawn"
    # safety net: caps were applied before randomisation, but another run may have sent since
    prior = history[history["run_id"] != run_id]
    cap_start = snapshot - pd.Timedelta(days=cfg["campaign"]["frequency_cap_days"])
    recently = set(prior.loc[prior["sent_on"] >= cap_start, "customer_id"])
    a.loc[approved & (a["suppressed_reason"] == "") & a["customer_id"].isin(recently), "suppressed_reason"] = "frequency_cap"
    a["delivered"] = approved & (a["suppressed_reason"] == "")
    a["valid_from"] = np.where(a["delivered"], str(snapshot.date()), "")
    till = snapshot + pd.to_timedelta(a["validity_days"] - 1, unit="D")
    a["valid_till"] = np.where(a["delivered"], till.dt.strftime("%Y-%m-%d"), "")

    d = a[a["delivered"]].merge(customers[["customer_id", "name", "language"]].rename(columns={"language": "cust_lang"}),
                                on="customer_id", how="left")
    channel = "in_app_push+shown_at_scan" + ("+sms_dlt" if cfg["compliance"].get("sms_fallback") else "")
    rows = []
    for r in d.itertuples(index=False):
        lang = r.cust_lang if r.cust_lang in ("hi", "en") else "en"
        first = str(r.name).split()[0]
        title = f"{r.merchant_name} par aapke liye offer" if lang == "hi" else f"An offer at {r.merchant_name}"
        hello = f"Namaste {first}! " if lang == "hi" else f"Hi {first}, "
        rows.append({"offer_id": r.offer_id, "customer_id": r.customer_id, "channel": channel, "language": lang,
                     "title": title,
                     "message": hello + offer_terms(r, lang, r.valid_till) + " " + OPT_OUT_TEXT[lang],
                     "why_this_offer": WHY_TEXT[lang], "offer_code": r.offer_id, "valid_till": r.valid_till,
                     "funder": r.funder})
    notes = pd.DataFrame(rows, columns=["offer_id", "customer_id", "channel", "language", "title", "message",
                                        "why_this_offer", "offer_code", "valid_till", "funder"])
    new_hist = pd.DataFrame({"customer_id": d["customer_id"], "sent_on": snapshot, "run_id": run_id,
                             "offer_id": d["offer_id"], "merchant_id": d["merchant_id"],
                             "max_merchant_liability": d["max_merchant_liability"],
                             "max_paytm_liability": d["max_paytm_liability"], "route": d["route"]})
    history = new_hist if prior.empty else pd.concat([prior, new_hist], ignore_index=True)
    return notes, a, history


def weekly_merchant_summaries(board: pd.DataFrame, merchants: pd.DataFrame, cfg: dict) -> list[dict]:
    """The weekly WhatsApp / app summary a shop gets. Bands, not exact counts."""
    m = merchants.set_index("merchant_id")
    out = []
    for r in board.itertuples(index=False):
        mm = m.loc[r.merchant_id]
        band = count_band(int(r.came_back), cfg["compliance"]["count_bands"])
        lang = mm["language"] if mm["language"] in ("hi", "en") else "en"
        suggest = (not bool(mm["auto_approve"])) and r.came_back > 0
        if lang == "hi":
            text = (f"Lost & Fund: {band} purane grahak wapas aaye, unse {_rs(r.sales_from_these_customers)} ki bikri. "
                    f"Cashback {_rs(r.cashback_netted)} settlement se kata."
                    + (" Autopilot on karein: budget ek baar set karein, baaki Paytm sambhalega." if suggest else ""))
        else:
            text = (f"Lost & Fund: {band} regulars came back, {_rs(r.sales_from_these_customers)} in sales from them. "
                    f"Cashback {_rs(r.cashback_netted)} netted from your settlement."
                    + (" Turn on autopilot: set a monthly budget once and Paytm handles the rest." if suggest else ""))
        out.append({"merchant_id": r.merchant_id, "language": lang, "text": text,
                    "channel": "whatsapp_template" if mm["whatsapp_opt_in"] else "business_app"})
    return out
