"""Clickable demo on top of the pipeline's latest campaign run.

    python run_pipeline.py demo        ->  http://127.0.0.1:8000

Three screens share one live state:
  Paytm Ops (P)   at-risk customer-shop pairs, WHY they are slipping and who acts, live KPIs, activity
  Merchant (A)    the WinBack alert: a count band, a budget, cautious numbers; approve / decline / autopilot
  Customer (B)    the offer in their language; pay the full bill, cashback credited by the live ledger

Every payment goes through the same OfferLedger the pipeline uses (min bill, one per day, streaks,
refund clawback, consent). Demo actions are stored in outputs/<run>/demo_state.json and never change
the pipeline's own files. "Export approvals" writes a decisions CSV that the real `decide` stage accepts.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from winback.config import load_config
from winback.diagnosis import ACTION_LABELS, REASON_LABELS
from winback.ledger import OfferLedger
from winback.messaging import WHY_TEXT, count_band, mask_name, merchant_texts, offer_mix_text, offer_terms

STATIC = Path(__file__).resolve().parent / "static"


def latest_run(cfg: dict) -> Path:
    runs = sorted(Path(cfg["paths"]["output_dir"]).glob("campaign_*"))
    if not runs:
        raise SystemExit("No campaign run found. Run `python run_pipeline.py all` first.")
    return runs[-1]


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe records (NaN -> None, numpy -> python)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _rs(x: float) -> str:
    return f"Rs {x:,.0f}"


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


class DecideBody(BaseModel):
    decision: str   # "approve" or "decline": the shop decides a budget for the whole group, never per person


class PayBody(BaseModel):
    offer_id: str
    amount: float


class RefundBody(BaseModel):
    txn_id: str


class Store:
    def __init__(self, cfg: dict, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        self.lock = threading.Lock()
        rd = run_dir
        self.summary = json.loads((rd / "run_summary.json").read_text(encoding="utf-8"))
        self.snapshot = pd.Timestamp(self.summary["snapshot"])
        res = rd / "09_campaign_results.json"
        self.results = json.loads(res.read_text(encoding="utf-8")) if res.exists() else {}
        card = Path(cfg["paths"]["model_dir"]) / "model_card.json"
        self.card = json.loads(card.read_text(encoding="utf-8")) if card.exists() else {}
        self.alerts = {a["merchant_id"]: a for a in json.loads((rd / "05_merchant_alerts.json").read_text(encoding="utf-8"))}

        data = Path(cfg["paths"]["data_dir"])
        self.merchants_df = pd.read_csv(data / "merchants.csv")
        customers = pd.read_csv(data / "customers.csv")
        if "language" not in customers.columns:
            customers["language"] = "en"
        self.merchants = self.merchants_df.set_index("merchant_id")

        a = pd.read_csv(rd / "04_campaign_assignments.csv")
        a = a.drop(columns=[c for c in ("merchant_decision", "decided_via", "delivered", "suppressed_reason",
                                        "valid_from", "valid_till", "name", "customer_locality", "language")
                            if c in a.columns])
        a = a.merge(customers[["customer_id", "name", "locality", "language"]].rename(
            columns={"locality": "customer_locality", "language": "cust_lang"}), on="customer_id")
        a["masked"] = a["name"].map(mask_name)
        a["first_name"] = a["name"].str.split().str[0]
        a["valid_from"] = str(self.snapshot.date())
        a["valid_till"] = (self.snapshot + pd.to_timedelta(a["validity_days"] - 1, unit="D")).dt.strftime("%Y-%m-%d")
        a["merchant_locality"] = a["merchant_id"].map(self.merchants["locality"])
        a["needs_merchant_approval"] = a["needs_merchant_approval"].astype(bool)
        self.assign = a.set_index("offer_id", drop=False)
        self.by_customer: dict[str, str] = dict(zip(a["customer_id"], a["offer_id"]))

        # at-risk table for the ops screen: one row per customer (their offer pair, else their riskiest pair)
        ar = pd.read_csv(rd / "01_at_risk_pairs.csv", usecols=["customer_id", "merchant_id", "lapse_prob", "risk_tier",
                                                                 "reasons", "route", "route_action", "diagnosis_evidence"])
        ar["has_offer"] = ar.set_index(["customer_id", "merchant_id"]).index.isin(
            a.set_index(["customer_id", "merchant_id"]).index)
        ar = ar.sort_values(["has_offer", "lapse_prob"], ascending=False).drop_duplicates("customer_id")
        ar = ar.merge(customers[["customer_id", "name", "locality", "offers_consent"]], on="customer_id", how="left")
        ar["name"] = ar["name"].map(mask_name)   # ops staff see first name + initial only (need-to-know)
        ar["merchant_name"] = ar["merchant_id"].map(self.merchants["name"])
        ar["offer_id"] = ar["customer_id"].map(self.by_customer)
        ar.loc[~ar["has_offer"], "offer_id"] = None
        dropped_path = rd / "03_offers_dropped.csv"
        dropped = pd.read_csv(dropped_path, usecols=lambda c: c in ("customer_id", "merchant_id", "drop_reason")) \
            if dropped_path.exists() and dropped_path.stat().st_size > 5 else pd.DataFrame(columns=["customer_id", "drop_reason"])
        drop_reason = dropped.drop_duplicates("customer_id").set_index("customer_id")["drop_reason"]
        ar["route_label"] = ar["route"].map(REASON_LABELS)
        ar["action_label"] = ar["route_action"].map(ACTION_LABELS)
        ar["no_offer_reason"] = np.where(ar["offer_id"].notna(), "",
                                         np.where(ar["route_action"].isin(["merchant_offer", "paytm_offer"]),
                                                  ar["customer_id"].map(drop_reason).fillna("shop not enrolled"),
                                                  ar["action_label"]))
        ar["has_offer"] = ar["offer_id"].notna()
        self.at_risk = ar.sort_values(["has_offer", "lapse_prob"], ascending=False).reset_index(drop=True)  # actionable first
        self.risk = self.at_risk.set_index("customer_id")

        # a little real history for realism on the phones
        tx = pd.read_csv(data / "transactions.csv", usecols=["customer_id", "merchant_id", "amount", "ts", "status"],
                         parse_dates=["ts"])
        tx = tx[tx["ts"] < self.snapshot]
        mine = tx[tx["customer_id"].isin(self.by_customer)].sort_values("ts")
        recent = mine.groupby("customer_id").tail(4).copy()
        recent["merchant_name"] = recent["merchant_id"].map(self.merchants["name"])
        recent["date"] = recent["ts"].dt.strftime("%d %b")
        self.recent = {c: _records(g[["merchant_name", "amount", "date", "status"]].iloc[::-1])
                       for c, g in recent.groupby("customer_id")}
        last30 = tx[(tx["ts"] >= self.snapshot - pd.Timedelta(days=30)) & tx["status"].eq("SUCCESS")
                    & tx["merchant_id"].isin(set(a["merchant_id"]))]
        self.m30 = last30.groupby("merchant_id").agg(sales_30d=("amount", "sum"), customers_30d=("customer_id", "nunique"))

        self.state_path = rd / "demo_state.json"
        self.state = self._load_state()
        self._rebuild_ledger()

    # ------------------------------------------------------------------ state + ledger
    def _empty_state(self) -> dict:
        return {"decisions": {}, "txns": [], "events": [], "opt_outs": {}, "demo_day": 0}

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                st = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(st.get("opt_outs"), list) or "txns" not in st:   # older demo format
                    return self._empty_state()
                for k, v in self._empty_state().items():
                    st.setdefault(k, v)
                return st
            except (json.JSONDecodeError, OSError):
                pass
        return self._empty_state()

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1, default=str), encoding="utf-8")

    def _rebuild_ledger(self) -> None:
        """Replay the demo's payments and refunds through a fresh ledger: same rules, same results."""
        treat = self.assign[self.assign["group"] == "treatment"]
        self.ledger = OfferLedger(treat.reset_index(drop=True), self.merchants_df, self.cfg,
                                  {c: pd.Timestamp(t) for c, t in self.state["opt_outs"].items()})
        for t in self.state["txns"]:
            if t["kind"] == "pay":
                t["result"] = self.ledger.on_payment(t)
            else:
                t["result"] = self.ledger.on_refund(t["txn_id"], t["refund_of"], t["ts"])

    def _event(self, kind: str, text: str) -> None:
        self.state["events"].append({"ts": datetime.now().strftime("%H:%M:%S"), "kind": kind, "text": text})
        self.state["events"] = self.state["events"][-200:]

    def now(self) -> pd.Timestamp:
        """Demo clock: the campaign day plus the presenter's 'next day' clicks, at the current time of day."""
        t = datetime.now()
        return self.snapshot + pd.Timedelta(days=int(self.state["demo_day"]), hours=t.hour, minutes=t.minute,
                                            seconds=t.second)

    def _ledger_offer(self, oid: str) -> dict | None:
        r = self.assign.loc[oid]
        return self.ledger.offer_for(r["customer_id"], r["merchant_id"])

    def status(self, oid: str) -> str:
        r = self.assign.loc[oid]
        if r["customer_id"] in self.state["opt_outs"]:
            return "opted_out"
        if r["group"] == "control":
            return "holdout"
        if r["group"] == "global_holdout":
            return "global_holdout"
        if r["needs_merchant_approval"]:
            d = self.state["decisions"].get(r["merchant_id"])
            if d is None and bool(self.merchants.loc[r["merchant_id"], "auto_approve"]):
                d = "approved"          # autopilot: the shop approved a monthly budget once
            if d is None:
                return "awaiting"
            if d == "rejected":
                return "rejected"
        o = self._ledger_offer(oid)
        prog = self.ledger.progress(o) if o else {"done": 0, "finished": False}
        if prog["finished"]:
            return "redeemed"
        if self.now().normalize() > pd.Timestamp(r["valid_till"]):
            return "expired"
        return "progress" if prog["done"] > 0 else "approved"

    def _decided_via(self, mid: str) -> str | None:
        if mid in self.state["decisions"]:
            return "merchant_app"
        return "autopilot" if bool(self.merchants.loc[mid, "auto_approve"]) else None

    # ------------------------------------------------------------------ views
    def overview(self) -> dict:
        t = self.assign[self.assign["group"] == "treatment"]
        st = Counter(self.status(o) for o in t["offer_id"])
        shop_t = t[t["needs_merchant_approval"]]
        mids = sorted(shop_t["merchant_id"].unique())
        decided_m = [m for m in mids if self._decided_via(m)]
        ls = self.ledger.stats()
        sales = sum(x["amount"] for x in self.state["txns"] if x["kind"] == "pay" and x["result"].get("applied"))
        v = self.card.get("validation", {})
        it = self.results.get("itt", {})
        r30 = it.get("returned_30") or {}
        return {
            "run": {"run_id": self.summary["run_id"], "snapshot": self.summary["snapshot"],
                    "demo_day": self.state["demo_day"], "demo_date": str(self.now().date())},
            "funnel": self.summary["funnel"],
            "routes": [{"route": k, "label": REASON_LABELS.get(k, k), "n": n,
                        "action": self.cfg["routing"][k]["action"],
                        "action_label": ACTION_LABELS.get(self.cfg["routing"][k]["action"], "")}
                       for k, n in sorted(self.summary.get("routes", {}).items(), key=lambda kv: -kv[1])],
            "actions": self.summary.get("actions", {}),
            "diagnosis_check": self.summary.get("diagnosis_vs_truth"),
            "model": {"auc": v.get("roc_auc"), "lift_top10": v.get("lift_top10pct"),
                      "calibration_gap": v.get("max_calibration_gap"),
                      "lapsers_active_on_app": self.card.get("shop_lapsers_still_active_on_app"),
                      "version": self.summary.get("model", {}).get("version"),
                      "max_psi": self.summary.get("model", {}).get("max_psi"),
                      "top_signals": [f["feature"] for f in self.card.get("feature_importance", [])[:4]]},
            "measured": {"simulated": self.results.get("simulated_outcomes", True),
                         "effect": r30.get("effect"), "ci": r30.get("ci95"), "significant": r30.get("significant"),
                         "treated": r30.get("treated"), "control": r30.get("control"),
                         "power": self.results.get("power"), "truth": self.results.get("simulator_truth")},
            "live": {
                "at_risk": int(len(self.at_risk)),
                "offers": int(len(t)),
                "shop_offers": int(len(shop_t)),
                "paytm_offers": int(len(t) - len(shop_t)),
                "holdout": int((self.assign["group"] == "control").sum()),
                "global_holdout": int((self.assign["group"] == "global_holdout").sum()),
                "awaiting": st["awaiting"],
                "live": st["approved"] + st["progress"] + st["redeemed"],
                "rejected": st["rejected"],
                "redeemed_customers": ls["customers_paid_cashback"],
                "merchants_total": len(mids),
                "merchants_responded": len(decided_m),
                "cashback_total": ls["cashback_total"],
                "cashback_shop": ls["merchant_funded"],
                "cashback_paytm": ls["paytm_funded"],
                "sales_recovered": round(sales, 2),
            },
            "compliance": {**self.summary.get("compliance", {}),
                           "opted_out_during_campaign": len(self.state["opt_outs"]),
                           "funnel": {k: self.summary["funnel"].get(k)
                                      for k in ("customers_total", "eligible_consented_adults")}},
            "events": self.state["events"][-25:][::-1],
        }

    def at_risk_rows(self, tier: str, status: str, route: str, q: str, offset: int, limit: int) -> dict:
        df = self.at_risk
        stat = df["offer_id"].map(lambda o: self.status(o) if isinstance(o, str) else "no_offer")
        mask = pd.Series(True, index=df.index)
        if tier:
            mask &= df["risk_tier"].eq(tier)
        if status:
            mask &= stat.eq(status)
        if route:
            mask &= df["route"].eq(route)
        if q:
            ql = q.lower()
            mask &= (df["name"].str.lower().str.contains(ql, regex=False)
                     | df["customer_id"].str.lower().str.contains(ql, regex=False)
                     | df["merchant_name"].fillna("").str.lower().str.contains(ql, regex=False))
        sub = df[mask].assign(status=stat[mask])
        page = sub.iloc[offset: offset + limit].copy()
        page["offer_type"] = page["offer_id"].map(lambda o: self.assign.loc[o, "offer_type"] if isinstance(o, str) else None)
        page["funder"] = page["offer_id"].map(lambda o: self.assign.loc[o, "funder"] if isinstance(o, str) else None)
        cols = ["customer_id", "name", "locality", "lapse_prob", "risk_tier", "reasons", "route", "route_label",
                "action_label", "diagnosis_evidence", "offer_id", "merchant_id", "merchant_name", "offer_type", "funder",
                "status", "no_offer_reason"]
        return {"total": int(len(sub)), "rows": _records(page[cols]), "status_counts": stat.value_counts().to_dict()}

    def merchant_list(self) -> list[dict]:
        out = []
        for mid, al in self.alerts.items():
            g = self.assign[(self.assign["merchant_id"] == mid) & (self.assign["group"] == "treatment")
                            & self.assign["needs_merchant_approval"]]
            st = Counter(self.status(o) for o in g["offer_id"])
            m = self.merchants.loc[mid]
            out.append({"merchant_id": mid, "name": m["name"], "category": m["category"], "locality": m["locality"],
                        "band": al["customers_band"], "awaiting": st["awaiting"] > 0,
                        "approved": (st["approved"] + st["progress"] + st["redeemed"]) > 0,
                        "rejected": st["rejected"] > 0, "redeemed": st["redeemed"], "autopilot": bool(m["auto_approve"])})
        return sorted(out, key=lambda r: (not r["awaiting"], r["name"]))

    def merchant_detail(self, mid: str) -> dict:
        """'app' is exactly what the shop's app receives (bands, budget, statements). 'presenter' is ops-only."""
        if mid not in self.alerts:
            raise HTTPException(404, "no WinBack alert for this shop")
        al = self.alerts[mid]
        m = self.merchants.loc[mid]
        g = self.assign[(self.assign["merchant_id"] == mid) & (self.assign["group"] == "treatment")
                        & self.assign["needs_merchant_approval"]]
        allm = self.assign[(self.assign["merchant_id"] == mid) & self.assign["needs_merchant_approval"]]
        st = Counter(self.status(o) for o in g["offer_id"])
        owner_first = str(m["owner_name"]).split()[0]
        texts = {lang: merchant_texts(owner_first, al["customers_band"], offer_mix_text(g, lang), al["expected_cost"],
                                      al["budget_cap"], al["extra_sales_range_per_month"], al["cofunded_by_paytm"],
                                      bool(m["auto_approve"]))[lang] for lang in ("hi", "en")}
        oids = set(g["offer_id"])
        pays = [x for x in self.state["txns"] if x["kind"] == "pay" and x["offer_id"] in oids and x["result"].get("applied")]
        came_back = len({x["customer_id"] for x in pays})
        lines = self.ledger.settlement_lines(self.cfg["compliance"]["settlement_netting"])
        lines = lines[lines["merchant_id"] == mid] if len(lines) else lines
        s30 = self.m30.loc[mid] if mid in self.m30.index else None
        return {
            "app": {
                "merchant": {"merchant_id": mid, "name": m["name"], "owner_first": owner_first, "category": m["category"],
                             "locality": m["locality"], "language": m["language"],
                             "sales_30d": float(s30["sales_30d"]) if s30 is not None else 0.0,
                             "customers_30d": int(s30["customers_30d"]) if s30 is not None else 0},
                "texts": texts,
                "band": al["customers_band"],
                "budget_cap": al["budget_cap"], "expected_cost": al["expected_cost"],
                "extra_sales_range": al["extra_sales_range_per_month"],
                "cofunded": al["cofunded_by_paytm"], "autopilot": bool(m["auto_approve"]),
                "segment": al["segment"],
                "offer_mix": {lang: offer_mix_text(g, lang) for lang in ("hi", "en")},
                "named": al.get("named_customers", []),
                "decision": ("awaiting" if st["awaiting"] else "declined" if st["rejected"] else "approved"),
                "decided_via": self._decided_via(mid),
                "results": {"came_back_band": count_band(came_back, [1, 5, 10, 20]) if came_back else "0",
                            "sales": round(sum(x["amount"] for x in pays), 2),
                            "cashback_netted": round(float(lines["winback_cashback_netted"].sum()) if len(lines) else 0.0, 2)},
                "statements": _records(lines[["week_starting", "statement_text"]]) if len(lines) else [],
            },
            "presenter": {"exact_customers": int(len(g)), "holdout_n": int((allm["group"] == "control").sum()),
                          "min_customers": self.cfg["compliance"]["min_customers_per_alert"],
                          "merchant_view": self.cfg["compliance"]["merchant_view"]},
        }

    def decide(self, mid: str, body: DecideBody) -> dict:
        choice = body.decision.strip().lower()
        if choice not in ("approve", "decline"):
            raise HTTPException(400, "decision must be 'approve' or 'decline'")
        with self.lock:
            if mid not in self.alerts:
                raise HTTPException(404, "no WinBack alert for this shop")
            val = "approved" if choice == "approve" else "rejected"
            autopilot = bool(self.merchants.loc[mid, "auto_approve"])
            if self.state["decisions"].get(mid) == val or (val == "approved" and autopilot and mid not in self.state["decisions"]):
                raise HTTPException(400, "this offer was already decided")
            self.state["decisions"][mid] = val
            name = self.merchants.loc[mid, "name"]
            owner = str(self.merchants.loc[mid, "owner_name"]).split()[0]
            band = self.alerts[mid]["customers_band"]
            if val == "approved":
                self._event("approve", f"{owner} ({name}) approved a budget of up to {_rs(self.alerts[mid]['budget_cap'])} "
                                       f"without seeing names. Paytm notified {band} regulars.")
            else:
                self._event("reject", f"{owner} ({name}) {'paused autopilot' if autopilot else 'declined the offer'}. "
                                      "No customers were contacted.")
            self._save()
        return self.merchant_detail(mid)

    def customer_list(self, merchant_id: str = "") -> list[dict]:
        """Paytm-side list for the demo pickers. The shop's app never receives this."""
        a = self.assign if not merchant_id else self.assign[self.assign["merchant_id"] == merchant_id]
        return [{"customer_id": r.customer_id, "masked": r.masked, "merchant_id": r.merchant_id,
                 "merchant_name": r.merchant_name, "funder": r.funder, "status": self.status(r.offer_id)}
                for r in a.itertuples()]

    def _offer_view(self, oid: str) -> dict:
        r = self.assign.loc[oid]
        lang = r["cust_lang"] if r["cust_lang"] in ("hi", "en") else "en"
        o = self._ledger_offer(oid)
        prog = self.ledger.progress(o) if o else {"done": 0, "need": 1, "finished": False}
        counted_today = bool(o and o["last_day"] is not None and o["last_day"] == self.now().normalize())
        title = (f"{r['merchant_name']} par aapke liye offer" if lang == "hi" else f"An offer at {r['merchant_name']}")
        return {
            "offer_id": oid, "merchant_id": r["merchant_id"], "merchant_name": r["merchant_name"],
            "merchant_locality": r["merchant_locality"], "category": r["category"], "offer_type": r["offer_type"],
            "cashback": _f(r["cashback"]), "pct": _f(r["pct"]), "cap": _f(r["cap"]), "uses": int(r["uses"]),
            "streak_visits": int(r["streak_visits"]) if pd.notna(r["streak_visits"]) else None,
            "min_bill": float(r["min_bill"]), "valid_till": r["valid_till"], "validity_days": int(r["validity_days"]),
            "avg_bill": float(r["avg_ticket_at_merchant"]), "status": self.status(oid), "group": r["group"],
            "funder": r["funder"], "language": lang, "progress": prog, "counted_today": counted_today,
            "title": title, "terms": offer_terms(r, lang, r["valid_till"]), "terms_en": offer_terms(r, "en", r["valid_till"]),
            "why": WHY_TEXT[lang],
        }

    def customer_detail(self, cid: str) -> dict:
        oid = self.by_customer.get(cid)
        if not oid:
            raise HTTPException(404, "customer has no WinBack offer")
        r0 = self.assign.loc[oid]
        risk = self.risk.loc[cid] if cid in self.risk.index else None
        return {
            "customer": {"customer_id": cid, "first_name": r0["first_name"], "masked": r0["masked"],
                         "locality": r0["customer_locality"], "language": r0["cust_lang"]},
            "risk": None if risk is None else {
                "lapse_prob": float(risk["lapse_prob"]), "risk_tier": risk["risk_tier"], "reasons": risk["reasons"],
                "route": risk["route"], "route_label": risk["route_label"], "action_label": risk["action_label"],
                "evidence": risk["diagnosis_evidence"]},
            "offers": [self._offer_view(oid)],
            "recent": self.recent.get(cid, []),
            "payments": [{k: x[k] for k in ("txn_id", "kind", "merchant_name", "amount", "ts_label", "refund_of")}
                         | {"cashback": x["result"].get("cashback", 0.0), "reason": x["result"].get("reason"),
                            "reversed": x["result"].get("reversed", 0.0),
                            "refunded": any(y.get("refund_of") == x["txn_id"] for y in self.state["txns"])}
                         for x in self.state["txns"] if x["customer_id"] == cid][::-1],
            "opted_out": cid in self.state["opt_outs"],
            "clock": {"day": self.state["demo_day"], "date": str(self.now().date())},
        }

    def pay(self, body: PayBody) -> dict:
        with self.lock:
            if body.offer_id not in self.assign.index:
                raise HTTPException(404, "unknown offer")
            r = self.assign.loc[body.offer_id]
            st = self.status(body.offer_id)
            if st not in ("approved", "progress", "redeemed", "expired"):
                raise HTTPException(400, f"offer is not live (status: {st})")
            amount = round(float(body.amount), 2)
            if not 1 <= amount <= 100000:
                raise HTTPException(400, "enter an amount between Rs 1 and Rs 1,00,000")
            ts = self.now()
            txn = {"kind": "pay", "txn_id": f"D{len(self.state['txns']) + 1:06d}", "offer_id": body.offer_id,
                   "customer_id": r["customer_id"], "merchant_id": r["merchant_id"], "merchant_name": r["merchant_name"],
                   "amount": amount, "ts": str(ts), "ts_label": ts.strftime("%d %b %H:%M"), "status": "SUCCESS",
                   "payer_app": "paytm", "refund_of": None}
            res = self.ledger.on_payment(txn)
            txn["result"] = res
            self.state["txns"].append(txn)
            reason = res.get("reason")
            if reason == "cashback_credited":
                who = {"merchant": "shop pays, netted weekly", "paytm": "Paytm pays",
                       "cofunded": "shop and Paytm split it"}.get(r["funder"], "")
                self._event("pay", f"{r['masked']} paid {_rs(amount)} at {r['merchant_name']} and got "
                                   f"{_rs(res['cashback'])} cashback ({who}).")
            elif reason == "streak_progress":
                p = res["progress"]
                self._event("pay", f"{r['masked']} paid {_rs(amount)} at {r['merchant_name']}: visit {p['done']} of {p['need']}.")
            else:
                self._event("pay", f"{r['masked']} paid {_rs(amount)} at {r['merchant_name']} (no cashback: "
                                   f"{str(reason).replace('_', ' ')}).")
            self._save()
            o = self._ledger_offer(body.offer_id)
            return {**txn, "result": res, "progress": self.ledger.progress(o) if o else None,
                    "soundbox": f"Paytm par {amount:,.0f} rupaye prapt hue"}

    def refund(self, body: RefundBody) -> dict:
        with self.lock:
            orig = next((x for x in self.state["txns"] if x["txn_id"] == body.txn_id and x["kind"] == "pay"), None)
            if orig is None:
                raise HTTPException(404, "unknown payment")
            if any(x.get("refund_of") == body.txn_id for x in self.state["txns"]):
                raise HTTPException(400, "this payment was already refunded")
            ts = self.now()
            txn = {"kind": "refund", "txn_id": f"DR{len(self.state['txns']) + 1:05d}", "offer_id": orig["offer_id"],
                   "customer_id": orig["customer_id"], "merchant_id": orig["merchant_id"],
                   "merchant_name": orig["merchant_name"], "amount": orig["amount"], "ts": str(ts),
                   "ts_label": ts.strftime("%d %b %H:%M"), "refund_of": body.txn_id}
            res = self.ledger.on_refund(txn["txn_id"], body.txn_id, ts)
            txn["result"] = res
            self.state["txns"].append(txn)
            masked = self.assign.loc[orig["offer_id"], "masked"]
            self._event("refund", f"{masked}'s {_rs(orig['amount'])} payment at {orig['merchant_name']} was refunded"
                                  + (f": {_rs(res['reversed'])} cashback clawed back." if res.get("reversed") else
                                     ": streak visit removed." if res.get("reason") == "streak_day_removed" else "."))
            self._save()
            return {**txn, "result": res}

    def opt_out(self, cid: str) -> dict:
        with self.lock:
            if cid not in self.by_customer:
                raise HTTPException(404, "unknown customer")
            if cid not in self.state["opt_outs"]:
                ts = self.now()
                self.state["opt_outs"][cid] = str(ts)
                self.ledger.opt_out(cid, ts)
                r = self.assign.loc[self.by_customer[cid]]
                self._event("optout", f"{r['masked']} turned off WinBack offers. The offer stopped at once and they are "
                                      "excluded from all future runs.")
                self._save()
        return self.customer_detail(cid)

    def next_day(self) -> dict:
        with self.lock:
            self.state["demo_day"] = int(self.state["demo_day"]) + 1
            self._event("clock", f"Demo clock moved to day {self.state['demo_day'] + 1} ({self.now():%d %b}).")
            self._save()
        return {"day": self.state["demo_day"], "date": str(self.now().date())}

    def reset(self) -> None:
        with self.lock:
            self.state = self._empty_state()
            self._rebuild_ledger()
            self._save()

    def export_decisions(self) -> dict:
        mids = sorted(self.alerts)
        rows = []
        for mid in mids:
            via = self._decided_via(mid)
            if via:
                d = self.state["decisions"].get(mid, "approved")
                rows.append({"merchant_id": mid, "decision": d, "decided_via": via})
        dec = pd.DataFrame(rows, columns=["merchant_id", "decision", "decided_via"])
        path = self.run_dir / "06_demo_merchant_decisions.csv"
        dec.to_csv(path, index=False)
        return {"path": str(path), "rows": len(dec),
                "command": f'python run_pipeline.py decide --run "{self.run_dir}" --decisions "{path}"'}


def create_app(cfg: dict | None = None, run_dir: Path | None = None) -> FastAPI:
    cfg = cfg or load_config()
    store = Store(cfg, Path(run_dir) if run_dir else latest_run(cfg))
    app = FastAPI(title="WinBack demo", docs_url="/api/docs")

    @app.middleware("http")
    async def no_cache(request, call_next):
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            resp.headers["Cache-Control"] = "no-cache"   # always pick up edited screens
        return resp

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/report")
    def report():
        p = store.run_dir / "report.html"
        if not p.exists():
            raise HTTPException(404, "no report yet: run the measure stage")
        return FileResponse(p)

    @app.get("/api/overview")
    def overview():
        return store.overview()

    @app.get("/api/at-risk")
    def at_risk(tier: str = "", status: str = "", route: str = "", q: str = "", offset: int = 0, limit: int = 25):
        return store.at_risk_rows(tier, status, route, q.strip(), max(offset, 0), min(max(limit, 1), 200))

    @app.get("/api/merchants")
    def merchants():
        return store.merchant_list()

    @app.get("/api/merchants/{mid}")
    def merchant(mid: str):
        return store.merchant_detail(mid)

    @app.post("/api/merchants/{mid}/decide")
    def decide(mid: str, body: DecideBody):
        return store.decide(mid, body)

    @app.get("/api/customers")
    def customers(merchant_id: str = ""):
        return store.customer_list(merchant_id)

    @app.get("/api/customers/{cid}")
    def customer(cid: str):
        return store.customer_detail(cid)

    @app.post("/api/pay")
    def pay(body: PayBody):
        return store.pay(body)

    @app.post("/api/refund")
    def refund(body: RefundBody):
        return store.refund(body)

    @app.post("/api/customers/{cid}/opt-out")
    def opt_out(cid: str):
        return store.opt_out(cid)

    @app.post("/api/clock/next")
    def next_day():
        return store.next_day()

    @app.post("/api/reset")
    def reset():
        store.reset()
        return {"ok": True}

    @app.post("/api/export-decisions")
    def export():
        return store.export_decisions()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(cfg: dict, run_dir: Path | None = None, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    app = create_app(cfg, run_dir)
    print(f"[demo] WinBack demo running at http://{host}:{port}   (Ctrl+C to stop)", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
