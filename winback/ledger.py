"""Offer ledger: what happens to money when a customer pays.

How the money moves (no change at the counter):
  1. The customer pays the FULL bill with the app. The Soundbox announces the full amount.
  2. If the payment qualifies, the app credits cashback to the customer at once.
  3. The shop's share of that cashback is netted from its settlement as ONE line per week
     ("Lost & Fund cashback this week: Rs 240"). There is no per-payment marker, so the shopkeeper
     can't tell which customer was picked. Paytm-funded cashback never touches the shop.

Rules checked at every payment (anti-abuse and fairness):
  * only successful payments made with our app, by a customer who still consents
  * inside the offer's validity, and the bill must reach the minimum (the app shows the real gap,
    no fake urgency)
  * at most one qualifying payment per customer per shop per day (stops splitting one bill in two)
  * streaks count distinct days; the reward is paid on the last one
  * refunds and chargebacks reverse the cashback (and the streak day)
  * the shop's share is checked against its monthly budget; if it would ever overflow, Paytm honours
    the promise to the customer and the overflow is flagged (budgets are reserved at worst case when
    offers are issued, so this should not happen)
  * every payment id is processed once (idempotent: retries and replays are safe)

In production this is a service backed by a database with row locks on the budget counters. Here it
is in-memory with the same rules; the measurement stage replays post-campaign payments through it
and the demo uses it live.
"""
from __future__ import annotations

import threading
from collections import Counter, defaultdict

import pandas as pd

OUR_APP = "paytm"


class OfferLedger:
    def __init__(self, offers: pd.DataFrame, merchants: pd.DataFrame, cfg: dict,
                 opt_outs: dict | None = None):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.one_per_day = cfg["abuse"]["one_redemption_per_day"]
        self.clawback = cfg["abuse"]["clawback_on_refund"]
        self.offers: dict[tuple, dict] = {}
        for r in offers.to_dict(orient="records"):
            r["valid_from"] = pd.Timestamp(r["valid_from"])
            r["valid_till"] = pd.Timestamp(r["valid_till"])
            r.update(uses_done=0, days=set(), rewarded=False, last_day=None, credited={})
            self.offers[(r["customer_id"], r["merchant_id"])] = r
        self.budget = merchants.set_index("merchant_id")["monthly_offer_budget"].to_dict()
        self.used = defaultdict(float)          # (merchant_id, "YYYY-MM") -> merchant share paid
        self.opt_outs = dict(opt_outs or {})    # customer_id -> timestamp consent was withdrawn
        self.entries: list[dict] = []
        self.results: dict[str, dict] = {}      # txn_id -> result (idempotency)
        self.applied_txn: dict[str, tuple] = {}  # txn_id -> (offer key, day) for payments that counted
        self.blocked = Counter()                 # rules that stopped a cashback (abuse / fairness checks)
        self.outside = Counter()                 # payments outside the offer's life (expired, used up)

    # -------------------------------------------------------------- helpers
    def opt_out(self, customer_id: str, ts) -> None:
        with self.lock:
            self.opt_outs.setdefault(customer_id, pd.Timestamp(ts))

    def offer_for(self, customer_id: str, merchant_id: str) -> dict | None:
        return self.offers.get((customer_id, merchant_id))

    def _need(self, o: dict) -> int:
        return int(o["streak_visits"]) if o["offer_type"] == "streak" else 1

    def progress(self, o: dict) -> dict:
        if o["offer_type"] == "streak":
            return {"done": len(o["days"]), "need": self._need(o), "finished": o["rewarded"]}
        if o["offer_type"] == "pct":
            return {"done": o["uses_done"], "need": int(o["uses"]), "finished": o["uses_done"] >= int(o["uses"])}
        return {"done": int(o["rewarded"]), "need": 1, "finished": o["rewarded"]}

    def _block(self, txn_id, reason, **extra) -> dict:
        # rule enforcement is counted; ordinary payments outside the offer's life are not "blocked"
        if reason not in ("expired", "not_started", "offer_used_up"):
            self.blocked[reason] += 1
        else:
            self.outside[reason] += 1
        res = {"txn_id": txn_id, "applied": False, "cashback": 0.0, "reason": reason, **extra}
        self.results[txn_id] = res
        return res

    # -------------------------------------------------------------- payments
    def on_payment(self, txn: dict) -> dict:
        """Process one payment. Returns what the customer sees (cashback credited, streak progress, or why not)."""
        with self.lock:
            tid = str(txn["txn_id"])
            if tid in self.results:                       # idempotent: a retry returns the same answer
                return self.results[tid]
            o = self.offers.get((txn["customer_id"], txn["merchant_id"]))
            if o is None:
                return {"txn_id": tid, "applied": False, "cashback": 0.0, "reason": "no_offer"}
            ts = pd.Timestamp(txn["ts"])
            amount = float(txn["amount"])
            if str(txn.get("status", "SUCCESS")).upper() != "SUCCESS":
                return self._block(tid, "payment_failed")
            if str(txn.get("payer_app", OUR_APP)).lower() != OUR_APP:
                return self._block(tid, "paid_with_another_app")
            oo = self.opt_outs.get(o["customer_id"])
            if oo is not None and ts >= oo:
                return self._block(tid, "consent_withdrawn")
            if ts < o["valid_from"]:
                return self._block(tid, "not_started")
            if ts.normalize() > o["valid_till"]:
                return self._block(tid, "expired")
            prog = self.progress(o)
            if prog["finished"]:
                return self._block(tid, "offer_used_up")
            if amount < float(o["min_bill"]):
                return self._block(tid, "below_min_bill", short_by=round(float(o["min_bill"]) - amount, 2))
            day = ts.normalize()
            if self.one_per_day and o["last_day"] == day:
                return self._block(tid, "already_counted_today")
            o["last_day"] = day
            self.applied_txn[tid] = ((o["customer_id"], o["merchant_id"]), day)

            if o["offer_type"] == "streak":
                o["days"].add(day)
                if len(o["days"]) < self._need(o):
                    res = {"txn_id": tid, "applied": True, "cashback": 0.0, "reason": "streak_progress",
                           "progress": self.progress(o)}
                    self._entry("STREAK_DAY", o, tid, ts, amount, 0.0)
                    self.results[tid] = res
                    return res
                cashback = float(o["cashback"])
                o["rewarded"] = True
            elif o["offer_type"] == "pct":
                cashback = round(min(float(o["pct"]) / 100.0 * amount, float(o["cap"])), 2)
                o["uses_done"] += 1
            else:
                cashback = float(o["cashback"])
                o["rewarded"] = True

            share = float(o.get("paytm_share") or 0.0)
            m_part, p_part = round(cashback * (1 - share), 2), round(cashback * share, 2)
            month = ts.strftime("%Y-%m")
            overflow = 0.0
            if m_part > 0:
                left = float(self.budget.get(o["merchant_id"], 0)) - self.used[(o["merchant_id"], month)]
                if m_part > left:   # should not happen (reserved at issue time): the customer still gets paid
                    overflow = round(m_part - max(left, 0), 2)
                    m_part, p_part = round(m_part - overflow, 2), round(p_part + overflow, 2)
                    self.blocked["budget_overflow_paid_by_paytm"] += 1
                self.used[(o["merchant_id"], month)] += m_part
            entry = self._entry("CASHBACK", o, tid, ts, amount, cashback, m_part, p_part)
            o["credited"][tid] = entry
            res = {"txn_id": tid, "applied": True, "cashback": cashback, "merchant_share": m_part,
                   "paytm_share": p_part, "reason": "cashback_credited", "progress": self.progress(o),
                   "overflow": overflow}
            self.results[tid] = res
            return res

    def on_refund(self, refund_txn_id: str, original_txn_id: str, ts) -> dict:
        """A refund or chargeback of a payment that earned cashback reverses that cashback."""
        with self.lock:
            rid = str(refund_txn_id)
            if rid in self.results:
                return self.results[rid]
            orig = self.results.get(str(original_txn_id))
            if not self.clawback or orig is None or not orig.get("applied"):
                res = {"txn_id": rid, "reversed": 0.0, "reason": "nothing_to_reverse"}
                self.results[rid] = res
                return res
            key, day = self.applied_txn[str(original_txn_id)]
            o = self.offers[key]
            reversed_amt = 0.0
            credited = o["credited"].pop(str(original_txn_id), None)
            if credited is not None:
                reversed_amt = credited["cashback"]
                month = pd.Timestamp(credited["ts"]).strftime("%Y-%m")
                self.used[(o["merchant_id"], month)] -= credited["merchant_share"]
                self._entry("REVERSAL", o, rid, ts, -credited["amount_paid"], -credited["cashback"],
                            -credited["merchant_share"], -credited["paytm_share"])
                if o["offer_type"] == "pct":
                    o["uses_done"] = max(0, o["uses_done"] - 1)
                else:
                    o["rewarded"] = False
            if o["offer_type"] == "streak":
                o["days"].discard(day)
            if o["last_day"] == day:
                o["last_day"] = None
            res = {"txn_id": rid, "reversed": reversed_amt,
                   "reason": "cashback_reversed" if reversed_amt else "streak_day_removed"}
            self.results[rid] = res
            return res

    def _entry(self, kind, o, tid, ts, amount, cashback, m_part=0.0, p_part=0.0) -> dict:
        e = {"entry_id": f"L{len(self.entries) + 1:07d}", "kind": kind, "offer_id": o["offer_id"], "txn_id": tid,
             "customer_id": o["customer_id"], "merchant_id": o["merchant_id"], "funder": o.get("funder", "merchant"),
             "ts": pd.Timestamp(ts), "amount_paid": round(amount, 2), "cashback": round(cashback, 2),
             "merchant_share": round(m_part, 2), "paytm_share": round(p_part, 2)}
        self.entries.append(e)
        return e

    # -------------------------------------------------------------- replay + reports
    def replay(self, txns: pd.DataFrame) -> None:
        """Feed a batch of payments (and refunds) in time order. Used by `measure` on post-campaign data."""
        cols = ["txn_id", "customer_id", "merchant_id", "amount", "ts", "status"]
        extra = [c for c in ("payer_app", "refund_of") if c in txns.columns]
        for r in txns.sort_values("ts", kind="stable")[cols + extra].to_dict(orient="records"):
            if str(r["status"]).upper() == "REFUNDED":
                if pd.notna(r.get("refund_of")):
                    self.on_refund(r["txn_id"], r["refund_of"], r["ts"])
            elif (r["customer_id"], r["merchant_id"]) in self.offers:
                self.on_payment(r)

    def frame(self) -> pd.DataFrame:
        cols = ["entry_id", "kind", "offer_id", "txn_id", "customer_id", "merchant_id", "funder", "ts",
                "amount_paid", "cashback", "merchant_share", "paytm_share"]
        return pd.DataFrame(self.entries, columns=cols)

    def settlement_lines(self, period: str = "weekly") -> pd.DataFrame:
        """One netting line per shop per week. No customer ids, no per-payment detail."""
        e = self.frame()
        e = e[e["merchant_share"] != 0]
        if e.empty:
            return pd.DataFrame(columns=["merchant_id", "week_starting", "winback_cashback_netted", "statement_text"])
        freq = "W-MON" if period == "weekly" else "D"
        e["week_starting"] = (e["ts"].dt.normalize() - pd.to_timedelta(e["ts"].dt.weekday, unit="D")
                              if freq == "W-MON" else e["ts"].dt.normalize())
        s = e.groupby(["merchant_id", "week_starting"])["merchant_share"].sum().round(2).reset_index()
        s = s.rename(columns={"merchant_share": "winback_cashback_netted"})
        s = s[s["winback_cashback_netted"].abs() >= 0.01].reset_index(drop=True)   # refunds can net a week to 0
        s["statement_text"] = [f"Lost & Fund cashback, week of {w:%d %b}: Rs {v:,.2f}" for w, v in
                               zip(s["week_starting"], s["winback_cashback_netted"])]
        return s

    def stats(self) -> dict:
        e = self.frame()
        cb = e[e["kind"] == "CASHBACK"]
        rv = e[e["kind"] == "REVERSAL"]
        return {
            "cashback_payments": int(len(cb)),
            "customers_paid_cashback": int(cb["customer_id"].nunique()) if len(cb) else 0,
            "cashback_total": round(float(e["cashback"].sum()), 2),
            "merchant_funded": round(float(e["merchant_share"].sum()), 2),
            "paytm_funded": round(float(e["paytm_share"].sum()), 2),
            "reversals": int(len(rv)),
            "reversed_amount": round(abs(float(rv["cashback"].sum())), 2),
            "streak_days_logged": int((e["kind"] == "STREAK_DAY").sum()),
            "blocked": dict(self.blocked),
            "outside_offer_window": dict(self.outside),
        }
