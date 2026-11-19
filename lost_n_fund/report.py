"""Self-contained HTML report for one campaign run (no external libraries)."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

from .diagnosis import ACTION_LABELS, REASON_LABELS

FEATURE_LABELS = {
    "category_code": "Shop category", "p_visits_30": "Visits to the shop, 30d", "p_visits_90": "Visits to the shop, 90d",
    "p_visits_14": "Visits to the shop, 14d", "p_trend": "Visit trend at the shop", "p_overdue": "Overdue vs usual gap",
    "p_alive": "BG/NBD: still a regular?", "p_silence": "Chance of this long a gap", "p_visit_share": "Share of visits at this shop",
    "p_gmv_share": "Share of spend at this shop", "p_days_since": "Days since last visit", "p_avg_gap": "Usual gap at the shop",
    "opens_ratio": "App opens trend", "act_ratio": "Bills + P2P trend", "c_other_trend": "Payments at other shops",
    "c_same_cat_shift": "Spend moved to a same-kind shop", "c_new_same_cat_30": "New same-kind shops",
    "c_qc_share_30": "Quick-commerce share", "c_qc_trend": "Quick-commerce trend", "c_away_share_14": "Paying in new places",
    "fail_rate_90": "Failed payments, 90d", "fail_rate_30": "Failed payments, 30d", "loc_ratio": "Locality activity",
    "m_trend": "Shop's own trend", "m_all_trend": "Shop's trend, all apps", "m_share_trend": "Our app's share at the shop",
    "m_offline_14": "Soundbox offline days", "avg_gap_days": "Usual gap between payments", "txn_30": "Payments, 30d",
    "tenure_days": "Tenure", "recency_days": "Days since any payment",
}
FUNNEL_LABELS = [
    ("customers_total", "All customers"),
    ("eligible_consented_adults", "Consented to offers, 18+"),
    ("regular_pairs_scored", "Regular customer-shop pairs scored"),
    ("at_risk_pairs", "At risk at a shop and overdue"),
    ("offer_candidates_pairs", "Routed to an offer (shop or Paytm pays)"),
    ("eligible_after_caps", "Pass anti-abuse and frequency caps"),
    ("offer_pays_for_itself", "Offer pays for itself for the funder"),
    ("within_budgets", "Within shop and Paytm budgets"),
    ("final_offers", "Shop big enough to stay anonymous"),
    ("treatment", "Treatment (holdouts removed)"),
    ("merchant_approved", "Approved (shop or Paytm programme)"),
    ("delivered_to_customer", "Delivered to the customer"),
    ("redeemed", "Got cashback"),
]

CSS = """
:root{color-scheme:light;--surface-0:#f2f6fb;--surface-1:#ffffff;--border:#e3e9f1;--grid:#edf1f6;
--text-primary:#0d1b2e;--text-secondary:#4a5a70;--text-muted:#8090a6;--series-1:#00baf2;--series-2:#002e6e;
--good:#0d8a4a;--critical:#c62b28;--banner:#fff3dc;--banner-border:#e39a00;--info:#e5f7fe;--info-border:#00baf2;
--navy:#002e6e;--sky:#00baf2}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--surface-0:#0b1322;
--surface-1:#111c2f;--border:#22314a;--grid:#1b2840;--text-primary:#eef3fa;--text-secondary:#b9c6d8;--text-muted:#8696ad;
--series-1:#00baf2;--series-2:#8fb0e3;--banner:#2e2610;--banner-border:#c98500;--info:#0d2238;--info-border:#00baf2}}
:root[data-theme="dark"]{color-scheme:dark;--surface-0:#0b1322;--surface-1:#111c2f;--border:#22314a;--grid:#1b2840;
--text-primary:#eef3fa;--text-secondary:#b9c6d8;--text-muted:#8696ad;--series-1:#00baf2;--series-2:#8fb0e3;
--banner:#2e2610;--banner-border:#c98500;--info:#0d2238;--info-border:#00baf2}
*{box-sizing:border-box}body{margin:0;background:var(--surface-0);color:var(--text-primary);
font:15px/1.5 "Inter",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.brandbar{background:linear-gradient(110deg,#002e6e,#0a4aa3 70%,#00a6e6);color:#fff}
.brandbar div{max-width:1080px;margin:0 auto;padding:14px 16px;display:flex;align-items:center;gap:12px}
.ptm{font-weight:800;letter-spacing:-.045em;font-size:26px;line-height:1}.ptm b{color:#fff}.ptm i{color:#00baf2;font-style:normal}
.brandbar span.t{font-weight:700;opacity:.9;border-left:1px solid rgba(255,255,255,.3);padding-left:12px}
main{max-width:1080px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.02em}h2{font-size:19px;margin:0 0 4px;display:flex;align-items:center;gap:8px}
h2::before{content:"";width:4px;height:18px;border-radius:2px;background:var(--sky)}h3{font-size:15px;margin:0 0 8px}
.sub{color:var(--text-secondary);margin:0 0 20px}.muted{color:var(--text-muted);font-size:13px}
section{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:20px;margin:16px 0}
.banner{background:var(--banner);border:1px solid var(--banner-border);border-radius:10px;padding:10px 14px;margin:12px 0;font-size:14px}
.note{background:var(--info);border:1px solid var(--info-border);border-radius:10px;padding:10px 14px;margin:12px 0;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:12px}
.tile{border:1px solid var(--border);border-radius:10px;padding:14px;background:var(--surface-1)}
.tile .v{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}.tile .l{color:var(--text-secondary);font-size:13px}
.tile .n{color:var(--text-muted);font-size:12px;margin-top:4px}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
.step{border:1px solid var(--border);border-radius:999px;padding:6px 12px;font-size:13px;background:var(--surface-0)}
.arrow{color:var(--text-muted)}
.bars{display:grid;grid-template-columns:minmax(140px,300px) 1fr;gap:6px 12px;align-items:center;margin-top:12px}
.bars .lab{font-size:13px;color:var(--text-secondary);text-align:right}
.track{position:relative;height:22px;border-left:1px solid var(--grid)}
.bar{position:absolute;left:0;top:3px;height:16px;border-radius:0 4px 4px 0;background:var(--series-1);min-width:2px}
.bar.s2{background:var(--series-2)}
.val{position:absolute;top:1px;font-size:12px;color:var(--text-primary);white-space:nowrap;padding-left:6px;font-variant-numeric:tabular-nums}
.pair .track{height:44px}.pair .bar{height:16px}.pair .bar.s2{top:23px}.pair .val.s2{top:21px}
[data-tip]{cursor:default}
#tip{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--surface-1);font-size:12px;
padding:6px 8px;border-radius:6px;opacity:0;transition:opacity .1s;max-width:260px;z-index:9}
.legend{display:flex;gap:16px;font-size:13px;color:var(--text-secondary);margin-top:8px;flex-wrap:wrap}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.party{border:1px solid var(--border);border-radius:10px;padding:14px}
.party .big{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}
.party ul{margin:8px 0 0;padding-left:18px;color:var(--text-secondary);font-size:13px}
.tbl{overflow-x:auto;margin-top:10px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
th{color:var(--text-secondary);font-weight:600}td.num,th.num{text-align:right}
.card{border:1px solid var(--border);border-radius:10px;padding:14px;margin-top:12px;background:var(--surface-0)}
.card .meta{font-size:12px;color:var(--text-muted);margin-bottom:6px}
.card .msg{font-size:14px}.card ul{margin:8px 0 0;padding-left:18px;font-size:13px;color:var(--text-secondary)}
.pill{display:inline-block;font-size:11px;padding:1px 8px;border-radius:999px;border:1px solid var(--border);color:var(--text-secondary);margin-left:6px}
.ok{color:var(--good)}.bad{color:var(--critical)}
code{font-size:12px;background:var(--surface-0);padding:1px 5px;border-radius:4px;border:1px solid var(--border)}
@media (max-width:600px){.bars{grid-template-columns:1fr}.bars .lab{text-align:left}}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
 el.addEventListener('mousemove',e=>{tip.textContent=el.dataset.tip;tip.style.opacity=1;
  tip.style.left=Math.min(e.clientX+12,innerWidth-270)+'px';tip.style.top=(e.clientY+14)+'px'});
 el.addEventListener('mouseleave',()=>tip.style.opacity=0);});
"""


def _e(x) -> str:
    return html.escape(str(x))


def _rs(x) -> str:
    if x is None:
        return "-"
    sign = "-" if x < 0 else ""
    return f"{sign}Rs {abs(x):,.0f}"


def _pct(x, digits=1) -> str:
    return "-" if x is None else f"{x * 100:.{digits}f}%"


def _eff(e: dict | None) -> str:
    if not e:
        return "-"
    u = " pts" if e["unit"] == "pts" else ""
    v = f"{e['effect']:+.1f}{u}" if e["unit"] == "pts" else _rs(e["effect"])
    return v


def _ci(e: dict | None) -> str:
    if not e:
        return "-"
    if e["unit"] == "pts":
        return f"{e['ci95'][0]:+.1f} to {e['ci95'][1]:+.1f}"
    return f"{_rs(e['ci95'][0])} to {_rs(e['ci95'][1])}"


def _hbars(rows: list[tuple[str, float, str]], fmt=lambda v: f"{v:,.0f}") -> str:
    mx = max([v for _, v, _ in rows] + [1e-9])
    out = ['<div class="bars">']
    for label, v, tip in rows:
        w = 100 * max(v, 0) / mx * 0.82
        out.append(f'<div class="lab">{_e(label)}</div><div class="track">'
                   f'<div class="bar" style="width:{w:.2f}%" data-tip="{_e(tip)}"></div>'
                   f'<span class="val" style="left:{w:.2f}%">{_e(fmt(v))}</span></div>')
    out.append("</div>")
    return "".join(out)


def _pair_bars(rows: list[tuple[str, float, float]], fmt) -> str:
    mx = max([max(a, b) for _, a, b in rows] + [1e-9])
    out = ['<div class="bars pair">']
    for label, a, b in rows:
        wa, wb = 100 * a / mx * 0.8, 100 * b / mx * 0.8
        out.append(
            f'<div class="lab">{_e(label)}</div><div class="track">'
            f'<div class="bar" style="width:{wa:.2f}%" data-tip="Treatment (offer): {_e(fmt(a))}"></div>'
            f'<span class="val" style="left:{wa:.2f}%">Treatment {_e(fmt(a))}</span>'
            f'<div class="bar s2" style="width:{wb:.2f}%" data-tip="Control (no offer): {_e(fmt(b))}"></div>'
            f'<span class="val s2" style="left:{wb:.2f}%">Control {_e(fmt(b))}</span></div>')
    out.append("</div>")
    out.append('<div class="legend"><span><span class="sw" style="background:var(--series-1)"></span>Treatment (offer)</span>'
               '<span><span class="sw" style="background:var(--series-2)"></span>Control holdout (no offer)</span></div>')
    return "".join(out)


def _table(df: pd.DataFrame, num_cols=()) -> str:
    head = "".join(f'<th class="{"num" if c in num_cols else ""}">{_e(c)}</th>' for c in df.columns)
    body = []
    for r in df.itertuples(index=False):
        cells = "".join(f'<td class="{"num" if c in num_cols else ""}">{_e(v)}</td>' for c, v in zip(df.columns, r))
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _redact(msg: str) -> str:
    """Customer messages start with a greeting and first name: keep names out of the report."""
    return re.sub(r"^(Hi|Namaste) [^,!]+[,!]", r"\1 [first name],", str(msg))


def _tile(value: str, label: str, note: str = "") -> str:
    n = f'<div class="n">{_e(note)}</div>' if note else ""
    return f'<div class="tile"><div class="v">{_e(value)}</div><div class="l">{_e(label)}</div>{n}</div>'


def _effect_rows(items: dict, label_map=None) -> pd.DataFrame:
    rows = []
    for k, e in items.items():
        if not e:
            continue
        rows.append({"Group": (label_map or {}).get(k, k), "Treated": e["treated"], "Control": e["control"],
                     "Treated %": f"{e['treated_mean']:.1f}", "Control %": f"{e['control_mean']:.1f}",
                     "Effect": _eff(e), "95% CI": _ci(e), "Shops": e["clusters"]})
    return pd.DataFrame(rows)


def build_report(run_dir: Path, cfg: dict) -> Path:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    results = json.loads((run_dir / "09_campaign_results.json").read_text(encoding="utf-8"))
    card = json.loads((Path(cfg["paths"]["model_dir"]) / "model_card.json").read_text(encoding="utf-8"))
    alerts = json.loads((run_dir / "05_merchant_alerts.json").read_text(encoding="utf-8"))
    notes = pd.read_csv(run_dir / "07_customer_notifications.csv")
    lines_path = run_dir / "08_settlement_lines.csv"
    lines = pd.read_csv(lines_path) if lines_path.exists() else pd.DataFrame()
    f = summary["funnel"]
    itt = results.get("itt", {})
    e = results.get("economics", {})
    pw = results.get("power", {})
    v = card["validation"]
    parts = []

    banner = ""
    if results.get("simulated_outcomes"):
        banner = ('<div class="banner"><b>Demo run.</b> Customers, shops and payments are synthetic, and the '
                  f'{cfg["campaign"]["measurement_days"]} days after the campaign are simulated. The pipeline code is the '
                  'same one that runs on real data; only the inputs change.</div>')
    parts.append(f"""<h1>Lost &amp; Fund: campaign report</h1>
<p class="sub">Run <code>{_e(summary['run_id'])}</code> · scored on {_e(summary['snapshot'])} with model
<code>{_e(summary['model']['version'])}</code> · measured {_e(results['measurement_window'][0])} to
{_e(results['measurement_window'][1])}</p>{banner}
<section><h2>How the loop works</h2>
<div class="flow"><span class="step">Find regulars slipping away from a shop</span><span class="arrow">→</span>
<span class="step">Diagnose why</span><span class="arrow">→</span>
<span class="step">Route: shop pays · Paytm pays · fix payments · wait · alert ops</span><span class="arrow">→</span>
<span class="step">Shop approves a budget (sees a count band, never names)</span><span class="arrow">→</span>
<span class="step">Customer gets cashback in-app; shop's share netted weekly</span><span class="arrow">→</span>
<span class="step">Measured against two holdouts</span></div></section>""")

    # ---------------- headline ----------------
    r30, rpo = itt.get("returned_30"), itt.get("returned_post_offer")
    if r30:
        truth = results.get("simulator_truth")
        sig = r30["significant"]
        tiles = (_tile(_eff(r30), "Came back to the shop within 30d", f"95% CI {_ci(r30)} · {r30['clusters']} shop clusters")
                 + _tile(_eff(rpo), "Still coming after the offer ended", f"95% CI {_ci(rpo)}")
                 + _tile(f"{pw.get('mde_pts', '-')} pts", "Smallest effect this run can detect",
                         f"a {pw.get('target_pts', 5):.0f}-pt effect needs ~{pw.get('runs_like_this_needed', '-')} runs like this")
                 + _tile(_rs(e.get("cashback_total")), "Cashback paid",
                         f"shops {_rs(e.get('merchant_funded_cashback'))} · Paytm {_rs(e.get('paytm_funded_cashback'))}")
                 + _tile(_rs(e.get("merchant_profit_after_cashback")), "Shops' profit after cashback", "90 days, point estimate"))
        if truth:
            tiles += _tile(f"{truth['returned_30_pts']:+.1f} pts", "Simulator's built-in effect",
                           f"{truth['returned_post_offer_pts']:+.1f} pts after the offer")
        verdict = ("" if sig else
                   f'<div class="note"><b>Not enough customers yet to tell.</b> With {r30["treated"]} treated and '
                   f'{r30["control"]} control customers, this run can only reliably detect effects of about '
                   f'{pw.get("mde_pts")} points or more. Detecting a {pw.get("target_pts", 5):.0f}-point effect needs about '
                   f'{pw.get("needed_control", 0):,} control and {pw.get("needed_treated", 0):,} treated customers, so pool '
                   f'~{pw.get("runs_like_this_needed")} weekly runs (or several cities) before deciding anything.'
                   + (f' In this demo the simulator\'s real effect is {truth["returned_30_pts"]:+.1f} points: the gap to '
                      f'the estimate is sampling noise, which is exactly why the power calculation matters.' if truth else "")
                   + '</div>')
        parts.append("<section><h2>Headline</h2><div class=\"tiles\">" + tiles + "</div>" + verdict
                     + "<p class=\"muted\" style=\"margin-top:10px\">Intent-to-treat (everyone assigned an offer vs "
                       "everyone held out), adjusted for pre-period visits (CUPED), standard errors clustered by shop.</p></section>")

    # ---------------- diagnosis ----------------
    routes = summary.get("routes", {})
    if routes:
        rows = [(f"{REASON_LABELS.get(k, k)} → {ACTION_LABELS.get(cfg['routing'][k]['action'], '')}", float(n),
                 f"{k}: {n:,} pairs") for k, n in sorted(routes.items(), key=lambda kv: -kv[1])]
        chk = summary.get("diagnosis_vs_truth")
        chk_html = ""
        if chk:
            rec = pd.DataFrame([{"Cause": REASON_LABELS.get(k, k), "Caught": _pct(x, 0)}
                                for k, x in chk["recall_by_cause"].items()])
            chk_html = ("<h3 style=\"margin-top:18px\">Diagnosis vs the simulator's hidden truth (demo only)</h3>"
                        f"<p class=\"muted\">{_pct(chk['truly_leaving_share'], 0)} of flagged pairs were truly leaving; on "
                        f"those the cause was right {_pct(chk['accuracy_on_leaving'], 0)} of the time. Of the customers a "
                        f"shop was asked to pay for, {_pct(chk['shop_funded_precision'], 0)} really had moved shop or "
                        "stopped for an unseen reason (the cases a shop offer can fix).</p>" + _table(rec, num_cols={"Caught"}))
        a = summary.get("actions", {})
        parts.append("<section><h2>Why regulars are slipping, and who acts</h2><p class=\"muted\">A shop-funded offer only "
                     "fits one cause: they moved to another shop. Everything else is routed elsewhere so shops don't pay "
                     "for problems that aren't theirs.</p>" + _hbars(rows)
                     + f"<p class=\"muted\" style=\"margin-top:10px\">This run: {a.get('paytm_fix', 0)} payment fixes queued, "
                       f"{a.get('ops_alert_shops', 0)} shops flagged to merchant ops, {a.get('wait', 0)} pairs waiting "
                       f"(travel or normal gaps), {a.get('skip', 0)} gone. Carry-over queue for small shops: "
                       f"{summary.get('carryover_queue', 0)} customers.</p>" + chk_html + "</section>")

    funnel_rows = [(lab, float(f[k]), f"{lab}: {f[k]:,}") for k, lab in FUNNEL_LABELS if f.get(k) is not None]
    parts.append("<section><h2>Pipeline funnel</h2><p class=\"muted\">Pairs or customers remaining after each stage.</p>"
                 + _hbars(funnel_rows) + "</section>")

    # ---------------- did it work ----------------
    if r30:
        bars = [("Came back to the shop, 30d", r30["treated_mean"] / 100, r30["control_mean"] / 100)]
        if rpo:
            bars.append(("Still coming after the offer", rpo["treated_mean"] / 100, rpo["control_mean"] / 100))
        ap = results.get("approved_shops", {})
        pooled = results.get("pooled") or {}
        main = _effect_rows({"All (intent-to-treat), 30d": r30, "All, after the offer ended": rpo,
                             "Approved shops only, 30d": ap.get("returned_30"),
                             f"Pilot so far ({pooled.get('runs', 0)} runs pooled), 30d": pooled.get("returned_30")})
        nums = {"Treated", "Control", "Treated %", "Control %", "Effect", "95% CI", "Shops"}
        parts.append("<section><h2>Did the offer work?</h2><p class=\"muted\">Share of customers who paid at the offer "
                     "shop. Customers were randomised within each shop, so the approved-shops read is also unbiased.</p>"
                     + _pair_bars(bars, _pct) + _table(main, num_cols=nums)
                     + "<h3 style=\"margin-top:18px\">By cause, offer type and risk (directional: small groups)</h3>"
                     + _table(pd.concat([_effect_rows(results.get("by_route", {}), {k: REASON_LABELS.get(k, k) for k in REASON_LABELS}),
                                         _effect_rows({f"Offer: {k}": x for k, x in results.get("by_offer_type", {}).items()}),
                                         _effect_rows({f"Risk: {k}": x for k, x in results.get("by_risk_tier", {}).items()})],
                                        ignore_index=True), num_cols=nums)
                     + (f"<p class=\"muted\">Global holdout sanity check: {results['global_holdout']['customers']} customers "
                        f"who get no offer all quarter vs this run's control: "
                        f"{_eff(results['global_holdout']['control_vs_global_holdout_returned_30'])} "
                        f"(CI {_ci(results['global_holdout']['control_vs_global_holdout_returned_30'])}); "
                        "both got no offer, so this should be near zero.</p>" if results.get("global_holdout") else "")
                     + "</section>")

        cof_m, cof_p = e.get("merchant_funded_cashback", 0), e.get("paytm_funded_cashback", 0)
        parts.append(f"""<section><h2>Who gains what (90 days)</h2><div class="grid2">
<div class="party"><h3>Shops (A)</h3><div class="big">{_rs(e.get('merchant_profit_after_cashback'))}</div>
<div class="muted">profit after their share of cashback (point estimate)</div><ul>
<li>Extra sales from slipping regulars: {_rs(e.get('incremental_merchant_gmv_90d'))}</li>
<li>Cashback netted from settlements: {_rs(cof_m)}</li>
<li>Blended gross margin: {_pct(e.get('merchant_blended_margin'), 0)}</li></ul></div>
<div class="party"><h3>Paytm (P)</h3><div class="big">{_rs(e.get('paytm_value'))}</div>
<div class="muted">value after its cashback (point estimate)</div><ul>
<li>Extra GMV on the app: {_rs(e.get('incremental_platform_gmv_90d'))}</li>
<li>Users kept active: {e.get('users_kept_active')}</li>
<li>Paytm-funded and co-funded cashback: {_rs(cof_p)}</li></ul></div>
<div class="party"><h3>Customers (B)</h3><div class="big">{_rs(e.get('customer_cashback_received'))}</div>
<div class="muted">cashback received</div><ul>
<li>{e.get('customers_paid_cashback')} customers got cashback ({_pct(e.get('redemption_rate_of_delivered'), 0)} of offers)</li>
<li>Paid the full bill at the counter; cashback credited by the app</li>
<li>{_rs(e.get('reversed_on_refund'))} clawed back on refunds</li></ul></div></div>
<p class="muted">With samples this small, money estimates swing widely: read them together with the confidence intervals above.</p></section>""")

        blocked = e.get("blocked_at_checkout", {})
        bl = pd.DataFrame([{"Rule": k.replace("_", " "), "Payments stopped": n} for k, n in blocked.items()])
        ln = lines.head(6)[["merchant_id", "statement_text"]] if len(lines) else pd.DataFrame()
        parts.append("<section><h2>Money movement and abuse controls</h2><p class=\"muted\">Customers pay the full bill; the "
                     "Soundbox announces the full amount. Cashback is credited by the app, and each shop's share is netted "
                     "from its settlement as one weekly line, so a redemption doesn't reveal who was picked.</p>"
                     + ("<h3>Rules that stopped a cashback</h3>" + _table(bl, num_cols={"Payments stopped"}) if len(bl) else "")
                     + ("<h3 style=\"margin-top:18px\">Settlement statement lines (sample)</h3>" + _table(ln) if len(ln) else "")
                     + "</section>")

    # ---------------- offer engine ----------------
    mix = pd.DataFrame(summary.get("offer_mix", []))
    if len(mix):
        mix.columns = ["Offer type", "Chosen by", "Offers"]
    fund = pd.DataFrame(summary.get("by_funder", []))
    if len(fund):
        fund.columns = ["Funder", "Offers", "Expected cashback (Rs)", "Shop EV (Rs)", "Paytm EV (Rs)"]
    drops = pd.DataFrame([{"Reason": k, "Dropped": n} for k, n in summary.get("drop_reasons", {}).items()])
    asm = summary.get("assumptions", {})
    parts.append("<section><h2>Offer engine</h2><p class=\"muted\">Cashback type from the customer's rhythm (streaks for "
                 "people who came 3+ times a month), size stepped down until it pays for whoever funds it. Expected cost "
                 "includes regulars who would have earned it anyway. "
                 f"{cfg['offers']['exploration_share']:.0%} of customers get a random type/size so the engine can learn "
                 f"what works. Assumptions: {_e(asm.get('source', ''))} (save rate {asm.get('save_rate')}).</p>"
                 + (_table(mix, num_cols={"Offers"}) if len(mix) else "")
                 + (_table(fund, num_cols={"Offers", "Expected cashback (Rs)", "Shop EV (Rs)", "Paytm EV (Rs)"}) if len(fund) else "")
                 + ("<h3 style=\"margin-top:18px\">Not offered</h3>" + _table(drops, num_cols={"Dropped"}) if len(drops) else "")
                 + "</section>")

    # ---------------- model ----------------
    imp = card["feature_importance"][:10]
    raw = card.get("validation_uncalibrated", {})
    mh = summary.get("model", {})
    shadow = mh.get("shadow")
    parts.append("<section><h2>Shop-level lapse model</h2>"
                 f"<p class=\"muted\">{_e(card['algorithm'])}. Unit: {_e(card.get('unit', ''))}. Target: {_e(card['target'])}. "
                 f"Trained on {', '.join(card['train_snapshots'])}, validated out-of-time on {_e(card['validation_snapshot'])}.</p>"
                 "<div class=\"tiles\">"
                 + _tile(f"{v['roc_auc']:.3f}", "ROC-AUC", f"logistic baseline {card['baseline_logistic_validation']['roc_auc']:.3f}")
                 + _tile(f"{v['lift_top10pct']}x", "Lift in top 10%", f"base lapse rate {_pct(v['churn_rate'])}")
                 + _tile(f"{v.get('max_calibration_gap', 0):.3f}", "Worst calibration gap (decile)",
                         f"uncalibrated {raw.get('max_calibration_gap', 0):.3f}; cross-fitted")
                 + _tile(_pct(card.get("shop_lapsers_still_active_on_app"), 0), "Shop lapsers still active on the app",
                         "an app-level churn model would call them safe")
                 + _tile(f"{mh.get('max_psi', 0):.2f}", "Max input drift (PSI)", ", ".join(list(mh.get("top_drift", {}))[:2]))
                 + "</div>"
                 + (f"<p class=\"muted\">Shadow model {_e(shadow['version'])}: top-10% overlap {_pct(shadow['top10pct_overlap'], 0)}, "
                    f"rank correlation {shadow['rank_correlation']}.</p>" if shadow else "")
                 + "<h3 style=\"margin-top:18px\">What drives the prediction</h3>"
                   "<p class=\"muted\">Drop in validation AUC when the signal is shuffled (permutation importance).</p>"
                 + _hbars([(FEATURE_LABELS.get(i["feature"], i["feature"]), max(i["importance"], 0.0),
                            f"{i['feature']}: AUC drop {i['importance']:.4f}") for i in imp], fmt=lambda x: f"{x:.3f}")
                 + "</section>")

    # ---------------- messages ----------------
    merchant_note = ("Shops see a count band, the offer and rounded costs: no names, numbers, exact counts or scores. "
                     "Bill and visit ranges appear only for groups of 10+."
                     if cfg["compliance"]["merchant_view"] == "anonymous" else
                     "Shops see a count band, plus first name + initial only for customers who agreed to be visible.")
    cards = []
    for a in alerts[:3]:
        s = a["segment"]
        lis = f"<li>{_e(a['customers_band'])} customers · {_e(a['offer_mix'])}</li>"
        if "typical_bill_range" in s:
            lis += (f"<li>usually ~{s['usual_visits_per_month']} visits/month · typical bill Rs {s['typical_bill_range'][0]}–"
                    f"{s['typical_bill_range'][1]}</li>")
        lis += "".join(f"<li>{_e(c['customer'])} (agreed to be visible) → {_e(c['offer'])}</li>"
                       for c in a.get("named_customers", [])[:3])
        cards.append(f'<div class="card"><div class="meta">To {_e(a["merchant_name"])} · {_e(a["category"])} · '
                     f'{_e(a["locality"])}<span class="pill">{_e(a["language"])}</span>'
                     f'{"<span class=pill>autopilot</span>" if a.get("autopilot") else ""}'
                     f'{"<span class=pill>co-funded</span>" if a.get("cofunded_by_paytm") else ""}</div>'
                     f'<div class="msg">{_e(a["message"])}</div><ul>{lis}</ul></div>')
    ncards = "".join(f'<div class="card"><div class="meta">{_e(r.channel)} · {_e(r.language)} · {_e(r.funder)}-funded</div>'
                     f'<div class="msg"><b>{_e(r.title)}</b><br>{_e(_redact(r.message))}</div></div>'
                     for r in notes.drop_duplicates(["language", "funder"]).head(4).itertuples())
    parts.append(f"""<section><h2>Messages</h2><div class="grid2"><div><h3>P → A: shop alerts</h3>
<p class="muted">{merchant_note}</p>{''.join(cards) or '<p class="muted">No shop alerts this run.</p>'}</div>
<div><h3>P → B: customer offers</h3><p class="muted">In the customer's language, real dates only, opt-out in every message,
at most one offer per {cfg['campaign']['frequency_cap_days']} days and {cfg['abuse']['max_offers_per_window']} per
{cfg['compliance']['abuse_history_days']} days. Names are redacted in this report.</p>{ncards}</div></div></section>""")

    board = pd.DataFrame(results.get("merchant_scoreboard", []))
    if not board.empty:
        board = board.head(10).rename(columns={
            "merchant_id": "ID", "merchant_name": "Shop", "category": "Category", "offers": "Offers",
            "delivered": "Delivered", "came_back": "Came back", "sales_from_these_customers": "Sales (Rs)",
            "cashback_netted": "Cashback netted (Rs)"})
        pooled_cat = results.get("pooled_by_category", {})
        pc = "; ".join(f"{k}: {_eff(x)} (CI {_ci(x)})" for k, x in pooled_cat.items() if x)
        parts.append("<section><h2>Shop scoreboard</h2><p class=\"muted\">One shop's own holdout is far too small to prove "
                     "anything, so shops are shown their own results next to the pooled effect for their category"
                     + (f" ({_e(pc)})" if pc else "") + ".</p>"
                     + _table(board, num_cols={"Offers", "Delivered", "Came back", "Sales (Rs)", "Cashback netted (Rs)"})
                     + "</section>")

    comp = summary.get("compliance")
    if comp:
        k = comp.get("min_customers_per_alert", 0)
        eligible = summary["funnel"].get("eligible_consented_adults", 0)
        rows = [
            ("Explicit consent", f"Only {eligible:,} of {comp['customers_total']:,} customers who opted in to offers based on "
             f"their payments. {comp['no_offers_consent']:,} without consent were not trained on, scored or messaged. "
             "Consent is re-checked before sending and at every payment.", "NPCI UPI rules · DPDP Act"),
            ("Children", f"{comp['minors']:,} under-18 customers excluded from training, scoring and offers.", "DPDP Act s.9"),
            ("What shops see", f"Count bands ({'/'.join(str(b) for b in comp.get('count_bands', []))}+), never exact counts; "
             f"ranges only for 10+; costs rounded to Rs {cfg['compliance']['cost_rounding']}. "
             f"{comp.get('names_shown_to_merchants', 0)} names and 0 phone numbers shared.", "NPCI UPI rules (no third-party sharing)"),
            ("Small shops", f"Alerts only when a shop has at least {k} customers (smallest: {comp.get('smallest_alert', 0)}). "
             f"{comp.get('customers_held_for_anonymity', 0)} customers held in a carry-over queue so no shop can guess who.",
             "Privacy by design"),
            ("Redemptions", "Shop's cashback share is netted weekly as one line: no per-payment marker that would reveal "
             "who was picked.", "Privacy by design"),
            ("Sensitive shops", f"{', '.join(comp.get('sensitive_categories_excluded', [])) or 'none'} never used "
             f"({comp.get('at_risk_pairs_skipped_sensitive', 0)} at-risk pairs skipped).", "DPDP Act (health data)"),
            ("Opt-out", "Every offer says how to stop; withdrawal stops sending and blocks cashback from that moment.", "DPDP Act"),
            ("No dark patterns", "Real expiry dates only; the 'add Rs X more' hint shows the true gap to the minimum bill.",
             "CCPA Dark Patterns Guidelines 2023"),
            ("Cross-app identity", "Off: customers are not linked to their UPI IDs on other apps."
             if not comp.get("cross_app_linking") else "On (requires legal sign-off).", "NPCI · DPDP Act"),
            ("Phone numbers", "None in any campaign file. Offers go out in-app.", "Data minimisation"),
            ("SMS", "On, via DLT-registered templates" if comp.get("sms_fallback") else
             "Off. Needs TRAI DLT-registered templates and DND scrubbing first.", "TRAI TCCCPR"),
            ("Deletion", (f"Customer-level files deleted on {comp['purged_on']}." if comp.get("purged_on") else
                          f"Customer-level files deleted after {comp['retention_days']} days (on or after "
                          f"{comp['delete_after']}); the cashback ledger is kept with hashed ids. "
                          "<code>erase</code> handles individual deletion requests."), "DPDP Act"),
            ("Data location", f"Must run on servers in {comp.get('data_residency', 'IN')}.", "RBI 2018 data localisation · NPCI"),
        ]
        body = "".join(f"<tr><td><b>{_e(a)}</b></td><td style='white-space:normal'>{b}</td>"
                       f"<td class='muted' style='white-space:normal'>{_e(c)}</td></tr>" for a, b, c in rows)
        parts.append("<section><h2>Compliance checks in this run</h2><p class=\"muted\">Guardrails applied "
                     "automatically. A legal review by the platform and its partner banks is still required.</p>"
                     f"<div class=\"tbl\"><table><thead><tr><th>Check</th><th>What this run did</th><th>Why</th></tr>"
                     f"</thead><tbody>{body}</tbody></table></div></section>")

    s = results.get("safety", {})
    g = cfg["guardrails"]
    learned = results.get("learned", {})
    health = [
        ("Opt-out rate", _pct(s.get("opt_out_rate")), f"pause above {_pct(g['max_opt_out_rate'])}",
         (s.get("opt_out_rate") or 0) <= g["max_opt_out_rate"]),
        ("Complaint rate", _pct(s.get("complaint_rate"), 2), f"pause above {_pct(g['max_complaint_rate'], 2)}",
         (s.get("complaint_rate") or 0) <= g["max_complaint_rate"]),
        ("Shop approval rate", _pct(s.get("approval_rate"), 0), f"warn below {_pct(g['min_merchant_approval_rate'], 0)}",
         (s.get("approval_rate") or 1) >= g["min_merchant_approval_rate"]),
        ("Shop budget used", _pct(s.get("merchant_budget_burn"), 0), "of the worst case reserved", True),
        ("Paytm budget used", _pct(s.get("paytm_budget_burn"), 1), f"of Rs {cfg['campaign']['paytm_monthly_budget']:,}", True),
    ]
    hb = "".join(f"<tr><td><b>{_e(a)}</b></td><td class='num'>{_e(b)}</td><td class='muted'>{_e(c)}</td>"
                 f"<td class='{'ok' if ok else 'bad'}'>{'OK' if ok else 'BREACH'}</td></tr>" for a, b, c, ok in health)
    lr = (f"Save rate measured {learned.get('save_rate_measured')} → used next run {learned.get('save_rate')} "
          f"(evidence weight {learned.get('evidence_weight')}, shrunk toward the prior {cfg['economics']['assumed_save_rate']}). "
          + ("Not applied: simulated outcomes." if results.get("simulated_outcomes")
             and not cfg["economics"].get("allow_simulated_learning") else ""))
    parts.append("<section><h2>Programme health and learning</h2><div class=\"tbl\"><table><thead><tr><th>Check</th>"
                 "<th class='num'>This run</th><th>Limit</th><th>Status</th></tr></thead><tbody>" + hb + "</tbody></table></div>"
                 f"<p class=\"muted\" style=\"margin-top:10px\">{_e(lr)}</p></section>")

    files = sorted(p.name for p in run_dir.glob("*") if p.is_file() and p.name != "report.html")
    parts.append("<section><h2>Files in this run</h2><p class=\"muted\">"
                 + " · ".join(f"<code>{_e(x)}</code>" for x in files) + "</p></section>")

    doc = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
           f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Lost &amp; Fund Campaign Report</title>"
           f"<style>{CSS}</style></head><body><header class=\"brandbar\"><div><span class=\"ptm\"><b>pay</b><i>tm</i></span>"
           f"<span class=\"t\">Lost &amp; Fund · campaign report</span></div></header><main>{''.join(parts)}</main><div id=\"tip\"></div>"
           f"<script>{JS}</script></body></html>")
    out = run_dir / "report.html"
    out.write_text(doc, encoding="utf-8")
    return out
