'use strict';
/* Lost & Fund static demo: the demo API and the offer ledger, running in the browser.
   Used when the site is hosted statically (Vercel). Same endpoints and rules as demo/server.py and
   lost_n_fund/ledger.py; the campaign snapshot comes from /data/demo.json and each viewer's clicks are
   kept in localStorage ("Reset demo" clears them). */
(function () {
  const STORE_KEY = 'lnf-demo-state-v1';
  const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  let D = null, ST = null, L = null, OFFERS = null, BY_CUST = null, AT_RISK = null;

  const pad = n => String(n).padStart(2, '0');
  const iso = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  const dayOf = ts => ts.slice(0, 10);
  const monthOf = ts => ts.slice(0, 7);
  const dm = day => `${day.slice(8, 10)} ${MON[+day.slice(5, 7) - 1]}`;
  const tsLabel = ts => `${dm(ts)} ${ts.slice(11, 16)}`;
  const r2 = x => Math.round(x * 100) / 100;
  const rs = x => 'Rs ' + Math.round(x).toLocaleString('en-IN');
  const sum = (xs, f) => xs.reduce((a, x) => a + f(x), 0);
  const fail = msg => { throw new Error(msg); };

  function weekStart(day) {
    const d = new Date(day + 'T00:00:00');
    d.setDate(d.getDate() - (d.getDay() + 6) % 7);   // Monday
    return iso(d).slice(0, 10);
  }
  function countBand(n, bands) {
    if (n < bands[0]) return 'a few';
    for (let i = 0; i < bands.length - 1; i++) if (n >= bands[i] && n < bands[i + 1]) return `${bands[i]}-${bands[i + 1] - 1}`;
    return `${bands[bands.length - 1]}+`;
  }

  /* ---------------------------------------------------------------- ledger (port of lost_n_fund/ledger.py) */
  class Ledger {
    constructor(offers, budgets, rules, optOuts) {
      this.rules = rules; this.budget = budgets; this.used = {}; this.optOuts = { ...optOuts };
      this.offers = new Map(); this.entries = []; this.results = {}; this.applied = {}; this.blocked = {};
      for (const o of offers) {
        this.offers.set(o.customer_id + '|' + o.merchant_id,
          { ...o, uses_done: 0, days: new Set(), rewarded: false, last_day: null, credited: {} });
      }
    }
    offerFor(c, m) { return this.offers.get(c + '|' + m); }
    need(o) { return o.offer_type === 'streak' ? o.streak_visits : 1; }
    progress(o) {
      if (o.offer_type === 'streak') return { done: o.days.size, need: this.need(o), finished: o.rewarded };
      if (o.offer_type === 'pct') return { done: o.uses_done, need: o.uses, finished: o.uses_done >= o.uses };
      return { done: o.rewarded ? 1 : 0, need: 1, finished: o.rewarded };
    }
    optOut(c, ts) { if (!(c in this.optOuts)) this.optOuts[c] = ts; }
    block(tid, reason, extra = {}) {
      this.blocked[reason] = (this.blocked[reason] || 0) + 1;
      return (this.results[tid] = { txn_id: tid, applied: false, cashback: 0, reason, ...extra });
    }
    entry(kind, o, tid, ts, amount, cashback, m = 0, p = 0) {
      const e = { kind, offer_id: o.offer_id, txn_id: tid, customer_id: o.customer_id, merchant_id: o.merchant_id, ts,
        amount_paid: r2(amount), cashback: r2(cashback), merchant_share: r2(m), paytm_share: r2(p) };
      this.entries.push(e);
      return e;
    }
    onPayment(t) {
      const tid = t.txn_id;
      if (this.results[tid]) return this.results[tid];            // idempotent
      const o = this.offerFor(t.customer_id, t.merchant_id);
      if (!o) return { txn_id: tid, applied: false, cashback: 0, reason: 'no_offer' };
      const ts = t.ts, amount = +t.amount, day = dayOf(ts);
      if ((t.status || 'SUCCESS') !== 'SUCCESS') return this.block(tid, 'payment_failed');
      if ((t.payer_app || 'paytm') !== 'paytm') return this.block(tid, 'paid_with_another_app');
      const oo = this.optOuts[o.customer_id];
      if (oo && ts >= oo) return this.block(tid, 'consent_withdrawn');
      if (day < o.valid_from) return this.block(tid, 'not_started');
      if (day > o.valid_till) return this.block(tid, 'expired');
      if (this.progress(o).finished) return this.block(tid, 'offer_used_up');
      if (amount < o.min_bill) return this.block(tid, 'below_min_bill', { short_by: r2(o.min_bill - amount) });
      if (this.rules.one_redemption_per_day && o.last_day === day) return this.block(tid, 'already_counted_today');
      o.last_day = day;
      this.applied[tid] = { key: o.customer_id + '|' + o.merchant_id, day };
      let cashback;
      if (o.offer_type === 'streak') {
        o.days.add(day);
        if (o.days.size < this.need(o)) {
          this.entry('STREAK_DAY', o, tid, ts, amount, 0);
          return (this.results[tid] = { txn_id: tid, applied: true, cashback: 0, reason: 'streak_progress', progress: this.progress(o) });
        }
        cashback = o.cashback; o.rewarded = true;
      } else if (o.offer_type === 'pct') {
        cashback = r2(Math.min(o.pct / 100 * amount, o.cap)); o.uses_done += 1;
      } else {
        cashback = o.cashback; o.rewarded = true;
      }
      const share = o.paytm_share || 0;
      let m = r2(cashback * (1 - share)), p = r2(cashback * share), overflow = 0;
      if (m > 0) {
        const k = o.merchant_id + '|' + monthOf(ts);
        const left = (this.budget[o.merchant_id] || 0) - (this.used[k] || 0);
        if (m > left) {        // budgets are reserved at worst case, so this shouldn't happen; Paytm honours it
          overflow = r2(m - Math.max(left, 0)); m = r2(m - overflow); p = r2(p + overflow);
          this.blocked.budget_overflow_paid_by_paytm = (this.blocked.budget_overflow_paid_by_paytm || 0) + 1;
        }
        this.used[k] = (this.used[k] || 0) + m;
      }
      o.credited[tid] = this.entry('CASHBACK', o, tid, ts, amount, cashback, m, p);
      return (this.results[tid] = { txn_id: tid, applied: true, cashback, merchant_share: m, paytm_share: p,
        reason: 'cashback_credited', progress: this.progress(o), overflow });
    }
    onRefund(rid, origId, ts) {
      if (this.results[rid]) return this.results[rid];
      const orig = this.results[origId];
      if (!this.rules.clawback_on_refund || !orig || !orig.applied) {
        return (this.results[rid] = { txn_id: rid, reversed: 0, reason: 'nothing_to_reverse' });
      }
      const { key, day } = this.applied[origId];
      const o = this.offers.get(key);
      let reversed = 0;
      const c = o.credited[origId];
      if (c) {
        delete o.credited[origId];
        reversed = c.cashback;
        const k = o.merchant_id + '|' + monthOf(c.ts);
        this.used[k] = (this.used[k] || 0) - c.merchant_share;
        this.entry('REVERSAL', o, rid, ts, -c.amount_paid, -c.cashback, -c.merchant_share, -c.paytm_share);
        if (o.offer_type === 'pct') o.uses_done = Math.max(0, o.uses_done - 1); else o.rewarded = false;
      }
      if (o.offer_type === 'streak') o.days.delete(day);
      if (o.last_day === day) o.last_day = null;
      return (this.results[rid] = { txn_id: rid, reversed, reason: reversed ? 'cashback_reversed' : 'streak_day_removed' });
    }
    settlementLines(mid) {   // one netting line per shop per week, no customer detail
      const by = {};
      for (const e of this.entries) {
        if (e.merchant_share === 0 || e.merchant_id !== mid) continue;
        const w = weekStart(dayOf(e.ts));
        by[w] = (by[w] || 0) + e.merchant_share;
      }
      return Object.keys(by).sort().filter(w => Math.abs(by[w]) >= 0.01).map(w => ({
        week_starting: w, cashback_netted: r2(by[w]),
        statement_text: `Lost & Fund cashback, week of ${dm(w)}: Rs ${by[w].toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      }));
    }
    stats() {
      const cb = this.entries.filter(e => e.kind === 'CASHBACK');
      return { customers_paid_cashback: new Set(cb.map(e => e.customer_id)).size,
        cashback_total: r2(sum(this.entries, e => e.cashback)),
        merchant_funded: r2(sum(this.entries, e => e.merchant_share)),
        paytm_funded: r2(sum(this.entries, e => e.paytm_share)) };
    }
  }

  /* ---------------------------------------------------------------- state */
  const empty = () => ({ run: D.overview.run.run_id, decisions: {}, txns: [], events: [], opt_outs: {}, demo_day: 0 });
  function loadState() {
    try {
      const s = JSON.parse(localStorage.getItem(STORE_KEY));
      if (s && s.run === D.overview.run.run_id) return s;
    } catch (e) { /* private window or blocked storage: start fresh */ }
    return empty();
  }
  function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(ST)); } catch (e) { /* ignore */ } }
  function rebuild() {
    L = new Ledger(D.offers.filter(o => o.group === 'treatment'), D.budgets, D.rules, ST.opt_outs);
    for (const t of ST.txns) t.result = t.kind === 'pay' ? L.onPayment(t) : L.onRefund(t.txn_id, t.refund_of, t.ts);
  }
  function event(kind, text) {
    const d = new Date();
    ST.events.push({ ts: `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`, kind, text });
    ST.events = ST.events.slice(-200);
  }
  function now() {   // demo clock: campaign day + "Next day" clicks, at the current time of day
    const d = new Date(D.snapshot + 'T00:00:00');
    d.setDate(d.getDate() + ST.demo_day);
    const n = new Date();
    d.setHours(n.getHours(), n.getMinutes(), n.getSeconds());
    return iso(d);
  }
  async function init() {
    if (D) return;
    const r = await fetch('/data/demo.json', { cache: 'no-cache' });
    if (!r.ok) fail('Could not load the demo data');
    D = await r.json();
    OFFERS = new Map(D.offers.map(o => [o.offer_id, o]));
    BY_CUST = new Map(D.offers.map(o => [o.customer_id, o.offer_id]));
    AT_RISK = D.at_risk.rows.map(row => Object.fromEntries(D.at_risk.cols.map((c, i) => [c, row[i]])));
    ST = loadState();
    rebuild();
  }

  const isAutopilot = mid => !!(D.merchants[mid] && D.merchants[mid].auto_approve);
  function status(oid) {
    const o = OFFERS.get(oid);
    if (o.customer_id in ST.opt_outs) return 'opted_out';
    if (o.group === 'control') return 'holdout';
    if (o.group === 'global_holdout') return 'global_holdout';
    if (o.needs_merchant_approval) {
      let d = ST.decisions[o.merchant_id];
      if (d == null && isAutopilot(o.merchant_id)) d = 'approved';
      if (d == null) return 'awaiting';
      if (d === 'rejected') return 'rejected';
    }
    const lo = L.offerFor(o.customer_id, o.merchant_id);
    const pr = lo ? L.progress(lo) : { done: 0, finished: false };
    if (pr.finished) return 'redeemed';
    if (dayOf(now()) > o.valid_till) return 'expired';
    return pr.done > 0 ? 'progress' : 'approved';
  }
  const decidedVia = mid => (mid in ST.decisions ? 'merchant_app' : isAutopilot(mid) ? 'autopilot' : null);
  const treatment = () => D.offers.filter(o => o.group === 'treatment');
  const shopOffers = mid => treatment().filter(o => o.needs_merchant_approval && o.merchant_id === mid);
  const counter = xs => xs.reduce((c, x) => (c[x] = (c[x] || 0) + 1, c), {});

  /* ---------------------------------------------------------------- endpoints */
  function overview() {
    const t = treatment();
    const st = counter(t.map(o => status(o.offer_id)));
    const shopT = t.filter(o => o.needs_merchant_approval);
    const mids = [...new Set(shopT.map(o => o.merchant_id))].sort();
    const s = L.stats();
    const sales = sum(ST.txns.filter(x => x.kind === 'pay' && x.result && x.result.applied), x => x.amount);
    const base = D.overview;
    return {
      ...base,
      run: { ...base.run, demo_day: ST.demo_day, demo_date: dayOf(now()) },
      live: {
        at_risk: AT_RISK.length, offers: t.length, shop_offers: shopT.length, paytm_offers: t.length - shopT.length,
        holdout: D.offers.filter(o => o.group === 'control').length,
        global_holdout: D.offers.filter(o => o.group === 'global_holdout').length,
        awaiting: st.awaiting || 0, live: (st.approved || 0) + (st.progress || 0) + (st.redeemed || 0), rejected: st.rejected || 0,
        redeemed_customers: s.customers_paid_cashback, merchants_total: mids.length,
        merchants_responded: mids.filter(decidedVia).length, cashback_total: s.cashback_total,
        cashback_shop: s.merchant_funded, cashback_paytm: s.paytm_funded, sales_recovered: r2(sales),
      },
      compliance: { ...base.compliance, opted_out_during_campaign: Object.keys(ST.opt_outs).length },
      events: ST.events.slice(-25).reverse(),
    };
  }

  function atRisk(q) {
    const tier = q.get('tier') || '', st = q.get('status') || '', route = q.get('route') || '', text = (q.get('q') || '').trim().toLowerCase();
    const offset = Math.max(+q.get('offset') || 0, 0), limit = Math.min(Math.max(+q.get('limit') || 25, 1), 200);
    const stat = AT_RISK.map(r => (r.offer_id ? status(r.offer_id) : 'no_offer'));
    const rows = [];
    AT_RISK.forEach((r, i) => {
      if (tier && r.risk_tier !== tier) return;
      if (st && stat[i] !== st) return;
      if (route && r.route !== route) return;
      if (text && ![r.name, r.customer_id, r.merchant_name || ''].some(v => String(v).toLowerCase().includes(text))) return;
      rows.push({ ...r, status: stat[i] });
    });
    const page = rows.slice(offset, offset + limit).map(r => {
      const o = r.offer_id ? OFFERS.get(r.offer_id) : null;
      return { ...r, offer_type: o ? o.offer_type : null, funder: o ? o.funder : null };
    });
    return { total: rows.length, rows: page, status_counts: counter(stat) };
  }

  function merchantList() {
    return Object.keys(D.alerts).map(mid => {
      const st = counter(shopOffers(mid).map(o => status(o.offer_id)));
      const m = D.merchants[mid];
      return { merchant_id: mid, name: m.name, category: m.category, locality: m.locality, band: D.alerts[mid].band,
        awaiting: (st.awaiting || 0) > 0, approved: ((st.approved || 0) + (st.progress || 0) + (st.redeemed || 0)) > 0,
        rejected: (st.rejected || 0) > 0, redeemed: st.redeemed || 0, autopilot: isAutopilot(mid) };
    }).sort((a, b) => (a.awaiting === b.awaiting ? a.name.localeCompare(b.name) : a.awaiting ? -1 : 1));
  }

  function merchantDetail(mid) {
    const al = D.alerts[mid] || fail('no Lost & Fund alert for this shop');
    const g = shopOffers(mid);
    const st = counter(g.map(o => status(o.offer_id)));
    const ids = new Set(g.map(o => o.offer_id));
    const pays = ST.txns.filter(x => x.kind === 'pay' && ids.has(x.offer_id) && x.result && x.result.applied);
    const back = new Set(pays.map(x => x.customer_id)).size;
    const lines = L.settlementLines(mid);
    const { presenter, ...app } = al;
    return {
      app: {
        ...app, merchant: D.merchants[mid],
        decision: st.awaiting ? 'awaiting' : st.rejected ? 'declined' : 'approved', decided_via: decidedVia(mid),
        results: { came_back_band: back ? countBand(back, [1, 5, 10, 20]) : '0', sales: r2(sum(pays, x => x.amount)),
          cashback_netted: r2(sum(lines, l => l.cashback_netted)) },
        statements: lines.map(l => ({ week_starting: l.week_starting, statement_text: l.statement_text })),
      },
      presenter,
    };
  }

  function decide(mid, body) {
    const choice = String(body.decision || '').trim().toLowerCase();
    if (!['approve', 'decline'].includes(choice)) fail("decision must be 'approve' or 'decline'");
    if (!D.alerts[mid]) fail('no Lost & Fund alert for this shop');
    const val = choice === 'approve' ? 'approved' : 'rejected';
    const auto = isAutopilot(mid);
    if (ST.decisions[mid] === val || (val === 'approved' && auto && !(mid in ST.decisions))) fail('this offer was already decided');
    ST.decisions[mid] = val;
    const m = D.merchants[mid], al = D.alerts[mid];
    if (val === 'approved') {
      event('approve', `${m.owner_first} (${m.name}) approved a budget of up to ${rs(al.budget_cap)} without seeing names. Paytm notified ${al.band} regulars.`);
    } else {
      event('reject', `${m.owner_first} (${m.name}) ${auto ? 'paused autopilot' : 'declined the offer'}. No customers were contacted.`);
    }
    save();
    return merchantDetail(mid);
  }

  function customerList(mid) {
    return D.offers.filter(o => !mid || o.merchant_id === mid).map(o => ({ customer_id: o.customer_id, masked: o.masked,
      merchant_id: o.merchant_id, merchant_name: o.merchant_name, funder: o.funder, status: status(o.offer_id) }));
  }

  function offerView(oid) {
    const o = OFFERS.get(oid);
    const lo = L.offerFor(o.customer_id, o.merchant_id);
    const pr = lo ? L.progress(lo) : { done: 0, need: 1, finished: false };
    return { ...o, status: status(oid), progress: pr, counted_today: !!(lo && lo.last_day && lo.last_day === dayOf(now())),
      why: D.why_text[o.language] || D.why_text.en };
  }

  function customerDetail(cid) {
    const oid = BY_CUST.get(cid) || fail('customer has no Lost & Fund offer');
    const o = OFFERS.get(oid);
    return {
      customer: { customer_id: cid, first_name: o.first_name, masked: o.masked, locality: o.customer_locality, language: o.language },
      risk: D.risk[cid] || null,
      offers: [offerView(oid)],
      recent: D.recent[cid] || [],
      payments: ST.txns.filter(x => x.customer_id === cid).map(x => ({
        txn_id: x.txn_id, kind: x.kind, merchant_name: x.merchant_name, amount: x.amount, ts_label: x.ts_label,
        refund_of: x.refund_of, cashback: (x.result && x.result.cashback) || 0, reason: x.result && x.result.reason,
        reversed: (x.result && x.result.reversed) || 0, refunded: ST.txns.some(y => y.refund_of === x.txn_id),
      })).reverse(),
      opted_out: cid in ST.opt_outs,
      clock: { day: ST.demo_day, date: dayOf(now()) },
    };
  }

  function pay(body) {
    const o = OFFERS.get(body.offer_id) || fail('unknown offer');
    const st = status(body.offer_id);
    if (!['approved', 'progress', 'redeemed', 'expired'].includes(st)) fail(`offer is not live (status: ${st})`);
    const amount = r2(+body.amount);
    if (!(amount >= 1 && amount <= 100000)) fail('enter an amount between Rs 1 and Rs 1,00,000');
    const ts = now();
    const txn = { kind: 'pay', txn_id: `D${pad(ST.txns.length + 1).padStart(6, '0')}`, offer_id: o.offer_id,
      customer_id: o.customer_id, merchant_id: o.merchant_id, merchant_name: o.merchant_name, amount, ts,
      ts_label: tsLabel(ts), status: 'SUCCESS', payer_app: 'paytm', refund_of: null };
    const res = L.onPayment(txn);
    txn.result = res;
    ST.txns.push(txn);
    if (res.reason === 'cashback_credited') {
      const who = { merchant: 'shop pays, netted weekly', paytm: 'Paytm pays', cofunded: 'shop and Paytm split it' }[o.funder] || '';
      event('pay', `${o.masked} paid ${rs(amount)} at ${o.merchant_name} and got ${rs(res.cashback)} cashback (${who}).`);
    } else if (res.reason === 'streak_progress') {
      event('pay', `${o.masked} paid ${rs(amount)} at ${o.merchant_name}: visit ${res.progress.done} of ${res.progress.need}.`);
    } else {
      event('pay', `${o.masked} paid ${rs(amount)} at ${o.merchant_name} (no cashback: ${String(res.reason).replace(/_/g, ' ')}).`);
    }
    save();
    const lo = L.offerFor(o.customer_id, o.merchant_id);
    return { ...txn, result: res, progress: lo ? L.progress(lo) : null, soundbox: `Paytm par ${Math.round(amount).toLocaleString('en-IN')} rupaye prapt hue` };
  }

  function refund(body) {
    const orig = ST.txns.find(x => x.txn_id === body.txn_id && x.kind === 'pay') || fail('unknown payment');
    if (ST.txns.some(x => x.refund_of === body.txn_id)) fail('this payment was already refunded');
    const ts = now();
    const txn = { kind: 'refund', txn_id: `DR${pad(ST.txns.length + 1).padStart(5, '0')}`, offer_id: orig.offer_id,
      customer_id: orig.customer_id, merchant_id: orig.merchant_id, merchant_name: orig.merchant_name, amount: orig.amount,
      ts, ts_label: tsLabel(ts), refund_of: body.txn_id };
    const res = L.onRefund(txn.txn_id, body.txn_id, ts);
    txn.result = res;
    ST.txns.push(txn);
    const masked = OFFERS.get(orig.offer_id).masked;
    event('refund', `${masked}'s ${rs(orig.amount)} payment at ${orig.merchant_name} was refunded` +
      (res.reversed ? `: ${rs(res.reversed)} cashback clawed back.` : res.reason === 'streak_day_removed' ? ': streak visit removed.' : '.'));
    save();
    return { ...txn, result: res };
  }

  function optOut(cid) {
    if (!BY_CUST.has(cid)) fail('unknown customer');
    if (!(cid in ST.opt_outs)) {
      const ts = now();
      ST.opt_outs[cid] = ts;
      L.optOut(cid, ts);
      event('optout', `${OFFERS.get(BY_CUST.get(cid)).masked} turned off Lost & Fund offers. The offer stopped at once and they are excluded from all future runs.`);
      save();
    }
    return customerDetail(cid);
  }

  function nextDay() {
    ST.demo_day += 1;
    event('clock', `Demo clock moved to day ${ST.demo_day + 1} (${dm(dayOf(now()))}).`);
    save();
    return { day: ST.demo_day, date: dayOf(now()) };
  }

  function reset() { ST = empty(); rebuild(); save(); return { ok: true }; }

  function exportDecisions() {
    const rows = Object.keys(D.alerts).sort().filter(decidedVia)
      .map(mid => [mid, ST.decisions[mid] || 'approved', decidedVia(mid)]);
    const csv = ['merchant_id,decision,decided_via', ...rows.map(r => r.join(','))].join('\n') + '\n';
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'merchant_decisions.csv';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    return { rows: rows.length, path: 'merchant_decisions.csv',
      command: 'python run_pipeline.py decide --decisions merchant_decisions.csv' };
  }

  async function route(path, body) {
    await init();
    const url = new URL(path, location.origin);
    const p = url.pathname, q = url.searchParams;
    let m;
    if (p === '/api/overview') return overview();
    if (p === '/api/at-risk') return atRisk(q);
    if (p === '/api/merchants') return merchantList();
    if ((m = p.match(/^\/api\/merchants\/([^/]+)\/decide$/))) return decide(decodeURIComponent(m[1]), body || {});
    if ((m = p.match(/^\/api\/merchants\/([^/]+)$/))) return merchantDetail(decodeURIComponent(m[1]));
    if (p === '/api/customers') return customerList(q.get('merchant_id') || '');
    if ((m = p.match(/^\/api\/customers\/([^/]+)\/opt-out$/))) return optOut(decodeURIComponent(m[1]));
    if ((m = p.match(/^\/api\/customers\/([^/]+)$/))) return customerDetail(decodeURIComponent(m[1]));
    if (p === '/api/pay') return pay(body || {});
    if (p === '/api/refund') return refund(body || {});
    if (p === '/api/clock/next') return nextDay();
    if (p === '/api/reset') return reset();
    if (p === '/api/export-decisions') return exportDecisions();
    fail('Not found');
  }

  // a deep copy, so screens never hold references into the ledger's live objects
  window.staticApi = (path, body) => route(path, body).then(r => JSON.parse(JSON.stringify(r)));
})();
