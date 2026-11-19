'use strict';
/* WinBack demo: Paytm Ops (P), Merchant app (A), Customer app (B), sharing one live state on the server. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const rs = n => '₹' + Number(n || 0).toLocaleString('en-IN', { maximumFractionDigits: Math.abs(n) < 100 && n % 1 ? 2 : 0 });
const num = n => Number(n || 0).toLocaleString('en-IN');
const pct = (x, d = 0) => (x == null ? '–' : (x * 100).toFixed(d) + '%');
const pts = x => (x == null ? '–' : (x > 0 ? '+' : '') + Number(x).toFixed(1) + ' pts');
const initials = s => String(s || '').split(/\s+/).filter(Boolean).slice(0, 2).map(w => w[0]).join('').toUpperCase();
const cap1 = s => String(s || '').charAt(0).toUpperCase() + String(s || '').slice(1);
const fmtDate = iso => new Date(iso + 'T00:00:00').toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
const LIVE = ['approved', 'progress', 'redeemed'];
const PAYABLE = ['approved', 'progress', 'expired'];

async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`);
  return data;
}

const S = {
  view: 'ops',
  merchantId: null,
  customerId: null,
  lang: {},                       // merchant_id -> 'hi' | 'en'
  ops: { tier: '', status: '', route: '', q: '', offset: 0 },
  cust: { screen: 'home', amount: null, last: null, banner: true },
  exportInfo: null,
};

const STATUS = {
  awaiting: ['Awaiting shop', 'warn'],
  approved: ['Offer live', 'info'],
  progress: ['Streak started', 'info'],
  redeemed: ['Got cashback', 'good'],
  expired: ['Offer expired', 'mute'],
  rejected: ['Shop declined', 'bad'],
  holdout: ['Holdout', 'hold'],
  global_holdout: ['Global holdout', 'hold'],
  no_offer: ['No offer', 'mute'],
  opted_out: ['Opted out', 'mute'],
};
const chip = (s, label) => `<span class="chip ${(STATUS[s] || ['', 'mute'])[1]}">${esc(label || (STATUS[s] || [s])[0])}</span>`;
const FUNDER = { merchant: ['Shop pays', 'info'], cofunded: ['Shop + Paytm', 'warn'], paytm: ['Paytm pays', 'good'] };
const funderChip = f => f ? `<span class="chip ${FUNDER[f][1]}">${FUNDER[f][0]}</span>` : '';
const ROUTE_CLASS = { shop_switch: 'info', unclear: 'warn', app_switch: 'good', payment_friction: 'bad', merchant_decline: 'bad',
  seasonal: 'hold', normal_gap: 'mute', gone: 'mute' };

const ICON = {
  bell: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M18 8a6 6 0 1 0-12 0c0 7-3 9-3 9h18s-3-2-3-9M13.7 21a2 2 0 0 1-3.4 0"/></svg>',
  speaker: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M11 5 6 9H2v6h4l5 4V5zM15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14"/></svg>',
  scan: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2M7 12h10"/></svg>',
  phone: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="6" y="2" width="12" height="20" rx="2"/><path d="M11 18h2"/></svg>',
  bank: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 10h18L12 4 3 10zM5 10v8M10 10v8M14 10v8M19 10v8M3 20h18"/></svg>',
  wallet: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M20 7H5a2 2 0 0 1 0-4h13v4M3 5v14a2 2 0 0 0 2 2h15V7"/><circle cx="16" cy="14" r="1.2"/></svg>',
  lock: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>',
  check: '<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 5 5L20 7"/></svg>',
  signal: '<svg width="54" height="12" viewBox="0 0 54 12" fill="currentColor"><rect x="0" y="8" width="3" height="4" rx="1"/><rect x="5" y="5" width="3" height="7" rx="1"/><rect x="10" y="2" width="3" height="10" rx="1"/><rect x="15" y="0" width="3" height="12" rx="1"/><rect x="29" y="1" width="21" height="10" rx="3" fill="none" stroke="currentColor" stroke-width="1.4"/><rect x="31" y="3" width="14" height="6" rx="1.5"/><rect x="51" y="4" width="2" height="4" rx="1"/></svg>',
  search: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
  bellLg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"><path d="M18 8a6 6 0 1 0-12 0c0 7-3 9-3 9h18s-3-2-3-9M13.7 21a2 2 0 0 1-3.4 0"/></svg>',
  qr: '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><path d="M14 14h3v3M21 14v.01M17 21h4v-4M14 18v3"/></svg>',
  home: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11 12 4l9 7v9a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z"/></svg>',
  gift: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="8" width="18" height="5" rx="1"/><path d="M5 13v7h14v-7M12 8v12M12 8S10 3 7.5 4.5 9 8 12 8zM12 8s2-5 4.5-3.5S15 8 12 8z"/></svg>',
  history: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5M12 7v5l3 2"/></svg>',
  user: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg>',
  mobile: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="7" y="2" width="10" height="20" rx="2"/><path d="M11 18h2"/></svg>',
  bulb: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"/></svg>',
  tv: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="2" y="6" width="20" height="13" rx="2"/><path d="m8 2 4 4 4-4"/></svg>',
  card: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="2" y="5" width="20" height="14" rx="2"/><path d="M2 10h20M6 15h4"/></svg>',
  shield: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z"/><path d="m9 12 2 2 4-4"/></svg>',
  users: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="9" cy="8" r="4"/><path d="M2 21a7 7 0 0 1 14 0M16 3.5a4 4 0 0 1 0 8M22 21a7 7 0 0 0-4-6.3"/></svg>',
  store: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M3 9 5 3h14l2 6M3 9h18v2a3 3 0 0 1-6 0 3 3 0 0 1-6 0 3 3 0 0 1-6 0zM5 13v8h14v-8"/></svg>',
  live: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M22 12h-4l-3 8L9 4l-3 8H2"/></svg>',
  rupee: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M6 4h12M6 9h12M14 4c3 0 4 2.5 4 5s-2 5-6 5H6l8 7"/></svg>',
  back: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="m15 18-6-6 6-6"/></svg>',
};
const PTM = dark => `<span class="ptm ${dark ? 'on-dark' : ''}"><b>pay</b><i>tm</i></span>`;

function toast(msg, kind = '') {
  const t = document.createElement('div');
  t.className = 'toast ' + kind;
  t.textContent = msg;
  $('#toasts').append(t);
  requestAnimationFrame(() => t.classList.add('show'));
  setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 300); }, 3600);
}

function speak(text) {
  if (!('speechSynthesis' in window)) { toast('This browser cannot play speech'); return; }
  const u = new SpeechSynthesisUtterance(text);
  const voices = speechSynthesis.getVoices();
  const v = voices.find(v => v.lang === 'en-IN') || voices.find(v => v.lang.startsWith('hi')) || voices.find(v => v.lang.startsWith('en'));
  if (v) u.voice = v;
  u.lang = v ? v.lang : 'en-IN';
  u.rate = 0.95;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

/* ---------------------------------------------------------------- routing */
function parseHash() {
  const [view, id] = location.hash.replace('#', '').split('/');
  if (['ops', 'merchant', 'customer', 'story'].includes(view)) S.view = view;
  if (view === 'merchant' && id) S.merchantId = id;
  if (view === 'customer' && id && id !== S.customerId) setCustomer(id);
}
function go(view, id) {
  const h = id ? `${view}/${id}` : view;
  if (location.hash.replace('#', '') === h) render(); else location.hash = h;
}
function setCustomer(id) {
  if (id !== S.customerId) S.cust = { screen: 'home', amount: null, last: null, banner: true };
  S.customerId = id;
}
window.addEventListener('hashchange', () => { parseHash(); window.scrollTo(0, 0); render(); });

async function render() {
  $$('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === S.view));
  const root = $('#view');
  try {
    const overview = await api('/api/overview');
    renderSteps(overview);
    if (S.view === 'ops') await renderOps(root, overview);
    else if (S.view === 'merchant') await renderMerchantView(root);
    else if (S.view === 'customer') await renderCustomerView(root);
    else await renderStory(root);
  } catch (e) {
    root.innerHTML = `<div class="error">${esc(e.message)}</div>`;
  }
}

function renderSteps(o) {
  const L = o.live;
  const steps = [
    ['ops', 'Find slipping regulars', true],
    ['merchant', 'Shop approves budget', L.merchants_responded > 0],
    ['customer', 'Customer gets cashback', L.redeemed_customers > 0 || L.sales_recovered > 0],
    ['ops', 'Measure vs holdout', L.redeemed_customers > 0],
  ];
  const cur = S.view === 'story' ? -1 : (S.view === 'merchant' ? 1 : S.view === 'customer' ? 2 : (L.redeemed_customers > 0 ? 3 : 0));
  $('#steps').innerHTML = steps.map(([v, label, done], i) =>
    `${i ? '<span class="step-sep">›</span>' : ''}<button class="step ${done ? 'done' : ''} ${i === cur ? 'current' : ''}" data-go="${v}">
      <span class="n">${done ? '✓' : i + 1}</span>${label}</button>`).join('');
  $$('#steps [data-go]').forEach(b => b.onclick = () => go(b.dataset.go, b.dataset.go === 'merchant' ? S.merchantId : b.dataset.go === 'customer' ? S.customerId : null));
}

/* ---------------------------------------------------------------- shared */
function phone(inner, kind = '') {
  return `<div class="phone ${kind}"><div class="notch"></div><div class="screen">
    <div class="statusbar"><span>9:41</span>${ICON.signal}</div>${inner}</div></div>`;
}

async function ensureMerchant() {
  const list = await api('/api/merchants');
  if (!list.length) throw new Error('No shop alerts in this run (every candidate shop was below the anonymity threshold).');
  if (!S.merchantId || !list.some(m => m.merchant_id === S.merchantId)) {
    S.merchantId = (list.find(m => m.awaiting) || list[0]).merchant_id;
  }
  return list;
}

function merchantPicker(list) {
  return `<label class="picker">Shop <select id="mpick">${list.map(m => `<option value="${m.merchant_id}" ${m.merchant_id === S.merchantId ? 'selected' : ''}>
    ${esc(m.name)} · ${esc(m.locality)}${m.autopilot ? ' · autopilot' : ''}${m.awaiting ? '' : ' ✓'}</option>`).join('')}</select></label>`;
}

/* ---------------------------------------------------------------- ops */
function kpi(v, l, s, hot, icon) {
  return `<div class="kpi ${hot ? 'hot' : ''}">${icon ? `<div class="ki">${ICON[icon]}</div>` : ''}<div class="v">${v}</div><div class="l">${esc(l)}</div>${s ? `<div class="s">${esc(s)}</div>` : ''}</div>`;
}

function compliancePanel(c) {
  if (!c || !c.customers_total) return '';
  const row = (ok, label, detail) => `<li><span class="ck ${ok ? 'ok' : 'warn'}">${ok ? '✓' : '!'}</span>
    <div><b>${esc(label)}</b><div class="sub">${esc(detail)}</div></div></li>`;
  const elig = c.funnel && c.funnel.eligible_consented_adults;
  return `<section class="panel"><div class="panel-head"><h2>Compliance</h2><span class="muted">checked every run</span></div><ul class="comp">
    ${row(c.processed_only_consented, 'Consent first', `${num(elig)} of ${num(c.customers_total)} opted in`)}
    ${row(c.minors_excluded, 'No under-18s', `${num(c.minors)} excluded`)}
    ${row(c.names_shown_to_merchants === 0 && c.exact_counts_shown_to_merchants === 0, 'Shops see count bands only', 'no names, numbers or exact counts')}
    ${row(true, `Alerts need ${c.min_customers_per_alert}+ customers`, `${num(c.customers_held_for_anonymity)} waiting in queue`)}
    ${row(true, 'Weekly settlement netting', 'no per-payment marker')}
    ${row(true, 'No sensitive shops', `${(c.sensitive_categories_excluded || []).join(', ') || 'none'} excluded`)}
    ${row(true, 'One-tap opt-out', `${num(c.opted_out_during_campaign)} opted out`)}
    ${row(true, `Deleted after ${c.retention_days} days`, `from ${fmtDate(c.delete_after)}`)}
    ${row(c.data_residency === 'IN', 'Data stays in India', 'RBI · NPCI')}
  </ul></section>`;
}

function routesPanel(o) {
  const mx = Math.max(...o.routes.map(r => r.n), 1);
  const chk = o.diagnosis_check;
  return `<section class="panel"><div class="panel-head"><h2>Why regulars are slipping</h2><span class="muted">and who acts</span></div>
    <div class="routes">${o.routes.map(r => `<div class="route-row" data-route="${r.route}">
      <div class="rl"><span class="chip ${ROUTE_CLASS[r.route] || 'mute'}">${esc(r.label)}</span><span class="sub">${esc(r.action_label)}</span></div>
      <div class="rbar"><span style="width:${(100 * r.n / mx).toFixed(1)}%"></span></div><b>${num(r.n)}</b></div>`).join('')}</div>
    ${chk ? `<p class="fine">Demo truth check: ${pct(chk.accuracy_on_leaving)} causes right · ${pct(chk.shop_funded_precision)} shop-funded precision</p>` : ''}
  </section>`;
}

async function renderOps(root, o) {
  const f = S.ops;
  const qs = new URLSearchParams({ tier: f.tier, status: f.status, route: f.route, q: f.q, offset: f.offset, limit: 25 });
  const [list, ms] = await Promise.all([api('/api/at-risk?' + qs), api('/api/merchants')]);
  const L = o.live, M = o.model, X = o.measured;
  const sc = list.status_counts;
  const statusOpt = Object.keys(STATUS).map(k => `<option value="${k}" ${f.status === k ? 'selected' : ''}>${STATUS[k][0]} (${num(sc[k] || 0)})</option>`).join('');
  const routeOpt = o.routes.map(r => `<option value="${r.route}" ${f.route === r.route ? 'selected' : ''}>${esc(r.label)} (${num(r.n)})</option>`).join('');

  const rows = list.rows.map(r => `
    <tr class="${r.offer_id ? 'clickable' : ''}" data-cust="${r.offer_id ? r.customer_id : ''}">
      <td><div class="who">${esc(r.name)}</div><div class="sub">${esc(r.customer_id)} · ${esc(r.locality)}</div></td>
      <td><div class="risk"><div class="riskbar"><span class="${r.risk_tier}" style="width:${(r.lapse_prob * 100).toFixed(0)}%"></span></div><b>${pct(r.lapse_prob)}</b></div>
          <div class="sub">at ${esc(r.merchant_name)}</div></td>
      <td class="why"><span class="chip ${ROUTE_CLASS[r.route] || 'mute'}">${esc(r.route_label)}</span><div class="sub" style="margin-top:4px">${esc(r.diagnosis_evidence)}</div></td>
      <td>${r.offer_id ? `<a class="mlink" data-m="${r.merchant_id}">${esc(r.offer_type)} cashback</a><div style="margin-top:3px">${funderChip(r.funder)}</div>`
                            : `<span class="sub">${esc(r.no_offer_reason)}</span>`}</td>
      <td>${chip(r.status)}</td>
    </tr>`).join('');

  const mrows = ms.map(m => {
    const st = m.awaiting ? '<span class="chip warn">Awaiting</span>' : m.rejected ? '<span class="chip bad">Declined</span>'
      : `<span class="chip ${m.redeemed ? 'good' : 'info'}">${m.autopilot ? 'Autopilot' : 'Approved'}${m.redeemed ? ` · ${m.redeemed} back` : ''}</span>`;
    return `<div class="mrow" data-m="${m.merchant_id}"><span class="avatar">${initials(m.name)}</span>
      <div class="grow"><div class="ellip" style="font-weight:600">${esc(m.name)}</div><div class="sub ellip">${cap1(m.category)} · ${esc(m.locality)} · ${esc(m.band)} regulars</div></div>${st}</div>`;
  }).join('') || '<div class="empty">No shop reached the anonymity threshold this run.</div>';

  const feed = o.events.length
    ? `<ul class="feed">${o.events.map(e => `<li><span class="dot ${e.kind}"></span><div class="grow">${esc(e.text)}</div><span class="t">${esc(e.ts)}</span></li>`).join('')}</ul>`
    : '<div class="empty">No activity yet.</div>';

  const p = X.power || {};
  const from = list.total ? f.offset + 1 : 0, to = Math.min(f.offset + 25, list.total);
  const shopRoutes = o.routes.filter(r => ['merchant_offer', 'paytm_offer'].includes(r.action)).reduce((a, r) => a + r.n, 0);
  root.innerHTML = `
    <section class="hero"><div class="hero-in">
      <div>
        <div class="hero-kicker">WinBack · campaign ${fmtDate(o.run.snapshot)} · model ${esc(M.version || '')}</div>
        <h1>${num(L.at_risk)} regulars slipping. Only ${num(shopRoutes)} need an offer.</h1>
        <div class="hero-actions"><a class="btn primary" href="/report" target="_blank" rel="noopener">Open campaign report</a>
          <button class="btn ghost" id="export">Export approvals for pipeline</button>
          <span class="btn ghost" style="cursor:default">Demo day ${o.run.demo_day + 1} · ${fmtDate(o.run.demo_date)}</span></div>
      </div>
      <div class="hero-stats">
        <div class="hstat"><b>${pct(M.lapsers_active_on_app)}</b><small>shop lapsers still on the app</small></div>
        <div class="hstat"><b>${num(L.offers)}</b><small>offers · ${L.shop_offers} shop / ${L.paytm_offers} Paytm</small></div>
        <div class="hstat"><b>${rs(L.cashback_total)}</b><small>cashback paid</small></div>
      </div>
    </div></section>
    ${S.exportInfo ? `<div class="notice">Saved ${S.exportInfo.rows} shop decisions. Feed them into the real pipeline with:<code>${esc(S.exportInfo.command)}</code></div>` : ''}
    <div class="kpis">
      ${kpi(num(L.at_risk), 'Regulars slipping', 'overdue at a shop they used to visit', false, 'users')}
      ${kpi(num(L.offers), 'Offers made', `${L.holdout} held out + ${L.global_holdout} global holdout`, false, 'gift')}
      ${kpi(num(L.merchants_responded) + ' / ' + num(L.merchants_total), 'Shops decided', `${L.awaiting} offers awaiting a shop`, false, 'store')}
      ${kpi(num(L.live), 'Offers live', L.rejected ? `${L.rejected} declined` : 'auto-applied at payment', L.live > 0, 'live')}
      ${kpi(num(L.redeemed_customers), 'Got cashback', `${rs(L.sales_recovered)} paid at the shops`, L.redeemed_customers > 0, 'qr')}
      ${kpi(rs(L.cashback_total), 'Cashback paid', `shops ${rs(L.cashback_shop)} · Paytm ${rs(L.cashback_paytm)}`, L.cashback_total > 0, 'rupee')}
    </div>
    <div class="ops-grid">
      <section class="panel">
        <div class="panel-head"><h2>Slipping regulars</h2></div>
        <div class="filters">
          <input id="q" type="search" placeholder="Search name, ID or shop" value="${esc(f.q)}">
          <select id="route"><option value="">All reasons</option>${routeOpt}</select>
          <select id="tier"><option value="">All risk levels</option>
            <option value="high" ${f.tier === 'high' ? 'selected' : ''}>High risk</option>
            <option value="medium" ${f.tier === 'medium' ? 'selected' : ''}>Medium risk</option></select>
          <select id="status"><option value="">All statuses</option>${statusOpt}</select>
        </div>
        <div class="table-wrap"><table class="tbl">
          <thead><tr><th>Customer (B) <span class="sub">masked</span></th><th>Lapse risk at shop</th><th>Why (diagnosis)</th><th>Action · who pays</th><th>Status</th></tr></thead>
          <tbody>${rows || '<tr><td colspan="5" class="empty">No customers match these filters.</td></tr>'}</tbody></table></div>
        <div class="pager"><span>${from}–${to} of ${num(list.total)}</span>
          <span class="btns"><button class="btn" id="prev" ${f.offset ? '' : 'disabled'}>Previous</button>
          <button class="btn" id="next" ${to < list.total ? '' : 'disabled'}>Next</button></span></div>
      </section>
      <div class="side">
        ${routesPanel(o)}
        <section class="panel"><div class="panel-head"><h2>Shops alerted</h2><span class="muted">${ms.length} shops</span></div>
          <div class="mlist">${mrows}</div></section>
        <section class="panel"><div class="panel-head"><h2>Live activity</h2></div>${feed}</section>
        ${compliancePanel(o.compliance)}
      </div>
    </div>
    <section class="panel impact"><div class="impact-grid">
      <div><h3>Shop-level lapse model</h3><div class="minis">
        <div class="mini"><b>${M.auc?.toFixed(3) ?? '–'}</b><small>ROC-AUC, out-of-time</small></div>
        <div class="mini"><b>${M.calibration_gap?.toFixed(3) ?? '–'}</b><small>worst calibration gap</small></div>
        <div class="mini"><b>${pct(M.lapsers_active_on_app)}</b><small>of shop lapsers still active on the app</small></div></div>
        <p class="fine">Top signals: ${M.top_signals.map(esc).join(', ')}</p></div>
      <div><h3>Measured impact <span class="tag">${X.simulated ? 'simulated' : 'measured'}</span></h3><div class="minis">
        <div class="mini"><b>${pts(X.effect)}</b><small>came back to the shop vs holdout (95% CI ${X.ci ? X.ci.map(v => v.toFixed(0)).join(' to ') : '–'})</small></div>
        <div class="mini"><b>${p.mde_pts ?? '–'} pts</b><small>smallest effect this run can detect</small></div>
        ${X.truth ? `<div class="mini"><b>${pts(X.truth.returned_30_pts)}</b><small>simulator's real effect</small></div>` : ''}</div>
        <p class="fine">${X.significant ? 'Significant at 95%.' : `Too few customers yet: ~${p.runs_like_this_needed ?? '–'} runs needed to detect ${p.target_pts ?? 5} pts.`}</p></div>
    </div></section>`;

  let t;
  $('#q', root).oninput = e => { clearTimeout(t); t = setTimeout(() => { f.q = e.target.value; f.offset = 0; f.typing = true; renderOps(root, o); }, 250); };
  $('#tier', root).onchange = e => { f.tier = e.target.value; f.offset = 0; renderOps(root, o); };
  $('#route', root).onchange = e => { f.route = e.target.value; f.offset = 0; renderOps(root, o); };
  $('#status', root).onchange = e => { f.status = e.target.value; f.offset = 0; renderOps(root, o); };
  $('#prev', root).onclick = () => { f.offset = Math.max(0, f.offset - 25); renderOps(root, o); };
  $('#next', root).onclick = () => { f.offset += 25; renderOps(root, o); };
  $$('.route-row', root).forEach(r => r.onclick = () => { f.route = r.dataset.route; f.offset = 0; renderOps(root, o); });
  $$('tr[data-cust]', root).forEach(tr => tr.onclick = () => { if (tr.dataset.cust) { setCustomer(tr.dataset.cust); go('customer', tr.dataset.cust); } });
  $$('[data-m]', root).forEach(el => el.onclick = e => { e.stopPropagation(); S.merchantId = el.dataset.m; go('merchant', el.dataset.m); });
  $('#export', root).onclick = async () => {
    S.exportInfo = await api('/api/export-decisions', {});
    toast(`Exported ${S.exportInfo.rows} decisions`, 'good');
    render();
  };
  if (f.typing) {  // keep the caret in the search box while typing
    const q = $('#q', root);
    q.focus();
    q.setSelectionRange(q.value.length, q.value.length);
    f.typing = false;
  }
}

/* ---------------------------------------------------------------- merchant app */
function merchantPhone(full) {
  const d = full.app, m = d.merchant, s = d.segment, R = d.results;
  const lang = S.lang[m.merchant_id] || m.language;
  const t = d.texts[lang] || d.texts.en;
  const result = d.decision === 'approved' ? `<div class="result-card"><div class="rc-title">Your WinBack results so far</div><div class="rc-grid g3">
      <div><b>${esc(R.came_back_band)}</b><small>regulars came back</small></div><div><b>${rs(R.sales)}</b><small>sales from them</small></div>
      <div><b>${rs(R.cashback_netted)}</b><small>cashback netted</small></div></div></div>` : '';
  const ranges = s.typical_bill_range ? `<div class="seg-grid">
        <div><small>Used to visit</small><b>~${Math.max(1, s.usual_visits_per_month)}×/month</b></div>
        <div><small>Typical bill</small><b>${rs(s.typical_bill_range[0])}–${rs(s.typical_bill_range[1])}</b></div>
        <div><small>Last visit</small><b>~${s.typical_days_since_last_visit} days</b></div></div>`
    : '';
  const named = d.named.length
    ? `<div class="sec-title">Chose to be visible to you <span>${d.named.length}</span></div>` + d.named.map(c =>
        `<div class="cust decided"><span class="avatar">${initials(c.customer)}</span><span class="info"><b>${esc(c.customer)}</b>
         <small>~${Math.max(1, Math.round(c.usual_visits_per_month))} visits/month</small><span class="offer-pill">${esc(c.offer)}</span></span></div>`).join('')
    : '';
  const segment = `<div class="seg-card">
      <div class="seg-top"><span class="seg-n">${esc(d.band)}</span>
        <div><b>regular customers are coming less</b></div></div>
      ${ranges}
      <div class="seg-mix">${d.offer_mix[lang].split('; ').map(x => `<span class="offer-pill">${esc(x)}</span>`).join('')}</div>
      <div class="seg-terms">Netted weekly from your settlement${d.cofunded ? ' · <b>Paytm pays half</b> for some' : ''}</div>
      <div class="privacy-note">${ICON.lock}<span>No names or numbers. You pay only when someone comes back.</span></div>
    </div>`;
  const statements = d.statements.length ? `<div class="sec-title">Settlement statement</div><div class="hist">${d.statements.map(x =>
      `<div class="hist-row"><div class="grow"><div class="nm">${esc(x.statement_text)}</div><small>deducted from your next settlement</small></div></div>`).join('')}</div>` : '';
  const bottom = d.decision === 'awaiting'
    ? `<div class="bottom-bar"><button class="btn primary block" id="approve">Approve budget up to ${rs(d.budget_cap)}</button>
        <button class="btn link" id="decline">Not now</button></div>`
    : d.decision === 'declined'
      ? `<div class="bottom-bar done" style="color:var(--ink-2)">${d.autopilot ? 'Autopilot paused.' : 'You declined this offer.'} No customers were contacted.</div>`
      : `<div class="bottom-bar done">✓ ${d.decided_via === 'autopilot' ? 'Autopilot: running within your monthly budget.' : 'Approved.'} Paytm notifies customers.
          ${d.decided_via === 'autopilot' ? '<button class="btn link" id="decline" style="display:block;margin:4px auto 0">Pause autopilot</button>' : ''}</div>`;

  return phone(`
    <div class="biz-head">
      <div class="biz-brand">${PTM(true)}<span class="ptm-sub" style="font-size:12px;font-weight:700">for Business</span>
        <span class="icons">${ICON.search}${ICON.bellLg}</span></div>
      <div class="biz-top"><span class="biz-logo">${initials(m.name)}</span>
        <div><div class="biz-name">${esc(m.name)}</div><div class="biz-sub">${esc(m.locality)} · ${cap1(m.category)}</div></div></div>
      <div class="biz-stats"><div><small>Collected, last 30 days</small><b>${rs(m.sales_30d)}</b></div><div><small>Paying customers</small><b>${num(m.customers_30d)}</b></div></div>
    </div>
    <div class="scroll">
      ${result}
      <div class="soundbox"><span class="sbx"></span><div class="t grow"><b>Soundbox</b><small>● Online</small></div>
        <button class="sound" data-say="${esc(t.voice)}">${ICON.speaker} Play alert</button></div>
      <div class="alert-card">
        <div class="alert-top"><span class="ic">${ICON.bell}</span><b>WinBack alert</b>
          ${d.autopilot ? '<span class="mini-pill">autopilot</span>' : ''}${d.cofunded ? '<span class="mini-pill">co-funded</span>' : ''}
          <span class="lang"><button data-lang="hi" class="${lang === 'hi' ? 'on' : ''}">Hinglish</button><button data-lang="en" class="${lang === 'en' ? 'on' : ''}">English</button></span></div>
        <p>${esc(t.short || t.message)}</p>
        <div class="est g3"><div><small>Budget, at most</small><b>${rs(d.budget_cap)}</b></div>
          <div><small>Expected cost</small><b>${rs(d.expected_cost)}</b></div>
          <div><small>Likely extra sales</small><b>${rs(d.extra_sales_range[0])}–${num(d.extra_sales_range[1])}</b><small>per month</small></div></div>
      </div>
      ${segment}
      ${named}
      ${statements}
    </div>${bottom}`, 'biz');
}

function wireMerchant(root, full, rerender) {
  const mid = full.app.merchant.merchant_id;
  const pick = $('#mpick', root);
  if (pick) pick.onchange = e => { S.merchantId = e.target.value; S.view === 'merchant' ? go('merchant', S.merchantId) : rerender(); };
  $$('[data-lang]', root).forEach(b => b.onclick = e => { e.preventDefault(); S.lang[mid] = b.dataset.lang; rerender(); });
  $$('.sound', root).forEach(b => b.onclick = e => { e.preventDefault(); speak(b.dataset.say); });
  const decide = async (decision, btn) => {
    btn.disabled = true;
    try {
      await api(`/api/merchants/${mid}/decide`, { decision });
      if (decision === 'approve') {
        // presenter convenience: open one of the customers Paytm just notified
        const cs = await api('/api/customers?merchant_id=' + mid);
        const f = cs.find(c => LIVE.includes(c.status) && c.funder !== 'paytm') || cs.find(c => LIVE.includes(c.status));
        if (f) setCustomer(f.customer_id);
        toast(`Approved. Paytm picked and notified ${full.app.band} regulars.`, 'good');
      } else {
        toast('No customers were contacted.');
      }
      rerender();
    } catch (e) { toast(e.message, 'bad'); btn.disabled = false; }
  };
  const ap = $('#approve', root), dc = $('#decline', root);
  if (ap) ap.onclick = () => decide('approve', ap);
  if (dc) dc.onclick = () => decide('decline', dc);
}

function merchantNotes(full) {
  const p = full.presenter, d = full.app;
  return `<div class="note-card"><div class="note-step">Step 2 · Shop</div>
    <h3>Approves a budget, never sees who</h3>
    <ul>
      <li>Count band "${esc(d.band)}", not the exact ${p.exact_customers}</li>
      <li>Costs rounded · sales shown as a range</li>
      <li>Cashback netted weekly, one line</li>
      <li>${d.autopilot ? '<b>Autopilot</b>: runs within monthly budget' : 'Autopilot offered after a good month'}</li>
      <li>${p.holdout_n} held out to measure impact</li>
    </ul>
    <div class="note-next">Next: <a data-go="customer">Paytm app</a> · <a data-go="story">Side by side</a></div></div>`;
}

async function renderMerchantView(root) {
  const list = await ensureMerchant();
  const d = await api('/api/merchants/' + S.merchantId);
  root.innerHTML = `<div class="stage">
    <div class="stage-col">${merchantPicker(list)}${merchantPhone(d)}</div>
    <aside class="notes">${merchantNotes(d)}</aside></div>`;
  wireMerchant(root, d, () => renderMerchantView(root));
  wireNoteLinks(root);
}

/* ---------------------------------------------------------------- customer app */
function headline(o) {
  if (o.offer_type === 'streak') return `${rs(o.cashback)} cashback`;
  if (o.offer_type === 'pct') return `${o.pct}% cashback`;
  return `${rs(o.cashback)} cashback`;
}
function subline(o) {
  if (o.offer_type === 'streak') return `pay on ${o.streak_visits} different days`;
  if (o.offer_type === 'pct') return `up to ${rs(o.cap)} on each of your next ${o.uses} payments`;
  return 'on your next payment';
}

/* What will happen if they pay this amount now (mirrors the ledger's rules; the server is the source of truth). */
function preview(o, amt) {
  const pr = o.progress;
  if (!(amt > 0)) return { kind: 'hint', text: 'Enter the bill amount' };
  if (o.status === 'expired') return { kind: 'hint', text: `This offer ended on ${fmtDate(o.valid_till)}` };
  if (pr.finished) return { kind: 'hint', text: 'Offer already used' };
  if (amt < o.min_bill) return { kind: 'hint', text: `Add ${rs(o.min_bill - amt)} more to qualify (minimum bill ${rs(o.min_bill)})` };
  if (o.counted_today) return { kind: 'hint', text: 'One qualifying payment per day: today is already counted' };
  if (o.offer_type === 'streak') {
    const n = pr.done + 1;
    return n >= pr.need ? { kind: 'cash', cash: o.cashback, text: `Visit ${n} of ${pr.need}: ${rs(o.cashback)} cashback` }
      : { kind: 'progress', text: `Counts as visit ${n} of ${pr.need}` };
  }
  if (o.offer_type === 'pct') {
    const c = Math.round(Math.min(o.pct / 100 * amt, o.cap) * 100) / 100;
    return { kind: 'cash', cash: c, text: `${o.pct}% cashback` };
  }
  return { kind: 'cash', cash: o.cashback, text: 'WinBack cashback' };
}

function breakdown(o, amt) {
  const p = preview(o, amt);
  return `<div class="bd-row"><span>Bill amount</span><span>${rs(amt || 0)}</span></div>
    <div class="bd-row total"><span>You pay (full bill)</span><span>${rs(amt || 0)}</span></div>
    ${p.kind === 'cash' ? `<div class="bd-row off"><span>${esc(p.text)}, credited after paying</span><span>+${rs(p.cash)}</span></div>`
      : p.kind === 'progress' ? `<div class="bd-row off"><span>${esc(p.text)}</span><span>✓</span></div>`
      : `<div class="bd-hint">${esc(p.text)}</div>`}`;
}

function dots(pr) {
  return `<span class="dots">${Array.from({ length: pr.need }, (_, i) => `<i class="${i < pr.done ? 'on' : ''}"></i>`).join('')}</span>`;
}

function customerPhone(d) {
  const c = d.customer, o = d.offers[0];
  const live = o && PAYABLE.includes(o.status);
  const pr = o ? o.progress : { done: 0, need: 1, finished: false };

  if (S.cust.screen === 'pay' && live) {
    if (S.cust.amount == null) S.cust.amount = Math.round(Math.max(o.avg_bill, o.min_bill));
    const amt = S.cust.amount;
    return phone(`
      <div class="pay-head"><button class="back" id="back" aria-label="Back">‹</button>
        <div class="grow"><b>${esc(o.merchant_name)}<span class="verified">✓</span></b><small>Paytm verified merchant · ${esc(o.merchant_locality)}</small></div>
        <span style="font-size:17px">${PTM(false)}</span></div>
      <div class="scroll pay-body">
        <div class="m-avatar">${initials(o.merchant_name)}</div>
        <div class="m-name">Paying ${esc(o.merchant_name)}</div>
        <div class="scan-offer">${ICON.gift}<span>WinBack: ${esc(headline(o))} · ${esc(subline(o))}</span></div>
        <div class="amount"><span>₹</span><input id="amt" type="number" inputmode="decimal" min="1" step="1" value="${amt}" aria-label="Bill amount"></div>
        <div class="amt-chips">${[100, Math.round(o.min_bill), 300, 500].filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b)
          .map(v => `<button data-amt="${v}">${rs(v)}</button>`).join('')}</div>
        <div class="breakdown" id="bd">${breakdown(o, amt)}</div>
      </div>
      <div class="bottom-bar"><div class="pay-from">${ICON.shield} Paying via UPI from bank account ••42</div>
        <button class="btn primary block" id="doPay">Pay ${rs(amt)}</button></div>`, 'pay');
  }

  if (S.cust.screen === 'done' && S.cust.last) {
    const p = S.cust.last, r = p.result;
    const ref = '6' + String(p.txn_id).replace(/\D/g, '').padStart(11, '2');
    const reward = r.reason === 'cashback_credited'
      ? `<div class="scratch ${S.cust.scratched ? 'revealed' : ''}" id="scratch">
           <div class="prize"><small>You've won</small><b>${rs(r.cashback)}</b><small>cashback · credited to your bank account</small></div>
           <div class="cover">${ICON.gift}<span>You've won a cashback!</span><small>Tap to scratch</small></div></div>`
      : r.reason === 'streak_progress'
        ? `<div class="saved">${dots(r.progress)} Visit ${r.progress.done} of ${r.progress.need} counted<br>
             <span style="font-weight:600;font-size:11.5px">${r.progress.need - r.progress.done} more visit${r.progress.need - r.progress.done === 1 ? '' : 's'} on another day unlock${r.progress.need - r.progress.done === 1 ? 's' : ''} ${rs(o.cashback)} cashback</span></div>`
        : `<div class="nosave">No cashback this time: ${esc(String(r.reason).replace(/_/g, ' '))}${r.short_by ? ` (${rs(r.short_by)} short of the minimum bill)` : ''}</div>`;
    return phone(`<div class="paid-screen">
      <div class="tick">${ICON.check}</div>
      <div class="paid-to">Paid Successfully</div>
      <div class="receipt"><div class="done-amt">${rs(p.amount)}</div>
        <div>to <b>${esc(p.merchant_name)}</b><span class="verified">✓</span></div>
        <div class="done-meta">${esc(p.ts_label)} · UPI Ref. No. ${ref}</div></div>
      ${reward}
      <div class="sb-line">${ICON.speaker}<span>Soundbox: “${esc(p.soundbox)}”</span></div>
      <button class="btn primary block" id="doneBtn">Done</button></div>`, 'pay paid');
  }

  const hist = [
    ...d.payments.map(p => ({ id: p.txn_id, kind: p.kind, name: p.merchant_name, date: p.ts_label, amount: p.amount,
      cash: p.cashback, reversed: p.reversed, refunded: p.refunded, reason: p.reason })),
    ...d.recent.map(r => ({ name: r.merchant_name, date: r.date, amount: r.amount, failed: r.status !== 'SUCCESS' })),
  ].slice(0, 6);
  const histHtml = hist.length ? `<div class="hist">${hist.map(h => `<div class="hist-row"><span class="avatar">${initials(h.name)}</span>
      <div class="grow"><div class="nm ellip">${h.kind === 'refund' ? 'Refund from ' : ''}${esc(h.name)}</div><small>${esc(h.date)}</small></div>
      <div class="amt">${h.kind === 'refund' ? '+' : ''}${rs(h.amount)}
        ${h.cash ? `<span class="saved-tag">+${rs(h.cash)} cashback</span>` : ''}
        ${h.reason === 'streak_progress' ? '<span class="saved-tag">streak visit</span>' : ''}
        ${h.reversed ? `<span class="failed-tag">−${rs(h.reversed)} cashback reversed</span>` : ''}
        ${h.failed ? '<span class="failed-tag">failed</span>' : ''}
        ${h.kind === 'pay' && !h.refunded ? `<button class="linkbtn dark" data-refund="${h.id}">refund</button>` : ''}</div></div>`).join('')}</div>`
    : '<div class="empty-offer">No payments yet</div>';

  const offer = live
    ? `<div class="offer-card ${pr.finished ? 'used' : ''}">
        <div class="oc-band"><div class="oc-top"><span class="oc-badge">WINBACK · ${o.funder === 'paytm' ? 'FROM PAYTM' : 'CASHBACK'}</span><span class="oc-valid">Valid till ${fmtDate(o.valid_till)}</span></div>
          <div class="oc-pct">${esc(headline(o))}</div><div class="oc-m">${esc(subline(o))}</div></div>
        <div class="oc-body">
          <div class="oc-shop"><span class="avatar">${initials(o.merchant_name)}</span><div class="grow"><b>${esc(o.merchant_name)}</b>
            <div class="sub">${esc(o.merchant_locality)} · auto-applied</div></div></div>
          ${o.offer_type === 'streak' ? `<div class="oc-terms">${dots(pr)} <b>${pr.done} of ${pr.need}</b> visits done</div>` : ''}
          <div class="oc-terms">Min bill ${rs(o.min_bill)}${o.offer_type === 'pct' ? ` · up to ${rs(o.cap)} each` : ''} · till ${fmtDate(o.valid_till)}</div>
          <button class="btn primary block" id="paynow" ${pr.finished ? 'disabled' : ''}>${pr.finished ? 'Cashback received' : `Pay at ${esc(o.merchant_name)}`}</button>
          <div class="oc-links"><button class="linkbtn" id="why">Why am I seeing this?</button><button class="linkbtn" id="optout">Stop these offers</button></div>
          ${S.cust.why ? `<div class="why-text">${esc(o.why)}</div>` : ''}</div></div>`
    : `<div class="empty-offer">${d.opted_out ? 'You turned off offers based on your payments. Turn them back on any time in Settings > Offers.' : o && o.status === 'redeemed' ? 'You got your WinBack cashback. Thanks for coming back!' : 'No offers right now'}</div>`;

  const push = live && !pr.finished && S.cust.banner
    ? `<div class="push" id="push"><div class="push-app"><i>p</i>Paytm · now</div><b>${esc(o.title)}</b><p>${esc(o.terms)}</p></div>` : '';

  return phone(`${push}
    <div class="pt-head"><span class="me">${initials(c.first_name)}</span>${PTM(false)}<span class="upi-pill">UPI</span>
      <span class="icons"><span>${ICON.search}</span><span>${ICON.bellLg}<i class="badge"></i></span></span></div>
    <div class="scroll">
      <div class="hello-strip"><span>${c.language === 'hi' ? 'Namaste' : 'Hi'} <b>${esc(c.first_name)}</b></span><span>${esc(c.locality)} · ${c.language === 'hi' ? 'Hinglish' : 'English'}</span></div>
      <div class="upi-card"><div class="ttl">UPI Money Transfer</div><div class="quick">
        <button class="q scan" id="scan"><span class="qi">${ICON.qr}</span>Scan &amp; Pay</button>
        <button class="q" data-deco><span class="qi">${ICON.phone}</span>To Mobile or Contact</button>
        <button class="q" data-deco><span class="qi">${ICON.bank}</span>To Bank A/c</button>
        <button class="q" data-deco><span class="qi">${ICON.wallet}</span>Balance &amp; History</button>
      </div></div>
      <div class="sec-title">Cashback &amp; Offers <a>View all</a></div>${offer}
      <div class="bills">
        <button class="q" data-deco><span class="qi">${ICON.mobile}</span>Mobile Recharge</button>
        <button class="q" data-deco><span class="qi">${ICON.bulb}</span>Electricity Bill</button>
        <button class="q" data-deco><span class="qi">${ICON.tv}</span>DTH Recharge</button>
        <button class="q" data-deco><span class="qi">${ICON.card}</span>Credit Card</button></div>
      <div class="sec-title">Recent payments</div>${histHtml}
    </div>
    <div class="pt-nav"><button class="on">${ICON.home}Home</button><button data-deco>${ICON.gift}Offers</button>
      <button class="scanbtn" id="scan2"><span>${ICON.qr}</span>Scan</button><button data-deco>${ICON.history}History</button>
      <button data-deco>${ICON.user}Profile</button></div>`, 'pay');
}

function wireCustomer(root, d, rerender) {
  const o = d.offers[0];
  const live = o && PAYABLE.includes(o.status);
  const pick = $('#cpick', root);
  if (pick) pick.onchange = e => { setCustomer(e.target.value); S.view === 'customer' ? go('customer', S.customerId) : rerender(); };
  const toPay = () => { S.cust.screen = 'pay'; S.cust.banner = false; S.cust.amount = null; rerender(); };
  const push = $('#push', root); if (push) push.onclick = toPay;
  const pn = $('#paynow', root); if (pn) pn.onclick = toPay;
  // the offer also shows when they scan that shop's QR, so it works even if the push never arrived
  $$('#scan, #scan2', root).forEach(b => b.onclick = () => live ? toPay() : toast('In this demo, Scan & Pay opens the shop with a live WinBack offer.'));
  const scratch = $('#scratch', root);
  if (scratch) scratch.onclick = () => { scratch.classList.add('revealed'); S.cust.scratched = true; };
  $$('[data-deco]', root).forEach(b => b.onclick = () => toast('Not part of this demo'));
  const back = $('#back', root); if (back) back.onclick = () => { S.cust.screen = 'home'; rerender(); };
  const amt = $('#amt', root);
  if (amt) {
    const upd = () => {
      S.cust.amount = parseFloat(amt.value) || 0;
      $('#bd', root).innerHTML = breakdown(o, S.cust.amount);
      $('#doPay', root).textContent = `Pay ${rs(S.cust.amount)}`;
    };
    amt.oninput = upd;
    $$('[data-amt]', root).forEach(b => b.onclick = () => { amt.value = b.dataset.amt; upd(); });
  }
  const why = $('#why', root);
  if (why) why.onclick = () => { S.cust.why = !S.cust.why; rerender(); };
  const opt = $('#optout', root);
  if (opt) opt.onclick = async () => {
    if (!confirm('Stop WinBack offers? Paytm will stop using your payments to send you these offers.')) return;
    try {
      await api(`/api/customers/${d.customer.customer_id}/opt-out`, {});
      S.cust.banner = false;
      toast('Done. You won\'t get these offers any more.');
      rerender();
    } catch (e) { toast(e.message, 'bad'); }
  };
  $$('[data-refund]', root).forEach(b => b.onclick = async () => {
    try {
      const r = await api('/api/refund', { txn_id: b.dataset.refund });
      toast(r.result.reversed ? `Refunded. ${rs(r.result.reversed)} cashback clawed back.` : 'Refunded.', r.result.reversed ? 'bad' : '');
      rerender();
    } catch (e) { toast(e.message, 'bad'); }
  });
  const pay = $('#doPay', root);
  if (pay) pay.onclick = async () => {
    if (!(S.cust.amount > 0)) { toast('Enter the bill amount', 'bad'); return; }
    pay.disabled = true;
    try {
      S.cust.last = await api('/api/pay', { offer_id: o.offer_id, amount: S.cust.amount });
      S.cust.screen = 'done';
      S.cust.scratched = false;
      rerender();
    } catch (e) { toast(e.message, 'bad'); pay.disabled = false; }
  };
  const done = $('#doneBtn', root); if (done) done.onclick = () => { S.cust.screen = 'home'; S.cust.amount = null; rerender(); };
  const nd = $('#nextday', root);
  if (nd) nd.onclick = async () => { const r = await api('/api/clock/next', {}); toast(`Demo clock: ${fmtDate(r.date)}`); rerender(); };
}

function customerNotes(d) {
  const o = d.offers[0], r = d.risk;
  const why = r ? `<div class="why-box"><b>${esc(r.route_label)}</b> → ${esc(r.action_label)} · ${pct(r.lapse_prob)} lapse risk<br>${esc(r.evidence)}</div>` : '';
  const clock = `<div class="clock">Demo day ${d.clock.day + 1} · ${fmtDate(d.clock.date)} <button class="btn" id="nextday">Next day →</button></div>`;
  let body;
  if (o.status === 'opted_out') {
    body = `<h3>Opted out</h3><p>Cashback stopped instantly; excluded from future runs.</p>`;
  } else if (o.status === 'holdout' || o.status === 'global_holdout') {
    body = `<h3>${o.status === 'holdout' ? 'Holdout' : 'Global holdout'}: no offer</h3><p>Compared with offered customers to measure real impact.</p>`;
  } else if (o.status === 'awaiting') {
    body = `<h3>Waiting for ${esc(o.merchant_name)}</h3>
      <div class="note-next"><a data-m="${o.merchant_id}">Open the shop app →</a></div>`;
  } else if (o.status === 'rejected') {
    body = `<h3>${esc(o.merchant_name)} declined</h3><p>No offer sent.</p>`;
  } else {
    const funder = o.funder === 'paytm' ? 'Paytm pays: switched apps' : o.funder === 'cofunded' ? 'Shop + Paytm split: unclear drop' : 'Shop pays: moved to another shop';
    body = `<h3>Cashback, not a discount</h3>
      <ul><li>${funder}</li>
      <li>Full bill at the counter, cashback after</li>
      ${o.offer_type === 'streak' ? '<li>Streak: pay, <b>Next day</b>, repeat</li>' : ''}
      <li>Try: low bill · same-day repeat · refund</li></ul>`;
  }
  return `<div class="note-card"><div class="note-step">Step 3 · Customer (B)</div>${body}${clock}${why ? `<div style="margin-top:12px">${why}</div>` : ''}</div>`;
}

function customerPickerHtml(list) {
  const groups = [['Offer live', LIVE.concat(['expired'])], ['Awaiting shop', ['awaiting']], ['Declined by shop', ['rejected']],
    ['Holdout (no offer)', ['holdout', 'global_holdout']], ['Opted out', ['opted_out']]];
  return `<label class="picker">Customer <select id="cpick">${groups.map(([label, sts]) => {
    const items = list.filter(c => sts.includes(c.status));
    return items.length ? `<optgroup label="${label} (${items.length})">${items.map(c =>
      `<option value="${c.customer_id}" ${c.customer_id === S.customerId ? 'selected' : ''}>${esc(c.masked)} · ${esc(c.merchant_name)} · ${FUNDER[c.funder] ? FUNDER[c.funder][0] : ''}</option>`).join('')}</optgroup>` : '';
  }).join('')}</select></label>`;
}

async function renderCustomerView(root) {
  const list = await api('/api/customers');
  if (!S.customerId || !list.some(c => c.customer_id === S.customerId)) {
    const first = list.find(c => LIVE.includes(c.status) && c.funder !== 'paytm') || list.find(c => LIVE.includes(c.status))
      || list.find(c => c.status === 'awaiting') || list[0];
    setCustomer(first.customer_id);
  }
  const d = await api('/api/customers/' + S.customerId);
  root.innerHTML = `<div class="stage">
    <div class="stage-col">${customerPickerHtml(list)}${customerPhone(d)}</div>
    <aside class="notes">${customerNotes(d)}</aside></div>`;
  wireCustomer(root, d, () => renderCustomerView(root));
  wireNoteLinks(root);
}

/* ---------------------------------------------------------------- side by side */
async function renderStory(root) {
  const list = await ensureMerchant();
  const [d, custs] = await Promise.all([api('/api/merchants/' + S.merchantId), api('/api/customers?merchant_id=' + S.merchantId)]);
  const mine = custs.filter(c => !['holdout', 'global_holdout'].includes(c.status) && c.funder !== 'paytm');
  if (!mine.some(c => c.customer_id === S.customerId)) {
    const live = mine.find(c => LIVE.includes(c.status));
    setCustomer((live || mine[0] || custs[0]).customer_id);
  }
  const cd = await api('/api/customers/' + S.customerId);
  const cpick = `<label class="picker">Customer <select id="cpick">${mine.map(c =>
    `<option value="${c.customer_id}" ${c.customer_id === S.customerId ? 'selected' : ''}>${esc(c.masked)} · ${(STATUS[c.status] || [c.status])[0]}</option>`).join('')}</select></label>`;
  root.innerHTML = `<div class="story">
    <div class="stage-col"><div class="phone-label">Shop (A)</div>${merchantPicker(list)}${merchantPhone(d)}</div>
    <div class="story-mid">${PTM(false)}<div class="flow"></div>picks &amp; notifies</div>
    <div class="stage-col"><div class="phone-label">Customer (B) · Paytm's view</div>${cpick}${customerPhone(cd)}</div></div>`;
  const rerender = () => renderStory(root);
  wireMerchant(root, d, rerender);
  wireCustomer(root, cd, rerender);
}

function wireNoteLinks(root) {
  $$('.notes [data-go]', root).forEach(a => a.onclick = () => go(a.dataset.go, a.dataset.go === 'customer' ? S.customerId : null));
  $$('.notes [data-m]', root).forEach(a => a.onclick = () => { S.merchantId = a.dataset.m; go('merchant', a.dataset.m); });
}

/* ---------------------------------------------------------------- boot */
$$('.tab').forEach(t => t.onclick = () => {
  const v = t.dataset.view;
  go(v, v === 'merchant' ? S.merchantId : v === 'customer' ? S.customerId : null);
});
$('#reset').onclick = async () => {
  if (!confirm('Reset the demo? This clears all approvals, payments and the demo clock.')) return;
  await api('/api/reset', {});
  S.cust = { screen: 'home', amount: null, last: null, banner: true };
  S.exportInfo = null;
  toast('Demo reset');
  render();
};
if ('speechSynthesis' in window) speechSynthesis.getVoices();
parseHash();
render();
