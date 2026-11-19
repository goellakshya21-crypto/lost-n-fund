# WinBack: bring regulars back to their shops

**The idea.** A payments app (Paytm, PhonePe, Google Pay, BHIM...) sees that customer **B** has
stopped coming to shop **A**, where B used to be a regular. It works out **why**, and acts only when
an offer can actually help:

- **B moved to another shop:** **A** approves a small cashback budget and the app sends B the offer.
- **B now pays with another app:** the app funds the cashback itself, and A is never asked.
- **B's payments are failing, B is travelling, the gap is normal for B, or A itself is failing:** no offer, just the right fix.

B pays the full bill at the counter. Cashback is credited by the app, and A's share is netted from
A's settlement as **one weekly line**. **A never learns who the customers are.**

This repo is the end-to-end pipeline and a clickable demo. It runs on synthetic data out of the box,
and the same code runs on real exports.

```
 payments ─► shop-level ─► why are they ─► route + ─► eligibility ─► cashback offer ─► holdouts ─► shop alert
 (consented   lapse model   fading?         who pays    (anti-abuse,   (type, size,      (run +       (count band,
  adults)    (customer×shop)(diagnosis)                 caps, budget)  pays for itself)   global)      budget)
                                                                                                         │
  learn ◄── measure vs ◄── ledger: cashback, refund clawback, ◄── customer pays ◄── app sends offer ◄────┘
  assumptions  holdouts      weekly settlement netting             full bill        (their language)
```

## Quickstart

```powershell
pip install -r requirements.txt
python run_pipeline.py all
```

This takes about 2 minutes. It generates data, trains and registers the model, runs a campaign,
simulates merchant approvals and 90 days of customer behaviour, and writes
`outputs/campaign_<date>/report.html`.

## Demo screens (for pitching)

```powershell
python run_pipeline.py demo
```

Then open http://127.0.0.1:8000. The screens use the Paytm look (navy `#002E6E`, sky `#00BAF2`, a text
wordmark). Swap in official brand assets only with permission. All four tabs share one live state:

- **Paytm Ops:** a hero summary, slipping regulars with **why** each one is fading and **who pays**,
  the reasons breakdown, the shops alerted, a live activity feed, compliance checks, and model + measured
  impact. It links to the full campaign report.
- **Paytm for Business:** the shop's alert. It shows a **count band** ("5–9 regulars"), a budget, and a
  *range* of likely sales, in Hinglish or English, with a Soundbox voice line. The owner taps
  **Approve budget**, or the shop is on **autopilot**. After the campaign it shows results and the
  weekly settlement statement.
- **Paytm app:** the customer's home screen with the WinBack offer under *Cashback & Offers*, a push
  notification, Scan & Pay, the pay screen (full bill plus a cashback preview), and a Paytm-style
  success screen with a **scratch card** for the cashback. Streak offers show visit dots.
  "Why am I seeing this?" and "Stop these offers" are on every offer.
- **Side by side:** the shop and customer phones together.

**A 3-minute pitch flow:**
1. **Ops:** the hero says *"5,331 regulars are slipping, only 754 need an offer"*. Click a reason
   (e.g. *Travelling*) to show why most of them get no offer.
2. **Paytm for Business:** play the Soundbox line, point at the count band and the price range, then approve.
3. **Paytm app:** open the streak offer and pay ₹120 to show the honest "add ₹X more" hint. Then pay
   enough, press **Next day** in the notes, and pay again. On the third visit, scratch the card. Pay
   twice on one day to show the one-per-day rule, and press **refund** to show the clawback.
4. **Ops:** the KPIs show cashback split between shop and Paytm. Open the report for the measured
   effect and the power calculation.

"Next day" is a demo clock, so multi-day streaks can be shown live. Demo clicks are saved in
`outputs/<run>/demo_state.json` ("Reset demo" clears them). Every payment goes through the same
`OfferLedger` the pipeline uses. "Export approvals for pipeline" writes a CSV that `decide --decisions`
accepts.

## How it works

**1. Shop-level lapse model** ([features.py](winback/features.py), [model.py](winback/model.py))
- *Unit:* a customer × shop pair where the customer is a regular (3+ visits in 90 days, last within 45 days).
- *Target:* the customer's spend at that shop in the next 30 days falls below 30% of their usual.
  **88% of shop lapsers are still active on the app**, so an "app churn" model would call them safe.
- *Features* (57, all point-in-time):
  - the pair's own rhythm, including a **BG/NBD "still a regular?"** probability
  - the customer's behaviour everywhere
  - **spend moving to a same-kind shop or quick-commerce**
  - payments in places they never used before
  - app opens and bill/P2P trends (optional table)
  - the shop's own trend and its share of payments from our app vs all apps, plus Soundbox uptime (acquirer-side, optional)
  - the locality's trend
- *Training:* gradient boosting on 4 past snapshots, validated **out-of-time**, plus **isotonic calibration**.
  Calibration is checked cross-fitted, so it isn't graded on its own homework. The offer economics
  multiply by this probability, so a miscalibrated model would overstate every shop's gain.
- *Registry:* each training run is saved under `models/registry/<version>/`. A new model is promoted only
  if it isn't worse on the new validation snapshot; otherwise it runs as a **shadow**. Campaigns report
  **input drift** (PSI per feature) and the shadow model's agreement.

**2. Why are they fading, and who pays?** ([diagnosis.py](winback/diagnosis.py))

| Cause | Signal | Action | Who pays |
|---|---|---|---|
| Shop declining | the shop's total footfall (all apps) down, Soundbox offline | alert merchant ops | no offer |
| Travelling / area quiet | paying in places never used before; shops in their locality quiet | wait, recheck | no offer |
| Payments failing | ≥10% of recent payments failed | fix routing / prompt bank switch | Paytm (no offer) |
| Normal gap | occasional visitor, or visits barely down | wait | no offer |
| Gone | payments down everywhere, app opens and bills down too | skip | no one |
| Paying with another app | scan-and-pay down everywhere, bills/P2P still on our app | cashback | **Paytm** (shop never asked) |
| Moved to another shop | the spend lost here shows up at a same-kind shop or quick-commerce | win-back offer | **the shop** |
| Unclear (often cash) | only this shop dropped, no replacement seen | win-back offer | **shop + Paytm 50/50** |

On the synthetic data the diagnosis is checked against the simulator's hidden truth and printed in every
run: it gets the cause right for 63% of truly-leaving pairs, and **44% of shop-funded customers really
moved shop or stopped for an unseen reason**. Before the diagnosis, the shop would have paid for every
at-risk customer.

**3. Eligibility and anti-abuse** ([offers.py](winback/offers.py)). These checks run before randomisation, so
the holdout and the shop counts only contain people who could really get an offer:
- account at least 60 days old, and a regular at that shop for at least 30 days
- a 21-day frequency cap, and **at most 2 win-back offers per 180 days**; the second is worth half
  (going quiet on purpose doesn't pay)
- a hash-based **global holdout** (5% of customers get nothing all quarter)

**4. Offer engine** ([offers.py](winback/offers.py)). Offers are **cashback**:

| Type | Example | Used for |
|---|---|---|
| streak | "Pay on 3 different days by 2 Dec (₹180+ each), get ₹35 cashback" | customers who came 3+ times a month: rebuilds the habit, hard to game |
| flat | "₹20 cashback on your next payment of ₹200+" | everyone else |
| % | "5% cashback, up to ₹40, on your next 2 payments" | exploration arm |

- *Sizing:*
  - generosity comes from risk × value, then steps down until the offer **pays for itself for whoever funds it**
  - the minimum bill is about 70% of their usual bill
  - validity is 1.75× their usual gap at that shop (7–21 days)
  - category caps apply: kirana at most 5% of the bill, fuel 2%
- *Expected cost uses the customer's own rhythm:* a weekly regular will finish a 3-visit streak anyway,
  and that cashback is pure cost (Poisson odds of qualifying visits within the validity window, including
  near-miss bill top-ups).
- *Expected gain:* calibrated P(lapse) × save rate × usual spend × a retention curve (100%/70%/50% for
  months 1–3, not a flat 3 months). Paytm's gain for app switchers includes the value of keeping a user
  (`paytm_value_per_user_month`).
- *Who it must pay for:*
  - shop-funded: shop EV > 0
  - co-funded: shop EV > 0, total EV > 0, and Paytm's expected cost ≤ ₹15
  - Paytm-funded: Paytm EV > 0
- *Learning:* 15% of customers get a random type or size (hash-based), so the engine can learn what
  works. Once enough measured campaigns exist, `train-uplift` builds a **T-learner uplift model**
  (who is *persuadable*, not just at risk) and the engine uses it instead of P(lapse) × save rate.

**5. Budgets, anonymity, holdouts**
- Worst-case liability must fit the shop's monthly budget (its own share only) and Paytm's programme
  budget, including what earlier runs this month already committed.
- A shop is alerted only about **5+ customers**. Smaller groups wait in a **carry-over queue** (up to
  21 days) until the shop has enough.
- 20% of customers at each shop are held out this run (hash-based, stratified by shop).

**6. Messages** ([messaging.py](winback/messaging.py))
- *To the shop:*
  - a **count band** ("5–9"), never an exact count
  - bill and visit ranges only for groups of 10+
  - costs rounded up to ₹50, so the count can't be backed out
  - sales as a **25th–75th percentile range**, never a rosy average (for a small group the low end is often ₹0)
  - co-funding stated
  - autopilot shops run within their budget, and the weekly summary suggests autopilot to others
  - WhatsApp only for shops that opted in
- *To the customer:*
  - in their language (Hinglish or English)
  - real dates, an opt-out line, and "why am I seeing this"
  - shown at Scan & Pay too, so it works even if the push never arrives

**7. Money movement** ([ledger.py](winback/ledger.py)). Every payment goes through the ledger, which:
- credits cashback only for successful payments made with our app by someone who still consents,
  inside the validity window and at or above the minimum bill
- counts **one qualifying payment per day** (no bill splitting)
- counts streak days, not payments
- **reverses cashback on refunds**
- checks the shop's budget at redemption, and Paytm honours any overflow
- is **idempotent** on payment IDs
- writes **one netting line per shop per week**

**8. Measurement and learning** ([measure.py](winback/measure.py))
- *Intent-to-treat*, adjusted for pre-period behaviour (**CUPED**), with **standard errors clustered by shop**.
- *Approved shops only:* also unbiased, because customers are randomised within each shop.
- *Did it stick?* "Came back within 30 days" counts the discounted visit itself. What matters is
  **still coming after the offer ended**, so that is measured too, along with month-by-month retention.
- *Breakdowns:* by cause, offer type and risk tier. The global holdout is used as a sanity check.
- *Power:* the smallest effect this sample can detect, and the sample a 5-point effect needs. The pilot
  readout **pools every run** inside the retention window.
- *Learned assumptions* (save rate, retention curve) are written to `state/learned_assumptions.json`
  and **shrunk toward the prior** by sample size, so one small run barely moves them. They are never
  learned from simulated runs (unless `allow_simulated_learning`).

**9. Kill switches** ([monitoring.py](winback/monitoring.py)). If the last measured run breached
**opt-outs >3%**, **complaints >0.5%** or **budget burn >100%**, `campaign` refuses to run until someone
investigates (`--force` overrides). Low shop approval and input drift are printed as warnings.

## Demo results (synthetic, see caveat)

| | |
|---|---|
| Model (out-of-time) | pair-level AUC 0.755 vs logistic 0.730; worst calibration gap 0.003; **88% of shop lapsers still active on the app** |
| Why | 6,321 at-risk pairs: 2,634 travelling, 2,343 normal gaps, 404 moved shop, 298 gone, 202 unclear, 168 payment failures, 148 other app, 124 at a failing shop |
| Offers | 214 offers (183 treatment / 31 holdout / 20 global holdout), 11 shops alerted, 103 Paytm-funded; 139 held in the anonymity queue |
| Measured | came back to the shop: **+9.8 pts** (95% CI −9.2 to +28.9); simulator's real effect **+12.2 pts** |
| Power | this run detects ≥25 pts; a 5-pt effect needs ~800 control / 4,700 treated (~26 weekly runs pooled) |
| Money | cashback ₹424 (shops ₹122, Paytm ₹302), point estimates only at this sample size |

> **Caveat:** how customers respond is an *assumption built into the simulator*. The results show the
> pipeline measuring correctly, including how noisy one small run is. They are **not** evidence that
> the idea works. Only a real pilot, pooled to the size the power calculation asks for, can show that.

## Compliance

All settings are in `compliance` in [winback/config.py](winback/config.py). Legal sign-off from the
platform and its partner PSP banks is still required.

| Rule | How WinBack handles it |
|---|---|
| **NPCI UPI rules** (consent; no third-party sharing; data in India) | Only `offers_consent` customers are trained on, scored or messaged. Shops see bands and ranges, never names, exact counts or scores. Cross-app identity linking is **off** (`cross_app_linking`) |
| **RBI data localisation** | Run in India (`data_residency: "IN"`). The model uses the payment mode only, never card numbers |
| **DPDP Act 2023 + Rules 2025** | Purpose-specific consent, re-checked before sending **and at every payment**. One-tap opt-out. No phone numbers. Minors excluded. `purge` deletes customer-level files after 120 days (the cashback ledger is kept with salted hashes for audit). `erase` handles deletion requests, and a hashed suppression list keeps those customers out |
| **CCPA dark-pattern guidelines 2023** | Real expiry dates only; the "add ₹X more" hint shows the true gap |
| **TRAI marketing SMS** | In-app delivery only; SMS off until DLT templates exist |
| **Sensitive inferences** | Pharmacies never used |
| **Guessing who** | Count bands, a 5-customer minimum, a carry-over queue, rounded costs, and weekly settlement netting (no per-payment marker) |

## Running on real data

Put the CSVs in `data/` (or point `paths.data_dir` at them via `--config`).

| file | | columns |
|---|---|---|
| `transactions.csv` | required (payer side) | `txn_id, customer_id, merchant_id, amount, ts, status (SUCCESS/FAILED), payment_mode` |
| `merchants.csv` | required | `merchant_id, name, owner_name, category, locality, enrolled, auto_approve, monthly_offer_budget, language` (+ `paytm_acquired`, `whatsapp_opt_in`) |
| `customers.csv` | required | `customer_id, name, locality, offers_consent, is_minor` (+ `share_with_merchants`, `language`) |
| `app_activity.csv` | optional | `customer_id, week_start, app_opens, bill_payments, p2p_payments` |
| `merchant_daily.csv` | optional (acquirer side) | `merchant_id, date, txns_all_apps, txns_paytm_app, gmv_all_apps, soundbox_online` |

- `platform.data_vantage`: `"both"` if you also acquire merchants (Paytm, PhonePe); `"payer"` if you only have app-user data (BHIM-like).
- Inputs are validated: duplicates, bad amounts, unknown statuses and unknown shops are dropped and reported, and stale data is flagged.
- At least 210 days of history is needed (90-day lookback + 4 training snapshots + an out-of-time month).
- Merchant decisions come in as `merchant_id,decision`.
- Post-campaign payments use the `transactions.csv` schema. Refunds are rows with status `REFUNDED` and a `refund_of` column.
- Opt-outs and complaints go in `--events` as `customer_id,event,ts`.

```powershell
python run_pipeline.py train
python run_pipeline.py campaign --snapshot 2026-11-19
python run_pipeline.py decide  --decisions merchant_decisions.csv
python run_pipeline.py measure --post-txns after.csv --events events.csv
python run_pipeline.py train-uplift
python run_pipeline.py purge
python run_pipeline.py erase --customers ids.csv
```

## Roadmap to production

1. **Legal sign-off.** The consent wording, the anonymous merchant view, weekly netting, and the GST
   treatment of merchant-funded cashback.
2. **Pilot sized by the power calculation.** Several localities or weekly waves, until the pooled
   holdout reaches ~800. Replace `economics.paytm_value_per_gmv` and `paytm_value_per_user_month` with
   real numbers.
3. **Services.**
   - Move features to the warehouse or feature store with point-in-time joins, and score daily.
   - Turn the ledger into a service with database row locks on the budget counters and idempotency keys.
   - Wire the consent service, push sender, business-app approve/autopilot API, Soundbox TTS, and
     settlement netting into the payout system.
   - Schedule every stage in Airflow on servers in India.
4. **For PhonePe or Google Pay:** check the status of NPCI's 30% UPI market-share cap, because
   "win back app share" may be sensitive for them.

## Layout

```
run_pipeline.py          CLI
winback/config.py        every knob (compliance, diagnosis, routing, offers, abuse, guardrails)
winback/data.py          data contract, validation, erasure suppression
winback/synthetic.py     demo data with hidden causes + 90-day outcome simulator (demo only)
winback/features.py      customer x shop features, BG/NBD, labels, reason codes
winback/model.py         train, calibrate, registry + promotion, drift, shadow scoring
winback/diagnosis.py     why they're fading -> route + who pays (+ truth check on demo data)
winback/matching.py      at-risk selection, carry-over, action queues, offer candidates
winback/offers.py        eligibility + anti-abuse, cashback design + economics, budgets, anonymity, holdouts
winback/uplift.py        T-learner uplift model from measured campaigns
winback/messaging.py     shop alerts (bands, ranges), autopilot, customer messages, weekly summaries
winback/ledger.py        redemption rules, refunds, weekly settlement netting
winback/measure.py       CUPED + clustered effects, power, pooled pilot, learned assumptions
winback/monitoring.py    programme health + kill switches
winback/report.py        HTML report
winback/pipeline.py      stages
demo/server.py           demo API on the latest run (uses the real ledger)
demo/static/             Paytm Ops, Paytm for Business, Paytm app screens (plain HTML/JS/CSS)
```
