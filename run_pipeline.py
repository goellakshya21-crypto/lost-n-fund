"""WinBack pipeline: find regulars slipping away from a shop -> diagnose why -> route to the right fix and
funder -> cashback offer -> measure against holdouts.

Examples
  python run_pipeline.py all                                   # full demo on synthetic data
  python run_pipeline.py generate                              # synthetic data only
  python run_pipeline.py train                                 # (re)train + register the shop-level lapse model
  python run_pipeline.py train-uplift                          # train the persuadability model (needs measured runs)
  python run_pipeline.py campaign --snapshot 2026-11-19        # score, diagnose, offers, merchant alerts
  python run_pipeline.py decide  --run outputs/campaign_2026-11-19 [--decisions approvals.csv]
  python run_pipeline.py measure --run outputs/campaign_2026-11-19 [--post-txns after.csv --events events.csv]
  python run_pipeline.py demo                                  # clickable demo screens at http://127.0.0.1:8000
  python run_pipeline.py purge [--as-of 2027-03-01]            # delete personal data older than retention_days
  python run_pipeline.py erase --customers ids.csv             # right to erasure (column: customer_id)
"""
from __future__ import annotations

import argparse
import sys

from winback.config import load_config
from winback.pipeline import (stage_all, stage_campaign, stage_decide, stage_erase, stage_generate, stage_measure,
                              stage_purge, stage_train, stage_train_uplift)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["all", "generate", "train", "train-uplift", "campaign", "decide", "measure",
                                      "demo", "purge", "erase"])
    ap.add_argument("--config", help="JSON file overriding winback/config.py defaults")
    ap.add_argument("--snapshot", help="scoring date YYYY-MM-DD (default: day after last transaction)")
    ap.add_argument("--run", help="campaign folder for decide/measure/demo (default: latest)")
    ap.add_argument("--decisions", help="CSV of merchant decisions: merchant_id,decision (or offer_id,decision)")
    ap.add_argument("--post-txns", help="CSV of real payments after the campaign (same schema as transactions.csv, "
                                        "+ optional refund_of for refunds with status REFUNDED)")
    ap.add_argument("--events", help="CSV of customer events after the campaign: customer_id,event,ts "
                                     "(event = opt_out | complaint)")
    ap.add_argument("--force", action="store_true", help="run a campaign even if a guardrail has paused the programme")
    ap.add_argument("--customers", help="CSV with a customer_id column, for `erase`")
    ap.add_argument("--port", type=int, default=8000, help="port for the demo screens (default 8000)")
    ap.add_argument("--as-of", help="date to purge as of, YYYY-MM-DD (default: today)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    def resolve_run():
        if args.run:
            return args.run
        runs = sorted(cfg["paths"]["output_dir"].glob("campaign_*"))
        if not runs:
            sys.exit("No campaign run found. Run the `campaign` stage first.")
        return runs[-1]

    if args.stage == "all":
        stage_all(cfg, args.snapshot)
    elif args.stage == "generate":
        stage_generate(cfg)
    elif args.stage == "train":
        stage_train(cfg)
    elif args.stage == "train-uplift":
        stage_train_uplift(cfg)
    elif args.stage == "campaign":
        stage_campaign(cfg, args.snapshot, force=args.force)
    elif args.stage == "decide":
        stage_decide(cfg, resolve_run(), args.decisions)
    elif args.stage == "measure":
        stage_measure(cfg, resolve_run(), args.post_txns, args.events)
    elif args.stage == "purge":
        stage_purge(cfg, args.as_of)
    elif args.stage == "erase":
        if not args.customers:
            sys.exit("erase needs --customers <csv with a customer_id column>")
        stage_erase(cfg, args.customers)
    elif args.stage == "demo":
        from demo.server import serve
        serve(cfg, resolve_run(), port=args.port)


if __name__ == "__main__":
    main()
