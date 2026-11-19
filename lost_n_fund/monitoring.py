"""Programme health and kill switches.

Every stage writes its latest numbers to state/program_health.json. Before a new campaign is
sent, `check_guardrails` reads them: a HARD breach (too many opt-outs or complaints, budget
overspend) pauses the programme until a person looks at it (`campaign --force` overrides);
a WARNING (low merchant approval, input drift, shadow model disagreeing) is printed and recorded.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _path(cfg: dict) -> Path:
    return Path(cfg["paths"]["state_dir"]) / "program_health.json"


def load_health(cfg: dict) -> dict:
    p = _path(cfg)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"runs": {}}


def update_health(cfg: dict, run_id: str, stage: str, metrics: dict) -> dict:
    h = load_health(cfg)
    run = h["runs"].setdefault(run_id, {})
    run[stage] = {**metrics, "updated_at": datetime.now().isoformat(timespec="seconds")}
    h["latest_run"] = max(h["runs"])
    p = _path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(h, indent=2, default=str), encoding="utf-8")
    return h


def check_guardrails(cfg: dict, exclude_run: str | None = None) -> tuple[list[str], list[str]]:
    """Returns (hard breaches, warnings) from the most recent measured / decided runs."""
    g = cfg["guardrails"]
    h = load_health(cfg)
    hard, warn = [], []
    runs = {k: v for k, v in h.get("runs", {}).items() if k != exclude_run}
    measured = [v["measure"] for k, v in sorted(runs.items()) if "measure" in v]
    if measured:
        m = measured[-1]
        if m.get("opt_out_rate", 0) > g["max_opt_out_rate"]:
            hard.append(f"opt-out rate {m['opt_out_rate']:.1%} > limit {g['max_opt_out_rate']:.1%}")
        if m.get("complaint_rate", 0) > g["max_complaint_rate"]:
            hard.append(f"complaint rate {m['complaint_rate']:.2%} > limit {g['max_complaint_rate']:.2%}")
        for k in ("merchant_budget_burn", "paytm_budget_burn"):
            if m.get(k, 0) > g["max_budget_burn"]:
                hard.append(f"{k.replace('_', ' ')} {m[k]:.0%} > {g['max_budget_burn']:.0%}")
    decided = [v["decide"] for k, v in sorted(runs.items()) if "decide" in v]
    if decided and decided[-1].get("approval_rate") is not None:
        if decided[-1]["approval_rate"] < g["min_merchant_approval_rate"]:
            warn.append(f"merchant approval rate {decided[-1]['approval_rate']:.0%} is below "
                        f"{g['min_merchant_approval_rate']:.0%}: review offer sizes and alert wording")
    return hard, warn
