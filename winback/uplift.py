"""Uplift model: who is PERSUADABLE, not just who is at risk.

High lapse risk is not the same as "an offer will change what they do": some at-risk customers are
lost causes, and some would come back anyway. Once campaigns have run with a randomised holdout,
`09_campaign_outcomes.csv` holds features + treatment + outcome for every customer, which is exactly
what a T-learner needs:

    uplift(x) = P(back at the shop | offer, x) - P(back at the shop | no offer, x)

The offer engine then uses this instead of P(lapse) x save rate. Outcomes are personal data, so only
runs still inside the retention window can be used (older runs have been purged).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_COLUMNS

OUTCOME = "returned_post_offer"   # still coming to the shop after the offer expired
MIN_PER_ARM = 300


def _qini(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Area between the Qini curve and random targeting, normalised by the number of customers."""
    order = np.argsort(-score)
    y, t = y[order], t[order]
    ct, cc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    qini = yt - yc * np.divide(ct, cc, out=np.zeros_like(ct, dtype=float), where=cc > 0)
    total = qini[-1]
    random_line = total * np.arange(1, len(y) + 1) / len(y)
    return float(np.mean(qini - random_line) / len(y))


def load_outcomes(cfg: dict) -> pd.DataFrame:
    frames = []
    allow_sim = cfg["economics"].get("allow_simulated_learning", False)
    for run in sorted(Path(cfg["paths"]["output_dir"]).glob("campaign_*")):
        f = run / "09_campaign_outcomes.csv"
        res = run / "09_campaign_results.json"
        if not f.exists():
            continue
        simulated = json.loads(res.read_text(encoding="utf-8")).get("simulated_outcomes", True) if res.exists() else True
        if simulated and not allow_sim:
            continue
        frames.append(pd.read_csv(f).assign(source_run=run.name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_uplift(cfg: dict) -> dict:
    df = load_outcomes(cfg)
    if df.empty:
        return {"trained": False, "reason": "no measured (non-simulated) campaign outcomes yet"}
    df = df[df["group"].isin(["treatment", "control"])]
    n_t, n_c = int((df["group"] == "treatment").sum()), int((df["group"] == "control").sum())
    if min(n_t, n_c) < MIN_PER_ARM:
        return {"trained": False, "reason": f"need {MIN_PER_ARM}+ customers per arm, have {n_t} treated / {n_c} control"}
    feats = [c for c in FEATURE_COLUMNS if c in df.columns]
    X, y, t = df[feats], df[OUTCOME].astype(int).to_numpy(), (df["group"] == "treatment").astype(int).to_numpy()
    half = np.array([int(hashlib.md5(str(c).encode()).hexdigest()[:4], 16) % 2 for c in df["customer_id"]])

    def fit(mask):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, min_samples_leaf=40,
                                           random_state=cfg["seed"])
        return m.fit(X[mask], y[mask])

    # honest check on the held-out half
    m1, m0 = fit((half == 0) & (t == 1)), fit((half == 0) & (t == 0))
    test = half == 1
    score = m1.predict_proba(X[test])[:, 1] - m0.predict_proba(X[test])[:, 1]
    q = _qini(y[test], t[test], score)
    q_risk = _qini(y[test], t[test], X[test]["p_overdue"].to_numpy()) if "p_overdue" in X else None
    # final models on everything
    m1, m0 = fit(t == 1), fit(t == 0)
    card = {"trained": True, "trained_at": datetime.now().isoformat(timespec="seconds"), "outcome": OUTCOME,
            "treated": n_t, "control": n_c, "runs": sorted(df["source_run"].unique().tolist()),
            "qini_heldout": round(q, 5), "qini_risk_ranking": None if q_risk is None else round(q_risk, 5),
            "features": feats}
    model_dir = Path(cfg["paths"]["model_dir"])
    joblib.dump({"m1": m1, "m0": m0, "features": feats}, model_dir / "uplift_model.joblib")
    (model_dir / "uplift_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


def load_uplift(cfg: dict) -> dict | None:
    p = Path(cfg["paths"]["model_dir"]) / "uplift_model.joblib"
    return joblib.load(p) if (p.exists() and cfg["offers"].get("use_uplift_model")) else None


def predict_uplift(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    Xf = X[bundle["features"]]
    return bundle["m1"].predict_proba(Xf)[:, 1] - bundle["m0"].predict_proba(Xf)[:, 1]
