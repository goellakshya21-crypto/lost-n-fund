"""Shop-level lapse model: train on past snapshots, validate out-of-time, calibrate, register, score.

* Target: will customer B's spend at shop A fall below 30% of usual in the next 30 days?
* Validation is out-of-time: trained on snapshots whose outcomes ended before the validation
  snapshot, which is exactly how it is used (train on the past, predict the next month).
* Calibration: raw gradient-boosting scores are mapped to real probabilities with isotonic
  regression fitted on the out-of-time snapshot. The offer economics multiply by this
  probability, so an over-confident model would overstate every merchant's expected gain.
  The reported calibration is cross-fitted (fit on one half of validation customers, checked
  on the other), so it is not graded on its own homework.
* Registry: every training run is saved under models/registry/<version>/. A new model only
  replaces the current one if it is not worse on the new validation snapshot; otherwise it
  stays as a "shadow" that campaigns also score with, for comparison.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, build_features, build_labels, reason_codes


def _snapshots(txns: pd.DataFrame, cfg: dict) -> tuple[list[pd.Timestamp], pd.Timestamp]:
    lb = cfg["features"]["lookback_days"]
    cw = cfg["labels"]["churn_window_days"]
    tc = cfg["training"]
    data_start = txns["ts"].min().normalize()
    data_end = txns["ts"].max().normalize() + pd.Timedelta(days=1)
    val_snap = data_end - pd.Timedelta(days=cw)
    train_snaps = []
    for i in range(tc["n_train_snapshots"]):
        s = val_snap - pd.Timedelta(days=cw + i * tc["snapshot_spacing_days"])
        if s - pd.Timedelta(days=lb) >= data_start:
            train_snaps.append(s)
    if not train_snaps:
        raise ValueError(f"Need at least {lb + 2 * cw} days of history to train; have {(data_end - data_start).days}.")
    return sorted(train_snaps), val_snap


def _labelled(data, snap, cfg):
    f, bg = build_features(data.txns, data.customers, snap, cfg, data.activity, data.merchant_daily)
    f["label"] = build_labels(data.txns, snap, f, cfg)
    return f, bg


def _half(ids: pd.Series) -> np.ndarray:
    return np.array([int(hashlib.md5(str(c).encode()).hexdigest()[:4], 16) % 2 for c in ids])


def evaluate(y: np.ndarray, p: np.ndarray, thresholds: dict | None = None) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    order = np.argsort(-p)
    n = len(p)
    k10, k20 = max(1, int(0.1 * n)), max(1, int(0.2 * n))
    base = y.mean()
    out = {
        "n": int(n),
        "churn_rate": round(float(base), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "precision_top10pct": round(float(y[order[:k10]].mean()), 4),
        "lift_top10pct": round(float(y[order[:k10]].mean() / base), 2),
        "recall_top20pct": round(float(y[order[:k20]].sum() / y.sum()), 4),
    }
    dec = pd.DataFrame({"y": y, "p": p})
    dec["decile"] = pd.qcut(dec["p"].rank(method="first"), 10, labels=False)
    cal = dec.groupby("decile").agg(predicted=("p", "mean"), actual=("y", "mean")).round(4)
    out["calibration_by_decile"] = cal.reset_index().to_dict(orient="records")
    out["max_calibration_gap"] = round(float((cal["predicted"] - cal["actual"]).abs().max()), 4)
    if thresholds:
        tiers = {}
        for name, thr in thresholds.items():
            sel = p >= thr
            tiers[name] = {
                "threshold": thr,
                "share_flagged": round(float(sel.mean()), 4),
                "precision": round(float(y[sel].mean()) if sel.any() else 0.0, 4),
                "recall": round(float(y[sel].sum() / y.sum()), 4),
            }
        out["tier_performance"] = tiers
    return out


def _drift_reference(X: pd.DataFrame) -> dict:
    ref = {}
    for c in X.columns:
        v = X[c].dropna().to_numpy()
        if len(v) == 0:
            continue
        edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        counts = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)[0]
        ref[c] = {"edges": edges.tolist(), "props": (counts / counts.sum()).tolist()}
    return ref


def psi(ref: dict, X: pd.DataFrame) -> dict:
    """Population stability index per feature: how far today's inputs moved from training."""
    out = {}
    for c, r in ref.items():
        if c not in X.columns:
            continue
        v = X[c].dropna().to_numpy()
        if len(v) == 0:
            continue
        edges = np.asarray(r["edges"])
        cur = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)[0] / len(v)
        base = np.asarray(r["props"])
        cur, base = np.clip(cur, 1e-4, None), np.clip(base, 1e-4, None)
        out[c] = round(float(np.sum((cur - base) * np.log(cur / base))), 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def train_model(data, cfg: dict) -> dict:
    seed = cfg["seed"]
    train_snaps, val_snap = _snapshots(data.txns, cfg)
    train = pd.concat([_labelled(data, s, cfg)[0] for s in train_snaps], ignore_index=True)
    val, _ = _labelled(data, val_snap, cfg)
    Xtr, ytr = train[FEATURE_COLUMNS], train["label"].to_numpy()
    Xva, yva = val[FEATURE_COLUMNS], val["label"].to_numpy()

    cat_idx = [FEATURE_COLUMNS.index("category_code")]
    gbm = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=60,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=seed, categorical_features=cat_idx)
    gbm.fit(Xtr, ytr)
    raw_va = gbm.predict_proba(Xva)[:, 1]

    # cross-fitted calibration check: calibrate on one half of validation customers, grade on the other
    half = _half(val["customer_id"])
    cal_va = np.empty_like(raw_va)
    for h in (0, 1):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_va[half != h], yva[half != h])
        cal_va[half == h] = iso.predict(raw_va[half == h])
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_va, yva)

    baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000))
    baseline.fit(Xtr, ytr)
    p_lr = baseline.predict_proba(Xva)[:, 1]

    thresholds = cfg["risk_tiers"]
    metrics = evaluate(yva, cal_va, {"high": thresholds["high"], "medium_or_high": thresholds["medium"]})
    metrics_raw = evaluate(yva, raw_va)
    metrics_baseline = evaluate(yva, p_lr)

    sub = np.random.default_rng(seed).choice(len(Xva), min(15000, len(Xva)), replace=False)
    imp = permutation_importance(gbm, Xva.iloc[sub], yva[sub], scoring="roc_auc", n_repeats=3, random_state=seed)
    importance = (pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False).round(5))

    # what a customer-level ("left the app") target would miss: shop lapsers still active on the app
    window = cfg["labels"]["churn_window_days"]
    fut = data.txns[(data.txns["ts"] >= val_snap) & (data.txns["ts"] < val_snap + pd.Timedelta(days=window))
                    & data.txns["status"].eq("SUCCESS")]
    still_active = val["customer_id"].isin(set(fut["customer_id"])).to_numpy()
    lapsers_active = float(still_active[yva == 1].mean()) if (yva == 1).any() else 0.0

    card = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "HistGradientBoostingClassifier + isotonic calibration",
        "target": (f"spend at the shop in the next {window} days below "
                   f"{cfg['labels']['pair_lapse_share']:.0%} of usual (population: regulars, "
                   f"{cfg['matching']['min_visits']}+ visits in {cfg['features']['lookback_days']} days)"),
        "unit": "customer x shop pair",
        "train_snapshots": [str(s.date()) for s in train_snaps],
        "validation_snapshot": str(val_snap.date()),
        "train_rows": int(len(train)),
        "validation": metrics,
        "validation_uncalibrated": {k: v for k, v in metrics_raw.items() if k != "tier_performance"},
        "baseline_logistic_validation": {k: v for k, v in metrics_baseline.items() if k != "calibration_by_decile"},
        "shop_lapsers_still_active_on_app": round(lapsers_active, 4),
        "feature_importance": importance.to_dict(orient="records"),
        "features": FEATURE_COLUMNS,
    }
    return {"model": gbm, "calibrator": calibrator, "features": FEATURE_COLUMNS,
            "drift_ref": _drift_reference(Xtr), "card": card, "val_X": Xva, "val_y": yva}


# ------------------------------------------------------------------ registry
def _registry(model_dir: Path) -> Path:
    return Path(model_dir) / "registry"


def current_version(model_dir: Path) -> str | None:
    p = Path(model_dir) / "current.json"
    return json.loads(p.read_text(encoding="utf-8"))["version"] if p.exists() else None


def load_model(model_dir: Path, version: str | None = None) -> dict:
    version = version or current_version(model_dir)
    if version is None:
        raise FileNotFoundError("No trained model. Run `python run_pipeline.py train` first.")
    path = _registry(model_dir) / version / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model version {version} not found in {path.parent}")
    bundle = joblib.load(path)
    bundle["version"] = version
    return bundle


def register_model(bundle: dict, model_dir: Path, cfg: dict) -> dict:
    """Save a newly trained model and decide whether it replaces the current one."""
    model_dir = Path(model_dir)
    version = datetime.now().strftime("v%Y%m%d-%H%M%S")
    vdir = _registry(model_dir) / version
    vdir.mkdir(parents=True, exist_ok=True)

    cur_v = current_version(model_dir)
    cand_auc = bundle["card"]["validation_uncalibrated"]["roc_auc"]
    decision = {"candidate_auc": cand_auc, "current_version": cur_v}
    if cur_v is None:
        promote, why = True, "no current model"
    else:
        cur = load_model(model_dir, cur_v)
        if list(cur["features"]) != list(bundle["features"]):
            promote, why = True, "feature set changed; old model cannot score the new inputs"
        else:
            cur_auc = float(roc_auc_score(bundle["val_y"], cur["model"].predict_proba(bundle["val_X"])[:, 1]))
            decision["current_auc_on_new_validation"] = round(cur_auc, 4)
            promote = cand_auc >= cur_auc - cfg["training"]["promote_max_auc_drop"]
            why = ("not worse than the current model" if promote else
                   f"AUC {cand_auc:.3f} vs current {cur_auc:.3f}: kept as a shadow model")
    decision.update({"promoted": bool(promote), "reason": why})
    bundle["card"]["version"] = version
    bundle["card"]["promotion"] = decision

    keep = {k: bundle[k] for k in ("model", "calibrator", "features", "drift_ref")}
    joblib.dump(keep, vdir / "model.joblib")
    (vdir / "model_card.json").write_text(json.dumps(bundle["card"], indent=2), encoding="utf-8")
    if promote:
        (model_dir / "current.json").write_text(json.dumps({"version": version, "promoted_at": datetime.now()
                                                            .isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
        shutil.copyfile(vdir / "model_card.json", model_dir / "model_card.json")
    for old in ("churn_model.joblib",):   # pre-registry file
        if (model_dir / old).exists():
            (model_dir / old).unlink()
    return decision


def shadow_version(model_dir: Path) -> str | None:
    """The newest registered model that was NOT promoted and is newer than the current one."""
    cur = current_version(model_dir)
    reg = _registry(model_dir)
    if not reg.exists():
        return None
    for v in sorted((p.name for p in reg.iterdir() if p.is_dir()), reverse=True):
        if cur is not None and v <= cur:
            return None
        card = json.loads((reg / v / "model_card.json").read_text(encoding="utf-8"))
        if not card.get("promotion", {}).get("promoted", False):
            return v
    return None


# ------------------------------------------------------------------ scoring
def score_pairs(data, bundle: dict, snapshot, cfg: dict) -> pd.DataFrame:
    f, _ = build_features(data.txns, data.customers, snapshot, cfg, data.activity, data.merchant_daily)
    raw = bundle["model"].predict_proba(f[bundle["features"]])[:, 1]
    f["lapse_prob_raw"] = raw.round(4)
    # isotonic steps can hit exactly 0 or 1 at the extremes; no customer is ever certain either way
    f["lapse_prob"] = np.clip(bundle["calibrator"].predict(raw), 0.01, 0.98).round(4)
    hi, med = cfg["risk_tiers"]["high"], cfg["risk_tiers"]["medium"]
    f["risk_tier"] = np.select([f["lapse_prob"] >= hi, f["lapse_prob"] >= med], ["high", "medium"], "low")
    f["reasons"] = reason_codes(f, data.merchants.set_index("merchant_id")["name"])
    return f.sort_values("lapse_prob", ascending=False).reset_index(drop=True)


def shadow_compare(primary: pd.DataFrame, shadow_probs: np.ndarray) -> dict:
    p = primary["lapse_prob_raw"].to_numpy()
    k = max(1, int(0.1 * len(p)))
    top_a = set(np.argsort(-p)[:k])
    top_b = set(np.argsort(-shadow_probs)[:k])
    rank_corr = pd.Series(p).rank().corr(pd.Series(shadow_probs).rank())
    return {"top10pct_overlap": round(len(top_a & top_b) / k, 4), "rank_correlation": round(float(rank_corr), 4)}
