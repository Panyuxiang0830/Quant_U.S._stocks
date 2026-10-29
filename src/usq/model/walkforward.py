from __future__ import annotations

import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from usq.data import PanelStore
from usq.utils import preprocess, sha256, json_dump

XGB_PARAMS = dict(objective="reg:squarederror", max_depth=7, learning_rate=.03,
                  n_estimators=500, subsample=1., colsample_bytree=1., gamma=.1,
                  reg_lambda=1., eval_metric="rmse", tree_method="hist", device="cuda",
                  max_bin=256, missing=None, seed=2964, nthread=48)


def half_years(start_year=2023, end_year=2025):
    for year in range(start_year, end_year + 1):
        yield f"{year}0101", f"{year}0701"
        yield f"{year}0701", f"{year + 1}0101"


def train_walkforward(store: PanelStore, factor_dir, selection_path, out_dir,
                      params=None, train_start="20150101", start_year=2023, end_year=2025,
                      batch_rows=200_000):
    """One expanding-window model each half-year, no CV/ensembles/early stopping.

    Labels are purged by their actual exit date. Past OOS labels may enter later
    retraining once observed; this is a fixed live-like policy, not factor reselection.
    """
    import xgboost as xgb
    cfg = {**XGB_PARAMS, **(params or {})}
    if cfg["n_estimators"] != 500:
        raise ValueError("this protocol fixes n_estimators=500")
    if cfg["tree_method"] != "hist":
        raise ValueError("tree_method must be hist")
    if cfg["device"].startswith("cuda") and not xgb.build_info().get("USE_CUDA", False):
        raise RuntimeError("CUDA requested but installed XGBoost has no CUDA support; CPU must be explicit")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(selection_path).read_text())
    items = selection["factors"]
    if not items or len({x["name"] for x in items}) != len(items):
        raise ValueError("selection must contain unique non-empty factors")
    if selection.get("frozen_through", "99999999") > "20221231":
        raise ValueError("factor selection must be frozen using <=2022 information")
    label = store.load("label_10")
    mask = store.load("universe", label).eq(1)
    end = pd.read_csv(store.root / "label_end.csv", dtype=str, index_col=0).iloc[:, 0].reindex(label.index)
    if end.isna().sum() > 11 or not end.dropna().gt(end.dropna().index).all():
        raise ValueError("invalid label_end; rerun prepare using the complete calendar")
    dates = label.index
    T, N = label.shape
    shape = (T * N, len(items))
    X = np.memmap(out / "features.f32", dtype="float32", mode="w+", shape=shape)
    hashes = {}
    for j, item in enumerate(items):
        fpath = Path(factor_dir) / f"{item['name']}.pkl"
        f = PanelStore(factor_dir).load(item["name"], label)
        if item["direction"] not in (-1, 1):
            raise ValueError("invalid frozen factor direction")
        X[:, j] = preprocess(f * item["direction"], mask).to_numpy().ravel()
        hashes[item["name"]] = sha256(fpath)
        del f
    X.flush()
    y = label.to_numpy(dtype="float32").ravel()
    allowed = mask.to_numpy().ravel()
    has_feature = np.zeros(T * N, dtype=bool)
    for lo in range(0, T * N, batch_rows):
        hi = min(T * N, lo + batch_rows)
        has_feature[lo:hi] = np.isfinite(X[lo:hi]).any(axis=1)
    scores = pd.DataFrame(np.nan, index=dates, columns=label.columns, dtype="float32")
    audit = []
    native = {k: v for k, v in cfg.items() if k not in ("n_estimators", "missing")}
    # Python None in the supplied screenshot means no finite sentinel. Native API
    # receives np.nan, preserving genuine zero factor values.
    for lo, hi in half_years(start_year, end_year):
        train_days = (dates >= train_start) & (dates < lo) & end.notna() & end.lt(lo)
        test_days = (dates >= lo) & (dates < hi)
        tr = np.flatnonzero(np.repeat(np.asarray(train_days), N) & allowed & np.isfinite(y) & has_feature)
        te = np.flatnonzero(np.repeat(np.asarray(test_days), N) & allowed & has_feature)
        if not len(tr) or not len(te):
            raise ValueError(f"empty train/test for {lo}; supply 2015-2025 inputs")
        daily_available = (allowed & has_feature).reshape(T, N).sum(axis=1)
        if (daily_available[np.asarray(test_days)] == 0).any():
            raise ValueError(f"missing all eligible predictions on a test date in {lo}")
        # QuantileDMatrix reduces resident histogram memory; matrix construction is
        # still in-memory. memmap avoids keeping every source factor in RAM.
        dtrain = xgb.QuantileDMatrix(np.asarray(X[tr]), label=y[tr], missing=np.nan,
                                    max_bin=cfg["max_bin"], nthread=cfg["nthread"])
        model = xgb.train(native, dtrain, num_boost_round=500, verbose_eval=False)
        actual_device = json.loads(model.save_config())["learner"]["generic_param"]["device"]
        if cfg["device"].startswith("cuda") and not actual_device.startswith("cuda"):
            raise RuntimeError("XGBoost silently fell back to CPU; CUDA run rejected")
        model.save_model(out / f"xgb_{lo}.ubj")
        pred = np.empty(len(te), dtype="float32")
        for offset in range(0, len(te), batch_rows):
            rows = te[offset:offset + batch_rows]
            dm = xgb.DMatrix(np.asarray(X[rows]), missing=np.nan, nthread=cfg["nthread"])
            pred[offset:offset + len(rows)] = model.predict(dm)
        flat = scores.to_numpy(copy=True).ravel()
        flat[te] = pred
        scores.iloc[:, :] = flat.reshape(T, N)
        audit.append({"period_start": lo, "period_end_exclusive": hi,
                      "train_first_signal": dates[np.flatnonzero(train_days)[0]],
                      "train_last_signal": dates[np.flatnonzero(train_days)[-1]],
                      "max_train_label_end": end[train_days].max(), "train_rows": len(tr),
                      "prediction_rows": len(te), "trees": model.num_boosted_rounds(),
                      "actual_device": actual_device})
        json_dump({"params": cfg, "feature_hashes": hashes, "selection_sha256": sha256(selection_path),
                   "periods": audit}, out / "training_audit.json")
        del dtrain, model
        gc.collect()
    scores.to_pickle(out / "scores.pkl", protocol=4)
    del X
    (out / "features.f32").unlink()
    return scores, audit
