"""Calibrate the differentiable action -> EEF-position model used by the guidance cost.

Reads traj/*.npz written by eval_libero.py (eef_pos[T+1], actions[T, 7]) and fits

  (A) nominal:   dp_t = 0.05 * clip(a_t[:3])                    (robosuite OSC output_max)
  (B) gain:      dp_t = G @ clip(a_t[:3])                        (3x3 least squares)
  (C) lag:       dp_t = G0 @ a_t + G1 @ a_{t-1} + G2 @ a_{t-2}   (first-order tracking lag)

and reports one-step R^2 plus the open-loop error of the predicted EEF position after
H = 5 and 10 steps (the executed / full chunk length of pi05_libero), starting from the
true position at chunk start and using only the commanded actions.

Usage: python benchmarks/calibrate_action_model.py runs/baseline/libero_spatial [more dirs...]
"""
import glob
import json
import sys

import numpy as np


def load(dirs):
    eps = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/traj/*.npz")):
            z = np.load(f)
            if len(z["actions"]) > 12:
                eps.append((np.clip(z["actions"][:, :3], -1, 1), z["eef_pos"]))
    return eps


def lagged(a, k):
    out = np.zeros_like(a)
    out[k:] = a[: len(a) - k]
    return out


def design(a, lags):
    return np.concatenate([lagged(a, k) for k in range(lags)], axis=1)


def fit(eps, lags):
    X = np.concatenate([design(a, lags) for a, _ in eps])
    Y = np.concatenate([np.diff(p, axis=0) for _, p in eps])
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W  # (3*lags, 3)


def predict_dp(a, W, lags):
    return design(a, lags) @ W


def r2(eps, pred_fn):
    Y = np.concatenate([np.diff(p, axis=0) for _, p in eps])
    P = np.concatenate([pred_fn(a) for a, _ in eps])
    return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def open_loop_err(eps, pred_fn, H):
    errs = []
    for a, p in eps:
        dp = pred_fn(a)
        for s in range(0, len(a) - H, 5):  # chunk starts every replan_steps=5
            pred = p[s] + dp[s : s + H].sum(0)
            errs.append(np.linalg.norm(pred - p[s + H]))
    e = np.array(errs)
    return {"mean_cm": 100 * e.mean(), "p90_cm": 100 * np.percentile(e, 90), "max_cm": 100 * e.max()}


def main():
    eps = load(sys.argv[1:])
    n_steps = sum(len(a) for a, _ in eps)
    print(f"{len(eps)} episodes, {n_steps} control steps")
    models = {
        "A_nominal_0.05": lambda a: 0.05 * a,
    }
    W1 = fit(eps, 1)
    models["B_gain3x3"] = lambda a, W=W1: predict_dp(a, W, 1)
    W3 = fit(eps, 3)
    models["C_lag2"] = lambda a, W=W3: predict_dp(a, W, 3)

    report = {"n_episodes": len(eps), "n_steps": n_steps, "G_gain3x3": W1.T.round(4).tolist()}
    for name, fn in models.items():
        row = {"r2_1step": round(float(r2(eps, fn)), 4)}
        for H in (5, 10):
            row[f"H{H}"] = {k: round(float(v), 3) for k, v in open_loop_err(eps, fn, H).items()}
        report[name] = row
        print(name, json.dumps(row))
    print("gain matrix G (rows = world xyz, cols = action xyz):\n", W1.T.round(4))
    json.dump(report, open(f"{sys.argv[1]}/action_model_calibration.json", "w"), indent=1)
    np.savez(f"{sys.argv[1]}/action_model_calibration.npz", G=W1.T, W_lag=W3)


if __name__ == "__main__":
    main()
