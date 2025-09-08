#!/usr/bin/env python3
# Predict evolution ratio r_t = x[t+1]/x[t] on a (positive) sine wave with a tiny LSTM

import os
import numpy as np
import tensorflow as tf

# Headless plotting (works in terminals/servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils, params

# ----------------------------
# 1) Generate a positive sine series
# ----------------------------
SEED         = 42
SERIES_LEN   = 2000          # total points
PERIOD       = 50            # samples per cycle
NOISE_STD    = 0.02          # noise on top of sine
OFFSET       = 2.0           # shift upward so values stay > 0 (needed for ratios/log)
AMP          = 1.0           # sine amplitude

WINDOW       = 50            # lookback
HORIZON      = 1             # predict t+1 ratio
TRAIN_SPLIT  = 0.8
VAL_SPLIT    = 0.1          # test is the rest

# tf.keras.utils.set_random_seed(SEED)
# rng = np.random.default_rng(SEED)

# t = np.arange(SERIES_LEN, dtype=np.float32)
# pure = AMP * np.sin(2*np.pi * t / PERIOD).astype(np.float32)
# series = (OFFSET + pure + rng.normal(0, NOISE_STD, SERIES_LEN)).astype(np.float32)
# assert np.all(series > 0), "Series must be positive for ratio/log-return."

df = utils.load_data(params.CRYPTO)
(df, feat_cols) = utils.prepare_data(df)

series = df["close"].to_numpy(dtype=float)
SERIES_LEN   = len(df)



# ----------------------------
# 2) Build target: log-ratio at t+1
#    y_t = log( x_{t+1} / x_t )
# ----------------------------
y_full = np.empty_like(series)
y_full[:] = np.nan
y_full[:-HORIZON] = np.log(series[HORIZON:] / series[:-HORIZON])

# Splits (time-based)
n_train = int(SERIES_LEN * TRAIN_SPLIT)
n_val   = int(SERIES_LEN * VAL_SPLIT)
n_test  = SERIES_LEN - n_train - n_val
train_slice = slice(0, n_train)
val_slice   = slice(n_train, n_train + n_val)
test_slice  = slice(n_train + n_val, SERIES_LEN)

# Standardize inputs using TRAIN ONLY
mu  = float(series[train_slice].mean())
std = float(series[train_slice].std() + 1e-8)
z = (series - mu) / std  # standardized inputs

# ----------------------------
# 3) Windowing aligned to window end
#    For a window ending at t, label is y_full[t] (which is about t+H ahead)
# ----------------------------
def make_windows(zseries, yseries, base_series, slc, window=50, horizon=1):
    start, end = slc.start, slc.stop
    Xs, ys, last_vals = [], [], []
    # Need i+window-1+horizon < end  -> i <= end - window - horizon
    for i in range(start, end - window - horizon + 1):
        t_end = i + window - 1
        Xs.append(zseries[i : i + window])
        ys.append(yseries[t_end])
        last_vals.append(base_series[t_end])  # last observed value before predicting t+1
    X = np.array(Xs, dtype=np.float32)[..., None]   # (n, window, 1)
    y = np.array(ys, dtype=np.float32)              # (n,)
    last_vals = np.array(last_vals, dtype=np.float32)  # original scale
    return X, y, last_vals

Xtr, ytr, last_tr = make_windows(z, y_full, series, train_slice, WINDOW, HORIZON)
Xva, yva, last_va = make_windows(z, y_full, series, val_slice,   WINDOW, HORIZON)
Xte, yte, last_te = make_windows(z, y_full, series, test_slice,  WINDOW, HORIZON)

print(f"Windows -> train:{len(Xtr)} val:{len(Xva)} test:{len(Xte)} (need ≥ {WINDOW+HORIZON} rows per split)")

# ----------------------------
# 4) Tiny LSTM that predicts log-ratio
# ----------------------------
inputs = tf.keras.Input(shape=(WINDOW, 1))
x = tf.keras.layers.LSTM(32)(inputs)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="mse", metrics=["mae"])

cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                    epochs=40, batch_size=64, callbacks=[cb], verbose=1)

# ----------------------------
# 5) Evaluate on test windows — ratio & next value
# ----------------------------
test_mse, test_mae = model.evaluate(Xte, yte, verbose=0)
print(f"Test  MSE (log-ratio): {test_mse:.6f}  |  MAE (log-ratio): {test_mae:.6f}")

yhat_log = model.predict(Xte, verbose=0).ravel()
ratio_true = np.exp(yte)        # r_t
ratio_pred = np.exp(yhat_log)   # \hat r_t

# Reconstruct next values from ratios
next_true = last_te * ratio_true
next_pred = last_te * ratio_pred

# ----------------------------
# 6) Plots
# ----------------------------
os.makedirs("artifacts", exist_ok=True)

# (a) Ratio true vs predicted
plt.figure()
plt.plot(ratio_true, label="true ratio (t→t+1)")
plt.plot(ratio_pred, label="pred ratio (t→t+1)")
plt.axhline(1.0, linestyle="--", linewidth=1)
plt.title("Evolution ratio r_t = x[t+1]/x[t] (test)")
plt.xlabel("test window index"); plt.ylabel("ratio")
plt.legend(); plt.tight_layout()
plt.savefig("artifacts/ratio_true_vs_pred.png", dpi=130)
print("Saved:", "artifacts/ratio_true_vs_pred.png")

# (b) Next value true vs predicted (easier to see)
plt.figure()
plt.plot(next_true, label="true next value")
plt.plot(next_pred, label="pred next value")
plt.title("Next value reconstructed from predicted ratio (test)")
plt.xlabel("test window index"); plt.ylabel("value")
plt.legend(); plt.tight_layout()
plt.savefig("artifacts/next_value_true_vs_pred.png", dpi=130)
print("Saved:", "artifacts/next_value_true_vs_pred.png")

# ----------------------------
# 7) Multi-step forecast from the last window
# ----------------------------
steps_ahead = 200
win_z = z[-WINDOW:].copy()
last_val = series[-1]
path_vals = []

for _ in range(steps_ahead):
    yhat_next_log = float(model.predict(win_z[None, :, None], verbose=0)[0, 0])
    r_pred = np.exp(yhat_next_log)
    next_val = last_val * r_pred
    path_vals.append(next_val)

    # Update window with the NEW predicted value (standardized)
    win_z = np.r_[win_z[1:], (next_val - mu) / std]
    last_val = next_val

path_vals = np.array(path_vals, dtype=np.float32)

plt.figure()
context = series[-300:]
plt.plot(np.arange(len(context)), context, label="true (last 300)")
plt.plot(np.arange(len(context)-1, len(context)-1 + steps_ahead),
         path_vals, label="forecast (next 200)")
plt.title("Multi-step forecast via predicted ratios")
plt.xlabel("time"); plt.ylabel("value")
plt.legend(); plt.tight_layout()
plt.savefig("artifacts/ratio_multistep_forecast.png", dpi=130)
print("Saved:", "artifacts/ratio_multistep_forecast.png")
