#!/usr/bin/env python3
# Minimal LSTM on a sine wave — train, evaluate, and plot

import os
import numpy as np
import tensorflow as tf

# Use a headless backend so it works in terminals/servers too
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# 1) Generate a sine time series
# ----------------------------
SEED         = 42
SERIES_LEN   = 2000          # total points
PERIOD       = 50            # sine period (samples per cycle)
NOISE_STD    = 0.05          # add a touch of noise
WINDOW       = 50            # lookback length
HORIZON      = 1             # predict next step
TRAIN_SPLIT  = 0.7
VAL_SPLIT    = 0.15          # test is the rest (0.15)

tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)

t = np.arange(SERIES_LEN, dtype=np.float32)
series = np.sin(2*np.pi * t / PERIOD).astype(np.float32) + rng.normal(0, NOISE_STD, SERIES_LEN).astype(np.float32) + 100.0

# ----------------------------
# 2) Train/val/test splits (by time)
# ----------------------------
n_train = int(SERIES_LEN * TRAIN_SPLIT)
n_val   = int(SERIES_LEN * VAL_SPLIT)
n_test  = SERIES_LEN - n_train - n_val

train_slice = slice(0, n_train)
val_slice   = slice(n_train, n_train + n_val)
test_slice  = slice(n_train + n_val, SERIES_LEN)

# Standardize using TRAIN ONLY (best practice)
mu  = float(series[train_slice].mean())
std = float(series[train_slice].std() + 1e-8)
z = (series - mu) / std  # standardized series

# ----------------------------
# 3) Windowing helper
# ----------------------------
def make_windows(zseries, slc, window=50, horizon=1):
    """Return (X, y) where X shape (n, window, 1), y shape (n,)"""
    start, end = slc.start, slc.stop
    xs, ys = [], []
    # window covers [i, i+window-1]; label is at i+window+h-1
    for i in range(start, end - window - horizon + 1):
        xs.append(zseries[i : i + window])
        ys.append(zseries[i + window + horizon - 1])
    X = np.array(xs, dtype=np.float32)[..., None]  # (n, window, 1)
    y = np.array(ys, dtype=np.float32)             # (n,)
    return X, y

Xtr, ytr = make_windows(z, train_slice, WINDOW, HORIZON)
Xva, yva = make_windows(z, val_slice,   WINDOW, HORIZON)
Xte, yte = make_windows(z, test_slice,  WINDOW, HORIZON)

print(f"Windows -> train:{len(Xtr)} val:{len(Xva)} test:{len(Xte)}")

# ----------------------------
# 4) Model: tiny LSTM
# ----------------------------
inputs = tf.keras.Input(shape=(WINDOW, 1))
x = tf.keras.layers.LSTM(32)(inputs)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="mse", metrics=["mae"])

cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=64, callbacks=[cb], verbose=1)

# ----------------------------
# 5) Evaluate on test windows
# ----------------------------
test_loss, test_mae = model.evaluate(Xte, yte, verbose=0)
print(f"Test MSE: {test_loss:.6f}  |  Test MAE: {test_mae:.6f}")

# Predicted next values (standardized) → inverse scale for plotting
yhat_z = model.predict(Xte, verbose=0).ravel()
yhat   = yhat_z * std + mu     # back to original scale
ytrue  = yte    * std + mu

# ----------------------------
# 6) Plot: true vs predicted (test)
# ----------------------------
os.makedirs("artifacts", exist_ok=True)

plt.figure()
plt.plot(ytrue, label="true (test)")
plt.plot(yhat,  label="pred (test)")
plt.title("Sine — next-step prediction on test windows")
plt.xlabel("test window index"); plt.ylabel("value")
plt.legend(); plt.tight_layout()
plt.savefig("artifacts/sine_test_true_vs_pred.png", dpi=130)
print("Saved:", "artifacts/sine_test_true_vs_pred.png")

# ----------------------------
# 7) Simple multi-step forecast demo
#    Start from the last available window, predict 200 steps ahead
# ----------------------------
steps_ahead = 200
win = z[-WINDOW:].copy()  # last standardized window from the whole series
pred_path = []
for _ in range(steps_ahead):
    y_next_z = float(model.predict(win[None, :, None], verbose=0)[0, 0])
    pred_path.append(y_next_z)
    # roll the window to include the new prediction
    win = np.r_[win[1:], y_next_z]

pred_path = np.array(pred_path, dtype=np.float32) * std + mu  # back to original scale

plt.figure()
# Plot the last 300 true points for context, plus the forecast path
context = series[-300:]
plt.plot(np.arange(len(context)), context, label="true (last 300)")
plt.plot(np.arange(len(context)-1, len(context)-1 + steps_ahead),
         pred_path, label="forecast (next 200)")
plt.title("Sine — multi-step forecast from last window")
plt.xlabel("time"); plt.ylabel("value")
plt.legend(); plt.tight_layout()
plt.savefig("artifacts/sine_multistep_forecast.png", dpi=130)
print("Saved:", "artifacts/sine_multistep_forecast.png")
