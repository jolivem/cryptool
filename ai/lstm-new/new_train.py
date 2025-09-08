#!/usr/bin/env python3
# Minimal LSTM on a sine wave — train, evaluate, and plot

import os, shutil
import numpy as np
import tensorflow as tf

# Use a headless backend so it works in terminals/servers too
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import params, utils

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

# tf.keras.utils.set_random_seed(SEED)
# rng = np.random.default_rng(SEED)

# t = np.arange(SERIES_LEN, dtype=np.float32)
# series = np.sin(2*np.pi * t / PERIOD).astype(np.float32) + rng.normal(0, NOISE_STD, SERIES_LEN).astype(np.float32) + 100.0


sav_path = "artifacts/" + params.CRYPTO
if os.path.exists(sav_path):
    shutil.rmtree(sav_path)

# -------------------------
# 1) Charger et nettoyer
# -------------------------

df = utils.load_data(params.CRYPTO)
(df, feat_cols) = utils.prepare_data(df)
df = df.sort_index()  # ensure chronological order
work = df[['open','high','low','close','volume']].copy()

# Light, causal feature engineering (all use only past/current info)
work['logret1']       = np.log(work['close']).diff()
work['hl_range']      = work['high'] - work['low']         # intraperiod range
work['oc_change']     = work['close'] - work['open']       # candle body
work['log_vol']       = np.log1p(work['volume'])           # compress heavy tails
work['volatility_10'] = work['logret1'].rolling(10).std()  # short-run vol

# Drop the initial NaNs created by diff/rolling
work = work.dropna().reset_index(drop=True)

FEATURE_COLS = [
    'open','high','low','close','log_vol','hl_range','oc_change','volatility_10','logret1'
]
F = len(FEATURE_COLS)

# Feature matrix and target series (close)
X_all = work[FEATURE_COLS].to_numpy(dtype=np.float32)   # (T, F)
y_all = work['close'].to_numpy(dtype=np.float32)        # (T,)

SERIES_LEN = len(work)

# ----------------------------
# 1) Time splits (same logic as yours)
# ----------------------------
n_train = int(SERIES_LEN * TRAIN_SPLIT)
n_val   = int(SERIES_LEN * VAL_SPLIT)
n_test  = SERIES_LEN - n_train - n_val

train_slice = slice(0, n_train)
val_slice   = slice(n_train, n_train + n_val)
test_slice  = slice(n_train + n_val, SERIES_LEN)

# ----------------------------
# 2) Standardize X per-feature using TRAIN ONLY
#    and standardize y (close) separately for inversion later
# ----------------------------
mu_x  = X_all[train_slice].mean(axis=0)
std_x = X_all[train_slice].std(axis=0) + 1e-8
Z = (X_all - mu_x) / std_x      # (T, F), standardized features

mu_y  = float(y_all[train_slice].mean())
std_y = float(y_all[train_slice].std() + 1e-8)
z_close = (y_all - mu_y) / std_y   # standardized label series

# ----------------------------
# 3) Windowing (multivariate X, univariate y)
# ----------------------------
def make_windows_mv(Z, z_label, slc, window=50, horizon=1):
    """Return (X, y); X shape (n, window, F), y shape (n,)"""
    start, end = slc.start, slc.stop
    xs, ys = [], []
    for i in range(start, end - window - horizon + 1):
        xs.append(Z[i : i + window, :])
        ys.append(z_label[i + window + horizon - 1])
    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return X, y

Xtr, ytr = make_windows_mv(Z, z_close, train_slice, WINDOW, HORIZON)
Xva, yva = make_windows_mv(Z, z_close, val_slice,   WINDOW, HORIZON)
Xte, yte = make_windows_mv(Z, z_close, test_slice,  WINDOW, HORIZON)

print(f"Windows -> train:{len(Xtr)} val:{len(Xva)} test:{len(Xte)}  |  features: {F}")

# ----------------------------
# 4) Model: slightly wider + dropout
# ----------------------------
inputs = tf.keras.Input(shape=(WINDOW, F))
x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
x = tf.keras.layers.Dropout(0.20)(x)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(16, activation="relu")(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="mse", metrics=["mae"])

cb = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
]
history = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                    epochs=80, batch_size=128, callbacks=cb, verbose=1)

# ----------------------------
# 5) Save model + scalers for inference
# ----------------------------
os.makedirs(sav_path, exist_ok=True)
model.save(sav_path + "/crypto_lstm_mv.keras")
np.savez(sav_path + "/scaler_stats.npz",
         mu_x=mu_x, std_x=std_x, mu_y=mu_y, std_y=std_y,
         feature_cols=np.array(FEATURE_COLS))
print("Model and scalers saved to ./" + sav_path)

# ----------------------------
# 6) Evaluate on test windows + inverse scale for plotting
# ----------------------------
test_loss, test_mae = model.evaluate(Xte, yte, verbose=0)
print(f"Test MSE: {test_loss:.6f}  |  Test MAE (z): {test_mae:.6f}")

yhat_z = model.predict(Xte, verbose=0).ravel()
yhat   = yhat_z * std_y + mu_y      # back to price
ytrue  = yte    * std_y + mu_y

plt.figure()
plt.plot(ytrue, label="true (test)")
plt.plot(yhat,  label="pred (test)")
plt.title("Next-close prediction (multivariate OHLCV)")
plt.xlabel("test window index"); plt.ylabel("price")
plt.legend(); plt.tight_layout()
plt.savefig(sav_path + "/test_true_vs_pred.png", dpi=130)
print("Saved:", sav_path + "/test_true_vs_pred.png")

# ----------------------------
# 7) One-step ahead inference from the latest window
# ----------------------------
last_win = Z[-WINDOW:, :]                   # (WINDOW, F)
next_close_z = float(model.predict(last_win[None, ...], verbose=0)[0, 0])
next_close   = next_close_z * std_y + mu_y
print("Next close prediction:", next_close)