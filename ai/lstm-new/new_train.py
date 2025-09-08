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
TRAIN_SPLIT  = 0.8
VAL_SPLIT    = 0.18          # test is the rest (0.15)

EPS      = 0.001    # ~0.1% band; tweak based on your asset/liquidity
SEED     = 1337
tf.keras.utils.set_random_seed(SEED)
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

df = df.sort_index()
work = df[['open','high','low','close','volume']].copy()
if 'volume' not in work:
    work['volume'] = 0.0

import numpy as np, tensorflow as tf, os, matplotlib.pyplot as plt

work['logret1']       = np.log(work['close']).diff()
work['hl_range']      = work['high'] - work['low']
work['oc_change']     = work['close'] - work['open']
work['log_vol']       = np.log1p(work['volume'])
work['volatility_10'] = work['logret1'].rolling(10).std()

# clean
work = (work
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True))

FEATURE_COLS = [
    'open','high','low','close',
    'log_vol','hl_range','oc_change','volatility_10','logret1'
]
X_all = work[FEATURE_COLS].to_numpy(np.float32)
SERIES_LEN = len(work)

# 1) Time splits
n_train = int(SERIES_LEN * TRAIN_SPLIT)
n_val   = int(SERIES_LEN * VAL_SPLIT)
n_test  = SERIES_LEN - n_train - n_val

train_slice = slice(0, n_train)
val_slice   = slice(n_train, n_train + n_val)
test_slice  = slice(n_train + n_val, SERIES_LEN)

# 2) Standardize X with TRAIN ONLY
mu_x  = X_all[train_slice].mean(axis=0)
std_x = X_all[train_slice].std(axis=0) + 1e-8
Z = (X_all - mu_x) / std_x

# 3) Build next-step *return* and binary labels
# ret1[t] = log(C_t) - log(C_{t-1}) sits at index t.
# For a window ending at time j-1, the "next-step" is ret1[j].
ret1 = np.log(work['close']).diff().to_numpy(np.float32)  # length T, ret1[0]=nan

# Binary label: 1 if UP, 0 if DOWN. We'll IGNORE near-zero moves via a mask.
y_bin  = (ret1 > 0).astype(np.int64)
mask   = (~np.isnan(ret1)) & (np.abs(ret1) >= EPS)

def make_windows_cls(Z, y, mask, slc, window=50, horizon=1):
    """Return (X, y) for classification; label index j=i+window+horizon-1 (next step)."""
    start, end = slc.start, slc.stop
    xs, ys = [], []
    for i in range(start, end - window - horizon + 1):
        j = i + window + horizon - 1
        if not mask[j]:
            continue  # skip tiny/undefined moves
        xs.append(Z[i : i + window, :])
        ys.append(y[j])
    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y

Xtr, ytr = make_windows_cls(Z, y_bin, mask, train_slice, WINDOW, HORIZON)
Xva, yva = make_windows_cls(Z, y_bin, mask, val_slice,   WINDOW, HORIZON)
Xte, yte = make_windows_cls(Z, y_bin, mask, test_slice,  WINDOW, HORIZON)

print(f"Windows -> train:{len(Xtr)} val:{len(Xva)} test:{len(Xte)} | features:{Z.shape[1]}")

# Optional: class weights to handle imbalance
neg = (ytr == 0).sum(); pos = (ytr == 1).sum(); total = neg + pos
class_weight = {0: total/(2*neg+1e-9), 1: total/(2*pos+1e-9)}
print("Class weights:", class_weight)

# 4) Model: same backbone, sigmoid head
inputs = tf.keras.Input(shape=(WINDOW, Z.shape[1]))
x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
x = tf.keras.layers.Dropout(0.20)(x)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(16, activation="relu")(x)
prob_up = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # P(trend==UP)

model = tf.keras.Model(inputs, prob_up)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                       tf.keras.metrics.AUC(name="auc")])

cb = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
]
history = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                    epochs=80, batch_size=128, callbacks=cb,
                    class_weight=class_weight, verbose=1)

# 5) Choose decision threshold on validation to maximize F1 (or another metric)
def bin_metrics(y_true, p, thr=0.5):
    yhat = (p >= thr).astype(np.int64)
    tp = ((y_true==1)&(yhat==1)).sum()
    tn = ((y_true==0)&(yhat==0)).sum()
    fp = ((y_true==0)&(yhat==1)).sum()
    fn = ((y_true==1)&(yhat==0)).sum()
    acc = (tp+tn)/max(len(y_true),1)
    prec = tp/max(tp+fp,1)
    rec  = tp/max(tp+fn,1)
    f1   = 2*prec*rec/max(prec+rec,1e-9)
    return acc, prec, rec, f1, (tp, fp, fn, tn)

p_va = model.predict(Xva, verbose=0).ravel()
ths  = np.linspace(0.2, 0.8, 121)
best = max((bin_metrics(yva, p_va, t)[3], t) for t in ths)  # by F1
best_thr = best[1]
print(f"Chosen threshold on val by F1: {best_thr:.3f}")

# 6) Evaluate on test
p_te = model.predict(Xte, verbose=0).ravel()
acc, prec, rec, f1, conf = bin_metrics(yte, p_te, best_thr)
print(f"[TEST] acc:{acc:.3f}  prec:{prec:.3f}  recall:{rec:.3f}  F1:{f1:.3f}  conf(TP,FP,FN,TN)={conf}")

# 7) Save model + scalers + threshold so inference is consistent
os.makedirs(sav_path, exist_ok=True)
model.save(sav_path + "/crypto_lstm_trend_binary.keras")
np.savez(sav_path + "/trend_artifacts.npz",
         mu_x=mu_x, std_x=std_x, feature_cols=np.array(FEATURE_COLS),
         eps=EPS, decision_threshold=best_thr)
print("Saved to", sav_path)

# 8) One-step ahead trend from the latest window
last_win = Z[-WINDOW:, :]
p_up = float(model.predict(last_win[None, ...], verbose=0)[0, 0])
trend = "UP" if p_up >= best_thr else "DOWN"
print(f"Next-step trend: {trend}  (P_up={p_up:.3f}, thr={best_thr:.3f})")

# yte: shape (n,), values {0,1}
# p_te: shape (n,), predicted P(UP)
# best_thr: float
# sav_path: folder to save the chart

idx = np.arange(len(yte))
yhat = (p_te >= best_thr).astype(int)
correct = (yhat == yte)

plt.figure(figsize=(11,4))
plt.plot(idx, p_te, label="Predicted P(UP)")
plt.plot(idx, yte, drawstyle="steps-post", alpha=0.6, label="Actual UP (0/1)")
plt.axhline(best_thr, linestyle="--", label=f"Decision threshold = {best_thr:.2f}")

# Light background shading where predictions are correct
ax = plt.gca()
plt.fill_between(
    idx, 0, 1, where=correct,
    alpha=0.08, step="pre",
    transform=ax.get_xaxis_transform(),
    label="Correct prediction"
)

plt.title("Next-step trend — prediction vs. actual (test set)")
plt.xlabel("Test window index")
plt.ylabel("Probability / Class")
plt.legend()
plt.tight_layout()

os.makedirs(sav_path, exist_ok=True)
out_path = os.path.join(sav_path, "trend_pred_vs_actual.png")
plt.savefig(out_path, dpi=130)
print("Saved:", out_path)