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
#SERIES_LEN   = 2000          # total points
PERIOD       = 10            # sine period (samples per cycle)
NOISE_STD    = 0.0          # add a touch of noise
WINDOW       = 50            # lookback length
HORIZON      = 1             # predict next step
TRAIN_SPLIT  = 0.8
VAL_SPLIT    = 0.18          # test is the rest (0.15)

EPS      = 0.001    # ~0.1% band; tweak based on your asset/liquidity
SEED     = 1337

tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)
# t = np.arange(SERIES_LEN, dtype=np.float32)
# series = np.sin(2*np.pi * t / PERIOD).astype(np.float32) + rng.normal(0, NOISE_STD, SERIES_LEN).astype(np.float32) + 100.0


sav_path = "artifacts/" + params.CRYPTO
if os.path.exists(sav_path):
    shutil.rmtree(sav_path)

os.makedirs(sav_path)

# -------------------------
# 1) Charger et nettoyer
# -------------------------

df = utils.load_data(params.CRYPTO)
(df, feat_cols) = utils.prepare_data(df)

# fake with sine
# SERIES_LEN = len(df['close'])
# print(f"SERIES_LEN {SERIES_LEN}")
# t = np.arange(SERIES_LEN, dtype=np.float32)
# series = np.sin(2*np.pi * t / PERIOD).astype(np.float32) + rng.normal(0, NOISE_STD, SERIES_LEN).astype(np.float32) + 100.0
# arr = np.asarray(series, dtype=np.float32).ravel()  # ensure 1-D
# df['close'] = arr
#df = df.sort_index()

#print(series[:10])
print(df['close'][:10])


work = df[['open','high','low','close', 'volume']].copy()
if 'volume' in df.columns:
    work['volume'] = df['volume'].astype(float)
else:
    work['volume'] = 0.0

# Causal light features
work['logret1']       = np.log(work['close']).diff()
work['hl_range']      = work['high'] - work['low']
work['oc_change']     = work['close'] - work['open']
work['log_vol']       = np.log1p(work['volume'])
work['volatility_10'] = work['logret1'].rolling(10).std()

# Clean NaNs/Infs created by diff/rolling/log
work = (work
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True))

FEATURE_COLS = [
    'open','high','low','close',
    'log_vol','hl_range','oc_change','volatility_10','logret1'
]
F = len(FEATURE_COLS)

X_all = work[FEATURE_COLS].to_numpy(np.float32)   # (T, F)
T = len(work)

# -------------------------
# 1) Labels (3-class)
# -------------------------
# Use already-clean logret1 as the next-step return series
ret1 = work['logret1'].to_numpy(np.float32)       # length T

# Map to 3 classes by EPS band
# 0 = DOWN, 1 = FLAT, 2 = UP  (targets for next-step from index j)
labels3 = np.where(ret1 >  EPS, 2,
           np.where(ret1 < -EPS, 0, 1)).astype(np.int64)

# -------------------------
# 2) Time splits (by index)
# -------------------------
n_train = int(T * TRAIN_SPLIT)
n_val   = int(T * VAL_SPLIT)
n_test  = T - n_train - n_val

train_slice = slice(0, n_train)
val_slice   = slice(n_train, n_train + n_val)
test_slice  = slice(n_train + n_val, T)

# -------------------------
# 3) Standardize X on TRAIN ONLY
# -------------------------
mu_x  = X_all[train_slice].mean(axis=0)
std_x = X_all[train_slice].std(axis=0) + 1e-8
Z = (X_all - mu_x) / std_x                          # (T, F)

# -------------------------
# 4) Windowing helper (multivariate X, 3-class y)
# -------------------------
def make_windows_cls3(Z, y_classes, slc, window=50, horizon=1):
    """
    Return X,y for classification.
    X shape: (n, window, F), y shape: (n,)
    Label index j = i + window + horizon - 1  (next-step for H=1).
    """
    start, end = slc.start, slc.stop
    xs, ys = [], []
    for i in range(start, end - window - horizon + 1):
        j = i + window + horizon - 1
        xs.append(Z[i : i + window, :])
        ys.append(y_classes[j])
    return np.asarray(xs, np.float32), np.asarray(ys, np.int64)

Xtr, ytr = make_windows_cls3(Z, labels3, train_slice, WINDOW, HORIZON)
Xva, yva = make_windows_cls3(Z, labels3, val_slice,   WINDOW, HORIZON)
Xte, yte = make_windows_cls3(Z, labels3, test_slice,  WINDOW, HORIZON)

print(f"Windows -> train:{len(Xtr)} val:{len(Xva)} test:{len(Xte)} | features:{F}")

# -------------------------
# 5) Optional: class weights for imbalance
# -------------------------
def compute_class_weights(y, n_classes=3):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    total = counts.sum()
    weights = {k: (total / (n_classes * counts[k])) if counts[k] > 0 else 1.0
               for k in range(n_classes)}
    return weights, counts

class_weight, train_counts = compute_class_weights(ytr, n_classes=3)
print("Train class counts (DOWN,FLAT,UP):", train_counts.tolist())
print("Class weights:", class_weight)

# -------------------------
# 6) Model: LSTM → Dense → softmax(3)
# -------------------------
inputs = tf.keras.Input(shape=(WINDOW, F))
x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
x = tf.keras.layers.Dropout(0.20)(x)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(16, activation="relu")(x)
logits = tf.keras.layers.Dense(3, activation="softmax")(x)

model = tf.keras.Model(inputs, logits)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

cb = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
]

history = model.fit(Xtr, ytr,
                    validation_data=(Xva, yva),
                    epochs=80,
                    batch_size=128,
                    callbacks=cb,
                    class_weight=class_weight,
                    verbose=1)

# -------------------------
# 7) Evaluate + save artifacts
# -------------------------
test_loss, test_acc = model.evaluate(Xte, yte, verbose=0)
print(f"[TEST] loss:{test_loss:.4f}  acc:{test_acc:.4f}")

model.save(os.path.join(sav_path, "crypto_lstm_trend3.keras"))
np.savez(os.path.join(sav_path, "trend3_artifacts.npz"),
         mu_x=mu_x, std_x=std_x,
         feature_cols=np.array(FEATURE_COLS),
         eps=EPS, window=WINDOW, horizon=HORIZON)
print("Saved model + scalers to:", sav_path)

# -------------------------
# 8) Predictions on test
# -------------------------
probs_te = model.predict(Xte, verbose=0)        # (n, 3)
yhat = np.argmax(probs_te, axis=1)              # (n,)
class_names = np.array(["DOWN","FLAT","UP"])

# -------------------------
# 9) Plots
# -------------------------
# (a) Predicted probabilities per class
plt.figure(figsize=(11,4))
idx = np.arange(len(yte))
for k, name in enumerate(class_names):
    plt.plot(idx, probs_te[:, k], label=f"P({name})")
plt.ylim(0, 1)
plt.title("3-class next-step trend — predicted probabilities (test)")
plt.xlabel("Test window index"); plt.ylabel("Probability")
plt.legend(); plt.tight_layout()
p1 = os.path.join(sav_path, "trend3_probabilities.png")
plt.savefig(p1, dpi=130)
print("Saved:", p1)

# (b) Predicted class vs actual (step charts) with shading for correct segments
plt.figure(figsize=(11,4))
plt.step(idx, yte,  where="post", label="Actual",   alpha=0.85)
plt.step(idx, yhat, where="post", label="Predicted",alpha=0.85)
correct = (yhat == yte)
ax = plt.gca()
plt.fill_between(idx, -0.5, 2.5, where=correct, alpha=0.08, step="pre",
                 transform=ax.get_xaxis_transform())
plt.yticks([0,1,2], class_names)
plt.ylim(-0.5, 2.5)
plt.title("3-class next-step trend — predicted vs actual (test)")
plt.xlabel("Test window index"); plt.ylabel("Class")
plt.legend(); plt.tight_layout()
p2 = os.path.join(sav_path, "trend3_pred_vs_actual.png")
plt.savefig(p2, dpi=130)
print("Saved:", p2)

# (c) Confusion matrix
try:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte, yhat, labels=[0,1,2])
except Exception:
    cm = np.zeros((3,3), dtype=int)
    for t, p in zip(yte, yhat):
        cm[int(t), int(p)] += 1

plt.figure(figsize=(4.8,4.8))
plt.imshow(cm)
plt.xticks([0,1,2], class_names); plt.yticks([0,1,2], class_names)
plt.title("Confusion matrix (test)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, int(v), ha="center", va="center")
plt.tight_layout()
p3 = os.path.join(sav_path, "trend3_confusion_matrix.png")
plt.savefig(p3, dpi=130)
print("Saved:", p3)

# (d) Running accuracy curve
run_acc = np.cumsum(correct) / (np.arange(len(correct)) + 1)
plt.figure(figsize=(11,3.5))
plt.plot(idx, run_acc, label="Running accuracy")
plt.ylim(0,1)
plt.title("Running accuracy on test")
plt.xlabel("Test window index"); plt.ylabel("Accuracy")
plt.legend(); plt.tight_layout()
p4 = os.path.join(sav_path, "trend3_running_accuracy.png")
plt.savefig(p4, dpi=130)
print("Saved:", p4)

# -------------------------
# 10) One-step ahead inference from latest window
# -------------------------
last_win = Z[-WINDOW:, :]                      # (WINDOW, F)
probs_next = model.predict(last_win[None, ...], verbose=0)[0]  # (3,)
pred_next  = int(np.argmax(probs_next))
print(f"Next-step predicted class: {class_names[pred_next]}  "
      f"(P_DOWN={probs_next[0]:.3f}, P_FLAT={probs_next[1]:.3f}, P_UP={probs_next[2]:.3f})")