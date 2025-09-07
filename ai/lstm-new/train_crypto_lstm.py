import os, shutil, math, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# sys.path.append("../../utils")
# import fetchonbinance

import utils
import params


# delete previous results
sav_path = "artifacts/" + params.CRYPTO
if os.path.exists(sav_path):
    shutil.rmtree(sav_path)

# -------------------------
# 1) Charger et nettoyer
# -------------------------

df = utils.load_data(params.CRYPTO)

(df, feat_cols) = utils.prepare_data(df)
period = 60 
A = 100.0              # amplitude
offset = 300.0         # vertical shift
i = np.arange(len(df))
df["close"] = offset + A * np.cos(2*np.pi * i / period)
df["target"] = offset + A * np.cos(2*np.pi * i / period)

# -------------------------
# 2) Split temporel
# -------------------------
N = len(df)
test_size = int(N * params.TEST_SPLIT)
trainval = df.iloc[: N - test_size].copy()
test = df.iloc[N - test_size :].copy()

# Séparer X / y
X_trainval = trainval[feat_cols].values.astype("float32")
y_trainval = trainval["target"].values.astype("float32")
X_test = test[feat_cols].values.astype("float32")
y_test = test["target"].values.astype("float32")

# -------------------------
# 3) Standardisation (fit sur train uniquement)
# -------------------------
train_size = int(len(X_trainval) * (1 - params.VAL_SPLIT))
X_train = X_trainval[:train_size]
y_train = y_trainval[:train_size]
X_val = X_trainval[train_size:]
y_val = y_trainval[train_size:]

# compute mu and sigma for all feat_cols
mu = X_train.mean(axis=0, keepdims=True)
sigma = X_train.std(axis=0, keepdims=True) + 1e-8
X_train = (X_train - mu) / sigma
X_val   = (X_val   - mu) / sigma
X_test  = (X_test  - mu) / sigma
print("sigma shape:", sigma.shape, "dtype:", sigma.dtype)
# 1) Print all values without truncation
np.set_printoptions(precision=6, suppress=True, linewidth=140, threshold=np.inf)
print("sigma raw:", sigma)

# # 2) Pretty table with feature names
# assert len(feat_cols) == sigma.shape[1], "feat_cols mismatch"
# sigma_series = pd.Series(sigma[0], index=feat_cols, name="sigma")
# print(sigma_series.to_string())

# # 3) Specific feature (e.g., 'close')
# close_idx = feat_cols.index("close")
# print("sigma['close'] =", float(sigma[0, close_idx]))
# -------------------------
# 4) Fenêtrage en séquences
# -------------------------
def make_windows(X, y, window, params.HORIZON):
    Xs, ys = [], []
    # On prédit y[t + params.HORIZON - 1] à partir de X[t-window+1 : t+1]
    for t in range(window - 1, len(X) - params.HORIZON + 1):
        Xs.append(X[t - window + 1 : t + 1])
        ys.append(y[t + params.HORIZON - 1])
    return np.array(Xs), np.array(ys)

Xtr_seq, ytr = make_windows(X_train, y_train, params.WINDOW, params.HORIZON)
Xval_seq, yval = make_windows(X_val, y_val, params.WINDOW, params.HORIZON)
Xte_seq, yte = make_windows(X_test, y_test, params.WINDOW, params.HORIZON)

# -------------------------
# 5) Datasets tf.data
# -------------------------
def make_ds(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
    return ds.batch(params.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_tr = make_ds(Xtr_seq, ytr, shuffle=True)
ds_val = make_ds(Xval_seq, yval)
ds_te = make_ds(Xte_seq, yte)

# -------------------------
# 6) Modèle LSTM
# -------------------------
tf.keras.utils.set_random_seed(42)

inputs = tf.keras.Input(shape=(params.WINDOW, Xtr_seq.shape[-1]))
x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params.LR),
    loss="mae",              # MAE = robuste sur séries
    metrics=["mse"]
)

model.summary()

# Early stopping
cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# train the model
history = model.fit(
    ds_tr,
    validation_data=ds_val,
    epochs=params.EPOCHS,
    callbacks=[cb],
    verbose=1
)

# -------------------------
# 7) Baseline naïve & évaluation
# -------------------------
# Baseline naïve: prédire "pas de changement" (0 pour returns, prix courant pour prix)
if params.USE_RETURNS:
    naive_pred = np.zeros_like(yte)
    mae_naive = np.mean(np.abs(naive_pred - yte))
else:
    # pour prix: y_pred = dernier "close" de la fenêtre
    last_close_idx = feat_cols.index("close")
    naive_pred = Xte_seq[:, -1, last_close_idx]
    mae_naive = np.mean(np.abs(naive_pred - yte))

# evaluate on the test set
eval_res = model.evaluate(ds_te, verbose=0)
mae_model, mse_model = eval_res[0], eval_res[1]
print(f"Test MAE (modèle): {mae_model:.6f}")
print(f"Test MSE (modèle): {mse_model:.6f}")
print(f"Test MAE baseline naïve: {mae_naive:.6f}")

# -------------------------
# 8) Prévision "récursive" sur les derniers pas
# -------------------------
def forecast_last_window(model, X_full, steps=10):
    """
    Prédit récursivement les 'steps' prochaines valeurs à partir de la
    dernière fenêtre disponible de X_full (déstandardisé en interne si besoin).
    """
    win = X_full[-params.WINDOW:].copy()
    preds = []
    for _ in range(steps):
        x_in = win[np.newaxis, ...]
        yhat = model.predict(x_in, verbose=0)[0,0]
        preds.append(yhat)
        # On simule l'arrivée d'un nouveau point -> on ajoute une "ligne" vide sauf target
        # Ici on n'a pas les futures features réelles; en prod, injectez des features à jour.
        # On fait un simple shift en réutilisant la dernière ligne (approximatif).
        new_row = win[-1].copy()
        win = np.vstack([win[1:], new_row])
    return np.array(preds)

future_preds = forecast_last_window(model, X_test, steps=10)
print("10 steps de prévision (espace modèle):", future_preds[:5], "...")

# -------------------------
# 9) Sauvegarde du modèle
# -------------------------
sav_path = "artifacts/" + params.CRYPTO
os.makedirs(sav_path, exist_ok=True)
model.save(sav_path + "/crypto_lstm.keras")
np.savez(sav_path + "/scaler_stats.npz", feat_cols=np.array(feat_cols))
print("Modèle et scaler sauvegardés dans ./" + sav_path)



# 1) Predict directly from numpy windows (avoid ds_te alignment gotchas)
yhat = model.predict(Xte_seq, batch_size=params.BATCH_SIZE, verbose=0).ravel()

# 2) Reconstruct last CLOSE (de-standardize the 'close' feature)
close_idx = feat_cols.index("close")
last_close_std = Xte_seq[:, -1, close_idx]
last_close = last_close_std

# 3) Build true & predicted price series from deltas
# ytrue_price = last_close + yte
# yhat_price  = last_close + yhat_delta


# for i, d in enumerate(yhat_delta):
#     print(f"{i}\t{d:.8f}")


#yhat_delta = model.predict(ds_te, verbose=0).ravel()   # predicts delta
ytrue_price = yte
yhat_price  = yhat
#for i, d in enumerate(yhat_delta):
for i, d in enumerate(ytrue_price):
    print(f"{i}\t{d:.8f}")
    if i == 20:
        break
for i, d in enumerate(yhat_price):
    print(f"{i}\t{d:.8f}")
    if i == 20:
        break
# Courbe série: vérité vs prédiction
plt.figure()
plt.plot(ytrue_price, label="vérité (test)")
plt.plot(yhat_price, label="prédiction (test)")
plt.title(f"Test — vérité vs prédiction ({'returns' if params.USE_RETURNS else 'prix'})")
plt.xlabel("index fenêtre test")
plt.ylabel("cible")
plt.legend()
plt.tight_layout()

#os.makedirs("artifacts", exist_ok=True)
plt.savefig(sav_path +"/pred_vs_true_test.png", dpi=120)
#plt.show()  # si vous êtes en console sans affichage, commentez et gardez seulement le savefig

print(f"[✓] Graphe sauvegardé: {sav_path +"/eval_pred_vs_true.png"}")