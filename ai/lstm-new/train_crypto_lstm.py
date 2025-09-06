import os, math, sys
import numpy as np
import pandas as pd
import tensorflow as tf
sys.path.append("../../utils")
import fetchonbinance

import utils
import inputs as inputs_

# -------------------------
# Paramètres principaux
# -------------------------
#CSV_PATH = "D:\DEVS\cryptool\datas\ETHUSDC-1s-2025-07.csv"       # votre fichier
# inputs_.TARGET_COL = "close"        # variable à prédire
# inputs_.USE_RETURNS = True          # True = prédire le rendement (log-return), False = prix
# WINDOW = 60                 # longueur de la fenêtre (pas de temps)
# inputs_.HORIZON = 1                 # prédire t+1 (même granularité)
# inputs_.BATCH_SIZE = 128
# inputs_.EPOCHS = 20
# inputs_.LR = 1e-3
# inputs_.VAL_SPLIT = 0.15            # portion de la fin du jeu d'entraînement pour la validation
# inputs_.TEST_SPLIT = 0.15           # portion de la fin totale pour test (walk-forward)

# -------------------------
# 1) Charger et nettoyer
# -------------------------

#CRYPTO = "SOLUSDC-1s-2025-07"

#print("GPUs:", tf.config.list_physical_devices("GPU"))
#tf.debugging.set_log_device_placement(True)

df = utils.load_data(inputs_.CRYPTO)

(df, feat_cols) = utils.prepare_data(df)

# # Features de base
# feat_cols = ["open","high","low","close","volume"]

# # (Optionnel) Ajouter indicateurs simples
# df["hl_spread"] = df["high"] - df["low"]
# df["oc_spread"] = (df["close"] - df["open"]).abs()
# feat_cols += ["hl_spread","oc_spread"]

# # Cible : prix ou log-returns
# if inputs_.USE_RETURNS:
#     # log-return: log(C_t / C_{t-1})
#     df["target"] = np.log(df[inputs_.TARGET_COL]).diff()
# else:
#     df["target"] = df[inputs_.TARGET_COL].shift(-inputs_.HORIZON)  # prédire le prix futur
# df = df.dropna().reset_index(drop=True)

# -------------------------
# 2) Split temporel
# -------------------------
N = len(df)
test_size = int(N * inputs_.TEST_SPLIT)
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
train_size = int(len(X_trainval) * (1 - inputs_.VAL_SPLIT))
X_train = X_trainval[:train_size]
y_train = y_trainval[:train_size]
X_val = X_trainval[train_size:]
y_val = y_trainval[train_size:]

mu = X_train.mean(axis=0, keepdims=True)
sigma = X_train.std(axis=0, keepdims=True) + 1e-8
X_train = (X_train - mu) / sigma
X_val   = (X_val   - mu) / sigma
X_test  = (X_test  - mu) / sigma

# -------------------------
# 4) Fenêtrage en séquences
# -------------------------
# def make_windows(X, y, window, inputs_.HORIZON):
#     Xs, ys = [], []
#     # On prédit y[t + inputs_.HORIZON - 1] à partir de X[t-window+1 : t+1]
#     for t in range(window - 1, len(X) - inputs_.HORIZON + 1):
#         Xs.append(X[t - window + 1 : t + 1])
#         ys.append(y[t + inputs_.HORIZON - 1])
#     return np.array(Xs), np.array(ys)

Xtr_seq, ytr = utils.make_windows(X_train, y_train, inputs_.WINDOW, inputs_.HORIZON)
Xval_seq, yval = utils.make_windows(X_val, y_val, inputs_.WINDOW, inputs_.HORIZON)
Xte_seq, yte = utils.make_windows(X_test, y_test, inputs_.WINDOW, inputs_.HORIZON)

# -------------------------
# 5) Datasets tf.data
# -------------------------
def make_ds(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
    return ds.batch(inputs_.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_tr = make_ds(Xtr_seq, ytr, shuffle=True)
ds_val = make_ds(Xval_seq, yval)
ds_te = make_ds(Xte_seq, yte)

# -------------------------
# 6) Modèle LSTM
# -------------------------
tf.keras.utils.set_random_seed(42)

inputs = tf.keras.Input(shape=(inputs_.WINDOW, Xtr_seq.shape[-1]))
x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=inputs_.LR),
    loss="mae",              # MAE = robuste sur séries
    metrics=["mse"]
)

model.summary()

# Early stopping
cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = model.fit(
    ds_tr,
    validation_data=ds_val,
    epochs=inputs_.EPOCHS,
    callbacks=[cb],
    verbose=1
)

# -------------------------
# 7) Baseline naïve & évaluation
# -------------------------
# Baseline naïve: prédire "pas de changement" (0 pour returns, prix courant pour prix)
if inputs_.USE_RETURNS:
    naive_pred = np.zeros_like(yte)
    mae_naive = np.mean(np.abs(naive_pred - yte))
else:
    # pour prix: y_pred = dernier "close" de la fenêtre
    last_close_idx = feat_cols.index("close")
    naive_pred = Xte_seq[:, -1, last_close_idx] * sigma[0,last_close_idx] + mu[0,last_close_idx]
    mae_naive = np.mean(np.abs(naive_pred - yte))

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
    win = X_full[-WINDOW:].copy()
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
sav_path = "artifacts/" + inputs_.CRYPTO
os.makedirs(sav_path, exist_ok=True)
model.save(sav_path + "/crypto_lstm.keras")
np.savez(sav_path + "/scaler_stats.npz", mu=mu, sigma=sigma, feat_cols=np.array(feat_cols))
print("Modèle et scaler sauvegardés dans ./" + sav_path)
