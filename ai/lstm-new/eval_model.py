#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import utils
import inputs

CRYPTO = "SOLUSDC-1s-2025-07"

def parse_args():
    p = argparse.ArgumentParser(description="Évaluation d'un modèle LSTM sauvegardé (graphe vérité vs prédiction)")
    #p.add_argument("--csv", default="data.csv", help="Chemin du CSV OHLCV utilisé à l'entraînement")
    p.add_argument("--model", default="artifacts/crypto_lstm.keras", help="Chemin du modèle sauvegardé")
    p.add_argument("--scaler", default="artifacts/scaler_stats.npz", help="Chemin du scaler sauvegardé (mu/sigma/feat_cols)")
    #p.add_argument("--target-col", default="close", help="Colonne cible utilisée à l'entraînement")
    # p.add_argument("--use-returns", action="store_true", default=True, help="Si True, cible = log-return (comme dans le script d'entraînement par défaut)")
    # p.add_argument("--no-returns", dest="use-returns", action="store_false", help="Si False, cible = prix futur (shift)")
    #p.add_argument("--horizon", type=int, default=1, help="Horizon utilisé à l'entraînement (par défaut 1)")
    #p.add_argument("--test-split", type=float, default=0.15, help="Part de test utilisée à l'entraînement (par défaut 0.15)")
    p.add_argument("--save", default="artifacts/eval_pred_vs_true.png", help="Chemin de sauvegarde du graphe")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Modèle introuvable: {args.model}")
    if not os.path.exists(args.scaler):
        raise FileNotFoundError(f"Scaler introuvable: {args.scaler}")
    # if not os.path.exists(args.csv):
    #     raise FileNotFoundError(f"CSV introuvable: {args.csv}")

    # -- Charger modèle et scaler --
    model = tf.keras.models.load_model(args.model)
    inp_shape = model.input_shape  # (None, WINDOW, N_FEATURES)
    WINDOW = int(inp_shape[1])
    NFEAT  = int(inp_shape[2])

    scal = np.load(args.scaler, allow_pickle=True)
    mu = scal["mu"]            # shape (1, n_features)
    sigma = scal["sigma"]      # shape (1, n_features)
    feat_cols = scal["feat_cols"].tolist()
    feat_cols = [fc.decode() if isinstance(fc, (bytes, bytearray)) else str(fc) for fc in feat_cols]

    if mu.shape[1] != len(feat_cols):
        raise ValueError("Incohérence scaler: mu/sigma et feat_cols n'ont pas la même taille.")
    if len(feat_cols) != NFEAT:
        raise ValueError(f"Incohérence features: modèle attend {NFEAT} features, scaler en a {len(feat_cols)}.")



    # -- Charger CSV & préparer features exactement comme à l'entraînement --

    df = utils.load_data(inputs.CRYPTO)

    (df, cols) = utils.prepare_data(df)

    # Recréer les features dérivées si nécessaires
    # if "hl_spread" in feat_cols and "hl_spread" not in df.columns:
    #     df["hl_spread"] = df["high"] - df["low"]
    # if "oc_spread" in feat_cols and "oc_spread" not in df.columns:
    #     df["oc_spread"] = (df["close"] - df["open"]).abs()

    # # Cible (même logique que l'entraînement)
    # if inputs.USE_RETURN:
    #     # log-return
    #     df["target"] = np.log(df[inputs.TARGET_COL]).diff()
    # else:
    #     df["target"] = df[inputs.TARGET_COL].shift(-inputs.HORIZON)

    # df = df.dropna().reset_index(drop=True)

    # -- Recréer le split temporel (test final) --
    N = len(df)
    test_size = int(N * inputs.TEST_SPLIT)
    test = df.iloc[N - test_size :].copy()

    # -- Préparer X_test / y_test avec les mêmes features et scaling sauvés --
    X_test = test[feat_cols].values.astype("float32")
    y_test = test["target"].values.astype("float32")

    # Appliquer le scaling du train (mu/sigma sauvegardés)
    X_test = (X_test - mu) / sigma

    # -- Fenêtrage identique --
    Xte_seq, yte = utils.make_windows(X_test, y_test, WINDOW, inputs.HORIZON)
    if len(Xte_seq) == 0:
        raise ValueError("Pas assez d'échantillons pour créer des fenêtres sur le set de test. Réduisez WINDOW/HORIZON ou fournissez plus de données.")

    # -- Prédictions --
    yhat = model.predict(Xte_seq, verbose=0).reshape(-1)

    # -- Métriques simples --
    mae = float(np.mean(np.abs(yhat - yte)))
    mse = float(np.mean((yhat - yte)**2))
    print(f"[Test] MAE: {mae:.6f} | MSE: {mse:.6f} | fenêtres: {len(yte)} | WINDOW={WINDOW} HORIZON={inputs.HORIZON}")

    # -- Graphe: vérité vs prédiction (série) --
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    plt.figure()
    plt.plot(yte, label="vérité")
    plt.plot(yhat, label="prédiction")
    title_suffix = "(returns)" if inputs.USE_RETURNS else "(prix)"
    plt.title(f"Évaluation test — vérité vs prédiction {title_suffix}")
    plt.xlabel("Index test")
    plt.ylabel("Cible")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save)
    try:
        plt.show()
    except Exception:
        pass
    print(f"[✓] Graphe sauvegardé: {args.save}")

if __name__ == "__main__":
    main()
