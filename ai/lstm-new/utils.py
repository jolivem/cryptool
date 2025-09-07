import os, math, sys
import numpy as np
import pandas as pd
sys.path.append("../../utils")
import fetchonbinance
import torch

import params

def load_data(cryptos):
    fetchonbinance.download_and_join_binance_file(cryptos)
    crypto_filename = cryptos + ".csv"
    df = fetchonbinance.load_crypto_month(crypto_filename)

    # Colonnes minimales
    required = {"open","high","low","close","volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")
        return None


    # Normaliser noms
    df.columns = df.columns.str.lower()

    # Trier par temps si présent
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values("timestamp")
        except Exception:
            df = df.reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Drop NA
    df = df.dropna().reset_index(drop=True)

    return df

def prepare_data(df):
    # Features de base
    #feat_cols = ["open","high","low","close","volume"]
    feat_cols = ["close"]

    # (Optionnel) Ajouter indicateurs simples
    # df["hl_spread"] = df["high"] - df["low"]
    # df["oc_spread"] = (df["close"] - df["open"]).abs()
    # feat_cols += ["hl_spread","oc_spread"]

    # Cible : prix ou log-returns
    if params.USE_RETURNS:
        # log-return: log(C_t / C_{t-1})
        df["target"] = np.log(df[params.TARGET_COL]).diff()
    else:
        #price delta
        df["target"] = df[params.TARGET_COL].shift(-params.HORIZON) - df[params.TARGET_COL]
    df = df.dropna().reset_index(drop=True)

    return (df, feat_cols)

def make_windows(X, y, window, horizon):
    Xs, ys = [], []
    for t in range(window - 1, len(X) - horizon):
        Xs.append(X[t - window + 1 : t + 1])
        ys.append(y[t])
    return np.array(Xs, dtype="float32"), np.array(ys, dtype="float32")

def test_gpu():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
        x = torch.randn(8000, 8000, device="cuda")
        y = x @ x
        torch.cuda.synchronize() 

#test_gpu()

def forecast_last_window_returns(model, mu, sigma, feat_cols, X_full_raw, steps=20):
    """
    Multi-step forecast for returns. 
    - model: loaded tf.keras model
    - mu, sigma: (1, n_features) arrays from scaler_stats.npz
    - feat_cols: list of feature column names (same as training)
    - X_full_raw: np.array of RAW (unstandardized) feature rows for the series (same columns/order as feat_cols)
    - steps: number of future steps to simulate
    Returns: (pred_returns, reconstructed_prices) where reconstructed_prices is optional (None if 'close' not in feat_cols)
    """
    import numpy as np

    # indices we need
    try:
        i_close = feat_cols.index("close")
        i_open  = feat_cols.index("open")
        i_high  = feat_cols.index("high")
        i_low   = feat_cols.index("low")
        i_vol   = feat_cols.index("volume")
    except ValueError:
        raise ValueError("The function expects feat_cols to contain open, high, low, close, volume.")

    WINDOW = model.input_shape[1]

    # seed windows (raw + standardized)
    win_raw = X_full_raw[-WINDOW:].astype("float32").copy()
    win_std = (win_raw - mu) / sigma

    # spreads (use medians from the seed window as simple priors)
    hl_med = float(np.median(win_raw[:, i_high] - win_raw[:, i_low]))
    vol_med = float(np.median(win_raw[:, i_vol]))

    pred_returns = []
    recon_prices = []

    last_close = float(win_raw[-1, i_close])

    for _ in range(steps):
        # predict next RETURN using the standardized window
        x_in = win_std[np.newaxis, ...]
        yhat = float(model.predict(x_in, verbose=0)[0, 0])  # predicted log-return
        pred_returns.append(yhat)

        # reconstruct next close from log-return
        next_close = last_close * np.exp(yhat)

        # build a plausible next raw row
        next_open = last_close
        oc_spread = abs(next_close - next_open)
        # optional derived features present in training?
        next_hl = hl_med if hl_med > 0 else oc_spread
        next_high = max(next_open, next_close) + 0.5 * next_hl
        next_low  = min(next_open, next_close) - 0.5 * next_hl
        next_vol  = vol_med

        new_raw = win_raw[-1].copy()  # start from last row
        new_raw[i_open]  = next_open
        new_raw[i_high]  = next_high
        new_raw[i_low]   = next_low
        new_raw[i_close] = next_close
        new_raw[i_vol]   = next_vol
        # update optional engineered features, if they exist
        if "hl_spread" in feat_cols:
            new_raw[feat_cols.index("hl_spread")] = next_high - next_low
        if "oc_spread" in feat_cols:
            new_raw[feat_cols.index("oc_spread")] = abs(next_close - next_open)

        # roll the window (raw + standardized)
        win_raw = np.vstack([win_raw[1:], new_raw])
        win_std = (win_raw - mu) / sigma

        last_close = next_close
        recon_prices.append(next_close)

    return np.array(pred_returns, dtype="float32"), np.array(recon_prices, dtype="float32")
