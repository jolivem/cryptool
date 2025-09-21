import pandas as pd
import numpy as np
import random
import simulate
import csv
import sys, os
from pathlib import Path
sys.path.append("../utils")
import fetchonbinance

def find_best_bot_with_params(historic, params_list, delete):

    try:
        fetchonbinance.download_and_join_binance_file(historic)
        crypto_filename = historic + ".csv"
        data = fetchonbinance.load_crypto_month(crypto_filename)

    except Exception  as e:
        # What to do if a ValueError happens
        print("find_best_bot_change_sell_gain: Oops! Could not find best bot:", e)
        return

    parts = crypto_filename.replace(".csv", "").split('-')
    date_part = "-".join(parts[2:-1])
    print(f" date_part: {date_part}\n")
    crypto = parts[0]

    # Paramètres fixes
    usdc_per_order = 25
    fee_pct = 0.00075

    # Liste des 5 meilleurs résultats
    top_results = []
    tries = 0
    print(f" CRYPTO: {crypto_filename}\n")
    for params in params_list:

        buy_drop_pct = params[0]
        buy_pullback_pct = params[1]
        sell_gain_pct = params[2]
        sell_pullback_pct = params[3]

        tries = tries + 1
        print(f"Try {tries}: -{100.0*buy_drop_pct:.3f}%, {100.0*buy_pullback_pct:.3f}% ; sell: +{100.0*sell_gain_pct:.3f}%, {100.0*sell_pullback_pct:.3f}%")

        # Appel de la simulation avec retour de la dernière position
        profit, last_pos = simulate.simulate(
            data,
            usdc_per_order,
            buy_drop_pct,
            buy_pullback_pct,
            sell_gain_pct,
            sell_pullback_pct,
            True
        )

        print(f"--> {profit:.2f} USDC, last_pos : {last_pos}")

        params = {
            "buy_drop_pct": buy_drop_pct,
            "buy_pullback_pct": buy_pullback_pct,
            "sell_gain_pct": sell_gain_pct,
            "sell_pullback_pct": sell_pullback_pct
        }

        # Ajout aux résultats avec position
        top_results.append((profit, params, last_pos))

        # Garde les 5 meilleurs profits
        top_results = sorted(top_results, key=lambda x: x[0], reverse=True)[:5]

    # Affichage final
    print("\n🏆 Top 5 meilleures configurations :")
    for i, (profit, params, last_pos) in enumerate(top_results, 1):
        print(f"{i}. Profit: {profit:.3f} USDC, last_pos : {last_pos}")
        for key, value in params.items():
            print(f"   {key}: {100.0*value:.4f}")

    result_file = fetchonbinance.result_folder + "bestbot-" + date_part+ ".csv"
    save_results(result_file, crypto, date_part, 111, top_results, 0, 0, 0, 0)


##
##
##
def find_best_bot_change_sell_gain(historic, delete):

    try:
        fetchonbinance.download_and_join_binance_file(historic)
        crypto_filename = historic + ".csv"
        data = fetchonbinance.load_crypto_month(crypto_filename)

    except Exception  as e:
        # What to do if a ValueError happens
        print("find_best_bot_change_sell_gain: Oops! Could not find crytpo data:", e)
        return

    # get some statistics
    ###############################
    nb_values = len(data)
    print(f"nb_values {nb_values}")
    mean_volume = data['quote asset volume'].mean()
    #print(f"mean_volume {mean_volume}")

    #volatility
    r = np.log(data['close']).diff().dropna()
    vol_ann = r.std() * np.sqrt(365*24*60*60)
    #print(f"vol_ann {vol_ann}")

    # regression linéaire
    line_slope = compute_slope_per_day(data)
    print("line_slope :", line_slope)

    parts = crypto_filename.replace(".csv", "").split('-')
    date_part = "-".join(parts[1:])
    crypto = parts[0]

    # prepare data
    ##########################

    #remove entries when volume = 0
    data = data[data["volume"] != 0]
    nan_pct = 100.0 * (1.0 - ((nb_values - len(data)) / nb_values))

    # simulation parameters
    ##########################
    
    # Paramètres fixes
    usdc_per_order = 25
    fee_pct = 0.00075

    # Liste des 5 meilleurs résultats
    top_results = []
    tries = 0
    print(f" CRYPTO: {crypto_filename}\n")
    buy_drop_pct = 0.015
    buy_pullback_pct = 0.0001
    sell_pullback_pct = 0.0001
    for sell_gain_pct in np.arange(0.015, 0.0351, 0.005): 

        tries = tries + 1
        print(f"Try {tries}: -{100.0*buy_drop_pct:.3f}%, {100.0*buy_pullback_pct:.3f}% ; sell: +{100.0*sell_gain_pct:.3f}%, {100.0*sell_pullback_pct:.3f}%")
        
        # Appel de la simulation avec retour de la dernière position
        profit, last_pos = simulate.simulate(
            data,
            usdc_per_order,
            buy_drop_pct,
            buy_pullback_pct,
            sell_gain_pct,
            sell_pullback_pct,
            False
        )

        print(f"--> {profit:.2f} USDC, last_pos : {last_pos}")

        params = {
            "buy_drop_pct": buy_drop_pct,
            "buy_pullback_pct": buy_pullback_pct,
            "sell_gain_pct": sell_gain_pct,
            "sell_pullback_pct": sell_pullback_pct
        }

        # Ajout aux résultats avec position
        top_results.append((profit, params, last_pos))

        # Garde les 5 meilleurs profits
        #top_results = sorted(top_results, key=lambda x: x[0], reverse=True)[:5]
        top_results = sorted(top_results, key=lambda x: x[0], reverse=True)[:1]

    # Affichage final
    print("\n🏆 Top des meilleures configurations :")
    for i, (profit, params, last_pos) in enumerate(top_results, 1):
        print(f"{i}. Profit: {profit:.3f} USDC, last_pos : {last_pos}")
        for key, value in params.items():
            print(f"   {key}: {100.0*value:.4f}")

    result_file = fetchonbinance.result_folder + "bestbot-" + date_part+ ".csv"
    save_results(result_file, crypto, date_part, 111, top_results, nb_values, nan_pct, mean_volume, vol_ann, line_slope)

    if delete == True:
        for p in Path(fetchonbinance.datas_folder).glob(crypto + "*.*"):   # non-recursive
            p.unlink()




def find_best_bot_random(nb_loop, crypto_filename):

    data = load_crypto_month(crypto_filename)

    parts = crypto_filename.replace(".csv", "").split('-')
    date_part = f"{parts[2]}-{parts[3]}"
    crypto = parts[0]

    # Paramètres fixes
    usdc_per_order = 25
    fee_pct = 0.00075

    # Liste des 5 meilleurs résultats
    top_results = []
    tries = 0
    print(f" CRYPTO: {crypto_filename}\n")
    for _ in range(nb_loop):  # Nombre d'essais

        # Génération aléatoire des paramètres
        #buy_drop_pct = random.uniform(0.005, 0.015)
        buy_drop_pct = 0.015
        buy_pullback_pct = random.uniform(0.0005, 0.01)
        sell_gain_pct = random.uniform(0.01, 0.03)
        sell_pullback_pct = random.uniform(0.001, 0.01)

        tries = tries + 1
        print(f"Try {tries}/{nb_loop}: -{100.0*buy_drop_pct:.3f}%, {100.0*buy_pullback_pct:.3f}% ; sell: +{100.0*sell_gain_pct:.3f}%, {100.0*sell_pullback_pct:.3f}%")

        # Appel de la simulation avec retour de la dernière position
        profit, last_pos = simulate.simulate(
            data,
            usdc_per_order,
            buy_drop_pct,
            buy_pullback_pct,
            sell_gain_pct,
            sell_pullback_pct,
            False
        )

        print(f"--> {profit:.2f} USDC, last_pos : {last_pos}")

        params = {
            "buy_drop_pct": buy_drop_pct,
            "buy_pullback_pct": buy_pullback_pct,
            "sell_gain_pct": sell_gain_pct,
            "sell_pullback_pct": sell_pullback_pct
        }

        # Ajout aux résultats avec position
        top_results.append((profit, params, last_pos))

        # Garde les 5 meilleurs profits
        top_results = sorted(top_results, key=lambda x: x[0], reverse=True)[:5]

    # Affichage final
    print("\n🏆 Top 5 meilleures configurations :")
    for i, (profit, params, last_pos) in enumerate(top_results, 1):
        print(f"{i}. Profit: {profit:.2f} USDC, last_pos : {last_pos}")
        for key, value in params.items():
            print(f"   {key}: {100.0*value:.4f}")

    save_results("res-"+ crypto_filename, crypto, date_part, nb_loop, top_results, 0, 0, 0, 0, 0)



def save_results(filename, crypto, date_part, nb_loop, top_results, nb_values, nan_pct, mean_volume, vol_ann, line_slope):

    if not os.path.exists(filename):
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([
                "crypto",
                "date",
                "nb_loop",
                "rank",
                "profit",
                "buy_drop_pct",
                "buy_pullback_pct",
                "sell_gain_pct",
                "sell_pullback_pct",
                "min_pos",
                "nb_values",
                "zero_pct",
                "mean_volume",
                "volatility",
                "line_slope"
            ])

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        
        rank = 1
        for profit, params, last_pos in top_results:

            writer.writerow([
                crypto,
                date_part,
                nb_loop,
                rank,
                profit,
                100.0*params['buy_drop_pct'],
                100.0*params['buy_pullback_pct'],
                100.0*params['sell_gain_pct'],
                100.0*params['sell_pullback_pct'],
                last_pos,
                nb_values, 
                nan_pct,
                mean_volume, 
                vol_ann, 
                line_slope
            ])
            rank = rank + 1


#
#

def _detect_unit_from_epoch(med: float) -> str:
    # s~1e9, ms~1e12, us~1e15, ns~1e18
    if med > 1e17: return 'ns'
    if med > 1e14: return 'us'
    if med > 1e11: return 'ms'
    return 's'

def _to_datetime_from_epoch(col: pd.Series, unit_override: str | None = None) -> pd.DatetimeIndex:
    s = pd.to_numeric(col, errors='coerce')
    if s.notna().sum() >= 0.5 * len(col):
        unit = unit_override or _detect_unit_from_epoch(float(s.dropna().median()))
        return pd.to_datetime(s, unit=unit, utc=True)
    return pd.to_datetime(col, errors='coerce', utc=True)

def compute_slope_per_day(
    data: pd.DataFrame,
    price_col: str = 'close',
    time_preference: tuple[str, ...] = ('close time', 'timespan'),
    unit_override: str | None = 'us',      # ← forcer microsecondes
    display_tz: str = 'UTC',               # 'Europe/Paris' si tu veux l’affichage local
):
    # 1) colonne temps
    time_col = 'timespan'

    # 2) index datetime (µs forcé), tri, dédoublonnage
    idx = _to_datetime_from_epoch(data[time_col], unit_override=unit_override)
    df = (data.assign(_ts=idx)
               .dropna(subset=['_ts'])
               .set_index('_ts')
               .sort_index())
    df = df[~df.index.duplicated(keep='first')]

    # Bornes du fichier (tous les timestamps valides)
    start_all_utc = df.index.min()
    end_all_utc   = df.index.max()

    # 3) série de prix positive (pour log)
    s = pd.to_numeric(df[price_col], errors='coerce').dropna()
    s = s[s > 0]

    if len(s) < 2:
        raise ValueError("Série trop courte après nettoyage (prix nuls/NaN ?).")

    # Bornes réellement utilisées pour la régression
    start_used_utc = s.index.min()
    end_used_utc   = s.index.max()

    # 4) temps en JOURS depuis le début utilisé (pas de 86 400 à multiplier)
    t = ((s.index - start_used_utc) / pd.Timedelta('1D')).astype('float64').to_numpy()
    y = np.log(s.to_numpy('float64'))

    # 5) moindres carrés (centrage)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    tc = t - t.mean()
    denom = float((tc * tc).sum())
    if denom == 0:
        raise ValueError("Variance du temps nulle (timestamps identiques).")
    slope_day = (tc * y).sum() / denom           # log-par-jour
    intercept = y.mean() - slope_day * t.mean()

    # 6) R²
    yhat = slope_day * t + intercept
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res/ss_tot

    # 7) conversions %
    pct_per_day = slope_day * 100
    pct_ann = (np.exp(slope_day * 365) - 1) * 100

    # pas médian (en secondes)
    step_s = (df.index.to_series().diff().median() / pd.Timedelta('1s'))

    # Affichage dans le fuseau demandé
    def _convert(ts):
        return ts.tz_convert(display_tz) if display_tz and ts.tzinfo else ts

    # return {
    #     "pct_per_day": float(pct_per_day),
    #     "pct_ann": float(pct_ann),
    #     "slope_log_per_day": float(slope_day),
    #     "r2": float(r2) if np.isfinite(r2) else np.nan,
    #     "n_points": int(len(t)),
    #     "median_step_seconds": float(step_s) if pd.notna(step_s) else None,
    #     "start_all": _convert(start_all_utc),
    #     "end_all": _convert(end_all_utc),
    #     "start_used": _convert(start_used_utc),
    #     "end_used": _convert(end_used_utc),
    # }

    return pct_per_day