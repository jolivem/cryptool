import pandas as pd
import numpy as np
import random
import simulate
import csv
import sys
sys.path.append("../utils")
import fetchonbinance

def find_best_bot_with_params(nb_loop, historic, params_list):

    try:
        fetchonbinance.download_and_join_binance_file(historic)
        crypto_filename = historic + ".csv"
        data = fetchonbinance.load_crypto_month(crypto_filename)

    except Exception  as e:
        # What to do if a ValueError happens
        print("find_best_bot_with_loop: Oops! Could not find best bot:", e)
        return

    parts = crypto_filename.replace(".csv", "").split('-')
    date_part = f"{parts[2]}-{parts[3]}" # ex 2025-07
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
        print(f"{i}. Profit: {profit:.3f} USDC, last_pos : {last_pos}")
        for key, value in params.items():
            print(f"   {key}: {100.0*value:.4f}")

    save_results(fetchonbinance.result_folder + "/bestbot.csv", crypto, date_part, nb_loop, top_results)


def find_best_bot_with_loop(nb_loop, historic):

    try:
        fetchonbinance.download_and_join_binance_file(historic)
        crypto_filename = historic + ".csv"
        data = fetchonbinance.load_crypto_month(crypto_filename)

    except Exception  as e:
        # What to do if a ValueError happens
        print("find_best_bot_with_loop: Oops! Could not find best bot:", e)
        return

    parts = crypto_filename.replace(".csv", "").split('-')
    date_part = f"{parts[2]}-{parts[3]}" # ex 2025-07
    crypto = parts[0]

    # Paramètres fixes
    usdc_per_order = 25
    fee_pct = 0.00075

    # Liste des 5 meilleurs résultats
    top_results = []
    tries = 0
    print(f" CRYPTO: {crypto_filename}\n")
    buy_drop_pct = 0.015
    buy_pullback_pct = 0.005
    sell_pullback_pct = 0.0001
    for sell_gain_pct in np.arange(0.0153, 0.0359, 0.004):

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
        print(f"{i}. Profit: {profit:.3f} USDC, last_pos : {last_pos}")
        for key, value in params.items():
            print(f"   {key}: {100.0*value:.4f}")

    save_results(fetchonbinance.result_folder + "/bestbot.csv", crypto, date_part, nb_loop, top_results)

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

    save_results("res-"+ crypto_filename, crypto, date_part, nb_loop, top_results)

def save_results(filename, crypto, date_part, nb_loop, top_results):

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        # writer.writerow([
        #     "crypto",
        #     "date",
        #     "nb_loop",
        #     "rank",
        #     "profit",
        #     "buy_drop_pct",
        #     "buy_pullback_pct",
        #     "sell_gain_pct",
        #     "sell_pullback_pct",
        #     "min_pos"
        # ])
        
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
                last_pos
            ])
            rank = rank + 1