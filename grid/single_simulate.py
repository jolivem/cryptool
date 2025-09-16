import pandas as pd
import old_simulate
import sys
sys.path.append("../utils")
import fetchonbinance
#crypto_filename = "INJUSDC-1s-2025-08.csv"


CRYPTO = "FLOKIUSDC-1s-2025-07"

try:
    fetchonbinance.download_and_join_binance_file(CRYPTO)
    crypto_filename = CRYPTO + ".csv"
    data = fetchonbinance.load_crypto_month(crypto_filename)

except Exception  as e:
    # What to do if a ValueError happens
    print("Oops! Could not find best bot:", e)
    exit



# Paramètres fixes
usdc_per_order = 20

buy_drop_pct = 0.01
buy_pullback_pct = 0.005
sell_gain_pct = 0.015
sell_pullback_pct = 0.005

# Appel de la simulation avec retour de la dernière position
profit, last_pos = old_simulate.simulate(
    data,
    usdc_per_order,
    buy_drop_pct,
    buy_pullback_pct,
    sell_gain_pct,
    sell_pullback_pct,
    True
)

print(f"--> {profit:.2f} USDC, last_pos : {last_pos}")
