import pandas as pd
import random
import simulate
import csv

crypto_filename = "FLOKIUSDC-1s-2025-07.csv"

# Génération aléatoire des paramètres
#-0.514%, 0.261% ; sell: +1.349%, 0.361%
buy_drop_pct = 0.015
buy_pullback_pct = 0.015
sell_gain_pct = 0.01
sell_pullback_pct = 0.005

columns = [
    "timespan",	
    "Open",	
    "High",	
    "Low",	
    "Close",	
    "Volume",	
    "Close time",
    "Quote asset volume",	
    "Number of trades",	
    "Taker buy base asset volume",	
    "Taker buy quote asset volume",	
    "Ignore"
]


#print(first_part)  # Output: ARBUSDC
#crypto = "ARBUSDC-1s-2025-06.csv"
root_dir = r"C:\\Users\\joliv\\Documents\\binance-data\\"
kline_file = root_dir + crypto_filename
data = pd.read_csv(kline_file, header=None, names=columns)
data['timestamp'] = pd.to_datetime(data['timespan'], unit='us')
data['price'] = data['Close']
data['volume'] = data['Volume']
data = data[data["volume"] != 0]

# Paramètres fixes
usdc_per_order = 25

# Liste des 5 meilleurs résultats
print(f" CRYPTO: {crypto_filename}\n")


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
