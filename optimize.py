import pandas as pd
import random
import simulate

# Chargement du CSV sans en-tête
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

root_dir = "/home/michel/binance-data/data/spot/monthly/klines/"
kline_file = root_dir + "SOLUSDC/1s/SOLUSDC-1s-2025-06/SOLUSDC-1s-2025-06.csv"
png_file = "SOLUSDC-1s-2025-06.png"
data = pd.read_csv(kline_file, header=None, names=columns)
data['timestamp'] = pd.to_datetime(data['timespan'], unit='us')
data['price'] = data['Close']
data['volume'] = data['Volume']
data = data[data["volume"] != 0]

# Paramètres de stratégie
usdc_per_order = 25
# buy_drop_pct = 0.015
# buy_pullback_pct = 0.005
# sell_gain_pct = 0.015
# sell_pullback_pct = 0.005
fee_pct = 0.001


best_profit = -float("inf")
best_params = None

for _ in range(100):  # 1000 essais

    buy_drop_pct = random.uniform(0.01, 0.02)
    buy_pullback_pct = random.uniform(0.002, 0.008)
    sell_gain_pct = random.uniform(0.01, 0.02)
    sell_pullback_pct = random.uniform(0.002, 0.008)
    print(f"\nSimulate with buy: -{100.0*buy_drop_pct:.3f}%, {100.0*buy_pullback_pct:.3f}% ; sell: +{100.0*sell_gain_pct:.3f}%, {100.0*sell_pullback_pct:.3f}%")
    profit = old_simulate.simulate(data, usdc_per_order, buy_drop_pct, buy_pullback_pct, sell_gain_pct, sell_pullback_pct, False)
    print(f"Profit total simulé : {profit:.2f} USDC")

    if profit > best_profit:
        best_profit = profit
        best_params = (buy_drop_pct, buy_pullback_pct, sell_gain_pct, sell_pullback_pct)

print("Meilleurs paramètres :", best_params)
print("Profit maximal :", best_profit)
