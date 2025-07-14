import pandas as pd
import matplotlib.pyplot as plt

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
df = pd.read_csv(kline_file, header=None, names=columns)
df['timestamp'] = pd.to_datetime(df['timespan'], unit='us')
df['price'] = df['Close']
df['volume'] = df['Volume']

# Paramètres de stratégie
usdc_per_order = 25
buy_drop_pct = 0.015
buy_pullback_pct = 0.005
sell_gain_pct = 0.015
sell_pullback_pct = 0.005
fee_pct = 0.001

positions = []
entry_price = None
profit = 0
log = []

# start_time = pd.Timestamp("2025-07-06 11:30:00+0200")
# df = df[df['timestamp'] >= start_time]

lowest_price = None
highest_price = None
top_price = None 

# Boucle principale
for i in range(1, len(df)):
    volume = df['volume'].iloc[i]
    if volume == 0:
        continue

    price = df['price'].iloc[i]
    time = df['timestamp'].iloc[i]
    # print(f"[{time}]\n")

    # === Logique d'achat (grille) ===
    if top_price is None or price > top_price:
        top_price = price
        lowest_price = None  # reset car nouveau sommet

    if lowest_price is None or price < lowest_price:
        lowest_price = price

    drop_from_top = 1 - price / top_price

    if drop_from_top >= buy_drop_pct:
        pull_back_price = lowest_price * (1 + buy_pullback_pct)
        if price >= lowest_price * (1 + buy_pullback_pct):
            sol_qty = usdc_per_order / price
            fee = sol_qty * price * fee_pct
            positions.append({
                'qty': sol_qty,
                'entry': price,
                'fee': fee,
                'highest': price  # suivi du plus haut après achat
            })
            log.append({'time': time, 'type': 'BUY', 'price': price, 'fee': fee})
            print(f"[{time}] ✅ Achat à {price:.3f} USDC (qty: {sol_qty:.4f})")
            print(f"drop_from_top:{drop_from_top:.4f} buy_drop_pct:{buy_drop_pct:.4f} price:{price:.4f} lowest_price:{lowest_price:.4f} pull_back_price:{pull_back_price:.4f}\n")
            # Reset cycle de détection
            top_price = None
            lowest_price = None
            last_pos_price = price

    # === Logique de vente (indépendante par position) ===
    to_close = []

    for pos in positions:
        if price > pos['highest']:
            pos['highest'] = price

        gain_pct = price / pos['entry'] - 1
        if gain_pct >= sell_gain_pct:
            if price <= pos['highest'] * (1 - sell_pullback_pct):
                usdc_out = pos['qty'] * price
                fee = usdc_out * fee_pct
                net_gain = usdc_out - fee - (pos['entry'] * pos['qty']) - pos['fee']
                profit += net_gain
                log.append({'time': time, 'type': 'SELL', 'price': price, 'fee': fee})
                print(f"[{time}] 💰 Vente à {price:.3f} (gain: {net_gain:.2f} USDC, total: {profit:.2f})")
                to_close.append(pos)

    for pos in to_close:
        positions.remove(pos)
        

print(f"\n🔚 Profit total simulé : {profit:.2f} USDC")

# ———————
# 📊 GRAPHIQUE
# ———————
df_plot = df.copy()

# Extraire les points d'achat et de vente
buy_points = [entry for entry in log if entry['type'] == 'BUY']
sell_points = [entry for entry in log if entry['type'] == 'SELL']

# Tracer le prix
plt.figure(figsize=(14, 6))
plt.plot(df_plot['timestamp'], df_plot['price'], label='Prix (Close)', linewidth=1)

# Tracer les BUY en vert
if buy_points:
    buy_times = [entry['time'] for entry in buy_points]
    buy_prices = [entry['price'] for entry in buy_points]
    plt.scatter(buy_times, buy_prices, color='cyan', marker='^', label='BUY', zorder=5)

# Tracer les SELL en rouge
if sell_points:
    sell_times = [entry['time'] for entry in sell_points]
    sell_prices = [entry['price'] for entry in sell_points]
    plt.scatter(sell_times, sell_prices, color='red', marker='v', label='SELL', zorder=5)

plt.title("Simulation Grid Scalping - Points d'achat/vente")
plt.xlabel("Temps")
plt.ylabel("Prix USDC")
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.savefig(png_file, dpi=300) 

plt.show()
