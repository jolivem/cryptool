import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Liste manuelle des fichiers à inclure
root_dir = "C:/Users/joliv/Documents/binance-data/"
file_list = [
    "BTCUSDC-1h-2025-06.csv",
    "ETHUSDC-1h-2025-06.csv",
    "PEPEUSDC-1h-2025-06.csv"
]

columns = [
    "timestamp",	
    "open",	
    "high",	
    "low",	
    "close",	
    "Volume",	
    "Close time",
    "Quote asset volume",	
    "Number of trades",	
    "Taker buy base asset volume",	
    "Taker buy quote asset volume",	
    "Ignore"
]
# 2. Lire et fusionner les fichiers
df_merged = None

for file in file_list:
    name = os.path.splitext(os.path.basename(file))[0]  # ex: BTCUSDC
    path = root_dir + file
    df = pd.read_csv(path, parse_dates=['timestamp'], header=None, names=columns)[['timestamp', 'close']]
    df = df.rename(columns={'close': name})
    
    if df_merged is None:
        df_merged = df
    else:
        df_merged = pd.merge(df_merged, df, on='timestamp', how='inner')

# 3. Rebasage à 100
df_base100 = df_merged.copy()
for col in df_base100.columns:
    if col != 'timestamp':
        df_base100[col] = df_base100[col] / df_base100[col].iloc[0] * 100

# 4. Tracé
plt.figure(figsize=(14, 6))
for col in df_base100.columns:
    if col != 'timestamp':
        plt.plot(df_base100['timestamp'], df_base100[col], label=col)

plt.title('Performance des cryptos (Base 100)')
plt.xlabel('Date')
plt.ylabel('Indice Base 100')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
