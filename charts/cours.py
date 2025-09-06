import pandas as pd
import plotly.graph_objects as go
import os

# 1. Liste manuelle des fichiers à inclure
root_dir = "C:/Users/joliv/Documents/binance-data/"
file_list = [
    "SOLUSDC-1m-2025-06.csv",
    "ETHUSDC-1m-2025-06.csv",
    "PEPEUSDC-1m-2025-06.csv"
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
# 4. Plotly chart
fig = go.Figure()

for col in df_base100.columns:
    if col != 'timestamp':
        fig.add_trace(go.Scatter(x=df_base100['timestamp'], y=df_base100[col],
                                 mode='lines', name=col))

fig.update_layout(
    title='Crypto Performance (Base 100)',
    xaxis_title='Date',
    yaxis_title='Index (Base 100)',
    hovermode='x unified',
    template='plotly_dark',
    xaxis=dict(rangeslider_visible=True),
    height=600
)

fig.show()