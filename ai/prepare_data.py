import pandas as pd



root_dir = "C:\\Users\\joliv\\Documents\\binance-data\\"


def load_crypto_data(crypto, name):

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

    file_path = root_dir + crypto + ".csv"
    df = pd.read_csv(file_path, header=None, names=columns)
    df = df[['timespan', 'Close']]
    df['timespan'] = pd.to_datetime(df['timespan'], unit='us')  # timestamp en ms
    df = df.rename(columns={'Close': f'{name}_Close'})
    return df.set_index('timespan')

# Charger les données

print("BTC...")
df_btc = load_crypto_data("BTCUSDC-1m-2025-06", "BTC")
print("ETH...")
df_eth = load_crypto_data("ETHUSDC-1m-2025-06", "ETH")
print("SOL...")
df_sol = load_crypto_data("SOLUSDC-1m-2025-06", "SOL")
print("PEPE...")
df_pepe = load_crypto_data("PEPEUSDC-1m-2025-06", "PEPE")

# Fusionner toutes les données sur le timestamp
df_all = df_btc.join([df_eth, df_sol, df_pepe], how='inner')

# Supprimer les lignes avec des valeurs manquantes (juste au cas où)
df_all = df_all.dropna()

# Affichage d'un aperçu
#print(df_all.head())

# Sauvegarde pour la suite
df_all.to_csv("merged_crypto_data.csv")
