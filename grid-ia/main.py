import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ta.volatility import AverageTrueRange
from ta.volatility import BollingerBands
from ta.trend import MACD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# -----------------------
# ETAPE 1 : Features + Label
# -----------------------

def prepare_features(df):
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std()
    
    # ATR (volatilité)
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()
    
    # MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd_diff()

    # Bollinger Band width
    #df['bb_width'] = df['high'].rolling(20).max() - df['low'].rolling(20).min()

    df.dropna(inplace=True)

    return df

def add_bb_width(df):
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
    return df

def create_target(df, threshold=0.02):
    """
    Crée un label binaire :
    - 1 = marché en range (volatilité faible)
    - 0 = marché en tendance
    """
    df = df.copy()
    df['target'] = (df['bb_width'].rolling(10).mean() < threshold).astype(int)
    print(df['target'].value_counts(normalize=True))
    return df

def train_model(df):
    features = ['returns', 'volatility', 'atr', 'macd', 'bb_width']
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Rapport de performance IA :")
    print(classification_report(y_test, y_pred))
    
    df['range_prediction'] = model.predict(X)
    return df, model

def simulate_grid(df, grid_spacing=0.01, grid_levels=3, base_qty=10):
    capital = 1000
    position = 0
    pnl_list = []
    trades = []

    for i in range(grid_levels, len(df) - 1):

        trade_executed = False

        row = df.iloc[i]
        price = row['close']
        timestamp = row['timestamp']
        range_signal = row['range_prediction']

        if range_signal == 1:
            grid_prices = [price * (1 + grid_spacing * (j - grid_levels // 2)) for j in range(grid_levels)]

            for level_price in grid_prices:
                # Simuler achat
                if level_price < price and capital >= base_qty and not trade_executed:
                    trade_executed = True
                    qty = base_qty / level_price
                    capital -= base_qty
                    position += qty
                    trades.append({
                        "timestamp": timestamp,
                        "type": "BUY",
                        "price": round(level_price, 4),
                        "qty": round(qty, 6),
                        "capital": round(capital, 2),
                        "position": round(position, 6),
                        "total_value": round(capital + position * price, 2)
                    })

                # Simuler vente
                elif level_price > price and position * level_price >= base_qty and not trade_executed:
                    trade_executed = True
                    qty = base_qty / level_price
                    capital += base_qty
                    position -= qty
                    trades.append({
                        "timestamp": timestamp,
                        "type": "SELL",
                        "price": round(level_price, 4),
                        "qty": round(qty, 6),
                        "capital": round(capital, 2),
                        "position": round(position, 6),
                        "total_value": round(capital + position * price, 2)
                    })

        total_value = capital + position * df.iloc[i + 1]['close']
        pnl_list.append(total_value)

    trades_df = pd.DataFrame(trades)
    return pnl_list, trades_df


# def simulate_grid(df, grid_spacing=0.01, grid_levels=3, base_qty=10):
#     """
#     Simule une grille d'achat/vente lorsque IA prédit un range.
#     """
#     capital = 1000
#     position = 0
#     pnl_list = []
#     active_orders = []

#     for i in range(grid_levels, len(df) - 1):
#         row = df.iloc[i]
#         price = row['close']
#         range_signal = row['range_prediction']

#         # Activer la grille seulement si marché en range
#         if range_signal == 1:
#             grid_prices = [price * (1 + grid_spacing * (j - grid_levels // 2)) for j in range(grid_levels)]
#             for level_price in grid_prices:
#                 if level_price < price:  # Simuler achat
#                     if capital >= base_qty:
#                         position += base_qty / level_price
#                         capital -= base_qty
#                 else:  # Simuler vente
#                     if position * level_price >= base_qty:
#                         capital += base_qty
#                         position -= base_qty / level_price

#         total_value = capital + position * df.iloc[i + 1]['close']
#         pnl_list.append(total_value)

#     return pnl_list

def plot_backtest(pnl_list):
    plt.figure(figsize=(12, 6))
    plt.plot(pnl_list, label="Valeur portefeuille")
    plt.title("Backtest Grid IA – SOL/USDC")
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.legend()
    plt.grid(True)
    plt.show()


def load_data(crypto):
    # Chargement du CSV sans en-tête
    columns = [
        "timespan",	
        "open",	
        "high",	
        "low",	
        "close",	
        "volume",	
        "close time",
        "quote asset volume",	
        "number of trades",	
        "taker buy base asset volume",	
        "taker buy quote asset volume",	
        "ignore"
    ]

    root_dir = "C:\\Users\\joliv\\Documents\\binance-data\\"
    kline_file = root_dir + crypto + ".csv"
    png_file = crypto + ".png"
    df = pd.read_csv(kline_file, header=None, names=columns)
    df['timestamp'] = pd.to_datetime(df['timespan'], unit='us')
    df['price'] = df['close']
    #df['volume'] = df['Volume']

    return df

df = load_data("SOLUSDC-15m-2025-06")

# Assume que df est déjà chargé (historique SOL/USDC 15m)
df = prepare_features(df)
df = add_bb_width(df)       

# for t in [0.01, 0.015, 0.02, 0.03]:
#     df2 = create_target(df, threshold=t)
#     print(f"Threshold {t} → % en range :", df2['target'].mean())

df = create_target(df, threshold=0.02)
df, model = train_model(df)
# pnl = simulate_grid(df)
pnl, trades_df = simulate_grid(df)
plot_backtest(pnl)

print(trades_df.head(10))
trades_df.to_csv("trades_simules.csv", index=False)
