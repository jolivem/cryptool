import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==== CONFIGURATION ====
root_dir = "C:\\Users\\joliv\\Documents\\binance-data\\"
files = [
    "SOLUSDC-5m-2025-04.csv",
    "SOLUSDC-5m-2025-05.csv",
    "SOLUSDC-5m-2025-06.csv",
    "SOLUSDC-5m-2025-07.csv",
]

LOOKBACK = 60
EPOCHS = 50
BATCH_SIZE = 32

# ==== 1. Load and Prepare Data ====

columns = [
    "timespan", "Open", "High", "Low", "close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
]

df_list = []

for fname in files:
    file_path = os.path.join(root_dir, fname)
    df_temp = pd.read_csv(file_path, header=None, names=columns)
    df_temp = df_temp[['timespan', 'Open', 'High', 'Low', 'close', 'Volume']]
    df_list.append(df_temp)

df = pd.concat(df_list, ignore_index=True)
df.sort_values('timespan', inplace=True)
df['timespan'] = pd.to_datetime(df['timespan'], unit='us')
df.reset_index(drop=True, inplace=True)

# ==== 2. Features and Target ====

# Features à utiliser
features = ['Open', 'High', 'Low', 'close', 'Volume']
data = df[features].values
price = df['close'].values.reshape(-1, 1)

# Normalisation des features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Log-returns comme cible
log_returns = np.log(price[1:] / price[:-1])
log_returns = np.vstack([[0], log_returns])  # alignement

# Normalisation de y (StandardScaler car centered)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(log_returns)

# ==== 3. Séquences ====
X, y = [], []
for i in range(LOOKBACK, len(data_scaled) - 1):  # on prédit t+1
    X.append(data_scaled[i - LOOKBACK:i])
    y.append(y_scaled[i + 1])  # prédiction à t+1

X, y = np.array(X), np.array(y)

# ==== 4. Split ====
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==== 5. LSTM Model ====
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1)
])

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()

# ==== 6. Entraînement ====
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)
model.save("SOL-pct2.h5")

#model = load_model("SOL-pct2.h5")

# ==== 7. Prédiction ====
predicted_scaled = model.predict(X_test)
predicted_returns = y_scaler.inverse_transform(predicted_scaled)
true_returns = y_scaler.inverse_transform(y_test)

# ==== 10. Visualisation combinée avec signaux d'achat/vente ====

# Aligner les prix et timestamps avec la taille des prédictions
prices_test = price[-len(true_returns):].reshape(-1)
times_test = df['timespan'][-len(true_returns):].reset_index(drop=True)

# Seuil pour achat/vente (peut être 0 ou un seuil minimum ex: 0.001)
buy_signals = predicted_returns.flatten() > 0
sell_signals = predicted_returns.flatten() < 0

plt.figure(figsize=(14, 6))
plt.plot(times_test, prices_test, label='Prix réel', color='blue')

# Points verts : achat
plt.scatter(
    times_test[buy_signals],
    prices_test[buy_signals],
    marker='^',
    color='green',
    label='Signal Achat',
    s=50
)

# Points rouges : vente
plt.scatter(
    times_test[sell_signals],
    prices_test[sell_signals],
    marker='v',
    color='red',
    label='Signal Vente',
    s=50
)

plt.title("Prix + Signaux d'achat/vente basés sur les log-returns prédits")
plt.xlabel("Temps")
plt.ylabel("Prix")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
