import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model

# ==== CONFIGURATION ====
root_dir = "C:\\Users\\joliv\\Documents\\binance-data\\"
files = [
    "SOLUSDC-5m-2025-04.csv",
    "SOLUSDC-5m-2025-05.csv",
    "SOLUSDC-5m-2025-06.csv",
    "SOLUSDC-5m-2025-07.csv",
]

LOOKBACK = 60
EPOCHS = 20
BATCH_SIZE = 32

# ==== 1. Load and Prepare Data ====

# Colonnes
columns = [
    "timespan", "Open", "High", "Low", "close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
]

# Liste des DataFrames
df_list = []

# Boucle sur les fichiers
for fname in files:
    file_path = os.path.join(root_dir, fname)
    df_temp = pd.read_csv(file_path, header=None, names=columns)
    df_temp = df_temp[['timespan', 'close']]
    df_list.append(df_temp)

# Concaténation
df = pd.concat(df_list, ignore_index=True)


# Optionnel : tri par date si ce n'était pas garanti
df.sort_values('timespan', inplace=True)
# Conversion timestamp
df['timespan'] = pd.to_datetime(df['timespan'], unit='us')
df.reset_index(drop=True, inplace=True)


# file_path = root_dir + crypto + ".csv"
# df = pd.read_csv(file_path, header=None, names=columns)
# df = df[['timespan', 'close']]
# df['timespan'] = pd.to_datetime(df['timespan'], unit='us')

data = df['close'].values.reshape(-1, 1)

# Normalisation des prix
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Calcul des returns (variation relative)
returns = (data[1:] - data[:-1]) / data[:-1]
returns = np.vstack([[0], returns])  # Alignement des dimensions

# ==== 2. Séquences pour LSTM ====
X, y = [], []
for i in range(LOOKBACK, len(data_scaled) - 1):  # -1 car on prédit t+1
    X.append(data_scaled[i - LOOKBACK:i])
    y.append(returns[i + 1])  # on prédit la variation du prochain pas

X, y = np.array(X), np.array(y)

# Division train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==== 3. Modèle LSTM ====
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ==== 4. Entraînement ====
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

model.save("SOL-pct2.h5")

#model = load_model("SOL-5m.h5")
# ==== 5. Prédiction ====

predicted_returns = model.predict(X_test)

# ==== 6. Affichage des returns ====
plt.figure(figsize=(14, 6))
plt.plot(y_test, label='Return réel')
plt.plot(predicted_returns, label='Return prédit')
plt.title("Prédiction de la variation relative (returns)")
plt.xlabel('Temps')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()