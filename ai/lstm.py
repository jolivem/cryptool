import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Charger les données
df = pd.read_csv("merged_crypto_data.csv", index_col="timespan", parse_dates=True)

# 2. Normaliser les données
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 3. Créer les séquences (ex: 24 pas de temps -> 1 jour si données horaires)
def create_sequences(data, sequence_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon):
        X_seq = data[i:i+sequence_length, :-1]  # BTC, ETH, SOL
        y_target = data[i+sequence_length + forecast_horizon - 1, -1]  # PEPE à t+n
        X.append(X_seq)
        y.append(y_target)
    return np.array(X), np.array(y)

# Paramètres
sequence_length = 60          # input: 60 min
forecast_horizon = 2         # target: PEPE à t + 2 min

# Séparation des features (X) et de la target (y)
X, y = create_sequences(scaled_data, sequence_length, forecast_horizon)

# 4. Séparer train/test (80/20)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Construire le modèle LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 6. Entraîner le modèle
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 7. Prédictions
y_pred = model.predict(X_test)

# 8. Inverser la normalisation pour PEPE uniquement
# On doit reconstruire un array complet pour inverser correctement
def invert_scaling(y_scaled, X_reference, scaler, target_index=-1):
    y_full = np.concatenate([X_reference, y_scaled], axis=1)
    y_inverted = scaler.inverse_transform(y_full)[:, target_index]
    return y_inverted

X_test_last = X_test[:, -1, :]  # Dernier pas de chaque séquence
y_pred_inv = invert_scaling(y_pred, X_test_last, scaler, target_index=-1)
y_test_inv = invert_scaling(y_test.reshape(-1, 1), X_test_last, scaler, target_index=-1)

# 9. Visualiser les résultats
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Réel (PEPE)")
plt.plot(y_pred_inv, label="Prédit (PEPE)")
plt.title("Prédiction du prix de PEPE avec LSTM")
plt.legend()
plt.show()
