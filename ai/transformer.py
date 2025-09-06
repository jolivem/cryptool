import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# === 1. Charger et agréger les données ===
df = pd.read_csv("merged_crypto_data.csv", index_col="timespan", parse_dates=True)

# === 2. Normalisation ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# === 3. Créer les séquences avec target à t+n ===
def create_sequences(data, sequence_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon):
        X.append(data[i:i+sequence_length, :-1])  # features
        y.append(data[i+sequence_length + forecast_horizon - 1, -1])  # PEPE
    return np.array(X), np.array(y)

sequence_length = 120
forecast_horizon = 2
X, y = create_sequences(scaled_data, sequence_length, forecast_horizon)

# === 4. Train / Test split ===
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# === 5. Modèle Transformer ===
def build_transformer_model(seq_len, n_features):
    inputs = layers.Input(shape=(seq_len, n_features))  # (60, 3)

    # Projection en dimension 64
    x = layers.Dense(64)(inputs)  # → (60, 64)

    # Embedding positionnel
    pos = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=64)(pos)
    x = x + pos_emb  # maintenant compatible : (60, 64) + (60, 64)

    # Bloc Transformer
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Feed-forward
    ff = layers.Dense(64, activation='relu')(x)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    # Global pooling et sortie
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(1)(x)

    model = models.Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model


model = build_transformer_model(X.shape[1], X.shape[2])
model.summary()

# === 6. Entraînement ===
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# === 7. Prédiction ===
y_pred = model.predict(X_test)

# === 8. Inverser la normalisation pour PEPE uniquement
X_test_last = X_test[:, -1, :]
X_ref_pred = np.concatenate([X_test_last, y_pred], axis=1)
X_ref_true = np.concatenate([X_test_last, y_test.reshape(-1, 1)], axis=1)
y_pred_inv = scaler.inverse_transform(X_ref_pred)[:, -1]
y_test_inv = scaler.inverse_transform(X_ref_true)[:, -1]

# === 9. Visualisation
plt.figure(figsize=(14, 6))
plt.plot(y_test_inv, label="PEPE réel")
plt.plot(y_pred_inv, label=f"PEPE prédit à t+{forecast_horizon} min")
plt.title("Prédiction Transformer - PEPE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
