import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


# ==== CONFIGURATION ====
root_dir = "C:\\Users\\joliv\\Documents\\binance-data\\"
crypto = "SOLUSDC-5m-2025-06"
FILE_PATH = 'binance_data.csv'  # Replace with your CSV file
LOOKBACK = 60  # Number of past timesteps to look at
EPOCHS = 20
BATCH_SIZE = 32

# ==== 1. Load and Prepare Data ====
# Load the CSV
columns = [
    "timespan",	
    "Open",	
    "High",	
    "Low",	
    "close",	
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
df = df[['timespan', 'close']]
df['timespan'] = pd.to_datetime(df['timespan'], unit='us')  # timestamp en ms

data = df['close'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Define 5-class labels based on future % return
def categorize_return(past, future):
    pct_change = (future - past) / past
    if pct_change <= -0.01:
        return 0  # Big Down
    elif pct_change <= -0.002:
        return 1  # Down
    elif pct_change < 0.002:
        return 2  # Flat
    elif pct_change < 0.01:
        return 3  # Up
    else:
        return 4  # Big Up

# Create sequences and labels
X = []
y = []

for i in range(LOOKBACK, len(data_scaled) - 1):
    X.append(data_scaled[i - LOOKBACK:i])
    label = categorize_return(data_scaled[i], data_scaled[i + 1])
    y.append(label)

X = np.array(X)
y = np.array(y)
y_cat = to_categorical(y, num_classes=5)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_cat[:split], y_cat[split:]
y_test_labels = y[split:]  # for evaluation

# ==== 2. Build LSTM Multi-Class Model ====
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
#     Dropout(0.2),
#     LSTM(50),
#     Dropout(0.2),
#     Dense(5, activation='softmax')  # 5-class classification
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # ==== 3. Train ====
# model.fit(
#     X_train, y_train,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     validation_data=(X_test, y_test)
# )
model = load_model("SOL-5m.h5")

# ==== 4. Predict ====
pred_probs = model.predict(X_test)
pred_classes = np.argmax(pred_probs, axis=1)

# ==== 5. Evaluation ====
print("Accuracy:", accuracy_score(y_test_labels, pred_classes))
print(classification_report(y_test_labels, pred_classes, digits=4))


# --- Assumes you have:
# - test_prices: real price (flattened)
# - pred_classes: predicted classes (0–4)
# - y_test_labels: actual classes (0–4)

test_prices = data[LOOKBACK + split + 1 : LOOKBACK + split + 1 + len(pred_classes)]
test_prices = test_prices.flatten()

plt.figure(figsize=(16, 10))

# Tracer les prix
plt.plot(test_prices, label='Prix Crypto', color='black', linewidth=1)

# Configuration pour l’espace vertical des classes (sous le prix)
y_min = np.min(test_prices)
y_range = np.max(test_prices) - y_min
base_line = y_min - 0.05 * y_range
class_spacing = 0.012 * y_range  # petit espacement entre les lignes

# Utiliser un code couleur unique par classe réelle
colors = ['tab:red', 'tab:orange', 'tab:gray', 'tab:green', 'tab:blue']

# Tracer 25 combinaisons (réelle, prédite)
for true_class in range(5):
    for pred_class in range(5):
        indices = np.where((y_test_labels == true_class) & (pred_classes == pred_class))[0]
        y_level = base_line - (true_class * 5 + pred_class) * class_spacing

        # Style : vert si correct, rouge sinon
        color = 'tab:green' if true_class == pred_class else 'tab:red'
        marker = 'o' if true_class == pred_class else 'x'
        alpha = 0.8 if true_class == pred_class else 0.3

        plt.scatter(indices, [y_level] * len(indices),
                    label=f"Réelle {true_class} / Prédite {pred_class}" if true_class == pred_class else None,
                    c=color, marker=marker, s=10, alpha=alpha)

# Légendes & Style
plt.title("Prix Crypto avec Prédictions vs Réalité (25 lignes)")
plt.xlabel("Index temporel")
plt.ylabel("Prix")
plt.legend(loc='upper left', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()







# plt.figure(figsize=(16, 8))

# # 1. Tracer le prix
# plt.plot(test_prices, label='Prix Crypto', color='black', linewidth=1)

# # 2. Affichage des classes en "bandes" en dessous
# n = len(pred_classes)
# y_min = np.min(test_prices)
# y_range = np.max(test_prices) - y_min

# # Position de base sous la courbe pour les classes
# base_line = y_min - 0.05 * y_range
# class_spacing = 0.01 * y_range  # espacement vertical entre les classes

# # 3. Afficher les classes prédites
# for i in range(5):
#     indices = np.where(pred_classes == i)[0]
#     y_level = base_line - i * class_spacing
#     plt.scatter(indices, [y_level] * len(indices), label=f'Prédite: Classe {i}', s=10, alpha=0.6)

# # 4. Afficher les classes réelles
# for i in range(5):
#     indices = np.where(y_test_labels == i)[0]
#     y_level = base_line - i * class_spacing - 0.5 * class_spacing  # décalage en dessous des prédictions
#     plt.scatter(indices, [y_level] * len(indices), label=f'Réelle: Classe {i}', s=10, marker='x', alpha=0.3)

# # 5. Étiquettes et style
# plt.title("Prix Crypto avec Prédictions et Réalité des Classes (0 = Big Down → 4 = Big Up)")
# plt.xlabel("Index temporel")
# plt.ylabel("Prix")
# plt.legend(loc='upper left', fontsize='small')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

