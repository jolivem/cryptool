import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

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
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(5, activation='softmax')  # 5-class classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==== 3. Train ====
model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)
model.save("SOL-5m.h5")

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
plt.figure(figsize=(16, 8))

# 1. Plot the real price
plt.plot(test_prices, label='Prix Crypto', color='black', linewidth=1)

# 2. Overlay class bands at the bottom
n = len(pred_classes)
y_min = np.min(test_prices)
y_range = np.max(test_prices) - y_min

# Calculate vertical position for class bands (flat area below price)
y_offset = y_min - 0.15 * y_range

# Create constant horizontal lines at fixed height per class
for i in range(5):
    indices = np.where(pred_classes == i)[0]
    plt.scatter(indices, [y_offset - i * 0.01] * len(indices), label=f'Classe prédite: {i}', s=10, alpha=0.6)

for i in range(5):
    indices = np.where(y_test_labels == i)[0]
    plt.scatter(indices, [y_offset - i * 0.02] * len(indices), marker='x', s=10, alpha=0.3)

# 3. Labels & style
plt.title("Prix Crypto avec Prédictions de Classes (0 = Forte baisse → 4 = Forte hausse)")
plt.xlabel("Temps")
plt.ylabel("Prix")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# # ==== 6. Plot Predicted vs Actual Classes ====
# # 1. Get the corresponding price series for the test set
# # Keep in mind: predictions start AFTER the lookback window
# test_prices = data[LOOKBACK + split + 1 : LOOKBACK + split + 1 + len(pred_classes)]
# test_prices = test_prices.flatten()

# # 2. Plot
# plt.figure(figsize=(16, 8))

# # Plot price
# plt.plot(test_prices, label='Crypto Price', color='black', linewidth=1)

# # Overlay classifications
# plt.scatter(range(len(pred_classes)), test_prices, c=pred_classes, cmap='viridis', label='Predicted Class', alpha=0.6, s=10)
# plt.scatter(range(len(y_test_labels)), test_prices, c=y_test_labels, cmap='cool', label='Actual Class', alpha=0.3, s=10, marker='x')

# # Labels & legend
# plt.title("Crypto Price with Actual vs Predicted Classifications")
# plt.xlabel("Time Step")
# plt.ylabel("Price")
# plt.colorbar(label='Class (0=Big Down, 4=Big Up)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(14, 6))
# plt.plot(y_test_labels, label='Actual')
# plt.plot(pred_classes, label='Predicted', alpha=0.7)
# plt.title('5-Class Price Movement Prediction')
# plt.xlabel('Time Steps')
# plt.ylabel('Class (0=Big Down → 4=Big Up)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
