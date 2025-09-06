import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==== CONFIGURATION ====
root_dir = "C:\\Users\\joliv\\Documents\\binance-data\\"
crypto = "SOLUSDC-1m-2025-06"
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

# df = pd.read_csv(FILE_PATH)

# # Optional: convert timestamp to datetime
# df['timestamp'] = pd.to_datetime(df['timestamp'])

# Use only the 'close' price
data = df['close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
X, y = [], []
for i in range(LOOKBACK, len(data_scaled)):
    X.append(data_scaled[i - LOOKBACK:i])
    y.append(data_scaled[i])
X, y = np.array(X), np.array(y)

# Split into train and test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==== 2. Build the LSTM Model ====
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ==== 3. Train the Model ====
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# ==== 4. Predict ====
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
real = scaler.inverse_transform(y_test)

# ==== 5. Plot the Results ====
plt.figure(figsize=(14, 6))
plt.plot(real, label='Actual Price')
plt.plot(predicted, label='Predicted Price')
plt.title('Crypto Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
