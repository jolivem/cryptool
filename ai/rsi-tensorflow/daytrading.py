import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go

# Sim some financial data
np.random.seed(7)
dates = pd.date_range(start='2024-01-01', periods=100)
data = np.random.randn(100).cumsum()
df = pd.DataFrame(data, columns=['Price'], index=dates)

# Prepare dataset for training
window_size = 5
features = []
labels = []

for i in range(window_size, len(df)):
    features.append(df.iloc[i-window_size:i, 0])
    labels.append(df.iloc[i, 0])

features, labels = np.array(features), np.array(labels)

# Split the dataset into training & testing
split = int(len(features) * 0.8)
train_features, test_features = features[:split], features[split:]
train_labels, test_labels = labels[:split], labels[split:]

# Build a simple LSTM model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(window_size,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_features, train_labels, epochs=10, batch_size=1)

# Make predictions
predictions = model.predict(test_features)

# Visualize the results
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates[split+window_size:], y=test_labels, mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=dates[split+window_size:], y=predictions.flatten(), mode='lines', name='Predicted'))

fig.update_layout(title='Financial Data Prediction',
                                  xaxis_title='Date',
                                  yaxis_title='Price')

fig.show()