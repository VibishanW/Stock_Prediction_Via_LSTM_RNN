import sys

# Redirect all output to a log file with UTF-8 encoding
output_file = open("output.log", "w", encoding="utf-8")
sys.stdout = output_file
sys.stderr = output_file

import yfinance as yf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define the ticker symbol
ticker = "SPY"

# Download SPY data from Yahoo Finance
data = yf.download(ticker, start="2020-01-01", end="2021-01-01", interval="1d")

# Select the relevant columns and normalize each one
data = data[['Open', 'Close', 'High', 'Low', 'Volume']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences of 60 days to predict the next day's values
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])  # Collecting past 60 days for each feature
    y.append(scaled_data[i])  # Target is the next day's values for all features

X, y = np.array(X), np.array(y)

# Step 2: Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=5))  # Output layer to predict next day's Open, Close, High, Low, and Volume

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 3: Train the Model
# Split data into training and test sets (80% train, 20% test)
split_ratio = 0.8
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 4: Make Predictions and Evaluate the Model
# Predict using the test set
predicted_data = model.predict(X_test)

# Inverse transform the predictions and the actual values
predicted_data = scaler.inverse_transform(predicted_data)
real_data = scaler.inverse_transform(y_test)

# Plot the results for each feature
plt.figure(figsize=(14, 8))

# Close Prices
plt.subplot(2, 3, 1)
plt.plot(real_data[:, 1], color='red', label='Real Close Price')
plt.plot(predicted_data[:, 1], color='blue', label='Predicted Close Price')
plt.title('Close Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# Open Prices
plt.subplot(2, 3, 2)
plt.plot(real_data[:, 0], color='red', label='Real Open Price')
plt.plot(predicted_data[:, 0], color='blue', label='Predicted Open Price')
plt.title('Open Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# High Prices
plt.subplot(2, 3, 3)
plt.plot(real_data[:, 2], color='red', label='Real High Price')
plt.plot(predicted_data[:, 2], color='blue', label='Predicted High Price')
plt.title('High Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# Low Prices
plt.subplot(2, 3, 4)
plt.plot(real_data[:, 3], color='red', label='Real Low Price')
plt.plot(predicted_data[:, 3], color='blue', label='Predicted Low Price')
plt.title('Low Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# Volume
plt.subplot(2, 3, 5)
plt.plot(real_data[:, 4], color='red', label='Real Volume')
plt.plot(predicted_data[:, 4], color='blue', label='Predicted Volume')
plt.title('Volume Prediction')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.legend()

plt.tight_layout()
plt.savefig("predictions_plot.png")  # Save the plot as an image instead of showing it

# Step 5: Predict the Next Day's Values
# Use the last 60 days from the dataset to predict the next day's values
last_sequence = scaled_data[-sequence_length:]
last_sequence = np.reshape(last_sequence, (1, sequence_length, X.shape[2]))

# Make the prediction and inverse transform
next_day_prediction = model.predict(last_sequence)
next_day_prediction = scaler.inverse_transform(next_day_prediction)

# Write the prediction to a file
with open("output.txt", "a", encoding="utf-8") as file:
    file.write("\nPredicted values for the next day (Open, Close, High, Low, Volume):\n")
    file.write(str(next_day_prediction[0]))

# Close the output redirection file at the end
output_file.close()
