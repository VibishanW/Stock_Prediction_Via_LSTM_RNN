import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Constants
sequence_length = 5
input_files = [f"data{i}.txt" for i in range(1, 11)]  # Input files: data1.txt to data10.txt
output_file = "predictions.txt"  # Output file for predictions


# Function to load and preprocess data from a file
def load_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    # Ignore the first line and parse the remaining data
    data = [list(map(float, line.strip().split(","))) for line in lines[1:]]
    return np.array(data)


# Function to create sequences for training/testing
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])  # Past `sequence_length` rows
        y.append(data[i])  # Current row
    return np.array(X), np.array(y)


# Function to build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=5))  # Predict Open, Close, High, Low, Volume
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Main process
def main():
    all_predictions = []  # To store predictions for all files

    for file in input_files:
        print(f"Processing {file}...")
        # Load and preprocess the data
        data = load_data(file)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create sequences
        X, y = create_sequences(scaled_data, sequence_length)

        # Split data into training and test sets
        split_ratio = 0.8
        train_size = int(len(X) * split_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train the LSTM model
        model = build_model((X.shape[1], X.shape[2]))
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predict using the last sequence of the file
        last_sequence = scaled_data[-sequence_length:]
        last_sequence = np.reshape(last_sequence, (1, sequence_length, X.shape[2]))
        prediction = model.predict(last_sequence)
        prediction = scaler.inverse_transform(prediction)[0]  # Inverse transform the prediction

        # Store the prediction in the required format
        all_predictions.append(",".join(map(str, prediction)))

    # Write all predictions to the output file
    with open(output_file, "w") as f:
        for prediction in all_predictions:
            f.write(prediction + "\n")

    print(f"Predictions written to {output_file}")


# Run the main function
if __name__ == "__main__":
    main()
