def calculate_column_accuracy_and_error(real_values, predictions):
    num_columns = len(real_values[0])
    accuracy_sums = [0] * num_columns
    error_sums = [0] * num_columns
    total_count = len(real_values)

    for real_row, predicted_row in zip(real_values, predictions):
        for i, (real, predicted) in enumerate(zip(real_row, predicted_row)):
            if real == 0:
                continue  # Avoid division by zero
            percent_error = abs(real - predicted) / real * 100
            percent_accuracy = 100 - percent_error
            accuracy_sums[i] += percent_accuracy
            error_sums[i] += percent_error

    # Calculate averages for each column
    average_accuracy = [accuracy_sum / total_count for accuracy_sum in accuracy_sums]
    average_error = [error_sum / total_count for error_sum in error_sums]

    # Calculate overall accuracy and error
    overall_accuracy = sum(accuracy_sums) / (total_count * num_columns)
    overall_error = sum(error_sums) / (total_count * num_columns)

    return average_accuracy, average_error, overall_accuracy, overall_error


def load_file(file_path):
    with open(file_path, "r") as file:
        return [
            list(map(float, line.strip().replace(" ", ",").split(",")))
            for line in file.readlines()
        ]


def main():
    # File names
    file_real = "outputs_real.txt"
    file_hw = "outputs_lstm_hw_U280.txt"
    file_sw = "outputs_lstm_sw.txt"
    output_metrics_file = "prediction_accuracy_metrics.txt"

    # Load data
    real_values = load_file(file_real)
    hw_predictions = load_file(file_hw)
    sw_predictions = load_file(file_sw)

    # Ensure files have matching lengths
    if len(real_values) != len(hw_predictions) or len(real_values) != len(sw_predictions):
        print("Error: Input files have different lengths.")
        return

    # Labels for the columns
    column_labels = ["Open", "Close", "High", "Low", "Volume"]

    # Calculate column-wise accuracy and error for hardware predictions
    hw_column_accuracy, hw_column_error, hw_overall_accuracy, hw_overall_error = calculate_column_accuracy_and_error(
        real_values, hw_predictions
    )

    # Calculate column-wise accuracy and error for software predictions
    sw_column_accuracy, sw_column_error, sw_overall_accuracy, sw_overall_error = calculate_column_accuracy_and_error(
        real_values, sw_predictions
    )

    # Write results to output file
    with open(output_metrics_file, "w") as output_file:
        # Hardware Predictions
        output_file.write(f"Hardware Predictions (U280):\n")
        for label, accuracy, error in zip(column_labels, hw_column_accuracy, hw_column_error):
            output_file.write(f"{label} - Percent Accuracy: {accuracy:.2f}%, Percent Error: {error:.2f}%\n")
        output_file.write(f"Overall Accuracy: {hw_overall_accuracy:.2f}%\n")
        output_file.write(f"Overall Error: {hw_overall_error:.2f}%\n\n")

        # Software Predictions
        output_file.write(f"Software Predictions (Python):\n")
        for label, accuracy, error in zip(column_labels, sw_column_accuracy, sw_column_error):
            output_file.write(f"{label} - Percent Accuracy: {accuracy:.2f}%, Percent Error: {error:.2f}%\n")
        output_file.write(f"Overall Accuracy: {sw_overall_accuracy:.2f}%\n")
        output_file.write(f"Overall Error: {sw_overall_error:.2f}%\n")

    print(f"Prediction accuracy metrics written to {output_metrics_file}")


if __name__ == "__main__":
    main()
