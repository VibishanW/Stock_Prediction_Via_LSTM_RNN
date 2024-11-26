/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

#include "testbench.h"
#include <iostream>
#include <fstream>
#include <string>

// Define input data sequence and hidden state arrays
fixed_type x_seq[SEQ_LENGTH][INPUT_SIZE];  // Input data sequence array
fixed_type h[HIDDEN_SIZE];                 // Hidden state array to store the RNN's final output

// Define feature names for labeling
const char* feature_names[INPUT_SIZE] = {"Open", "Close", "High", "Low", "Volume"};

// Function to load data from data.txt into x_seq array
void load_data(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Read data line by line into x_seq
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            file >> x_seq[i][j];
            if (file.fail()) {
                std::cerr << "Error reading data at position (" << i << ", " << j << ")" << std::endl;
                file.close();
                return;
            }
        }
    }
    file.close();
    std::cout << "Data successfully loaded from " << filename << std::endl;
}

// Function to predict the next day's values using the RNN
void predict_next_day(fixed_type input_seq[SEQ_LENGTH][INPUT_SIZE], fixed_type next_day_prediction[INPUT_SIZE]) {
    // Run the RNN on the input sequence
    rnn_sequence(input_seq, h);

    // Assuming that the final hidden state (h) can serve as a basis for predicting the next day's values.
    for (int i = 0; i < INPUT_SIZE; i++) {
        next_day_prediction[i] = h[i % HIDDEN_SIZE];  // Map hidden state to each feature for next day
    }
}

// Function to verify golden output
bool compare_files(const char* predicted_file, const char* golden_file) {
    std::ifstream pred(predicted_file);
    std::ifstream gold(golden_file);

    if (!pred.is_open() || !gold.is_open()) {
        std::cerr << "Error: Could not open one of the files for comparison." << std::endl;
        return false;
    }

    std::string pred_line, gold_line;
    int line_num = 0;
    bool match = true;

    while (std::getline(pred, pred_line) && std::getline(gold, gold_line)) {
        line_num++;
        if (pred_line != gold_line) {
            std::cerr << "Mismatch at line " << line_num << ":" << std::endl;
            std::cerr << "Predicted: " << pred_line << std::endl;
            std::cerr << "Golden:    " << gold_line << std::endl;
            match = false;
        }
    }

    // Check if files have different lengths
    if ((pred.eof() != gold.eof()) || (!pred.eof() && gold.eof()) || (pred.eof() && !gold.eof())) {
        std::cerr << "Files have different lengths." << std::endl;
        match = false;
    }

    pred.close();
    gold.close();

    return match;
}

int main() {
    // Load data from file
    load_data("data.txt");

    // Array to store the next day's predicted values for each feature
    fixed_type next_day_prediction[INPUT_SIZE];

    // Call the prediction function
    predict_next_day(x_seq, next_day_prediction);

    // Write the predicted values for the next day to an output file
    std::ofstream output_file("prediction_output.txt");
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file." << std::endl;
        return 1;
    }
    output_file << "Predicted values for the next day:" << std::endl;
    for (int i = 0; i < INPUT_SIZE; i++) {
        output_file << feature_names[i] << ": " << next_day_prediction[i] << std::endl;
    }
    output_file.close();
    std::cout << "Predicted values written to prediction_output.txt" << std::endl;

    // Compare prediction output with golden output file
    if (compare_files("prediction_output.txt", "out.gold.dat")) {
        std::cout << "Test Passed: Prediction matches the golden output." << std::endl;
    } else {
        std::cerr << "Test Failed: Prediction does not match the golden output." << std::endl;
    }

    return 0;
}
