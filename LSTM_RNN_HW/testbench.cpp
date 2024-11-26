/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include "lstm_rnn.h"

int main() {
    const char* input_file = "data.txt";
    const char* golden_output_file = "out.gold.dat";

    fixed_type input_data[SEQ_LENGTH][INPUT_SIZE];
    fixed_type output_data[INPUT_SIZE] = {0};
    fixed_type hidden_state[HIDDEN_SIZE] = {0};

    // Read input data
    std::ifstream infile(input_file);
    for (int t = 0; t < SEQ_LENGTH; t++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            infile >> input_data[t][i];
        }
    }
    infile.close();

    std::cout << "Input Data:" << std::endl;
    for (int t = 0; t < SEQ_LENGTH; t++) {
        std::cout << "Timestep " << t << ": ";
        for (int i = 0; i < INPUT_SIZE; i++) {
            std::cout << input_data[t][i] << " ";
        }
        std::cout << std::endl;
    }

    // Process input sequence
    lstm_sequence(input_data, hidden_state, output_data);

    // Read golden output
    std::ifstream golden_file(golden_output_file);
    fixed_type golden_data[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
        golden_file >> golden_data[i];
    }
    golden_file.close();

    // Compare output
    int error_count = 0;
    for (int i = 0; i < INPUT_SIZE; i++) {
        if (std::fabs((float)(output_data[i] - golden_data[i])) > 0.0001) {
            std::cerr << "Error: Mismatch at index " << i
                      << " | Expected: " << golden_data[i]
                      << " | Got: " << output_data[i] << std::endl;
            error_count++;
        }
    }

    if (error_count == 0) {
        std::cout << "TEST PASSED" << std::endl;
    } else {
        std::cout << "TEST FAILED with " << error_count << " errors" << std::endl;
    }

    return error_count;
}