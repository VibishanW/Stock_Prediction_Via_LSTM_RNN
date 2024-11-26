/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

#ifndef TESTBENCH_H
#define TESTBENCH_H

#include "rnn.h"          // Include the main RNN header file
#include <iostream>
#include <fstream>

typedef ap_fixed<32, 16> fixed_type;

// Define constants
#define SEQ_LENGTH 60    // Number of time steps (or days) in the input sequence
#define INPUT_SIZE 5     // Number of features per day (Open, Close, High, Low, Volume)
#define HIDDEN_SIZE 16   // Size of the RNN hidden state

// Declare input data array and hidden state
extern fixed_type x_seq[SEQ_LENGTH][INPUT_SIZE];
extern fixed_type h[HIDDEN_SIZE];

// Function declarations
void load_data(const char* filename);
void predict_next_day(fixed_type input_seq[SEQ_LENGTH][INPUT_SIZE], fixed_type next_day_prediction[INPUT_SIZE]);

#endif // TESTBENCH_H
