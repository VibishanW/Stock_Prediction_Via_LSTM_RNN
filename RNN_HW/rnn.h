/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

#ifndef RNN_H
#define RNN_H

#include <hls_math.h>

typedef ap_fixed<32, 16> fixed_type;

// Define parameters
#define INPUT_SIZE 5      // Size of each input vector x_t
#define HIDDEN_SIZE 16    // Size of each hidden state vector h_t
#define SEQ_LENGTH 60     // Length of the input sequence

// Declare weight matrices and bias vector for the RNN cell
extern fixed_type W[HIDDEN_SIZE][INPUT_SIZE];   // Input weight matrix
extern fixed_type U[HIDDEN_SIZE][HIDDEN_SIZE];  // Hidden state weight matrix
extern fixed_type b[HIDDEN_SIZE];               // Bias vector

// Function declarations
void rnn_cell(fixed_type x[INPUT_SIZE], fixed_type h_prev[HIDDEN_SIZE], fixed_type h[HIDDEN_SIZE]);
void rnn_sequence(fixed_type x_seq[SEQ_LENGTH][INPUT_SIZE], fixed_type h[HIDDEN_SIZE]);

#endif // RNN_H
