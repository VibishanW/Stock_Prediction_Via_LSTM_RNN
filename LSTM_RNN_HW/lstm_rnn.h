/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

#ifndef LSTM_RNN_H
#define LSTM_RNN_H

#include <hls_math.h>

typedef ap_fixed<32, 16> fixed_type;

// Define constants
#define INPUT_SIZE 5      // Input feature size
#define HIDDEN_SIZE 16    // Hidden state size
#define SEQ_LENGTH 60     // Sequence length

// Weight matrices and biases for LSTM gates
extern fixed_type W_i[HIDDEN_SIZE][INPUT_SIZE];
extern fixed_type U_i[HIDDEN_SIZE][HIDDEN_SIZE];
extern fixed_type b_i[HIDDEN_SIZE];

extern fixed_type W_f[HIDDEN_SIZE][INPUT_SIZE];
extern fixed_type U_f[HIDDEN_SIZE][HIDDEN_SIZE];
extern fixed_type b_f[HIDDEN_SIZE];

extern fixed_type W_c[HIDDEN_SIZE][INPUT_SIZE];
extern fixed_type U_c[HIDDEN_SIZE][HIDDEN_SIZE];
extern fixed_type b_c[HIDDEN_SIZE];

extern fixed_type W_o[HIDDEN_SIZE][INPUT_SIZE];
extern fixed_type U_o[HIDDEN_SIZE][HIDDEN_SIZE];
extern fixed_type b_o[HIDDEN_SIZE];

// Function declarations
void lstm_cell(fixed_type x[INPUT_SIZE], fixed_type h_prev[HIDDEN_SIZE], fixed_type c_prev[HIDDEN_SIZE], 
               fixed_type h[HIDDEN_SIZE], fixed_type c[HIDDEN_SIZE]);
void lstm_sequence(fixed_type x_seq[SEQ_LENGTH][INPUT_SIZE], fixed_type h[HIDDEN_SIZE]);

#endif // LSTM_RNN_H