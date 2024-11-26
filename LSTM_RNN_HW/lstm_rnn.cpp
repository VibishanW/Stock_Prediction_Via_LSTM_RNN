/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

#include "lstm_rnn.h"
#include <cstdlib>
#include <cmath>

// Xavier initialization
fixed_type xavier_initialization(int input_size, int output_size) {
    float limit = std::sqrt(6.0f / (input_size + output_size));
    float random_value = ((float)std::rand() / RAND_MAX) * 2.0f * limit - limit; // Uniform distribution
    return fixed_type(random_value);
}

// Weight matrices and bias initialization using Xavier initialization
fixed_type W_i[HIDDEN_SIZE][INPUT_SIZE];
fixed_type U_i[HIDDEN_SIZE][HIDDEN_SIZE];
fixed_type b_i[HIDDEN_SIZE];

fixed_type W_f[HIDDEN_SIZE][INPUT_SIZE];
fixed_type U_f[HIDDEN_SIZE][HIDDEN_SIZE];
fixed_type b_f[HIDDEN_SIZE];

fixed_type W_c[HIDDEN_SIZE][INPUT_SIZE];
fixed_type U_c[HIDDEN_SIZE][HIDDEN_SIZE];
fixed_type b_c[HIDDEN_SIZE];

fixed_type W_o[HIDDEN_SIZE][INPUT_SIZE];
fixed_type U_o[HIDDEN_SIZE][HIDDEN_SIZE];
fixed_type b_o[HIDDEN_SIZE];

// Initialize weights and biases
void initialize_weights_and_biases() {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            W_i[i][j] = xavier_initialization(INPUT_SIZE, HIDDEN_SIZE);
            W_f[i][j] = xavier_initialization(INPUT_SIZE, HIDDEN_SIZE);
            W_c[i][j] = xavier_initialization(INPUT_SIZE, HIDDEN_SIZE);
            W_o[i][j] = xavier_initialization(INPUT_SIZE, HIDDEN_SIZE);
        }
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            U_i[i][j] = xavier_initialization(HIDDEN_SIZE, HIDDEN_SIZE);
            U_f[i][j] = xavier_initialization(HIDDEN_SIZE, HIDDEN_SIZE);
            U_c[i][j] = xavier_initialization(HIDDEN_SIZE, HIDDEN_SIZE);
            U_o[i][j] = xavier_initialization(HIDDEN_SIZE, HIDDEN_SIZE);
        }
        b_i[i] = 0;
        b_f[i] = 0;
        b_c[i] = 0;
        b_o[i] = 0;
    }
}

inline fixed_type sigmoid(fixed_type x) {
    return (fixed_type)1.0 / ((fixed_type)1.0 + hls::exp(-x));
}

// LSTM cell implementation
void lstm_cell(fixed_type x[INPUT_SIZE], fixed_type h_prev[HIDDEN_SIZE], fixed_type c_prev[HIDDEN_SIZE],
               fixed_type h[HIDDEN_SIZE], fixed_type c[HIDDEN_SIZE]) {
    #pragma HLS PIPELINE II=1

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        fixed_type input_gate = b_i[i];
        fixed_type forget_gate = b_f[i];
        fixed_type candidate = b_c[i];
        fixed_type output_gate = b_o[i];

        for (int j = 0; j < INPUT_SIZE; j++) {
            #pragma HLS UNROLL
            input_gate += W_i[i][j] * x[j];
            forget_gate += W_f[i][j] * x[j];
            candidate += W_c[i][j] * x[j];
            output_gate += W_o[i][j] * x[j];
        }

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            #pragma HLS UNROLL
            input_gate += U_i[i][j] * h_prev[j];
            forget_gate += U_f[i][j] * h_prev[j];
            candidate += U_c[i][j] * h_prev[j];
            output_gate += U_o[i][j] * h_prev[j];
        }

        input_gate = sigmoid(input_gate);
        forget_gate = sigmoid(forget_gate);
        candidate = hls::tanh(candidate);
        output_gate = sigmoid(output_gate);

        c[i] = forget_gate * c_prev[i] + input_gate * candidate;
        h[i] = output_gate * hls::tanh(c[i]);
    }
}

// LSTM sequence processing
void lstm_sequence(fixed_type x_seq[SEQ_LENGTH][INPUT_SIZE], fixed_type h[HIDDEN_SIZE]) {
    #pragma HLS ARRAY_PARTITION variable=W_i complete dim=2
    #pragma HLS ARRAY_PARTITION variable=U_i complete dim=2
    #pragma HLS ARRAY_PARTITION variable=b_i complete dim=1
    #pragma HLS ARRAY_PARTITION variable=h complete dim=1

    fixed_type c[HIDDEN_SIZE] = {0};
    fixed_type h_prev[HIDDEN_SIZE] = {0};
    fixed_type c_prev[HIDDEN_SIZE] = {0};

    for (int t = 0; t < SEQ_LENGTH; t++) {
        lstm_cell(x_seq[t], h_prev, c_prev, h, c);

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            h_prev[i] = h[i];
            c_prev[i] = c[i];
        }
    }
}