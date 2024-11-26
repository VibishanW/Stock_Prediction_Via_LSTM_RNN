/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

#include "lstm_rnn.h"

// Weight matrices and bias initialization
fixed_type W[HIDDEN_SIZE][INPUT_SIZE] = {
    {0.0543, -0.0234, 0.0675, -0.0812, 0.0398},
    {-0.0457, 0.0321, -0.0104, 0.0932, -0.0671},
    {0.0223, -0.0785, 0.0416, -0.0123, 0.0887},
    {-0.0371, 0.0465, -0.0568, 0.0721, -0.0352}
};

fixed_type U[HIDDEN_SIZE][HIDDEN_SIZE] = {
    {0.0625, -0.0492, 0.0108, -0.0715},
    {-0.0537, 0.0304, -0.0417, 0.0552},
    {0.0185, -0.0673, 0.0241, -0.0811},
    {0.0456, -0.0327, 0.0732, -0.0554}
};

fixed_type b[HIDDEN_SIZE] = {0.0, 0.01, -0.02, 0.03};

// Function to compute one RNN cell step
void rnn_cell(fixed_type x[INPUT_SIZE], fixed_type h_prev[HIDDEN_SIZE], fixed_type h[HIDDEN_SIZE]) {
    #pragma HLS PIPELINE II=1
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        fixed_type sum = b[i];

        // Compute W * x_t
        for (int j = 0; j < INPUT_SIZE; j++) {
            #pragma HLS UNROLL
            sum += W[i][j] * x[j];
        }

        // Compute U * h_{t-1}
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            #pragma HLS UNROLL
            sum += U[i][j] * h_prev[j];
        }

        // Apply the tanh activation function
        h[i] = hls::tanh(sum);
    }
}

// Function to process an entire sequence with the RNN
void rnn_sequence(fixed_type x_seq[SEQ_LENGTH][INPUT_SIZE], fixed_type h[HIDDEN_SIZE]) {
    #pragma HLS ARRAY_PARTITION variable=W complete dim=2
    #pragma HLS ARRAY_PARTITION variable=U complete dim=2
    #pragma HLS ARRAY_PARTITION variable=b complete dim=1
    #pragma HLS ARRAY_PARTITION variable=h complete dim=1

    fixed_type h_prev[HIDDEN_SIZE] = {0}; // Initialize hidden state to zero

    // Process each time step in the sequence
    for (int t = 0; t < SEQ_LENGTH; t++) {
        fixed_type x[INPUT_SIZE];

        // Load input for the current time step
        for (int i = 0; i < INPUT_SIZE; i++) {
            x[i] = x_seq[t][i];
        }

        // Compute the RNN cell for the current time step
        rnn_cell(x, h_prev, h);

        // Update h_prev for the next time step
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            h_prev[i] = h[i];
        }
    }
}


