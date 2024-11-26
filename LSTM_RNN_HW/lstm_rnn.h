/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

#ifndef LSTM_RNN_H
#define LSTM_RNN_H

#include <hls_math.h>

typedef ap_fixed<32, 16> fixed_type;

// Define parameters
#define INPUT_SIZE 5      // Size of each input vector x_t
#define HIDDEN_SIZE 16    // Size of each hidden state vector h_t
#define SEQ_LENGTH 60     // Length of the input sequence

// Declare weight matrices and bias vectors for the LSTM gates
extern fixed_type Wi[HIDDEN_SIZE][INPUT_SIZE];  // Input gate weight matrix (input)
extern fixed_type Ui[HIDDEN_SIZE][HIDDEN_SIZE]; // Input gate weight matrix (hidden)
extern fixed_type bi[HIDDEN_SIZE];             // Input gate bias vector

extern fixed_type Wf[HIDDEN_SIZE][INPUT_SIZE];  // Forget gate weight matrix (input)
extern fixed_type Uf[HIDDEN_SIZE][HIDDEN_SIZE]; // Forget gate weight matrix (hidden)
extern fixed_type bf[HIDDEN_SIZE];             // Forget gate bias vector

extern fixed_type Wo[HIDDEN_SIZE][INPUT_SIZE];  // Output gate weight matrix (input)
extern fixed_type Uo[HIDDEN_SIZE][HIDDEN_SIZE]; // Output gate weight matrix (hidden)
extern fixed_type bo[HIDDEN_SIZE];             // Output gate bias vector

extern fixed_type Wc[HIDDEN_SIZE][INPUT_SIZE];  // Cell state weight matrix (input)
extern fixed_type Uc[HIDDEN_SIZE][HIDDEN_SIZE]; // Cell state weight matrix (hidden)
extern fixed_type bc[HIDDEN_SIZE];             // Cell state bias vector

// Function declarations
void lstm_cell(
    fixed_type x[INPUT_SIZE], 
    fixed_type h_prev[HIDDEN_SIZE], 
    fixed_type c_prev[HIDDEN_SIZE], 
    fixed_type h[HIDDEN_SIZE], 
    fixed_type c[HIDDEN_SIZE]
);

void lstm_sequence(
    fixed_type x_seq[SEQ_LENGTH][INPUT_SIZE], 
    fixed_type h[HIDDEN_SIZE]
);

#endif // LSTM_RNN_H