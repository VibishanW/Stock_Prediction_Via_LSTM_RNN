#ifndef LSTM_RNN_H
#define LSTM_RNN_H

#include <hls_math.h>
#include <vector>
#include <string>
#include <fstream>

// Define fixed-point type
typedef ap_fixed<64, 32> fixed_type;

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

// LSTM-related functions
void lstm_cell(fixed_type x[INPUT_SIZE], fixed_type h_prev[HIDDEN_SIZE], fixed_type c_prev[HIDDEN_SIZE],
               fixed_type h[HIDDEN_SIZE], fixed_type c[HIDDEN_SIZE],
               fixed_type i_gate[HIDDEN_SIZE], fixed_type f_gate[HIDDEN_SIZE],
               fixed_type g_gate[HIDDEN_SIZE], fixed_type o_gate[HIDDEN_SIZE]);

void lstm_sequence(fixed_type x_seq[SEQ_LENGTH][INPUT_SIZE], fixed_type h[HIDDEN_SIZE],
                   fixed_type c[HIDDEN_SIZE], fixed_type output_data[INPUT_SIZE],
                   fixed_type i_gate[HIDDEN_SIZE], fixed_type f_gate[HIDDEN_SIZE],
                   fixed_type o_gate[HIDDEN_SIZE], fixed_type g_gate[HIDDEN_SIZE]);

void initialize_weights_and_biases();

// Weight management functions
void save_weights(const std::string &filename, fixed_type weights[][INPUT_SIZE], int rows, int cols);
bool load_weights(const std::string &filename, fixed_type weights[][INPUT_SIZE], int rows, int cols);

#endif // LSTM_RNN_H