#include "lstm_rnn.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

// Xavier initialization
fixed_type xavier_initialization(int input_size, int output_size) {
    float limit = std::sqrt(12.0f / (input_size + output_size)); 
    float random_value = ((float)std::rand() / RAND_MAX) * 2.0f * limit - limit; 
    
    return fixed_type(random_value);
}


// Clipping function
inline fixed_type clip(fixed_type x, fixed_type min_val, fixed_type max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

// Weight matrices and bias initialization
fixed_type W_i[HIDDEN_SIZE][INPUT_SIZE], U_i[HIDDEN_SIZE][HIDDEN_SIZE], b_i[HIDDEN_SIZE];
fixed_type W_f[HIDDEN_SIZE][INPUT_SIZE], U_f[HIDDEN_SIZE][HIDDEN_SIZE], b_f[HIDDEN_SIZE];
fixed_type W_c[HIDDEN_SIZE][INPUT_SIZE], U_c[HIDDEN_SIZE][HIDDEN_SIZE], b_c[HIDDEN_SIZE];
fixed_type W_o[HIDDEN_SIZE][INPUT_SIZE], U_o[HIDDEN_SIZE][HIDDEN_SIZE], b_o[HIDDEN_SIZE];

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
        b_i[i] = xavier_initialization(1, HIDDEN_SIZE);
        b_f[i] = xavier_initialization(1, HIDDEN_SIZE);
        b_c[i] = xavier_initialization(1, HIDDEN_SIZE);
        b_o[i] = xavier_initialization(1, HIDDEN_SIZE);
    }
}

// Save weights to a file
void save_weights(const std::string &filename, fixed_type weights[][INPUT_SIZE], int rows, int cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for saving weights." << std::endl;
        return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << weights[i][j].to_double() << " ";
        }
        file << "\n";
    }

    file.close();
}

// Load weights from a file
bool load_weights(const std::string &filename, fixed_type weights[][INPUT_SIZE], int rows, int cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double value;
            if (!(file >> value)) {
                return false;
            }
            weights[i][j] = fixed_type(value);
        }
    }

    file.close();
    return true;
}

// Activation functions
inline fixed_type sigmoid(fixed_type x) {
    fixed_type result = (fixed_type)1.0 / ((fixed_type)1.0 + hls::exp(-x));
    return result;
}

// LSTM cell implementation with gate outputs
void lstm_cell(fixed_type x[INPUT_SIZE], fixed_type h_prev[HIDDEN_SIZE], fixed_type c_prev[HIDDEN_SIZE],
               fixed_type h[HIDDEN_SIZE], fixed_type c[HIDDEN_SIZE],
               fixed_type i_gate[HIDDEN_SIZE], fixed_type f_gate[HIDDEN_SIZE],
               fixed_type g_gate[HIDDEN_SIZE], fixed_type o_gate[HIDDEN_SIZE]) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        fixed_type input_gate = b_i[i];
        fixed_type forget_gate = b_f[i];
        fixed_type candidate = b_c[i];
        fixed_type output_gate = b_o[i];

        for (int j = 0; j < INPUT_SIZE; j++) {
            input_gate += W_i[i][j] * x[j];
            forget_gate += W_f[i][j] * x[j];
            candidate += W_c[i][j] * x[j];
            output_gate += W_o[i][j] * x[j];
        }

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            input_gate += U_i[i][j] * h_prev[j];
            forget_gate += U_f[i][j] * h_prev[j];
            candidate += U_c[i][j] * h_prev[j];
            output_gate += U_o[i][j] * h_prev[j];
        }

        i_gate[i] = sigmoid(input_gate);
        f_gate[i] = sigmoid(forget_gate);
        g_gate[i] = hls::tanh(candidate);
        o_gate[i] = sigmoid(output_gate);

        c[i] = clip(f_gate[i] * c_prev[i] + i_gate[i] * g_gate[i], -50.0, 50.0);
        h[i] = o_gate[i] * hls::tanh(c[i]);
    }
}

// LSTM sequence processing with gate debugging
void lstm_sequence(fixed_type x_seq[SEQ_LENGTH][INPUT_SIZE], fixed_type h[HIDDEN_SIZE], fixed_type c[HIDDEN_SIZE],
                   fixed_type output_data[INPUT_SIZE],
                   fixed_type i_gate[HIDDEN_SIZE], fixed_type f_gate[HIDDEN_SIZE],
                   fixed_type g_gate[HIDDEN_SIZE], fixed_type o_gate[HIDDEN_SIZE]) {
    for (int t = 0; t < SEQ_LENGTH; t++) {
        lstm_cell(x_seq[t], h, c, h, c, i_gate, f_gate, g_gate, o_gate);
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        output_data[i] = h[i];
    }
}