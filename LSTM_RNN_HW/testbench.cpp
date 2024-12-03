#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <sstream>
#include <algorithm>
#include "lstm_rnn.h"

// Global weight definitions
extern fixed_type W_i[HIDDEN_SIZE][INPUT_SIZE], U_i[HIDDEN_SIZE][HIDDEN_SIZE], b_i[HIDDEN_SIZE];
extern fixed_type W_f[HIDDEN_SIZE][INPUT_SIZE], U_f[HIDDEN_SIZE][HIDDEN_SIZE], b_f[HIDDEN_SIZE];
extern fixed_type W_c[HIDDEN_SIZE][INPUT_SIZE], U_c[HIDDEN_SIZE][HIDDEN_SIZE], b_c[HIDDEN_SIZE];
extern fixed_type W_o[HIDDEN_SIZE][INPUT_SIZE], U_o[HIDDEN_SIZE][HIDDEN_SIZE], b_o[HIDDEN_SIZE];

// Function to save weights to a file
void save_weights_to_file() {
    std::ofstream weight_file("weights.dat", std::ios::binary);
    if (!weight_file.is_open()) {
        std::cerr << "Error: Could not open weights file for saving!" << std::endl;
        return;
    }
    weight_file.write(reinterpret_cast<const char *>(W_i), sizeof(W_i));
    weight_file.write(reinterpret_cast<const char *>(U_i), sizeof(U_i));
    weight_file.write(reinterpret_cast<const char *>(b_i), sizeof(b_i));
    weight_file.write(reinterpret_cast<const char *>(W_f), sizeof(W_f));
    weight_file.write(reinterpret_cast<const char *>(U_f), sizeof(U_f));
    weight_file.write(reinterpret_cast<const char *>(b_f), sizeof(b_f));
    weight_file.write(reinterpret_cast<const char *>(W_c), sizeof(W_c));
    weight_file.write(reinterpret_cast<const char *>(U_c), sizeof(U_c));
    weight_file.write(reinterpret_cast<const char *>(b_c), sizeof(b_c));
    weight_file.write(reinterpret_cast<const char *>(W_o), sizeof(W_o));
    weight_file.write(reinterpret_cast<const char *>(U_o), sizeof(U_o));
    weight_file.write(reinterpret_cast<const char *>(b_o), sizeof(b_o));
    weight_file.close();
}

// Function to initialize or load weights
void initialize_or_load_weights() {
    std::ifstream weight_file("weights.dat", std::ios::binary);
    if (weight_file.is_open()) {
        weight_file.read(reinterpret_cast<char *>(W_i), sizeof(W_i));
        weight_file.read(reinterpret_cast<char *>(U_i), sizeof(U_i));
        weight_file.read(reinterpret_cast<char *>(b_i), sizeof(b_i));
        weight_file.read(reinterpret_cast<char *>(W_f), sizeof(W_f));
        weight_file.read(reinterpret_cast<char *>(U_f), sizeof(U_f));
        weight_file.read(reinterpret_cast<char *>(b_f), sizeof(b_f));
        weight_file.read(reinterpret_cast<char *>(W_c), sizeof(W_c));
        weight_file.read(reinterpret_cast<char *>(U_c), sizeof(U_c));
        weight_file.read(reinterpret_cast<char *>(b_c), sizeof(b_c));
        weight_file.read(reinterpret_cast<char *>(W_o), sizeof(W_o));
        weight_file.read(reinterpret_cast<char *>(U_o), sizeof(U_o));
        weight_file.read(reinterpret_cast<char *>(b_o), sizeof(b_o));
        weight_file.close();
    } else {
        std::cout << "Weights file not found. Initializing new weights..." << std::endl;
        initialize_weights_and_biases();
        save_weights_to_file();
    }
}

// Function to normalize data
void normalize_data(const std::vector<std::vector<double>> &raw_data, std::vector<std::vector<fixed_type>> &normalized_data, std::vector<double> &means, std::vector<double> &std_devs) {
    int num_features = raw_data[0].size();
    int num_samples = raw_data.size();

    means.resize(num_features, 0.0);
    std_devs.resize(num_features, 0.0);

    for (const auto &row : raw_data) {
        for (int i = 0; i < num_features; ++i) {
            means[i] += row[i];
        }
    }
    for (int i = 0; i < num_features; ++i) {
        means[i] /= num_samples;
    }

    for (const auto &row : raw_data) {
        for (int i = 0; i < num_features; ++i) {
            std_devs[i] += (row[i] - means[i]) * (row[i] - means[i]);
        }
    }
    for (int i = 0; i < num_features; ++i) {
        std_devs[i] = std::sqrt(std_devs[i] / num_samples);
    }

    for (const auto &row : raw_data) {
        std::vector<fixed_type> normalized_row;
        for (int i = 0; i < num_features; ++i) {
            normalized_row.push_back((row[i] - means[i]) / std_devs[i]);
        }
        normalized_data.push_back(normalized_row);
    }
}

// Function to load data from data.txt
void load_data(const std::string &file_name, int &prediction_days, std::vector<std::vector<double>> &raw_data) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_name << std::endl;
        return;
    }

    std::string line;
    if (std::getline(file, line)) {
        prediction_days = std::stoi(line);
    }

    std::getline(file, line); // Skip header
    std::getline(file, line); // Skip tickers
    std::getline(file, line); // Skip dates

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        std::string value;

        int column = 0;
        while (std::getline(iss, value, ',')) {
            if (column++ == 0) continue;
            row.push_back(std::stod(value));
        }
        if (!row.empty()) {
            raw_data.push_back(row);
        }
    }
    file.close();
}

int main() {
    const std::string file_name = "data.txt";
    const std::string output_file_name = "out.dat";
    const std::string debug_file_name = "debug_output.dat";

    initialize_or_load_weights();

    int prediction_days = 0;
    std::vector<std::vector<double>> raw_data;
    load_data(file_name, prediction_days, raw_data);

    if (raw_data.empty()) {
        std::cerr << "Error: No data loaded!" << std::endl;
        return -1;
    }

    std::vector<std::vector<fixed_type>> normalized_data;
    std::vector<double> means, std_devs;
    normalize_data(raw_data, normalized_data, means, std_devs);

    fixed_type input_seq[SEQ_LENGTH][INPUT_SIZE] = {0};
    for (int i = 0; i < SEQ_LENGTH && i < normalized_data.size(); ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            input_seq[i][j] = normalized_data[i][j];
        }
    }

    fixed_type h[HIDDEN_SIZE] = {0};
    fixed_type c[HIDDEN_SIZE] = {0};
    fixed_type output_data[INPUT_SIZE] = {0};

    fixed_type i_gate[HIDDEN_SIZE], f_gate[HIDDEN_SIZE], o_gate[HIDDEN_SIZE], g_gate[HIDDEN_SIZE];

    std::ofstream output_file(output_file_name), debug_file(debug_file_name);
    for (int day = 0; day < prediction_days; ++day) {
        // Perform the LSTM sequence operation
        lstm_sequence(input_seq, h, c, output_data, i_gate, f_gate, o_gate, g_gate);

        // Log debug information
        debug_file << "Day " << day + 1 << " Debug Information:\n";
        debug_file << "Hidden States (h): ";
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            debug_file << h[i].to_double() << " ";
        }
        debug_file << "\nCell States (c): ";
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            debug_file << c[i].to_double() << " ";
        }
        debug_file << "\n";

        // Log predictions to the output file with denormalization
        output_file << "Day " << day + 1 << ": ";
        for (int i = 0; i < INPUT_SIZE; ++i) {
            // Apply denormalization to each output
            double denormalized_value = output_data[i].to_double() * std_devs[i] + means[i];
            output_file << denormalized_value << " ";
        }
        output_file << "\n";

        // Shift input sequence for the next prediction
        for (int i = 0; i < SEQ_LENGTH - 1; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                input_seq[i][j] = input_seq[i + 1][j];
            }
        }
        for (int j = 0; j < INPUT_SIZE; ++j) {
            input_seq[SEQ_LENGTH - 1][j] = output_data[j];
        }
    }

    save_weights_to_file();
    output_file.close();
    debug_file.close();

    return 0;
}