#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

// Utility function to read data file
std::vector<std::vector<float>> read_data_file(const std::string &file_path, int &prediction_days) {
    std::cout << "Debug: Reading text data file: " << file_path << std::endl;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open data file " << file_path << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<float>> data;
    std::string line;

    // Read the number of predictions (first line)
    if (std::getline(file, line)) {
        prediction_days = std::stoi(line);
        std::cout << "Debug: Prediction days: " << prediction_days << std::endl;
    }

    // Process the remaining lines (numeric data)
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<float> row;
        std::string value;

        while (std::getline(iss, value, ',')) {
            row.push_back(std::stof(value));
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();
    std::cout << "Debug: Data file read successfully. Rows: " << data.size() << std::endl;
    return data;
}

// Utility function to normalize data
void normalize_data(std::vector<std::vector<float>> &data, std::vector<float> &means, std::vector<float> &std_devs) {
    size_t num_features = data[0].size();
    size_t num_samples = data.size();

    means.assign(num_features, 0.0f);
    std_devs.assign(num_features, 0.0f);

    // Calculate means
    for (const auto &row : data) {
        for (size_t i = 0; i < num_features; ++i) {
            means[i] += row[i];
        }
    }
    for (size_t i = 0; i < num_features; ++i) {
        means[i] /= num_samples;
    }

    // Calculate standard deviations
    for (const auto &row : data) {
        for (size_t i = 0; i < num_features; ++i) {
            std_devs[i] += (row[i] - means[i]) * (row[i] - means[i]);
        }
    }
    for (size_t i = 0; i < num_features; ++i) {
        std_devs[i] = std::sqrt(std_devs[i] / num_samples);
    }

    // Normalize data
    for (auto &row : data) {
        for (size_t i = 0; i < num_features; ++i) {
            row[i] = (row[i] - means[i]) / std_devs[i];
        }
    }

    std::cout << "Debug: Data normalization complete." << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <XCLBIN File> <Data File>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string xclbin_file = argv[1];
    const std::string data_file = argv[2];

    // Read and normalize data
    int prediction_days = 0;
    auto raw_data = read_data_file(data_file, prediction_days);
    if (raw_data.empty()) {
        std::cerr << "Error: Data file is empty or invalid." << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<float> means, std_devs;
    normalize_data(raw_data, means, std_devs);

    // Initialize device and load XCLBIN
    std::cout << "Debug: Initializing device and loading XCLBIN..." << std::endl;
    auto device = xrt::device(0);
    auto xclbin_uuid = device.load_xclbin(xclbin_file);
    std::cout << "Debug: XCLBIN loaded successfully." << std::endl;

    // Create kernel
    auto kernel = xrt::kernel(device, xclbin_uuid, "lstm_sequence");
    std::cout << "Debug: Kernel 'lstm_sequence' created successfully." << std::endl;

    // Allocate buffers
    size_t input_size = raw_data.size() * raw_data[0].size() * sizeof(float);
    size_t output_size = raw_data[0].size() * sizeof(float);

    auto input_bo = xrt::bo(device, input_size, kernel.group_id(0));
    auto output_bo = xrt::bo(device, output_size, kernel.group_id(1));
    std::cout << "Debug: Input and output buffers allocated." << std::endl;

    // Write initial data to input buffer
    input_bo.write(raw_data.data());
    input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::cout << "Debug: Input data transferred to device." << std::endl;

    std::ofstream output_file("output.dat");
    std::vector<float> predictions(raw_data[0].size());

    // Run kernel for each prediction day
    for (int day = 0; day < prediction_days; ++day) {
        auto run = xrt::run(kernel);
        run.set_arg(0, input_bo);
        run.set_arg(1, output_bo);
        run.start();
        run.wait();
        std::cout << "Debug: Kernel execution for day " << day + 1 << " complete." << std::endl;

        // Read and save predictions
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        output_bo.read(predictions.data());

        // Denormalize and write to output
        for (size_t i = 0; i < predictions.size(); ++i) {
            float denormalized = predictions[i] * std_devs[i] + means[i];
            output_file << denormalized << " ";
        }
        output_file << std::endl;

        // Update input sequence with latest predictions
        std::vector<float> new_input(predictions);
        input_bo.write(new_input.data());
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    output_file.close();
    std::cout << "Debug: Results written to 'output.dat'." << std::endl;

    return EXIT_SUCCESS;
}

