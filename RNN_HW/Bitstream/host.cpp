#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// Define constants
#define SEQ_LENGTH 60
#define INPUT_SIZE 5
#define HIDDEN_SIZE 16

typedef float fixed_type; // Data type for compatibility

void load_data(const char *filename, std::vector<fixed_type> &data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    fixed_type value;
    while (file >> value) {
        data.push_back(value);
    }
    file.close();
    if (data.size() != SEQ_LENGTH * INPUT_SIZE) {
        std::cerr << "Error: Input size mismatch. Expected " << SEQ_LENGTH * INPUT_SIZE 
                  << ", got " << data.size() << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <xclbin file> <data file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbin_path = argv[1];
    std::string data_path = argv[2];

    // Load input data
    std::vector<fixed_type> input_data;
    load_data(data_path.c_str(), input_data);
    std::vector<fixed_type> output_data(HIDDEN_SIZE, 0);

    // Open the device and load the xclbin
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbin_path);

    // Open the kernel
    auto kernel = xrt::kernel(device, uuid, "rnn_sequence");

    // Allocate device buffers
    auto in_buffer = xrt::bo(device, input_data.size() * sizeof(fixed_type), kernel.group_id(0));
    auto out_buffer = xrt::bo(device, output_data.size() * sizeof(fixed_type), kernel.group_id(1));

    // Write input data to the input buffer
    in_buffer.write(input_data.data());
    in_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Run the kernel
    auto run = kernel(in_buffer, out_buffer, SEQ_LENGTH, INPUT_SIZE, HIDDEN_SIZE);
    run.wait();

    // Read output data from the output buffer
    out_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    out_buffer.read(output_data.data());

    // Print output
    std::cout << "Kernel output:" << std::endl;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        std::cout << "Feature[" << i << "]: " << output_data[i] << std::endl;
    }

    return EXIT_SUCCESS;
}

