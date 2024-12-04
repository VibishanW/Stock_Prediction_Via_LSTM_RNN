#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cstring>

// Utility function to read binary file
std::vector<unsigned char> read_binary_file(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open binary file " << file_path << std::endl;
        exit(EXIT_FAILURE);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer(size);
    file.read(reinterpret_cast<char *>(buffer.data()), size);
    file.close();
    return buffer;
}

// Utility function to read data file
std::vector<std::vector<float>> read_data_file(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open data file " << file_path << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<float> row;
        float value;

        while (iss >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }

    file.close();
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
}

// Utility function to denormalize predictions
std::vector<float> denormalize_predictions(const std::vector<float> &predictions, const std::vector<float> &means, const std::vector<float> &std_devs) {
    std::vector<float> denormalized;
    for (size_t i = 0; i < predictions.size(); ++i) {
        denormalized.push_back(predictions[i] * std_devs[i] + means[i]);
    }
    return denormalized;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <xclbin_file> <data_file> <weights_file>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string xclbin_file = argv[1];
    const std::string data_file = argv[2];
    const std::string weights_file = argv[3];

    cl_int status;

    // Get platform and device
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_platforms, num_devices;

    status = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Failed to get platform IDs." << std::endl;
        return EXIT_FAILURE;
    }

    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, &num_devices);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Failed to get device IDs." << std::endl;
        return EXIT_FAILURE;
    }

    // Create context and command queue
    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &status);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Failed to create context." << std::endl;
        return EXIT_FAILURE;
    }
    
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, 0, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, properties, &status);
	
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Failed to create command queue." << std::endl;
        return EXIT_FAILURE;
    }

    // Read binary file and prepare for creating the program
    std::vector<unsigned char> binary_data = read_binary_file(xclbin_file);
    const unsigned char *binary_data_ptr = binary_data.data();
    size_t binary_sizes = binary_data.size();

    // Create the program from the binary
    cl_int binary_status; // To capture binary-specific status
    cl_program program = clCreateProgramWithBinary(
        context,
        1,               // Number of devices
        &device_id,      // Device ID
        &binary_sizes,   // Size of the binary
        &binary_data_ptr,// Pointer to binary data
        &binary_status,  // Binary loading status
        &status          // Overall function status
    );

    if (status != CL_SUCCESS) {
        std::cerr << "Error: Failed to create program from binary. Status: " << status << std::endl;
        return EXIT_FAILURE;
    }

    if (binary_status != CL_SUCCESS) {
        std::cerr << "Error: Binary failed to load for the device. Binary status: " << binary_status << std::endl;
        return EXIT_FAILURE;
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "lstm", &status);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Failed to create kernel." << std::endl;
        return EXIT_FAILURE;
    }

    // Load and normalize data
    auto raw_data = read_data_file(data_file);
    std::vector<float> means, std_devs;
    normalize_data(raw_data, means, std_devs);

    // Create input/output buffers
    size_t data_size = raw_data.size() * raw_data[0].size() * sizeof(float);
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data_size, raw_data.data(), &status);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, nullptr, &status);

    if (status != CL_SUCCESS) {
        std::cerr << "Error: Failed to create buffers." << std::endl;
        return EXIT_FAILURE;
    }

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);

    // Execute kernel
    size_t global_work_size = raw_data.size();
    status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Failed to execute kernel." << std::endl;
        return EXIT_FAILURE;
    }

    // Read and denormalize results
    std::vector<float> predictions(raw_data[0].size());
    clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, data_size, predictions.data(), 0, nullptr, nullptr);

    auto denormalized_predictions = denormalize_predictions(predictions, means, std_devs);

    // Write output to file
    std::ofstream output_file("output.dat");
    for (const auto &value : denormalized_predictions) {
        output_file << value << " ";
    }
    output_file << std::endl;

    // Cleanup
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
