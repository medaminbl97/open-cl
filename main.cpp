#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cassert>

std::string getDeviceTypeString(cl_device_type type) {
    switch (type) {
        case CL_DEVICE_TYPE_CPU: return "CPU";
        case CL_DEVICE_TYPE_GPU: return "GPU";
        case CL_DEVICE_TYPE_ACCELERATOR: return "Accelerator";
        case CL_DEVICE_TYPE_DEFAULT: return "Default";
        default: return "Unknown";
    }
}

const char* kernelSource = R"CLC(
__kernel void compute_sine(__global const float* input, __global float* output) {
    int gid = get_global_id(0);
    output[gid] = sin(input[gid]);
}
)CLC";

int main() {
    constexpr size_t count = 8'000'000;
    std::vector<float> input(count), output_cpu(count), output_gpu(count);

    // Fill input array
    for (size_t i = 0; i < count; ++i)
        input[i] = i * 0.0001f;

    // ------------------- DEVICE INFO ---------------------
    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::cout << "OpenCL platforms found: " << platformCount << std::endl;

    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    cl_device_id selectedDevice = nullptr;
    cl_platform_id selectedPlatform = nullptr;

    for (cl_uint i = 0; i < platformCount; ++i) {
        std::cout << "\n=== Platform " << i << " ===" << std::endl;

        char buffer[1024];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buffer), buffer, nullptr);
        std::cout << "Platform Name: " << buffer << std::endl;

        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buffer), buffer, nullptr);
        std::cout << "Platform Vendor: " << buffer << std::endl;

        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(buffer), buffer, nullptr);
        std::cout << "Platform Version: " << buffer << std::endl;

        cl_uint deviceCount;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);

        for (cl_uint j = 0; j < deviceCount; ++j) {
            std::cout << "\n  >> Device " << j << ":" << std::endl;

            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
            std::cout << "    Name: " << buffer << std::endl;

            cl_device_type type;
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
            std::cout << "    Type: " << getDeviceTypeString(type) << std::endl;

            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(buffer), buffer, nullptr);
            std::cout << "    Vendor: " << buffer << std::endl;

            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(buffer), buffer, nullptr);
            std::cout << "    Driver Version: " << buffer << std::endl;

            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(buffer), buffer, nullptr);
            std::cout << "    OpenCL Version: " << buffer << std::endl;

            if (selectedDevice == nullptr && type == CL_DEVICE_TYPE_GPU) {
                selectedDevice = devices[j];
                selectedPlatform = platforms[i];
            }
        }
    }

    if (!selectedDevice) {
        std::cerr << "No GPU device found, aborting." << std::endl;
        return 1;
    }

    // ------------------- CPU Benchmark ---------------------
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < count; ++i)
        output_cpu[i] = std::sin(input[i]);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "\n[CPU] Time: " << cpu_time << " ms" << std::endl;

    // ------------------- GPU Benchmark ---------------------
    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &selectedDevice, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, selectedDevice, 0, &err);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    err = clBuildProgram(program, 1, &selectedDevice, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, selectedDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, selectedDevice, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build Error:\n" << log.data() << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "compute_sine", &err);
    cl_mem buf_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * count, input.data(), &err);
    cl_mem buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
    size_t globalSize = count;

    auto t3 = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, sizeof(float) * count, output_gpu.data(), 0, nullptr, nullptr);
    auto t4 = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
    char device_name[256];
    clGetDeviceInfo(selectedDevice, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    std::cout << "[GPU] Device Used: " << device_name << std::endl;
    std::cout << "[GPU] Time: " << gpu_time << " ms" << std::endl;


    // ------------------- Comparison ---------------------
    double max_diff = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double diff = std::abs(output_cpu[i] - output_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max difference between CPU and GPU: " << max_diff << std::endl;

    // Cleanup
    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
