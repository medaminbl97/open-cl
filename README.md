# OpenCL Sine Benchmark

This project demonstrates how to use OpenCL to benchmark the computation of the sine function.

## Structure

- `main.cpp`: Main source file for the benchmark.
- `cl/`: Contains OpenCL SDK headers, libraries, and tools.
  - `include/CL/`: OpenCL header files.
  - `lib/`: OpenCL libraries.
  - `bin/`: OpenCL tools and DLLs.

## Building

Ensure you have CMake installed. From the project root:

```sh
cmake -S . -B build
cmake --build build