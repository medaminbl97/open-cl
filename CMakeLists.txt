cmake_minimum_required(VERSION 3.10)
project(OpenCL_SineBenchmark)

# Your source file
add_executable(OpenCL_SineBenchmark main.cpp)

# Path to the Khronos OpenCL SDK headers and lib
set(OPENCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/cl/include")
set(OPENCL_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/cl/lib")

# Include and link OpenCL
target_include_directories(OpenCL_SineBenchmark PRIVATE ${OPENCL_INCLUDE_DIR})
target_link_directories(OpenCL_SineBenchmark PRIVATE ${OPENCL_LIBRARY_DIR})
target_link_libraries(OpenCL_SineBenchmark PRIVATE OpenCL)
