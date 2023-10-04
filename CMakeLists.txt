cmake_minimum_required(VERSION 3.10)

# Project name
project(lbvh)

# Enable C++ and CUDA
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Add libigl to the project
add_subdirectory(/seoh_fast_cephfs/lib/libigl ${CMAKE_BINARY_DIR}/_deps/libigl)

# NVCC flags (for CUDA)
# You can uncomment this if you want to switch between Debug and Release flags
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-extended-lambda")

# Debug flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O3 --expt-extended-lambda")

# Add your main.cu and other required files
add_executable(lbvh main.cu)

# Link to igl
target_link_libraries(lbvh igl::core)

# If there are other include directories or other libraries you need,
# you can add them using target_include_directories and target_link_libraries respectively.

# Since you have "include" directory, you might want to add it to the include directories
target_include_directories(lbvh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
