#!/bin/bash

# Ensure external directory exists
rm -rf external
rm -rf build

mkdir -p external

# Clone libigl into external directory
echo "Cloning libigl into ./external..."
git clone https://github.com/libigl/libigl.git external/libigl

# Set environment variable for libigl (temporary for current session)
export LIBIGL_DIR=$(pwd)/external/libigl

# Suggest adding it permanently to user's bashrc or equivalent shell config
echo "To make the LIBIGL_DIR environment variable permanent, add the following line to your ~/.bashrc or shell configuration:"
echo "export LIBIGL_DIR=$(pwd)/external/libigl"

# CUDA env variable
echo "Set cuda ENV variable"
export PATH=$PATH:/usr/local/cuda/bin
export CUDACXX=/usr/local/cuda/bin/nvcc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda

echo "Compiling code"
mkdir -p build
cd build
cmake .. -DCMAKE-TYPE=Release