#!/usr/bin/env bash

rm -rf build
mkdir -p build && cd build
cmake \
  -D CMAKE_BUILD_TYPE=Debug \
  -D CMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper \
  -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic" \
  -D CMAKE_PREFIX_PATH=$KOKKOS_DIR \
  -D Cabana_ENABLE_MPI=ON \
  -D Cabana_ENABLE_Cuda=ON \
  -D Cabana_ENABLE_Serial=OFF \
  -D Cabana_ENABLE_OpenMP=OFF \
  -D Cabana_ENABLE_TESTING=ON \
  ../
make -j4
ctest --output-on-failure --no-compress-output -T Test
