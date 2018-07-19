#ifndef CABANA_TEST_CUDAUVM_CATEGORY_HPP
#define CABANA_TEST_CUDAUVM_CATEGORY_HPP

#include <Kokkos_Cuda.hpp>

#include <gtest/gtest.h>

namespace Test {

class cuda_uvm : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

} // namespace Test

#define TEST_CATEGORY cuda_uvm
#define TEST_EXECSPACE Kokkos::Cuda
#define TEST_MEMSPACE Kokkos::CudaUVMSpace

#endif // end CABANA_TEST_CUDAUVM_CATEGORY_HPP
