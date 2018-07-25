#ifndef CABANA_TEST_OPENMP_CATEGORY_HPP
#define CABANA_TEST_OPENMP_CATEGORY_HPP

#include <Cabana_Types.hpp>

#include <Kokkos_OpenMP.hpp>

#include <gtest/gtest.h>

namespace Test {

class openmp : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

} // namespace Test

#define TEST_CATEGORY openmp
#define TEST_EXECSPACE Kokkos::OpenMP
#define TEST_MEMSPACE Cabana::HostSpace

#endif // end CABANA_TEST_OPENMP_CATEGORY_HPP
