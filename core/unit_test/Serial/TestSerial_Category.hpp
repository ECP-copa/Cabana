#ifndef CABANA_TEST_SERIAL_CATEGORY_HPP
#define CABANA_TEST_SERIAL_CATEGORY_HPP

#include <Kokkos_Serial.hpp>

#include <gtest/gtest.h>

namespace Test {

class serial : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

} // namespace Test

#define TEST_CATEGORY serial
#define TEST_EXECSPACE Kokkos::Serial
#define TEST_MEMSPACE Kokkos::Serial::memory_space

#endif // end CABANA_TEST_SERIAL_CATEGORY_HPP
