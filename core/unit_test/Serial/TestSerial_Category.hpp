#ifndef CABANA_TEST_SERIAL_CATEGORY_HPP
#define CABANA_TEST_SERIAL_CATEGORY_HPP

#include <Kokkos_Serial.hpp>

#define TEST_EXECSPACE Kokkos::Serial
#define TEST_MEMSPACE Kokkos::Serial::memory_space

#endif // end CABANA_TEST_SERIAL_CATEGORY_HPP
