#ifndef CAJITA_TEST_SERIAL_CATEGORY_HPP
#define CAJITA_TEST_SERIAL_CATEGORY_HPP

#define TEST_CATEGORY serial
#define TEST_EXECSPACE Kokkos::Serial
#define TEST_MEMSPACE Kokkos::HostSpace
#define TEST_DEVICE Kokkos::Device<Kokkos::Serial,Kokkos::HostSpace>

#endif // end CAJITA_TEST_SERIAL_CATEGORY_HPP
