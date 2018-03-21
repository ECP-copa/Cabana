#ifndef CABANA_TEST_OPENMP_CATEGORY_HPP
#define CABANA_TEST_OPENMP_CATEGORY_HPP

#include <Kokkos_OpenMP.hpp>

#define TEST_EXECSPACE Kokkos::OpenMP
#define TEST_MEMSPACE Kokkos::OpenMP::memory_space

#endif // end CABANA_TEST_OPENMP_CATEGORY_HPP
