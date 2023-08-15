# Change Log

## (dev)

- Updated minimum Kokkos dependency to version 3.6
- Updated heFFTe version from 2.1 to 2.3

## 0.5.0

**New Features**

- Particle migration using Cajita grid added
- Random particle generation added
- Complete Cajita tutorial examples added
- Cajita performance benchmarks added

**Bug Fixes and Improvements**

- Remove all uses of `Kokkos::Impl`
- Redesign `SimdPolicy` to not modify the underlying `Kokkos::TeamPolicy`
- Rename `Cabana_REQUIRE_`{`PTHREAD` -> `THREADS`}
- Rename clang-format build rule `format` -> `cabana-format`
- Improved Doxygen coverage
- Improved wiki documentation

**Minimum dependency version updates**

- CMake minimum 3.16 required (previously 3.9)
- Optional dependency heFFTe minimum 2.1 (previously 2.0)
- Optional dependency HYPRE minimum 2.22.1 (previously 2.22.0)

**Experimental Features (subject to change in future releases)**

- Distributed particle output with SILO library interface
- Cajita load balancing added through ALL library interface

## 0.4.0

**New Features**

- C++14 required
- Updated minimum Kokkos dependency to version 3.2
- AMD HIP support and continuous integration testing
- Intel SYCL support and continuous integration testing
- OpenMP-Target support (with some disabled features)
- Hybrid particle-grid capability through the Cajita interfaces. Features include:
    - 2D/3D structured grid data structures
    - particle-grid interpolation
    - particle-grid communication
    - multidimensional distributed FFTs via heFFTe (including host, CUDA, and HIP)
    - linear solvers and preconditions via HYPRE (including host and CUDA)

**Bug Fixes and Improvements**

- Removed deprecated portability macros in favor of Kokkos macros (e.g. KOKKOS_INLINE_FUNCTION)
- General performance improvements including neighbor list and particle communication updates
- Improved Doxygen coverage, wiki documentation, and tutorials

**Experimental Features (subject to change in future releases)**

- Sparse grids support in Cajita
- Structured grid data I/O in Cajita

## 0.3.0

**New Features**

- Updated minimum Kokkos dependency to version 3.1
- CUDA and HIP support and testing in continuous integration
- Mirror view capability for AoSoA
- New performance benchmarks for sorting, communication, and neighbor lists
- Improving AoSoA memory managment with empty() and shrinkToFit()
- Second level neighbor parallel for and reduce algorithms for triplet operations
- Unmanaged AoSoA for wrapping user memory

**Bug Fixes and Improvements**
- Using new CMake target for linking Kokkos
- Removed numerous instances of default allocation of Kokkos Views
- Eliminated use of user-defined MPI tags in communication algorithms
- Cleaned usage of deprecated Kokkos code
- Update for compilation with C++14
- Significant performance enhancements to communication code

**Experimental Features (subject to change in future releases)**
- Tree-based neighbor lists using ArborX


## 0.2.0

**New Features**

- An optional MPI dependency has been added. Note that when CUDA is enabled the MPI implementation is expected to be CUDA-aware. [#45](https://github.com/ECP-copa/Cabana/pull/45)
- Particle redistribution via MPI. Implemented in the `Cabana::Distributor` [#43](https://github.com/ECP-copa/Cabana/pull/43)
- Particle halo exchange via MPI. Implemented in the `Cabana::Halo` [#43](https://github.com/ECP-copa/Cabana/pull/43)
- Parallel for concept for 2D indexing in AoSoA loops. Implemented via `Cabana::simd_parallel_for`. Includes a new execution space concept `Cabana::SimdPolicy`. [#49](https://github.com/ECP-copa/Cabana/pull/49)
- Parallel for concept for traversing neighbor lists. Implemented via `Cabana::neighbor_parallel_for` [#49](https://github.com/ECP-copa/Cabana/pull/49)
- Continuous integration for pull requests via GitHub [#9](https://github.com/ECP-copa/Cabana/pull/9)
- Support the ECP continuous integration infrastructure [#66](https://github.com/ECP-copa/Cabana/pull/66)
- New example using scafacos for long-range solvers [#46](https://github.com/ECP-copa/Cabana/pull/46)
- Additional tutorials documentation on the [Wiki](https://github.com/ECP-copa/Cabana/wiki)

**Bug Fixes and Improvements**
- Fixed a bug in the construction of slices on uninitialized `AoSoA` containers [#80](https://github.com/ECP-copa/Cabana/pull/80)
- Construct Verlet lists over the specified range of indices [#70](https://github.com/ECP-copa/Cabana/pull/70)
- Removed aliases of Kokkos macros and classes [#58](https://github.com/ECP-copa/Cabana/pull/58)

## 0.1.0

**New Features**

- Core portable data structures: `Tuple`, `SoA`, `AoSoA`, and `Slice`
- `DeepCopy`: deep copy of data between `AoSoA` data structures
- Sorting and binning `AoSoA` data structures
- `LinkedCellList`: linked cell list implementation
- Portable neighbor list interface
- `VerletList` - linked cell accelerated Verlet list implementation
- Basic tutorial

**Experimental Features (subject to change in future releases)**

- `parallel_for` - portable parallel loops over supported execution spaces
- `neighbor_parallel_for` - portable parallel loops over neighbor lists in supported execution spaces
- `RangePolicy` - defines an index range for parallel loops
