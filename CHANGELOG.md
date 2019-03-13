# Change Log

## [0.2.0]()

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

## [0.1.0]()

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
