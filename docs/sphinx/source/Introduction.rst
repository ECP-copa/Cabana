Cabana Introduction
===================

Cabana is a performance portable library for particle-based simulations. It
provides particle data structures, algorithms, and utilities to enable
simulations on a variety of architectures including many-core/multi-core
architectures and GPUs.

Cabana is developed as part of the Co-Design Center for Particle Applications
(CoPA) within the Exascale Computing Project (ECP) under the U.S. Department
of Energy. CoPA is a multi-institution project with developers from ORNL,
LANL, SNL, LLNL, PPPL, and ANL.

Key features of Cabana include:

- General and portable Array-of-Structs-of-Arrays data structure for
  node-level representations of particle data.

- Support for general multidimensional particle data members.

- Support for mixed precision particle data members.

- Node-level parallel particle operations with Serial, OpenMP, and CUDA
  back-end implementations. (Indicate Pthread support and other Kokkos flavors
  as we add them).

- Support for both standard CUDA global memory and unified-virtual-memory (UVM).

- List other features as we add them... (e.g. algorithms, distributed data
  structures, etc.)
