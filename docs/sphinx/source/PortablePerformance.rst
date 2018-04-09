.. role:: cpp(code)
   :language: c++

Portable Performance
====================

Cabana is designed and implemented for portable performance, meaning that the
user programs with the library through a single interface while implementation
details allow the library to execute on multiple node architectures in a
performant manner. In general, programming on different architectures requires
use of architecture specific programming models (i.e. the use of CUDA for
NVIDIA GPUs) and designing data structures specifically to obtain performance
on a given architecture. The variation in performance that leads to different
data structures giving better performance on different architectures is
largely due to threading models and memory access patterns best suited for a
given hardware design.

Cabana builds on the Kokkos programming model to encapsulate different
concepts related to performance portability. These concepts are used to
customize data structures and parallel kernels for a given architecture. From
the perspective of the Cabana user, three concepts are critical for
understanding how to use the library: memory spaces, execution spaces, and
parallel work units.


Memory Space
------------

The memory space defines where in memory Cabana data structures are
allocated. For example, if a user wanted to operate on a group of particles
using the GPU, the particles must exist in a GPU-compatible memory
space. Memory spaces define policies for allocating, deallocating, and copying
data on a given architecture as well as define rules for copying data between
different memory spaces (e.g. a copy between host memory and GPU memory). Each
memory space also has its own default execution space (see next section) with
which it is associated. For example, particles defined in a memory space on the
GPU by default are associated with a GPU execution space (although this may be
overridden by the user if desired). Equivalently, all execution spaces have a
default memory space, allowing users to define one or the other in many cases.

The following memory spaces are supported in Cabana (for advanced users, these
are simply aliases of Kokkos memory spaces):

* ``HostSpace``

  - The host space is compatible with standard CPU architectures. Memory is
    allocated and deallocated in using standard calls to ``malloc`` and
    ``free``.

* ``CudaUVMSpace``

  - Defines a memory space that uses Unified-Virtual-Memory (UVM) for NVIDIA
    GPUs. UVM allocated memory is accessible from both the host and the device
    and therefore does not requiring explicit copying of data between host and
    device. However, a user of such memory must be sure to synchronize the
    device appropriately to avoid accessing this memory on the host while
    simultaneously accessing it in a device kernel. Memory is allocated with
    ``cudaMallocManaged`` and deallocated with ``cudaFree``.

When needed, these memory spaces will appear as template parameters in Cabana
functions and data structures.

Execution Space
---------------

The execution space defines where and how parallel kernels are executed. For
example, if a user wants to run a kernel on an NVIDIA GPU, a GPU-compatible
execution space will launch a CUDA kernel with the appropriate launch syntax
and settings. Per the memory space description above, each execution space has
a default memory space (defined by the underlying Kokkos implementation) and
each memory space has a default execution space.

The following execution spaces are supported in Cabana (for advanced users,
these are simply aliases of Kokkos execution spaces):

* ``Serial``

  - Executes kernels in serial (i.e. no parallelism) on standard CPU
    architectures.

* ``OpenMP``

  - Executes kernels in parallel using OpenMP as the threading
    implementation. Standard variables such as ``OMP_NUM_THREADS`` and
    ``OMP_PROC_BIND`` can be used to set execution parameters for the OpenMP
    backend.

* ``Cuda``

  - Executes kernels in parallel using NVIDIA CUDA as the threading
    implementation. Kernels are launched within a ``__global__`` function and
    thread and block parameters are typically set automatically by the
    library.

When needed, these execution spaces will appear as template parameters in
Cabana functions and data structures.

Parallel Work Units
-------------------

On-node parallelism in Cabana is expressed through the concept of parallel
work units. A work unit is a single set of instructions to be applied to a
given particle, a pair of particles, or some combination of a particle and
other objects. Examples of work units include:

* Calculating the force between two particles in a molecular dynamics
  simulation.
* Interpolating the charge of one particle to a single grid node in a
  particle-in-cell simulation.
* Computing the stress tensor of a single particle by evaluating a
  constitutive model using that particle's properties in a material point
  method simulation.

From the perspective of Cabana, work units are defined by **callable objects**
which can be evaluated in a given execution space to perform work on particles
that are defined in a compatible memory space. Examples of callable objects
include functors and lambda functions. Within a callable object, all of the
necessary computation is performed for a single unit of work. Parallelism in
Cabana is always expressed over groups of particles and the particle operated
on in a given unit of work is identified by an index to that particle within
the group.

Two core types of work units exist in particle algorithms in practice: work
units that apply operations on individual particles and work units that apply
operations on particles and their adjacencies. Examples of particle
adjacencies include:

* Neighboring particles that satisfy some criteria (e.g. within a certain
  distance).
* Neighbors of neighbor particles.
* Neighboring cells, faces, nodes, or edges in a computational grid
  (e.g. deposition in a particle-in-cell algorithm).

Work units can be evaluated in a variety of general parallel algorithm
concepts including those discussed below. The signature of a callable object
defining a work unit will be a function of whether it is for a particle
operation or a particle-adjacency operation in addition to the type of parallel
algorithm for which it is defined. Specific details of these signatures are
defined in later sections as well as examples of various ways in which a user
may define a callable object.


Parallel For
~~~~~~~~~~~~

A parallel for is a for loop evaluated in parallel with the callable object
representing the work unit executed as the body of a loop. For example,
considering the following pseudo-code:

.. highlight:: c++

::

   // Define particles in the memory space of choice.
   Particles p;

   // Define a work unit that will operate on particles in an execution space
   // of choice.
   auto my_work_unit = [=]( const int i )
   {
       p(i) = some_work_on_particle_i;
   };

   // Parallelize this loop to run in the execution space of choice.
   for ( int i = 0; i < num_particle; ++i )
   {
       my_work_unit( i );
   }

In this case it is assumed that the loop is embarrassingly parallel (i.e. the
loop can be evaluated on an independent parallel thread for each index). In
many cases, these loops may be multidimensional depending on the data
structures used to represent particles or in the case of loops over
adjacencies and multidimensional parallelism may be exploited.


Parallel Reduce
~~~~~~~~~~~~~~~

Parallel Scan
~~~~~~~~~~~~~
