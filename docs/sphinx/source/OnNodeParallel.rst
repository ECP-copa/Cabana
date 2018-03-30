.. role:: cpp(code)
   :language: c++

On-Node Parallelism
===================

Expression of on-node parallelism in Cabana allows for the execution of
parallel kernels on AoSoA data through a single interface which dispatches the
kernel to the appropriate back-end threading implementation (e.g. OpenMP or
CUDA). Users need to be familiar with 4 core concepts to write their own
on-node parallel code: execution policies, callable objects, parallel
execution patterns, and parallel algorithms.

Execution Policy
----------------

The **execution policy** defines the range of indices over which a parallel
work unit will be executed. In addition, the execution policy also specifies
the execution space in which the work unit will be executed. The
``Cabana::IndexRangePolicy``, the most basic of execution policies, is one in
which a ``begin`` and ``end`` index is specified along with the desired
execution space. For example, to define an execution policy over an entire
AoSoA in a CUDA compatible memory space on would write:

.. highlight:: c++

::

   enum FieldNames { Stress = 0,
                     Temperature,
                     MatId };

   using DataTypes = Cabana::MemberDataTypes<double[3][3], // stress tensor
                                             float[2],     // two-phase temperature
                                             int>;         // material id

   Cabana::AoSoA<DataTypes,Cabana::CudaUVMSpace> aosoa( num_particle );

   Cabana::IndexRangePolicy<Cabana::Cuda> exec_policy( aosoa.begin(), aosoa.end() );

Note that using the beginning and the end of the AoSoA is not required with
the index range policy. One may choose any sequence of indices as long as
``begin <= end``. This may be useful in cases where operations over a subset
of particles in an AoSoA is desired rather than over the entire container.

Callable Objects
----------------

Callable objects are the means by which a user represents a work
kernel. Callable objects are often used to operate on Cabana-allocated data
(e.g. objects of type ``Cabana::AoSoA`` and ``Cabana::MemberSlice``). The
signatures of these callable objects will depend on the context in which they
are used. In addition, these callable objects in general have some
encapsulation of state (e.g. a closure) in which local data is captured for
use within the work unit. There are two primary ways of generating callable
objects: lambda functions and functors.

Lambda Functions
~~~~~~~~~~~~~~~~

Lambda functions (available in the C++ standard since C++11) allow for the
simplest means of generating a callable object for dispatch within a parallel
algorithm. In Cabana, lambda functions in general are used to capture local
variables by *value* and, as of more recent versions of CUDA, if the lambda is
called within a class function it can also be used to capture the current
state of the class via ``this``. Users write Cabana lambda functions in an
identical manner to any other C++ lambda functions except the captures are
replaced by the macro ``CABANA_LAMBDA`` which sets proper captures and other
needed keyword parameters for portability (e.g. adds the ``__device__``
keyword for CUDA-enabled builds). The following is an example of a Cabana
lambda function that initializes all of the values in an AoSoA:

.. highlight:: c++

::

   enum FieldNames { Stress = 0,
                     Temperature,
                     MatId };

   using DataTypes = Cabana::MemberDataTypes<double[3][3], // stress tensor
                                             float[2],     // two-phase temperature
                                             int>;         // material id

   Cabana::AoSoA<DataTypes,Cabana::CudaUVMSpace> aosoa( num_particle );

   auto init_kernel = CABANA_LAMBDA( const Cabana::Index idx )
   {
       // Initialize stress
       for ( int i = 0; i < aosoa.extent(Stress,0); ++i )
           for ( int j = 0; j < aosoa.extent(Stress,1); ++j )
              aosoa.get<Stress>( idx, i, j ) = 1.0;

       // Initialize temperature
       for ( int i = 0; i < aosoa.extent(Temperature,0); ++i )
           aosoa.get<Temperature>( idx, i ) = 2.3;

       // Initialize material id.
       aosoa.get<MatId>( idx ) = 5;
   };

There are a few salient features to note:

* First, the lambda ``init_kernel`` is assigned as an ``auto`` type. In
  general this is recommended so the compiler may interpret this to be the most
  appropriate type of closure.
* Second, the local ``aosoa`` variable is captured by value into the scope of
  the lambda function. Because the ``Cabana::AoSoA`` is essentially a pointer
  to a large block of memory, this pointer is essentially moved to the scope
  of the calling lambda instead of copying the entire block of memory. The
  user can then operate on this memory within the lambda through the use of
  ``AoSoA::get``.
* Third, the unit of work here is an initialization operation on a particle at
  index ``idx`` which can proceed in parallel independent of all other
  particles in the AoSoA.

We can also write lambdas using member slices. Consider the same example, this
time with slices:

.. highlight:: c++

::

   enum FieldNames { Stress = 0,
                     Temperature,
                     MatId };

   using DataTypes = Cabana::MemberDataTypes<double[3][3], // stress tensor
                                             float[2],     // two-phase temperature
                                             int>;         // material id

   Cabana::AoSoA<DataTypes,Cabana::CudaUVMSpace> aosoa( num_particle );

   auto stress = Cabana::slice<Stress>( aosoa );
   auto temperature = Cabana::slice<Temperature>( aosoa );
   auto matid = Cabana::slice<MatId>( aosoa );

   auto init_kernel = CABANA_LAMBDA( const Cabana::Index idx )
   {
       // Initialize stress
       for ( int i = 0; i < stress.extent(0); ++i )
           for ( int j = 0; j < stress.extent(1); ++j )
              stress( idx, i, j ) = 1.0;

       // Initialize temperature
       for ( int i = 0; i < temperature.extent(0); ++i )
           temperature( idx, i ) = 2.3;

       // Initialize material id.
       matid( idx ) = 5;
   };

In this case, again, the slices are captured by value but because they simply
wrap operations around the pointer inside of an AoSoA this is yet another copy
of an address into the scope of the lambda function rather than a copy of the
entire memory block.

Functors
~~~~~~~~

Functors are a slightly more cumbersome way to produce callable objects for
use with Cabana on-node parallelism, however, there are cases in which they
are useful and result in overall code reduction. A functor is an object
(typically a class) that specifically has an implementation of ``operator()``
defined with the proper arguments for the given parallel algorithm. Unlike
lambdas, functors do have the advantage of being class objects and therefore
can be used in situations where class syntax and persistent state is
desired. The following is an example of a functor that can be used for
initialization as in our previous example.

.. highlight:: c++

::

   template<class AoSoA_t>
   class InitFunctor
   {
     public:

       // Constructor.
       InitFunctor( AoSoA_t aosoa )
           : _aosoa( aosoa )
       {}

       // Initialize.
       CABANA_INLINE_FUNCTION
       void operator()( const Cabana::Index idx ) const
       {
           // Initialize stress
           for ( int i = 0; i < aosoa.extent(Stress,0); ++i )
               for ( int j = 0; j < aosoa.extent(Stress,1); ++j )
                  aosoa.get<Stress>( idx, i, j ) = 1.0;

           // Initialize temperature
           for ( int i = 0; i < aosoa.extent(Temperature,0); ++i )
               aosoa.get<Temperature>( idx, i ) = 2.3;

           // Initialize material id.
           aosoa.get<MatId>( idx ) = 5;
       }

     private:

       // The AoSoA the functor will initialize.
       AoSoA_t _aosoa;
   };

There are a couple of key features to note in this functor definition:

* First, the functor has state and maintains a copy of an AoSoA on which to
  operate. Again, because the AoSoA is simply a pointer to large block of
  memory this does not copy the entire memory block into the functor.
* Second, the functor has a definition of ``operator()`` which defines a
  single parallel work unit on a particle at index ``idx``. Like the lambda
  function definition, this operation can occur independently for all
  particles.
* Third, the functor uses another Cabana macro for performance portability,
  ``CABANA_INLINE_FUNCTION``. This macro prefixes the function definition with
  proper inlining keywords as well as keywords for portability
  (i.e. ``__device__`` in a CUDA-enabled build).
* Fourth, the functor is templated on a general type ``AoSoA_t``. The
  intention behind this template parameter is to allow this function to
  operate on any variety of AoSoA instances with different template
  parameters. A user may choose to use any template parameters they desire
  with their functors - the parameter shown here is simply for purposes of
  demonstration.

In the following section, we will demonstrate how to use both lambdas and
functors to execute work in parallel.

Parallel Execution Patterns
---------------------------

Loops in Cabana over AoSoA data structures are fundamentally two dimensional:
there is an outer loop over each of the structs and within each struct there
is a loop over the arrays within that struct:

.. highlight:: c++

::

   Cabana::AoSoA<DataTypes,MemorySpace> aosoa;

   for ( int s = 0; s < aosoa.numSoa(); ++s )
   {
       for ( int i = 0; i < aosoa.arraySize(s); ++i )
       {
            // Do work on particle at array element i in struct s...
       }
   }

There are a number of ways to parallelize this two dimensional loop, each of
which may have performance benefits on different architectures by producing
different amounts of parallelism and different memory access patterns. Next we
define the various types of loop parallelism available in Cabana.

Struct Parallel
~~~~~~~~~~~~~~~

Struct parallelism is a 1-dimensional pattern that parallelizes the outer loop
over structs while the inner loop over arrays is executed sequentially within
the thread executing the work for a given struct. Computationally this is
equivalent to:

.. highlight:: c++

::

   Cabana::AoSoA<DataTypes,MemorySpace> aosoa;

   parallel for ( int s = 0; s < aosoa.numSoa(); ++s )
   {
       for ( int i = 0; i < aosoa.arraySize(s); ++i )
       {
            // Do work on particle at array element i in struct s...
       }
   }

A pattern like this can be convenient, for example, when the inner array size
is the size of a vector unit allowing for threading over structs and, within a
thread, vectorization of the work for the struct operated on by that
thread. For Cabana interfaces that allow for the selection of parallel loop
patterns, this pattern is dispatched with the tag
``Cabana::StructParallelTag``.

Array Parallel
~~~~~~~~~~~~~~

Array parallel is a 1-dimensional pattern that parallelizes the inner loops
over arrays. This equates to an outer sequential loop over structs and a
parallel loop dispatch over arrays for each struct. In many cases, if the
parallel dispatch of inner array loops can be done asynchronously, all work
for all structs may be dispatched at once. This is computationally equivalent
to:

.. highlight:: c++

::

   Cabana::AoSoA<DataTypes,MemorySpace> aosoa;

   for ( int s = 0; s < aosoa.numSoa(); ++s )
   {
       parallel for ( int i = 0; i < aosoa.arraySize(s); ++i )
       {
            // Do work on particle at array element i in struct s...
       }
   }

A pattern like this works well for systems where memory access patterns favor
working on large contiguous member data arrays (e.g. systems where
Struct-of-Arrays works well). For Cabana interfaces that allow for the
selection of parallel loop patterns, this pattern is dispatched with the tag
``Cabana::ArrayParallelTag``.

Struct and Array Parallel
~~~~~~~~~~~~~~~~~~~~~~~~~

Struct and Array parallel is a 2-dimensional pattern that parallelizes over
both the out loop over structs and the inner loop over arrays:

::

   Cabana::AoSoA<DataTypes,MemorySpace> aosoa;

   for ( int s = 0; s < aosoa.numSoa(); ++s )
   {
       parallel for ( int i = 0; i < aosoa.arraySize(s); ++i )
       {
            // Do work on particle at array element i in struct s...
       }
   }

A pattern like this tends works well for systems where there is a natural
2-dimensional indexing of threads and their associated data access patterns
(e.g. an NVIDIA GPU). For Cabana interfaces that allow for the selection of
parallel loop patterns, this pattern is dispatched with the tag
``Cabana::StructAndArrayParallelTag``.

Particle Parallel Algorithms
----------------------------

The most rudimentary class of work units in Cabana perform operations on a
single particle at a given ``Cabana::Index``. The signature of those work
units and the means by which they are dispatched will be a function of the
parallel algorithm type. Next we describe the basic parallel algorithms
supported by Cabana.

Parallel For
~~~~~~~~~~~~

A **parallel for** is the execution of a loop over particles in parallel with
the work unit representing the body of the parallel loop. The work unit has
the following function prototype for operating on a particle at a given index:

.. highlight:: c++

::

   void parallel_for_loop_body_prototype( const Cabana::Index idx );

These prototypes may be realized by lambda functions, functors, or other
callable objects. Parallel for loops are dispatched via
``Cabana::parallel_for``. The following is an example of performing a parallel
initialization of particles using lambda function executed with CUDA using a
variety of syntax options for ``Cabana::parallel_for``:

.. highlight:: c++

::

   enum FieldNames { Stress = 0,
                     Temperature,
                     MatId };

   using DataTypes = Cabana::MemberDataTypes<double[3][3], // stress tensor
                                             float[2],     // two-phase temperature
                                             int>;         // material id

   Cabana::AoSoA<DataTypes,Cabana::CudaUVMSpace> aosoa( num_particle );

   auto stress = Cabana::slice<Stress>( aosoa );
   auto temperature = Cabana::slice<Temperature>( aosoa );
   auto matid = Cabana::slice<MatId>( aosoa );

   auto init_kernel = CABANA_LAMBDA( const Cabana::Index idx )
   {
       // Initialize stress
       for ( int i = 0; i < stress.extent(0); ++i )
           for ( int j = 0; j < stress.extent(1); ++j )
              stress( idx, i, j ) = 1.0;

       // Initialize temperature
       for ( int i = 0; i < temperature.extent(0); ++i )
           temperature( idx, i ) = 2.3;

       // Initialize material id.
       matid( idx ) = 5;
   };

   // Create a range policy.
   Cabana::IndexRangePolicy<Cabana::Cuda> exec_policy( aosoa.begin(), aosoa.end() );

   // Launch with auto-dispatch of execution pattern.
   Cabana::parallel_for( exec_policy, init_kernel, "auto dispatch init" );

   // Launch specifically with StructParallel pattern.
   Cabana::parallel_for( exec_policy, init_kernel,
                         Cabana::StructParallelTag(), "struct parallel init" );

   // Launch specifically with ArrayParallel pattern.
   Cabana::parallel_for( exec_policy, init_kernel,
                         Cabana::ArrayParallelTag(), "array parallel init" );

   // Launch specifically with StructAndArrayParallel pattern.
   Cabana::parallel_for( exec_policy, init_kernel,
                         Cabana::StructAndArrayParallelTag(),
                         struct and array parallel init );

There are a number of important features of ``parallel_for``:

* First, all parallel algorithms operate on an execution policy (in this case
  a ``Cabana::IndexRangePolicy``) which defines the bounds of the parallel
  loop and the execution space in which the loop will operate.
* Second, the lambda function (or any callable object to be executed by
  ``parallel_for``) is passed by value into the function call where it will
  then be inserted as the parallel loop body.
* Third, in the last three calls to ``parallel_for`` an optional argument for
  the parallel execution algorithm is passed to allow users the option to
  choose their own loop pattern. In the first call to ``parallel_for`` no loop
  pattern tag is provided and the library will select a default pattern for
  the given execution space.
* Finally, the last argument provides an optional string for the parallel
  kernel. Using a string can aid in both debugging as well as performance
  timing using Kokkos services.

We can also perform the same parallel for using the functor ``InitFunctor``
that we defined above:

.. highlight:: c++

::

   enum FieldNames { Stress = 0,
                     Temperature,
                     MatId };

   using DataTypes = Cabana::MemberDataTypes<double[3][3], // stress tensor
                                             float[2],     // two-phase temperature
                                             int>;         // material id

   using AoSoA_t = Cabana::AoSoA<DataTypes,Cabana::CudaUVMSpace>;
   AoSoA_t aosoa( num_particle );

   auto stress = Cabana::slice<Stress>( aosoa );
   auto temperature = Cabana::slice<Temperature>( aosoa );
   auto matid = Cabana::slice<MatId>( aosoa );

   // Create a functor instance.
   InitFunctor<AoSoA_t> init_functor( aosoa );

   // Create a range policy.
   Cabana::IndexRangePolicy<Cabana::Cuda> exec_policy( aosoa.begin(), aosoa.end() );

   // Launch with auto-dispatch of execution pattern.
   Cabana::parallel_for( exec_policy, init_functor, "auto dispatch init" );

   // Launch specifically with StructParallel pattern.
   Cabana::parallel_for( exec_policy, init_functor,
                         Cabana::StructParallelTag(), "struct parallel init" );

   // Launch specifically with ArrayParallel pattern.
   Cabana::parallel_for( exec_policy, init_functor,
                         Cabana::ArrayParallelTag(), "array parallel init" );

   // Launch specifically with StructAndArrayParallel pattern.
   Cabana::parallel_for( exec_policy, init_functor,
                         Cabana::StructAndArrayParallelTag(),
                         struct and array parallel init );

Here the primary difference is the the creation of an ``InitFunctor`` object
with ``aosoa`` assigned as member data as an alternative to writing a lambda
and capturing ``aosoa`` by value within the closure of that lambda.

Parallel Scan
~~~~~~~~~~~~~

Parallel Reduce
~~~~~~~~~~~~~~~
