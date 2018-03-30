Configure, Build, Test and Install
==================================

In this section we will review configure, building, testing, and install
Cabana so it may be used within another library or application. We will first
review the dependencies used by the library, provide instructions for cloning
the library, and then configure, build, test, and install the library for
multiple node architectures.

Dependencies
------------

Cabana has a minimal number of dependencies:

==========  =======  ========  =======
Dependency  Version  Required  Details
----------  -------  --------  -------
CMake       xxxxx    Yes       Build system
CUDA        9.x      No        Programming model for NVIDIA GPUs
MPI         xxxxx    Soon      Provides distributed parallelism.
Boost       1.66.0   Yes       Provides unit test harness, C++ MPI wrapper, and other utilities.
Kokkos      2.6.0    Yes       Provides portable on-node parallelism.
==========  =======  ========  =======

Next we will provide examples of how to build the dependencies.

MPI
~~~

Coming soon once we have an MPI dependency...
Be sure to discuss properly setting up `nvcc_wrapper` to use with MPI.

Boost
~~~~~

After downloading Boost to a directory, ``BOOST_SRC_DIR``, simply use the
``bootstrap`` script to configure:

.. code:: bash

::

    cd $BOOST_SRC_DIR
    ./bootstrap.sh --prefix=$BOOST_INSTALL_DIR

Here we have defined an installation directory for Boost,
``BOOST_INSTALL_DIR``. Next build and install the library using the generated
``b2`` binary using your available number of threads, ``BUILD_NUM_THREADS``:

.. code:: bash

::

   ./b2 -j $BUILD_NUM_THREADS
   ./b2 install

Verify your install by navigating to ``BOOST_INSTALL_DIR`` and check that
the libraries have been installed in ``BOOST_INSTALL_DIR/lib`` and the headers
have been installed in ``BOOST_INSTALL_DIR/include``.

Kokkos
~~~~~~

Clone the the latest version of Kokkos into a new directory which we will
refer to as ``KOKKOS_SRC_DIR``:

.. code:: bash

::

   git clone https://github.com/kokkos/kokkos.git

For CUDA builds, we will be using the Kokkos ``nvcc_wrapper`` as a wrapper
around the C++ host compiler and the NVIDIA compiler. This wrapper is located
at ``$KOKKOS_SRC_DIR/bin/nvcc_wrapper``. To configure the wrapper, simply set
the environment variable ``NVCC_WRAPPER_DEFAULT_COMPILER`` to wrap around the
C++ host compiler of your choice. For example:

.. code:: bash

::

   export NVCC_DEFAULT_COMPILER_WRAPPER=/usr/bin/g++

For CUDA-enabled builds ``nvcc_wrapper`` is then used as the compiler instead
of the host C++ compiler and the wrapper will automatically dispatch the host
compiler and ``nvcc`` as necessary.

Next we will configure Kokkos. For NVIDIA GPU configuration to use the Kokkos
CUDA backend, we need to know the architecture capability of the GPU. We can
check this by using NVIDIA device query:

::

  /usr/local/cuda-9.0/samples/1_Utilities/deviceQuery/deviceQuery Starting...

   CUDA Device Query (Runtime API) version (CUDART static linking)

  Detected 1 CUDA Capable device(s)

  Device 0: "GeForce GTX 770"
    CUDA Driver Version / Runtime Version          9.1 / 9.0
    CUDA Capability Major/Minor version number:    3.0
    Total amount of global memory:                 4035 MBytes (4231200768 bytes)
    ( 8) Multiprocessors, (192) CUDA Cores/MP:     1536 CUDA Cores
    GPU Max Clock rate:                            1189 MHz (1.19 GHz)
    Memory Clock rate:                             3505 Mhz
    Memory Bus Width:                              256-bit
    L2 Cache Size:                                 524288 bytes
    Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
    Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
    Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
    Total amount of constant memory:               65536 bytes
    Total amount of shared memory per block:       49152 bytes
    Total number of registers available per block: 65536
    Warp size:                                     32
    Maximum number of threads per multiprocessor:  2048
    Maximum number of threads per block:           1024
    Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
    Maximum memory pitch:                          2147483647 bytes
    Texture alignment:                             512 bytes
    Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
    Run time limit on kernels:                     Yes
    Integrated GPU sharing Host Memory:            No
    Support host page-locked memory mapping:       Yes
    Alignment requirement for Surfaces:            Yes
    Device has ECC support:                        Disabled
    Device supports Unified Addressing (UVA):      Yes
    Supports Cooperative Kernel Launch:            No
    Supports MultiDevice Co-op Kernel Launch:      No
    Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
    Compute Mode:
       < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

  deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 9.1, CUDA Runtime Version = 9.0, NumDevs = 1
  Result = PASS

The output we are looking for is ``CUDA Capability Major/Minor version number``
which for this card is version `3.0`. Based on this, we will pass the
``-arch=sm_30`` flag to the CUDA compiler, ``nvcc``, to ensure the compiled
code is compatible with a given architecture. For other capability versions
this flag is changed to match. For example, if your NVIDIA card is of
capability ``6.0`` you will use ``-arch=sm_60`` instead.

Next we will configure Kokkos for three execution policies: Serial, OpenMP,
and CUDA. Make a new directory ``KOKKOS_BUILD_DIR`` and use that directory for
configuration. The Kokkos build directory must be different than the source
directory. We use the following to configure Kokkos:

.. code:: bash

::

    cd $KOKKOS_BUILD_DIR
    $KOKKOS_SRC_DIR/generate_makefile.bash \
      --prefix=$KOKKOS_INSTALL_DIR \
      --with-cuda --with-openmp --with-serial \
      --compiler=$KOKKOS_SRC_DIR/bin/nvcc_wrapper \
      --cxxflags="-arch=sm_30" \
      --with-cuda-options=enable_lambda

There are a couple of key features to this configuration:

* ``generate_makefile.bash`` is the mechanism by which the Kokkos makefiles are
  generated and the options are passed into this script.
* We have set ``KOKKOS_INSTALL_DIR`` as the location for the installation of
  Kokkos
* Because we are doing a CUDA-enabled build we are using ``nvcc_wrapper`` to
  compile Kokkos. If you are not using CUDA you can simply use your host C++
  compiler in this case.
* We have specified the NVIDIA GPU compute capability as an extra compiler
  flag.
* We would like use lambda functions with Kokkos within Cabana for CUDA builds
  so we enable that option as well.

Extra Kokkos options may always be added or options removed from this
configuration depending on the architecture for which Kokkos and Cabana are
being configured.

Now we can build and install Kokkos:

.. code:: bash

::

   make -j $BUILD_NUM_THREADS
   make install

Verify your install by checking out the libraries installed in
``$KOKKOS_INSTALL_DIR/lib`` and the headers installed in
``$KOKKOS_INSTALL_DIR/include``.

Cloning the Library
-------------------

The first step to obtaining Cabana is cloning the ``master`` branch of the
library into a directory we will refer to as ``CABANA_SRC_DIR``:

.. code:: bash

::

   git clone git@code.ornl.gov:CoPA/Cabana.git

Next, we need to download the Cabana build system (TriBITS) which is contained
within a git submodule:

.. code:: bash

::

   cd $CABANA_SRC_DIR
   git submodule init
   git submodule update

Your entire output should look something like this:

::

   bash$ git clone git@code.ornl.gov:CoPA/Cabana.git
    Cloning into 'Cabana'...
    remote: Counting objects: 307, done.
    remote: Compressing objects: 100% (286/286), done.
    remote: Total 307 (delta 177), reused 53 (delta 18)
    Receiving objects: 100% (307/307), 63.69 KiB | 893.00 KiB/s, done.
    Resolving deltas: 100% (177/177), done.
   bash$ cd Cabana/
   bash$ git submodule init
     Submodule 'cmake/tribits' (https://github.com/TriBITSPub/TriBITS.git) registered for path 'cmake/tribits'
   bash$ git submodule update
     Cloning into '/Users/uy7/scratch/Cabana/cmake/tribits'...
     Submodule path 'cmake/tribits': checked out 'f624a59c872f792ffe30db2f696f98aedd53cc5a'

Next we will configure the library.

Configure
---------

To configure Cabana first make a new directory ``CABANA_BUILD_DIR`` which is
different from the source directory. Cabana uses CMake and we recommend
writing a small configuration script in this new build directory. For this
example will configure using the same Serial, OpenMP, and CUDA backend
implementations that we built with Kokkos. The following is an example of such
a script for these backends (which we will call ``cmake_configure.sh``):

::

   #!/bin/sh

   SOURCE=$CABANA_SRC_DIR
   INSTALL=$CABANA_INSTALL_DIR
   BUILD="RELEASE"

   cmake \
     -D CMAKE_BUILD_TYPE:STRING="$BUILD" \
     -D CMAKE_INSTALL_PREFIX:PATH=$INSTALL \
     -D CMAKE_CXX_COMPILER:PATH=$KOKKOS_INSTALL_DIR/bin/nvcc_wrapper \
     -D CMAKE_CXX_FLAGS="-arch=sm_30" \
     -D TPL_ENABLE_CUDA:BOOL=ON \
     -D TPL_ENABLE_BoostOrg:BOOL=ON \
     -D Boost_LIBRARY_DIRS:PATH=$BOOST_INSTALL_DIR/lib \
     -D Boost_INCLUDE_DIRS:PATH=$BOOST_INSTALL_DIR/include \
     -D TPL_ENABLE_Kokkos:BOOL=ON \
     -D Kokkos_LIBRARY_DIRS:PATH=$KOKKOS_INSTALL_DIR/lib \
     -D Kokkos_INCLUDE_DIRS:PATH=$KOKKOS_INSTALL_DIR/include \
     -D Cabana_CXX11_FLAGS="-std=c++11 --expt-extended-lambda" \
     -D Cabana_EXTRA_LINK_FLAGS:STRING="-ldl" \
     -D Cabana_ENABLE_Serial:BOOL=ON \
     -D Cabana_ENABLE_OpenMP:BOOL=ON \
     -D Cabana_ENABLE_Cuda:BOOL=ON \
     -D Cabana_ENABLE_TESTS:BOOL=ON \
     -D Cabana_ENABLE_EXAMPLES:BOOL=ON \
     \
     ${SOURCE}

There are a couple of key features of this configuration:

* This script is performing a release build which will enable some level of
  optimization. A debug build may be enabled by simply changing ``RELEASE`` to
  ``DEBUG``.
* We will be installing Cabana in a directory ``CABANA_INSTALL_DIR``.
* The ``nvcc_wrapper`` is designated as the C++ compiler.
* The C++ flags here are again assign the CUDA compute capability flag as was
  done in the Kokkos configuration. This must be consistent with the Kokkos
  configuration to ensure compatible binaries.
* We enable Boost and Kokkos as dependencies and provide paths to their
  installation of headers and libraries.
* The Cabana-specific options are prefixed with ``CABANA_``.
* The Cabana C++11 flags include ``--expt-extended-lambda`` which allows for
  the use of lambda functions with CUDA.
* We enable Serial, OpenMP, and CUDA backend implementations.
* Cabana unit tests and examples are turned on and off.

The script is then executed to configure the library.

.. code:: bash

::

   cd $CABANA_BUILD_DIR
   sh cmake_configure.sh

Build, Test, and Install
------------------------

Once configured build and install Cabana in ``CABANA_BUILD_DIR`` with:

.. code:: bash

::

   make -j $BUILD_NUM_THREADS
   make install

Ensure installation by checking the installed libraries an headers in
``CABANA_INSTALL_DIR``. If tests are enable you can run the Cabana unit test
suite with ``ctest``. A successful test output should look something like
this:

::

   bash$ ctest
   Test project build/Cabana/release
         Start  1: Core_Version_test
    1/12 Test  #1: Core_Version_test ................   Passed    0.10 sec
         Start  2: Core_Index_test
    2/12 Test  #2: Core_Index_test ..................   Passed    0.06 sec
         Start  3: Core_SoA_test
    3/12 Test  #3: Core_SoA_test ....................   Passed    0.05 sec
         Start  4: Core_AoSoA_test_Serial
    4/12 Test  #4: Core_AoSoA_test_Serial ...........   Passed    0.07 sec
         Start  5: Core_MemberSlice_test_Serial
    5/12 Test  #5: Core_MemberSlice_test_Serial .....   Passed    0.07 sec
         Start  6: Core_Parallel_test_Serial
    6/12 Test  #6: Core_Parallel_test_Serial ........   Passed    0.11 sec
         Start  7: Core_AoSoA_test_OpenMP
    7/12 Test  #7: Core_AoSoA_test_OpenMP ...........   Passed    0.08 sec
         Start  8: Core_MemberSlice_test_OpenMP
    8/12 Test  #8: Core_MemberSlice_test_OpenMP .....   Passed    0.06 sec
         Start  9: Core_Parallel_test_OpenMP
    9/12 Test  #9: Core_Parallel_test_OpenMP ........   Passed    0.10 sec
         Start 10: Core_AoSoA_test_CudaUVM
   10/12 Test #10: Core_AoSoA_test_CudaUVM ..........   Passed    0.08 sec
         Start 11: Core_MemberSlice_test_CudaUVM
   11/12 Test #11: Core_MemberSlice_test_CudaUVM ....   Passed    0.06 sec
         Start 12: Core_Parallel_test_CudaUVM
   12/12 Test #12: Core_Parallel_test_CudaUVM .......   Passed    0.11 sec

   100% tests passed, 0 tests failed out of 12

   Label Time Summary:
   Core    =   0.96 sec (12 tests)

   Total Test time (real) =   0.97 sec

If any tests fail, start by checking your configuration to ensure proper
installation of each back-end implementation. Individual tests my be run with
more detailed output is well to help with checking configuration issues. For
example, if all of the CUDA tests are failing they can be run with more
verbose output for diagnostics as:

::

   bash$ ctest -VV -R Cuda
   UpdateCTestConfiguration  from :/home/uy7/build/Cabana/release/DartConfiguration.tcl
   Parse Config file:/home/uy7/build/Cabana/release/DartConfiguration.tcl
   UpdateCTestConfiguration  from :/home/uy7/build/Cabana/release/DartConfiguration.tcl
   Parse Config file:/home/uy7/build/Cabana/release/DartConfiguration.tcl
   Test project /home/uy7/build/Cabana/release
   Constructing a list of tests
   Done constructing a list of tests
   Checking test dependency graph...
   Checking test dependency graph end
   test 10
       Start 10: Core_AoSoA_test_CudaUVM

   10: Test command: /home/uy7/build/Cabana/release/core/unit_test/Core_AoSoA_test_CudaUVM.exe
   10: Test timeout computed to be: 1500
   10: Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set
   10:   In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
   10:   For best performance with OpenMP 3.1 set OMP_PROC_BIND=true
   10:   For unit testing set OMP_PROC_BIND=false
   10: Running 2 test cases...
   10:
   10: *** No errors detected
   1/3 Test #10: Core_AoSoA_test_CudaUVM ..........   Passed    0.12 sec
   test 11
       Start 11: Core_MemberSlice_test_CudaUVM

   11: Test command: /home/uy7/build/Cabana/release/core/unit_test/Core_MemberSlice_test_CudaUVM.exe
   11: Test timeout computed to be: 1500
   11: Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set
   11:   In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
   11:   For best performance with OpenMP 3.1 set OMP_PROC_BIND=true
   11:   For unit testing set OMP_PROC_BIND=false
   11: Running 1 test case...
   11:
   11: *** No errors detected
   2/3 Test #11: Core_MemberSlice_test_CudaUVM ....   Passed    0.07 sec
   test 12
       Start 12: Core_Parallel_test_CudaUVM

   12: Test command: /home/uy7/build/Cabana/release/core/unit_test/Core_Parallel_test_CudaUVM.exe
   12: Test timeout computed to be: 1500
   12: Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set
   12:   In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
   12:   For best performance with OpenMP 3.1 set OMP_PROC_BIND=true
   12:   For unit testing set OMP_PROC_BIND=false
   12: Running 1 test case...
   12:
   12: *** No errors detected
   3/3 Test #12: Core_Parallel_test_CudaUVM .......   Passed    0.11 sec

   The following tests passed:
        Core_AoSoA_test_CudaUVM
        Core_MemberSlice_test_CudaUVM
        Core_Parallel_test_CudaUVM

   100% tests passed, 0 tests failed out of 3

   Label Time Summary:
   Core    =   0.30 sec (3 tests)

   Total Test time (real) =   0.30 sec

If test failures still persist please contact the Cabana developers.
