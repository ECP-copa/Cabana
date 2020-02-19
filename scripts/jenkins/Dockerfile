FROM nvidia/cuda:10.1-devel-ubuntu18.04
RUN apt-get update && apt-get install -y \
        bc \
        wget \
        openssh-client \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install CMake
ENV CMAKE_DIR=/opt/cmake
RUN CMAKE_VERSION=3.13.4 && \
    CMAKE_KEY=2D2CEF1034921684 && \
    CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION} && \
    CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    CMAKE_SHA256=cmake-${CMAKE_VERSION}-SHA-256.txt && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SHA256} && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SHA256}.asc && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SCRIPT} && \
    gpg --recv-keys ${CMAKE_KEY} && \
    gpg --verify ${CMAKE_SHA256}.asc ${CMAKE_SHA256} && \
    grep ${CMAKE_SCRIPT} ${CMAKE_SHA256} | sha256sum --check && \
    mkdir -p ${CMAKE_DIR} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_DIR} && \
    rm ${CMAKE_SCRIPT}
ENV PATH=${CMAKE_DIR}/bin:$PATH

# Install CUDA-aware Open MPI
RUN export OPENMPI_VERSION=4.0.0 && \
    export OPENMPI_VERSION_SHORT=4.0 && \
    export OPENMPI_SHA1=fee1d0287abfb150bae16957de342752c9bdd4e8 && \
    export OPENMPI_URL=https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSION_SHORT}/downloads/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    export OPENMPI_ARCHIVE=openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    export OPENMPI_SOURCE_DIR=openmpi-${OPENMPI_VERSION} && \
    export OPENMPI_INSTALL_PREFIX=/usr && \
    wget --quiet ${OPENMPI_URL} --output-document=${OPENMPI_ARCHIVE} && \
    echo "${OPENMPI_SHA1} ${OPENMPI_ARCHIVE}" | sha1sum -c && \
    mkdir -p ${OPENMPI_SOURCE_DIR} && \
    tar -xf ${OPENMPI_ARCHIVE} -C ${OPENMPI_SOURCE_DIR} --strip-components=1 && \
    cd ${OPENMPI_SOURCE_DIR} && mkdir build && cd build && \
    ../configure --prefix=${OPENMPI_INSTALL_PREFIX} --with-cuda && \
    make -j8 install && \
    rm -rf ${OPENMPI_ARCHIVE} ${OPENMPI_SOURCE_DIR}

# Install Kokkos
ENV KOKKOS_DIR=/opt/kokkos
RUN KOKKOS_VERSION=master && \
    KOKKOS_URL=https://github.com/kokkos/kokkos/archive/${KOKKOS_VERSION}.tar.gz && \
    KOKKOS_ARCHIVE=kokkos-${KOKKOS_VERSION}.tar.gz && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${KOKKOS_URL} --output-document=${KOKKOS_ARCHIVE} && \
    mkdir -p kokkos && \
    tar -xf ${KOKKOS_ARCHIVE} -C kokkos --strip-components=1 && \
    cd kokkos && \
    mkdir -p build && cd build && \
    cmake \
      -D CMAKE_INSTALL_PREFIX=${KOKKOS_DIR} \
      -D CMAKE_BUILD_TYPE=Debug \
      -D CMAKE_CXX_COMPILER=/scratch/kokkos/bin/nvcc_wrapper \
      -D Kokkos_ENABLE_CUDA=ON \
      -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
      -D Kokkos_ARCH_VOLTA70=ON
    .. && \
    make && make install && \
    rm -rf ${SCRATCH_DIR}
