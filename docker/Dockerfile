ARG BASE=nvidia/cuda:10.2-devel
FROM $BASE

ARG NPROCS=4

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
    gpg --keyserver keys.gnupg.net --recv-keys ${CMAKE_KEY} && \
    gpg --verify ${CMAKE_SHA256}.asc ${CMAKE_SHA256} && \
    grep ${CMAKE_SCRIPT} ${CMAKE_SHA256} | sha256sum --check && \
    mkdir -p ${CMAKE_DIR} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_DIR} && \
    rm ${CMAKE_SCRIPT}
ENV PATH=${CMAKE_DIR}/bin:$PATH

# Install CUDA-aware Open MPI
ENV OPENMPI_DIR=/opt/openmpi
RUN OPENMPI_VERSION=4.0.2 && \
    OPENMPI_VERSION_SHORT=4.0 && \
    OPENMPI_SHA1=32ce3761288575fb8e4f6296c9105c3a25cf3235 && \
    OPENMPI_URL=https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION_SHORT}/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    OPENMPI_ARCHIVE=openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    [ ! -z "${CUDA_VERSION}" ] && CUDA_OPTIONS=--with-cuda || true && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${OPENMPI_URL} --output-document=${OPENMPI_ARCHIVE} && \
    echo "${OPENMPI_SHA1} ${OPENMPI_ARCHIVE}" | sha1sum -c && \
    mkdir -p openmpi && \
    tar -xf ${OPENMPI_ARCHIVE} -C openmpi --strip-components=1 && \
    mkdir -p build && cd build && \
    ../openmpi/configure --prefix=${OPENMPI_DIR} ${CUDA_OPTIONS} CFLAGS=-w && \
    make -j${NPROCS} install && \
    rm -rf ${SCRATCH_DIR}
ENV PATH=${OPENMPI_DIR}/bin:$PATH

# Install Kokkos
ARG KOKKOS_VERSION=3.1.00
ENV KOKKOS_DIR=/opt/kokkos
RUN KOKKOS_URL=https://github.com/kokkos/kokkos/archive/${KOKKOS_VERSION}.tar.gz && \
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
      -D CMAKE_CXX_COMPILER=${SCRATCH_DIR}/kokkos/bin/nvcc_wrapper \
      -D Kokkos_ENABLE_CUDA=ON \
      -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
      -D Kokkos_ARCH_VOLTA70=ON \
    .. && \
    make -j${NPROCS} install && \
    rm -rf ${SCRATCH_DIR}

# Install ArborX
ENV ARBORX_DIR=/opt/arborx
RUN ARBORX_VERSION=4834bff44c23c9510c6ed93366638dcdf85ab217 && \
    ARBORX_URL=https://github.com/arborx/ArborX/archive/${ARBORX_VERSION}.tar.gz && \
    ARBORX_ARCHIVE=arborx.tar.gz && \
    wget --quiet ${ARBORX_URL} --output-document=${ARBORX_ARCHIVE} && \
    mkdir arborx && \
    tar -xf ${ARBORX_ARCHIVE} -C arborx --strip-components=1 && \
    cd arborx && \
    mkdir -p build && cd build && \
    cmake \
      -D CMAKE_INSTALL_PREFIX=${ARBORX_DIR} \
      -D CMAKE_BUILD_TYPE=Debug \
      -D CMAKE_CXX_COMPILER=${KOKKOS_DIR}/bin/nvcc_wrapper \
      -D CMAKE_CXX_EXTENSIONS=OFF \
      -D CMAKE_PREFIX_PATH=${KOKKOS_DIR} \
    .. && \
    make -j${NPROCS} install && \
    cd ../.. && rm -r arborx
