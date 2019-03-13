FROM nvidia/cuda:10.0-devel-ubuntu18.04
RUN apt-get update && apt-get install -y \
        bc \
        wget \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install CMake
RUN export CMAKE_VERSION=3.13.4 && \
    export CMAKE_VERSION_SHORT=3.13 && \
    export CMAKE_URL=https://cmake.org/files/v${CMAKE_VERSION_SHORT}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    export CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    wget --quiet ${CMAKE_URL} --output-document=${CMAKE_SCRIPT} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=/usr && \
    rm ${CMAKE_SCRIPT}

# Install Kokkos
ENV KOKKOS_DIR=/opt/kokkos
RUN export KOKKOS_VERSION=2.8.00 && \
    export KOKKOS_URL=https://github.com/kokkos/kokkos/archive/${KOKKOS_VERSION}.tar.gz && \
    export KOKKOS_ARCHIVE=kokkos-${KOKKOS_VERSION}.tar.gz && \
    export KOKKOS_SOURCE_DIR=kokkos-${KOKKOS_VERSION} && \
    wget --quiet ${KOKKOS_URL} --output-document=${KOKKOS_ARCHIVE} && \
    mkdir -p ${KOKKOS_SOURCE_DIR} && \
    tar -xf ${KOKKOS_ARCHIVE} -C ${KOKKOS_SOURCE_DIR} --strip-components=1 && \
    cd ${KOKKOS_SOURCE_DIR} && mkdir build && cd build && \
    ../generate_makefile.bash \
        --with-cuda \
        --with-cuda-options=enable_lambda \
        --arch=Volta70 \
        --prefix=${KOKKOS_DIR} \
        --debug \
    && \
    make && make install && \
    rm -rf ${KOKKOS_ARCHIVE} ${KOKKOS_SOURCE_DIR}
