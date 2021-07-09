FROM ghcr.io/ecp-copa/ci-containers/fedora:latest 

COPY kokkos/ /home/kokkos/kokkos
RUN cmake -S kokkos -B kokkos-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DKokkos_ENABLE_OPENMP=On -DKokkos_ENABLE_HWLOC=ON
RUN cmake --build kokkos-build -j2
RUN sudo cmake --install kokkos-build
# some bug in in-place cmake target file
RUN rm -rf kokkos-build 

COPY arborx/ /home/kokkos/arborx
RUN cmake -S arborx -B arborx-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
RUN cmake --build arborx-build -j2
RUN sudo cmake --install arborx-build
# some bug in in-place cmake target file
RUN rm -rf arborx-build 

COPY heffte/ /home/kokkos/heffte
RUN cmake -S heffte -B heffte-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_CXX_STANDARD="11" -DHeffte_ENABLE_FFTW=ON
RUN cmake --build heffte-build -j2
RUN sudo cmake --install heffte-build
# some bug in in-place cmake target file
RUN rm -rf heffte-build 

COPY hypre/ /home/kokkos/hypre
RUN cmake -S hypre/src -B hypre-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DHYPRE_WITH_MPI=ON -DHYPRE_WITH_OPENMP=ON
RUN cmake --build hypre-build -j2
RUN sudo cmake --install hypre-build
# some bug in in-place cmake target file
RUN rm -rf hypre-build 

COPY cabana/ /home/kokkos/cabana
RUN cmake -S cabana -B cabana-build -DCMAKE_BUILD_TYPE=Release -DCabana_ENABLE_TESTING=ON -DCabana_ENABLE_EXAMPLES=ON -DCabana_ENABLE_PERFORMANCE_TESTING=ON -DCabana_PERFORMANCE_EXPECTED_FLOPS=0
RUN cmake --build cabana-build -j2
RUN cmake --build cabana-build --target test
RUN sudo cmake --install cabana-build
