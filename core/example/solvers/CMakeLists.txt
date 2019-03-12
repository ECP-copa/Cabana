add_executable(SPME particles.cpp direct.cpp ewald.cpp pme.cpp main.cpp)
target_link_libraries(SPME cabanacore fftw3)
