add_executable(SPME particles.cpp direct.cpp ewald.cpp pme.cpp main.cpp)
find_package(FFTW3 REQUIRED)
target_link_libraries(SPME cabanacore ${FFTW3_LIBRARIES})
