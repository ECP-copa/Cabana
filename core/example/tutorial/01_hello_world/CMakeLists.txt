# the next for lines are just, so that we can use this as
# a standalone CMakeLists.txt for testing of the Target export
cmake_minimum_required(VERSION 3.9)
if(NOT TARGET Cabana::cabanacore)
  find_package(Cabana)
endif()

add_executable(HelloWorld hello_world.cpp)
target_link_libraries(HelloWorld Cabana::cabanacore)
add_test(NAME Core_tutorial_01 COMMAND ${NONMPI_PRECOMMAND} HelloWorld)