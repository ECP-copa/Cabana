INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

TRIBITS_CONFIGURE_FILE(${PROJECT_NAME}${PACKAGE_NAME}_config.hpp)

SET(CABANA_INCDIR ${CMAKE_INSTALL_PREFIX}/${${PROJECT_NAME}_INSTALL_INCLUDE_DIR})
#-----------------------------------------------------------------------------

SET(HEADERS_PUBLIC "")
SET(HEADERS_PRIVATE "")
SET(SOURCES "")

#-----------------------------------------------------------------------------

FILE(GLOB HEADERS_PUBLIC *.hpp)
FILE(GLOB SOURCES *.cpp)

#-----------------------------------------------------------------------------

FILE(GLOB HEADERS_IMPL impl/*.hpp)
FILE(GLOB SOURCES_IMPL impl/*.cpp)

LIST(APPEND HEADERS_PRIVATE ${HEADERS_IMPL} )
LIST(APPEND SOURCES         ${SOURCES_IMPL} )

INSTALL(FILES ${HEADERS_IMPL} DESTINATION ${CABANA_INCDIR}/impl/)

#-----------------------------------------------------------------------------

TRIBITS_ADD_LIBRARY(
  cabanacore
  HEADERS ${HEADERS_PUBLIC}
  NOINSTALLHEADERS ${HEADERS_PRIVATE}
  SOURCES ${SOURCES}
  DEPLIBS
  )
