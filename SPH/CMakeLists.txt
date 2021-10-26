if(NOT MSVC)
    set(CMAKE_CXX_CLANG_TIDY clang-tidy)
endif()

set(HEADERS
  SPH.hpp
)

set(SOURCES)

add_library(SPH INTERFACE)
target_include_directories(SPH INTERFACE .)
