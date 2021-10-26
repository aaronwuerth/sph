if(NOT MSVC)
  set(CMAKE_CXX_CLANG_TIDY clang-tidy)
endif()

configure_file(copyright.hpp.in include/copyright.hpp)

add_compile_options("$<$<CONFIG:Debug>:-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE>")

set(HEADERS
  ${CMAKE_CURRENT_BINARY_DIR}/include/copyright.hpp
  kernel.hpp
  neighborsearch.hpp
  render.hpp
)

set(SOURCES
)

add_executable(simulator ${HEADERS} ${SOURCES} main.cpp)
target_link_libraries(simulator SPH ${CMAKE_DL_LIBS}  glad glfw glm::glm spdlog::spdlog TCLAP stb $<$<NOT:$<BOOL:${MSVC}>>:tbb>)
target_include_directories(simulator PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

if(MSVC)
  target_compile_options(simulator PRIVATE /W4)
else()
  target_compile_options(simulator PRIVATE -Wall -Wextra -Wpedantic)
endif()
