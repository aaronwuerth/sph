cmake_minimum_required(VERSION 3.2)

project(SPH)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

file(READ LICENSE GPL_RAW_TEXT)
string(REPLACE "\r" "" GPL_RAW_TEXT ${GPL_RAW_TEXT})
string(REGEX REPLACE "^(.*)[^\n]$" "\\1\n" GPL_RAW_TEXT ${GPL_RAW_TEXT})
string(REGEX REPLACE "([^\r\n]*\n)" "R\"(\\1)\"\n" GPL_TEXT ${GPL_RAW_TEXT})


file(READ LEGAL LEGAL_RAW_TEXT)
string(REPLACE "\r" "" LEGAL_RAW_TEXT ${LEGAL_RAW_TEXT})
string(REGEX REPLACE "^(.*)[^\n]$" "\\1\n" LEGAL_RAW_TEXT ${LEGAL_RAW_TEXT})
string(REGEX REPLACE "([^\r\n]*\n)" "R\"(\\1)\"\n" LEGAL_TEXT ${LEGAL_RAW_TEXT})

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third-party/glfw)
add_subdirectory(third-party/glm)
add_subdirectory(third-party/glad)
if(MSVC)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(third-party/googletest)
add_subdirectory(third-party/spdlog)
add_subdirectory(SPH)
add_subdirectory(src)
add_subdirectory(test)

add_library(TCLAP INTERFACE)
target_include_directories(TCLAP SYSTEM INTERFACE third-party/tclap/include)
add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE third-party/stb)
