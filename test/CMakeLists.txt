if(NOT MSVC)
	set(CMAKE_CXX_CLANG_TIDY clang-tidy)
endif()

add_executable(${CMAKE_PROJECT_NAME}_test neighborsearch_and_kernel_test.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME}_test SPH glad glfw glm gtest_main gmock)
target_include_directories(${CMAKE_PROJECT_NAME}_test SYSTEM PRIVATE ${GLAD_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/libs/glm})
target_include_directories(${CMAKE_PROJECT_NAME}_test PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(${CMAKE_PROJECT_NAME}_test)
