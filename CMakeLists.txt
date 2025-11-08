cmake_minimum_required(VERSION 3.10)
project(CannyPerformanceTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find OpenCV from submodule
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/external/opencv/build")
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/external/opencv/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Create executable
add_executable(canny_performance_test main.cpp)

# Link OpenCV libraries
target_link_libraries(canny_performance_test ${OpenCV_LIBS})

# Print OpenCV information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Enable testing and add sobel test
enable_testing()

# Collect all test sources in ./test
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/test/*.cpp")

foreach(test_src ${TEST_SOURCES})
	get_filename_component(test_name ${test_src} NAME_WE)
	add_executable(${test_name} ${test_src})
	target_link_libraries(${test_name} ${OpenCV_LIBS})
	add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()