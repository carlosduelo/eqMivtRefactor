# The name of our project is "mivt". CMakeLists files in this project can
# refer to the root source directory of the project as ${mivt_SOURCE_DIR} and 
# to the root binary directory of the project as ${mivt_BINARY_DIR}. 

cmake_minimum_required (VERSION 2.8)

project (mivt)

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${mivt_SOURCE_DIR}/bin )

set(CMAKE_BUILD_TYPE Debug)

# CUDA
find_package(CUDA 5.0 REQUIRED)
if (CUDA_FOUND)
	include_directories(${CUDA_INCLUDE_DIRS})
	cuda_include_directories(${mivt_SOURCE_DIR}/inc)
else()
	message(ERROR "CUDA NOT FOUND")
endif()

# MIVT
include_directories(${mivt_SOURCE_DIR}/inc)
add_subdirectory(${mivt_SOURCE_DIR}/src)
