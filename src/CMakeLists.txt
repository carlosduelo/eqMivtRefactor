# The name of our project is "mivt". CMakeLists files in this project can
# refer to the root source directory of the project as ${mivt_SOURCE_DIR} and 
# to the root binary directory of the project as ${mivt_BINARY_DIR}.

#definitios

# SETTING CUDA PARAMETERS
#list(APPEND CUDA_NVCC_FLAGS -arch=sm_21)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	list(APPEND CUDA_NVCC_FLAGS -G)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-O0)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wall)
	#add_definitions(-DDISK_TIMING)
else()
	list(APPEND CUDA_NVCC_FLAGS -O3)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wall)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-O3)
endif()

#MIVT
set(MIVT_SOURCE 
		visibleCubes.cpp
	)




# TEST VISIBLE CUBES
set(TEST_VISIBLE_CUBES_SOURCE
		visibleCubes.cpp
   )

set(TEST_VISIBLE_CUBES_CUDA_SOURCE 
		testVisibleCubes_cuda.cu
	)

set(LIBS_NEEDED_TEST_VISIBLE_CUBES ${CUDA_LIBRARIES}) 

cuda_compile(TEST_VISIBLE_CUBES_CUDA_OBJECTS ${TEST_VISIBLE_CUBES_CUDA_SOURCE})
add_executable(testVisibleCubes ${TEST_VISIBLE_CUBES_SOURCE} ${TEST_VISIBLE_CUBES_CUDA_OBJECTS} testVisibleCubes.cpp) 
target_link_libraries(testVisibleCubes ${LIBS_NEEDED_TEST_VISIBLE_CUBES} )

# TEST CONTROL PLANE CACHE 
set(TEST_CONTROL_PLANE_CACHE_SOURCE
		controlPlaneCache.cpp
		linkedList.cpp
		hdf5File.cpp
		memoryCheck.cpp
   )

set(LIBS_NEEDED_TEST_CONTROL_PLANE_CACHE ${LUNCHBOX_LIBRARIES} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES}) 

add_executable(testControlPlaneCache ${TEST_CONTROL_PLANE_CACHE_SOURCE}  testControlPlaneCache.cpp) 
target_link_libraries(testControlPlaneCache ${LIBS_NEEDED_TEST_CONTROL_PLANE_CACHE} )

# TEST CACHE 
set(TEST_CACHE_SOURCE
		visibleCubes.cpp
		linkedList.cpp
		cache.cpp
		controlPlaneCache.cpp
		controlCubeCache.cpp
		hdf5File.cpp
		memoryCheck.cpp
   )

set(TEST_CACHE_CUDA_SOURCE
	testVisibleCubes_cuda.cu
)

set(LIBS_NEEDED_TEST_CACHE ${LUNCHBOX_LIBRARIES} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES}) 

cuda_compile(TEST_CACHE_CUDA_OBJECTS ${TEST_CACHE_CUDA_SOURCE})
add_executable(testCache ${TEST_CACHE_SOURCE} ${TEST_CACHE_CUDA_OBJECTS} testCache.cpp) 
target_link_libraries(testCache ${LIBS_NEEDED_TEST_CACHE} )

# TEST CONTROL CUBE CACHE 
set(TEST_CONTROL_CUBE_CACHE_SOURCE
		controlPlaneCache.cpp
		controlCubeCache.cpp
		hdf5File.cpp
		linkedList.cpp
		memoryCheck.cpp
   )

set(LIBS_NEEDED_TEST_CONTROL_CUBE_CACHE ${LUNCHBOX_LIBRARIES} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES}) 

add_executable(testControlCubeCache ${TEST_CONTROL_CUBE_CACHE_SOURCE} testControlCubeCache.cpp) 
target_link_libraries(testControlCubeCache ${LIBS_NEEDED_TEST_CONTROL_CUBE_CACHE} )
