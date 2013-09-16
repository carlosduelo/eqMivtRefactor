# The name of our project is "mivt". CMakeLists files in this project can
# refer to the root source directory of the project as ${mivt_SOURCE_DIR} and 
# to the root binary directory of the project as ${mivt_BINARY_DIR}.

#definitios

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
   )

set(LIBS_NEEDED_TEST_CONTROL_PLANE_CACHE ${LUNCHBOX_LIBRARIES} ${CUDA_LIBRARIES}) 

add_executable(testConstrolPlaneCache ${TEST_CONTROL_PLANE_CACHE_SOURCE}  testControlPlaneCache.cpp) 
target_link_libraries(testConstrolPlaneCache ${LIBS_NEEDED_TEST_CONTROL_PLANE_CACHE} )
