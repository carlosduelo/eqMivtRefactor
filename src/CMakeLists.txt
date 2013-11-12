# The name of our project is "mivt". CMakeLists files in this project can
# refer to the root source directory of the project as ${mivt_SOURCE_DIR} and 
# to the root binary directory of the project as ${mivt_BINARY_DIR}.

#definitios
	#add_definitions(-DDISK_TIMING)
	add_definitions(-DTIMING)

# SETTING CUDA PARAMETERS
list(APPEND CUDA_NVCC_FLAGS -arch=sm_13)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#	list(APPEND CUDA_NVCC_FLAGS -G)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-O0)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wall)
	set( CMAKE_CXX_FLAGS "-Wall -g -O0 -Wno-unknown-pragmas" )
else()
	list(APPEND CUDA_NVCC_FLAGS -O3)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wall)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-O3)
	set( CMAKE_CXX_FLAGS "-Wall -O3 -Wno-unknown-pragmas" )
endif()

#MIVT
set(MIVT_SOURCE 
	renderOGL.cpp
	renderWorkers.cpp
	render.cpp
	resourcesManager.cpp
	colorManager.cpp
	cuda_help.cpp
	octreeManager.cpp
	cacheManager.cpp
	controlCache.cpp
	linkedList.cpp
	cache.cpp
	controlPlaneCache.cpp
	controlCubeCPUCache.cpp
	controlCubeCache.cpp
	hdf5File.cpp
	memoryCheck.cpp
	octree.cpp
	octreeContainer.cpp
	channel.cpp
	error.cpp
	view.cpp
	node.cpp
	pipe.cpp
	window.cpp
	surfaceInfo.cpp
	frameData.cpp
	initData.cpp
	localInitData.cpp
	config.cpp
	eqMivt.cpp
	)

set(MIVT_CUDA_SOURCE
	deviceCode.cu
)

set(LIBS_NEEDED_MIVT ${EQUALIZER_LIBRARIES} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES} ) 

cuda_compile(MIVT_CUDA_OBJECTS ${MIVT_CUDA_SOURCE})
add_executable(mivt ${MIVT_SOURCE} ${MIVT_CUDA_OBJECTS} main.cpp) 
target_link_libraries(mivt ${LIBS_NEEDED_MIVT} )

# CREATE OCTREE 
set(CREATE_OCTREE_SOURCE
		cuda_help.cpp
		controlCache.cpp
		octreeConstructor.cpp
		linkedList.cpp
		cache.cpp
		controlPlaneCache.cpp
		controlCubeCache.cpp
		controlCubeCPUCache.cpp
		hdf5File.cpp
		memoryCheck.cpp
   )

set(CREATE_OCTREE_CUDA_SOURCE
	createOctree_cuda.cu
)

set(LIBS_NEEDED_CREATE_OCTREE ${LUNCHBOX_LIBRARIES} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES} ) 

cuda_compile(CREATE_OCTREE_CUDA_OBJECTS ${CREATE_OCTREE_CUDA_SOURCE})
add_executable(createOctree ${CREATE_OCTREE_SOURCE} ${CREATE_OCTREE_CUDA_OBJECTS} createOctree.cpp) 
target_link_libraries(createOctree ${LIBS_NEEDED_CREATE_OCTREE} )

# TRANSFER FUNCTION CREATOR
add_executable(transferFunctionCreator transferFunctionCreator.cpp) 
target_link_libraries(transferFunctionCreator GL glut) 

# PRINT OCTREE
add_executable(printOctree printOctree.cpp) 


# TEST CONTROL PLANE CACHE 
set(TEST_CONTROL_PLANE_CACHE_SOURCE
		controlCache.cpp
		controlPlaneCache.cpp
		linkedList.cpp
		hdf5File.cpp
		memoryCheck.cpp
   )

set(LIBS_NEEDED_TEST_CONTROL_PLANE_CACHE ${LUNCHBOX_LIBRARIES} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES}) 

add_executable(testControlPlaneCache ${TEST_CONTROL_PLANE_CACHE_SOURCE}  testControlPlaneCache.cpp) 
target_link_libraries(testControlPlaneCache ${LIBS_NEEDED_TEST_CONTROL_PLANE_CACHE} )

# TEST CONTROL CUBE CPU CACHE 
set(TEST_CONTROL_CUBE_CPU_CACHE_SOURCE
		cuda_help.cpp
		controlCache.cpp
		controlPlaneCache.cpp
		controlCubeCPUCache.cpp
		hdf5File.cpp
		linkedList.cpp
		memoryCheck.cpp
   )

set(LIBS_NEEDED_TEST_CONTROL_CUBE_CPU_CACHE ${LUNCHBOX_LIBRARIES} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES}) 

add_executable(testControlCubeCPUCache ${TEST_CONTROL_CUBE_CPU_CACHE_SOURCE} testControlCubeCPUCache.cpp) 
target_link_libraries(testControlCubeCPUCache ${LIBS_NEEDED_TEST_CONTROL_CUBE_CPU_CACHE} )

# TEST CONTROL CUBE CACHE 
set(TEST_CONTROL_CUBE_CACHE_SOURCE
		cuda_help.cpp
		controlCache.cpp
		controlPlaneCache.cpp
		controlCubeCPUCache.cpp
		controlCubeCache.cpp
		hdf5File.cpp
		linkedList.cpp
		memoryCheck.cpp
   )

set(LIBS_NEEDED_TEST_CONTROL_CUBE_CACHE ${LUNCHBOX_LIBRARIES} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES}) 

add_executable(testControlCubeCache ${TEST_CONTROL_CUBE_CACHE_SOURCE} testControlCubeCache.cpp) 
target_link_libraries(testControlCubeCache ${LIBS_NEEDED_TEST_CONTROL_CUBE_CACHE} )

# TEST RAYCASTING 
set(TEST_RAYCASTING_SOURCE
	renderWorkers.cpp
	render.cpp
	renderPNG.cpp
	resourcesManager.cpp
	colorManager.cpp
	cuda_help.cpp
	octreeManager.cpp
	cacheManager.cpp
	controlCache.cpp
	linkedList.cpp
	cache.cpp
	controlPlaneCache.cpp
	controlCubeCPUCache.cpp
	controlCubeCache.cpp
	hdf5File.cpp
	memoryCheck.cpp
	octree.cpp
	octreeContainer.cpp
   )

set(TEST_RAYCASTING_CUDA_SOURCE
	deviceCode.cu
)

set(LIBS_NEEDED_TEST_RAYCASTING ${LUNCHBOX_LIBRARIES} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES} freeimage) 

cuda_compile(TEST_RAYCASTING_CUDA_OBJECTS ${TEST_RAYCASTING_CUDA_SOURCE})
add_executable(testRayCasting ${TEST_RAYCASTING_SOURCE} ${TEST_RAYCASTING_CUDA_OBJECTS} testRayCasting.cpp) 
target_link_libraries(testRayCasting ${LIBS_NEEDED_TEST_RAYCASTING} )

# TEST MULTITHREADING 
set(TEST_MULTITHREADING_SOURCE
	renderWorkers.cpp
	render.cpp
	renderPNG.cpp
	resourcesManager.cpp
	colorManager.cpp
	cuda_help.cpp
	octreeManager.cpp
	cacheManager.cpp
	controlCache.cpp
	linkedList.cpp
	cache.cpp
	controlPlaneCache.cpp
	controlCubeCache.cpp
	hdf5File.cpp
	memoryCheck.cpp
	octree.cpp
	octreeContainer.cpp
   )

set(TEST_MULTITHREADING_CUDA_SOURCE
	rayCaster_cuda.cu
	octree_cuda.cu
)

set(LIBS_NEEDED_TEST_MULTITHREADING ${LUNCHBOX_LIBRARIES} ${CUDA_LIBRARIES} ${HDF5_LIBRARIES} freeimage) 

#cuda_compile(TEST_MULTITHREADING_CUDA_OBJECTS ${TEST_RAYCASTING_CUDA_SOURCE})
#add_executable(testMultithreading ${TEST_MULTITHREADING_SOURCE} ${TEST_RAYCASTING_CUDA_OBJECTS} testMultithreading.cpp) 
#target_link_libraries(testMultithreading ${LIBS_NEEDED_TEST_MULTITHREADING} )
