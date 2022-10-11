cmake_minimum_required ( VERSION 3.14 )

project (PatchMatchSDK CXX CUDA )

set(SKDNAME PathcMatchImp)
set(DEMONAME demo)

if(CUDA_ENABLE)
    enable_language(CUDA)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)



if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
	message("${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

#add_subdirectory(PatchMatchImp)
add_subdirectory(Demo)

