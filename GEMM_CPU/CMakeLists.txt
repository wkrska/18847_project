# cmake_minimum_required(VERSION 2.8)
# project( benchmark )
# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )
# add_executable( benchmark benchmark.cpp )
# target_link_libraries( benchmark ${OpenCV_LIBS} )


project( gemm_cpu )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( gemm_cpu gemm_cpu.cpp )
target_link_libraries( gemm_cpu ${OpenCV_LIBS} )