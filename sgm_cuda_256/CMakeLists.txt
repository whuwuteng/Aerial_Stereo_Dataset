#    This file is part of sgm. (https://github.com/dhernandez0/sgm).
#
#    Copyright (c) 2016 Daniel Hernandez Juarez.

#    sgm is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    sgm is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 2.4)
project(sgm)

#set(CMAKE_CXX_FLAGS "-lpthread")
#set(CMAKE_CXX_FLAGS "-dl")
#set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)
find_package(PNG REQUIRED)

include_directories(${PNG_INCLUDE_DIR})

# because this depends on the GPU card
# use cmake -DCMAKE_CUDA_FLAGS=”-arch=sm_50” ..
#set(
#    CUDA_NVCC_FLAGS
#    ${CUDA_NVCC_FLAGS};
#    -O3 -lineinfo
#    -gencode
#    arch=compute_50,code=sm_50
#    )

include_directories($ENV{HOME}/thirdparty)
MESSAGE(STATUS "$ENV{HOME}/thirdparty")
include_directories(${CMAKE_CURRENT_LIST_DIR}/opencv)

cuda_add_executable(
    sgm
    main.cu median_filter.cu hamming_cost.cu disparity_method.cu debug.cu costs.cu
    opencv/core.cpp opencv/cvstd.cpp opencv/imgproc.cpp)

# do not use OpenCV
#target_link_libraries( sgm ${OpenCV_LIBS} )
#target_link_libraries(sgm Threads::Threads ${CMAKE_DL_LIBS})
#target_link_libraries(sgm lpthread dl)
target_link_libraries(sgm ${PNG_LIBRARY})
