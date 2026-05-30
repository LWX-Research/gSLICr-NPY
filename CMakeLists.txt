cmake_minimum_required(VERSION 3.10)
project(gSLICr)

IF(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

find_package(CUDA REQUIRED)

# OpenCV is optional — only needed for webcam demo and legacy PNG tools
find_package(OpenCV QUIET)

set(GSLICR_CUDA_ARCH "86" CACHE STRING "CUDA compute capability, e.g. 86 for RTX 3090, 89 for RTX 4080 SUPER")

include_directories(${CUDA_INCLUDE_DIRS})
add_subdirectory(ORUtils)

set(GSLICR_LIB
  gSLICr_Lib/engines/gSLICr_core_engine.h
  gSLICr_Lib/engines/gSLICr_seg_engine.h
  gSLICr_Lib/engines/gSLICr_seg_engine_GPU.h
  gSLICr_Lib/engines/gSLICr_seg_engine_shared.h
  gSLICr_Lib/engines/gSLICr_core_engine.cpp
  gSLICr_Lib/engines/gSLICr_seg_engine.cpp
  gSLICr_Lib/engines/gSLICr_seg_engine_GPU.cu
  gSLICr_Lib/objects/gSLICr_settings.h
  gSLICr_Lib/objects/gSLICr_spixel_info.h
  gSLICr_Lib/gSLICr_defines.h
  gSLICr_Lib/gSLICr.h
)

list(APPEND "-std=c++11 -ftree-vectorize")
SOURCE_GROUP(engines FILES ${GSLICR_LIB})

cuda_add_library(gSLICr_lib
  ${GSLICR_LIB}
  NVTimer.h
  OPTIONS -gencode arch=compute_${GSLICR_CUDA_ARCH},code=sm_${GSLICR_CUDA_ARCH})
target_link_libraries(gSLICr_lib ${CUDA_LIBRARY})

# === Native NPY demo — no OpenCV needed ==========
add_executable(gslicr_npy demo_npy.cpp)
target_link_libraries(gslicr_npy gSLICr_lib ${CUDA_LIBRARY})

# === Python ctypes bridge — no OpenCV needed ======
add_library(gslicr_bridge SHARED gslicr_bridge.cpp)
target_link_libraries(gslicr_bridge gSLICr_lib ${CUDA_LIBRARY})
set_target_properties(gslicr_bridge PROPERTIES POSITION_INDEPENDENT_CODE ON)

# === Legacy OpenCV-based demos (only if OpenCV found) ===
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})

  add_executable(demo demo.cpp)
  target_link_libraries(demo gSLICr_lib ${OpenCV_LIBS})

  add_executable(demo_image demo_image.cpp)
  target_link_libraries(demo_image gSLICr_lib ${OpenCV_LIBS})

  add_executable(demo_benchmark demo_benchmark.cpp)
  target_link_libraries(demo_benchmark gSLICr_lib ${OpenCV_LIBS})

  message(STATUS "OpenCV found — building legacy demo/demo_image/demo_benchmark")
else()
  message(STATUS "OpenCV not found — skipping legacy demos (gslicr_npy + gslicr_bridge still built)")
endif()
