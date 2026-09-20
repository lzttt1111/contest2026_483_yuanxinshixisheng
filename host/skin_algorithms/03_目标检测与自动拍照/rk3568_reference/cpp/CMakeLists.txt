cmake_minimum_required(VERSION 3.10)
project(medvision_rk3568 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(RKNN_INCLUDE_DIR "" CACHE PATH "Directory containing rknn_api.h")
set(RKNN_LIBRARY "" CACHE FILEPATH "Path to librknnrt.so")
set(TURBOJPEG_INCLUDE_DIR "" CACHE PATH "Directory containing turbojpeg.h")
set(TURBOJPEG_LIBRARY "" CACHE FILEPATH "Path to libturbojpeg.a")
set(RGA_INCLUDE_DIR "" CACHE PATH "Directory containing im2d.h")
set(RGA_LIBRARY "" CACHE FILEPATH "Path to librga.so")
if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
  message(FATAL_ERROR "Set RKNN_INCLUDE_DIR to the RKNN 1.5.2 include directory")
endif()
if(NOT EXISTS "${RKNN_LIBRARY}")
  message(FATAL_ERROR "Set RKNN_LIBRARY to the RK356X aarch64 librknnrt.so")
endif()
if(NOT EXISTS "${TURBOJPEG_INCLUDE_DIR}/turbojpeg.h")
  message(FATAL_ERROR "Set TURBOJPEG_INCLUDE_DIR to the cached turbojpeg headers")
endif()
if(NOT EXISTS "${TURBOJPEG_LIBRARY}")
  message(FATAL_ERROR "Set TURBOJPEG_LIBRARY to the aarch64 libturbojpeg.a")
endif()
if(NOT EXISTS "${RGA_INCLUDE_DIR}/im2d.h")
  message(FATAL_ERROR "Set RGA_INCLUDE_DIR to the cached RK356X RGA headers")
endif()
if(NOT EXISTS "${RGA_LIBRARY}")
  message(FATAL_ERROR "Set RGA_LIBRARY to the RK356X aarch64 librga.so")
endif()

add_library(medvision_rknn_backend STATIC
  src/rknn_backend.cpp
)
target_include_directories(medvision_rknn_backend
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${RKNN_INCLUDE_DIR}"
)
target_compile_options(medvision_rknn_backend PRIVATE
  -Wall -Wextra -Wpedantic
)

add_library(medvision_rk3568_pipeline STATIC
  src/rk3568_pipeline.cpp
)
target_include_directories(medvision_rk3568_pipeline
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_include_directories(medvision_rk3568_pipeline
  SYSTEM PUBLIC
    "${TURBOJPEG_INCLUDE_DIR}"
    "${RGA_INCLUDE_DIR}"
)
target_compile_options(medvision_rk3568_pipeline PRIVATE
  -Wall -Wextra -Wpedantic
)
target_link_libraries(medvision_rk3568_pipeline PUBLIC
  medvision_rknn_backend
  "${TURBOJPEG_LIBRARY}"
  "${RGA_LIBRARY}"
)

add_executable(medvision_rk3568 src/medvision_rk3568.cpp)
target_link_libraries(medvision_rk3568 PRIVATE
  medvision_rk3568_pipeline
  "${RKNN_LIBRARY}"
  pthread
  dl
)
set_target_properties(medvision_rk3568 PROPERTIES
  BUILD_RPATH "/usr/lib"
  INSTALL_RPATH "/usr/lib"
)

add_executable(rknn_probe src/rknn_probe.cpp)
target_link_libraries(rknn_probe PRIVATE
  medvision_rknn_backend
  "${RKNN_LIBRARY}"
)

add_executable(rknn_batch_probe src/rknn_batch_probe.cpp)
target_link_libraries(rknn_batch_probe PRIVATE
  medvision_rknn_backend
  "${RKNN_LIBRARY}"
)
