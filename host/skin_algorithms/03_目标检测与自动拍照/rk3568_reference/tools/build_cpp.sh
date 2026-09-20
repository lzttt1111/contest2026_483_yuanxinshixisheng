#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
DEPLOY_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
CACHE_ROOT="${MEDVISION_RK3568_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/medvision-rk3568}"
TOOLCHAIN_ROOT="${CACHE_ROOT}/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu"
CXX="${TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-g++"
BUILD_DIR="${DEPLOY_ROOT}/build"

RKNN_ROOT="${CACHE_ROOT}/rknpu2/runtime/RK356X/Linux/librknn_api"
RKNN_INCLUDE="${RKNN_ROOT}/include"
RKNN_LIBRARY="${RKNN_ROOT}/aarch64/librknnrt.so"
ZOO_3RDPARTY="${CACHE_ROOT}/rknn_model_zoo/3rdparty"
TURBOJPEG_INCLUDE="${ZOO_3RDPARTY}/jpeg_turbo/include"
TURBOJPEG_LIBRARY="${ZOO_3RDPARTY}/jpeg_turbo/Linux/aarch64/libturbojpeg.a"
RGA_INCLUDE="${ZOO_3RDPARTY}/librga/include"
RGA_LIBRARY="${ZOO_3RDPARTY}/librga/Linux/aarch64/librga.so"

REQUIRED_FILES=(
  "${CXX}"
  "${RKNN_INCLUDE}/rknn_api.h"
  "${RKNN_LIBRARY}"
  "${TURBOJPEG_INCLUDE}/turbojpeg.h"
  "${TURBOJPEG_LIBRARY}"
  "${RGA_INCLUDE}/im2d.h"
  "${RGA_LIBRARY}"
)
for required in "${REQUIRED_FILES[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "BUILD_CPP=FAIL missing=${required}" >&2
    exit 1
  fi
done

mkdir -p "${BUILD_DIR}"
COMMON_FLAGS=(
  -std=c++14
  -O2
  -Wall
  -Wextra
  -Wpedantic
  -I "${DEPLOY_ROOT}/cpp/include"
  -I "${RKNN_INCLUDE}"
)

PROBE_COMMAND=(
  "${CXX}"
  "${COMMON_FLAGS[@]}"
  "${DEPLOY_ROOT}/cpp/src/rknn_backend.cpp"
  "${DEPLOY_ROOT}/cpp/src/rknn_probe.cpp"
  "${RKNN_LIBRARY}"
  -Wl,-rpath,/usr/lib
  -o "${BUILD_DIR}/rknn_probe"
)
"${PROBE_COMMAND[@]}"

BATCH_PROBE_COMMAND=(
  "${CXX}"
  "${COMMON_FLAGS[@]}"
  "${DEPLOY_ROOT}/cpp/src/rknn_backend.cpp"
  "${DEPLOY_ROOT}/cpp/src/rknn_batch_probe.cpp"
  "${RKNN_LIBRARY}"
  -Wl,-rpath,/usr/lib
  -o "${BUILD_DIR}/rknn_batch_probe"
)
"${BATCH_PROBE_COMMAND[@]}"

SERVICE_COMMAND=(
  "${CXX}"
  "${COMMON_FLAGS[@]}"
  -isystem "${TURBOJPEG_INCLUDE}"
  -isystem "${RGA_INCLUDE}"
  "${DEPLOY_ROOT}/cpp/src/rknn_backend.cpp"
  "${DEPLOY_ROOT}/cpp/src/rk3568_pipeline.cpp"
  "${DEPLOY_ROOT}/cpp/src/medvision_rk3568.cpp"
  "${RKNN_LIBRARY}"
  "${TURBOJPEG_LIBRARY}"
  "${RGA_LIBRARY}"
  -Wl,-rpath,/usr/lib
  -lpthread
  -ldl
  -o "${BUILD_DIR}/medvision_rk3568"
)
"${SERVICE_COMMAND[@]}"

echo "BUILD_CPP=PASS"
sha256sum "${BUILD_DIR}/rknn_probe" "${BUILD_DIR}/rknn_batch_probe" "${BUILD_DIR}/medvision_rk3568"
