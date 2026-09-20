#!/bin/sh
set -eu

DEPLOY_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
CONFIG_PATH="${DEPLOY_ROOT}/config/service.env"
RUN_DIR="${DEPLOY_ROOT}/run"
LOG_DIR="${DEPLOY_ROOT}/logs"
PID_PATH="${RUN_DIR}/medvision_rk3568.pid"
GOVERNOR_PATH="${RUN_DIR}/governors.env"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "START=FAIL missing_config=${CONFIG_PATH}" >&2
  exit 1
fi
. "${CONFIG_PATH}"

for required in "${DEPLOY_ROOT}/bin/medvision_rk3568" "${DEPLOY_ROOT}/${DEVICE_MODEL}" "${DEPLOY_ROOT}/${CONTACT_MODEL}"; do
  if [ ! -f "${required}" ]; then
    echo "START=FAIL missing=${required}" >&2
    exit 1
  fi
done

device_actual_sha256="$(sha256sum "${DEPLOY_ROOT}/${DEVICE_MODEL}" | awk '{print $1}')"
contact_actual_sha256="$(sha256sum "${DEPLOY_ROOT}/${CONTACT_MODEL}" | awk '{print $1}')"
if [ "${device_actual_sha256}" != "${DEVICE_SHA256}" ]; then
  echo "START=FAIL device_sha256_mismatch expected=${DEVICE_SHA256} actual=${device_actual_sha256}" >&2
  exit 1
fi
if [ "${contact_actual_sha256}" != "${CONTACT_SHA256}" ]; then
  echo "START=FAIL contact_sha256_mismatch expected=${CONTACT_SHA256} actual=${contact_actual_sha256}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
if [ -f "${PID_PATH}" ]; then
  old_pid="$(cat "${PID_PATH}")"
  old_exe="$(readlink "/proc/${old_pid}/exe" 2>/dev/null || true)"
  if [ "${old_exe##*/}" = "medvision_rk3568" ]; then
    echo "START=PASS already_running=true pid=${old_pid}"
    exit 0
  fi
  rm -f "${PID_PATH}"
fi

NPU_GOVERNOR=/sys/class/devfreq/fde40000.npu/governor
DMC_GOVERNOR=/sys/class/devfreq/dmc/governor
CPU_GOVERNOR=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
if [ "${ENABLE_PERFORMANCE_MODE:-0}" = "1" ]; then
  {
    echo "NPU_PREVIOUS=$(cat "${NPU_GOVERNOR}")"
    echo "DMC_PREVIOUS=$(cat "${DMC_GOVERNOR}")"
    echo "CPU_PREVIOUS=$(cat "${CPU_GOVERNOR}")"
  } > "${GOVERNOR_PATH}"
  echo performance > "${NPU_GOVERNOR}"
  echo performance > "${DMC_GOVERNOR}"
  echo performance > "${CPU_GOVERNOR}"
fi

cd "${DEPLOY_ROOT}"
export LD_LIBRARY_PATH="${DEPLOY_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
set -- ./bin/medvision_rk3568
set -- "$@" --device-model "${DEVICE_MODEL}"
set -- "$@" --contact-model "${CONTACT_MODEL}"
set -- "$@" --device-sha256 "${DEVICE_SHA256}"
set -- "$@" --contact-sha256 "${CONTACT_SHA256}"
set -- "$@" --quantization "${MODEL_QUANTIZATION}"
set -- "$@" --board-model "${BOARD_MODEL}"
set -- "$@" --host "${SERVICE_HOST}"
set -- "$@" --port "${SERVICE_PORT}"
set -- "$@" --device-confidence "${DEVICE_CONFIDENCE}"
set -- "$@" --nms-iou "${NMS_IOU}"
set -- "$@" --device-stale-ms "${DEVICE_STALE_MS}"
set -- "$@" --contact-stale-ms "${CONTACT_STALE_MS}"
set -- "$@" --contact-interval "${CONTACT_INTERVAL_VALID_FRAMES}"
nohup "$@" > "${LOG_DIR}/service.log" 2>&1 &
service_pid=$!
echo "${service_pid}" > "${PID_PATH}"
sleep 1

service_exe="$(readlink "/proc/${service_pid}/exe" 2>/dev/null || true)"
if [ "${service_exe##*/}" != "medvision_rk3568" ]; then
  echo "START=FAIL service_exited=true" >&2
  "${DEPLOY_ROOT}/deploy/stop.sh" || true
  tail -n 40 "${LOG_DIR}/service.log" >&2 || true
  exit 1
fi

echo "START=PASS pid=${service_pid} url=http://${SERVICE_HOST}:${SERVICE_PORT}"
