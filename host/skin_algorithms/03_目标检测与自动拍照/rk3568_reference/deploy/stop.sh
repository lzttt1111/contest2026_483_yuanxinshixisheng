#!/bin/sh
set -eu

DEPLOY_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
RUN_DIR="${DEPLOY_ROOT}/run"
PID_PATH="${RUN_DIR}/medvision_rk3568.pid"
GOVERNOR_PATH="${RUN_DIR}/governors.env"

if [ -f "${PID_PATH}" ]; then
  service_pid="$(cat "${PID_PATH}")"
  service_exe="$(readlink "/proc/${service_pid}/exe" 2>/dev/null || true)"
  if [ "${service_exe##*/}" = "medvision_rk3568" ]; then
    kill "${service_pid}" 2>/dev/null || true
    attempts=0
    while [ "${attempts}" -lt 50 ] && kill -0 "${service_pid}" 2>/dev/null; do
      sleep 0.1
      attempts=$((attempts + 1))
    done
    service_exe="$(readlink "/proc/${service_pid}/exe" 2>/dev/null || true)"
    if [ "${service_exe##*/}" = "medvision_rk3568" ]; then
      kill -9 "${service_pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${PID_PATH}"
fi

if [ -f "${GOVERNOR_PATH}" ]; then
  . "${GOVERNOR_PATH}"
  NPU_GOVERNOR=/sys/class/devfreq/fde40000.npu/governor
  DMC_GOVERNOR=/sys/class/devfreq/dmc/governor
  CPU_GOVERNOR=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
  if [ -n "${NPU_PREVIOUS:-}" ] && [ -w "${NPU_GOVERNOR}" ]; then
    echo "${NPU_PREVIOUS}" > "${NPU_GOVERNOR}"
  fi
  if [ -n "${DMC_PREVIOUS:-}" ] && [ -w "${DMC_GOVERNOR}" ]; then
    echo "${DMC_PREVIOUS}" > "${DMC_GOVERNOR}"
  fi
  if [ -n "${CPU_PREVIOUS:-}" ] && [ -w "${CPU_GOVERNOR}" ]; then
    echo "${CPU_PREVIOUS}" > "${CPU_GOVERNOR}"
  fi
  rm -f "${GOVERNOR_PATH}"
fi

echo "STOP=PASS"
