"use strict";

const API_ROOT = "http://127.0.0.1:18080";
const CAPTURE_WIDTH = 960;
const CAPTURE_HEIGHT = 540;
const JPEG_QUALITY = 0.70;

const elementIds = [
  "connection-state", "inference-toggle", "camera-stop", "camera", "view",
  "empty-state", "fps", "rtt", "board-time", "camera-select",
  "camera-start", "inference-start", "stream-reset", "runtime",
  "board-model", "rknn-version", "driver-version", "rail-connection",
  "connection-age", "device-model", "device-sha", "contact-model",
  "contact-sha", "model-quantization", "timing-decode", "timing-pre",
  "timing-npu",
  "timing-post", "timing-total", "timing-rtt", "event-log", "clear-log",
];
const elements = Object.fromEntries(
  elementIds.map(function (id) { return [id, document.getElementById(id)]; })
);

const captureCanvas = document.createElement("canvas");
captureCanvas.width = CAPTURE_WIDTH;
captureCanvas.height = CAPTURE_HEIGHT;
const captureContext = captureCanvas.getContext("2d", { alpha: false });
const viewContext = elements.view.getContext("2d", { alpha: false });

const state = {
  stream: null,
  connected: false,
  connectedAt: 0,
  inferenceActive: false,
  inFlight: false,
  latest: null,
  frameId: 0,
  responseCount: 0,
  fpsWindowStarted: performance.now(),
  inferTimer: 0,
  healthTimer: 0,
  lastErrorLogAt: 0,
};

function byId(id) {
  return elements[id];
}

function timestamp() {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    fractionalSecondDigits: 3,
    hour12: false,
  }).format(new Date());
}

function log(message, kind) {
  const item = document.createElement("li");
  const time = document.createElement("time");
  const text = document.createElement("span");
  time.textContent = timestamp();
  text.textContent = message;
  if (kind) text.className = kind;
  item.append(time, text);
  byId("event-log").prepend(item);
  while (byId("event-log").children.length > 8) {
    byId("event-log").lastElementChild.remove();
  }
}

function shortHash(value) {
  if (!value || value.length < 20) return value || "—";
  return value.slice(0, 12) + "…" + value.slice(-8);
}

function modelName(path) {
  return path ? path.split(/[\\/]/).pop() : "—";
}

function formatMs(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(1) : "—";
}

function setConnection(connected, label) {
  state.connected = connected;
  const header = byId("connection-state");
  const rail = byId("rail-connection");
  header.className = "connection-state " + (connected ? "connected" : "failed");
  header.querySelector("span").textContent = label;
  rail.className = connected ? "connected" : "failed";
  rail.replaceChildren();
  const dot = document.createElement("span");
  dot.className = "status-dot";
  dot.setAttribute("aria-hidden", "true");
  rail.append(dot, document.createTextNode(connected ? "已连接" : "未连接"));
  if (!connected && state.inferenceActive) {
    stopInference("板卡连接中断，推理已停止");
  }
  updateControls();
}

function updateControls() {
  const cameraReady = Boolean(state.stream);
  const canInfer = cameraReady && state.connected;
  byId("camera-start").disabled = cameraReady;
  byId("camera-stop").disabled = !cameraReady;
  byId("inference-start").disabled = !canInfer || state.inferenceActive;
  byId("inference-toggle").disabled = !canInfer;
  byId("stream-reset").disabled = !state.connected;
  byId("camera-select").disabled = cameraReady;
  const toggle = byId("inference-toggle");
  toggle.classList.toggle("active", state.inferenceActive);
  toggle.querySelector("span").textContent =
    state.inferenceActive ? "停止推理" : "启动推理";
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(function () { controller.abort(); }, timeoutMs);
  try {
    return await fetch(url, Object.assign({}, options || {}, {
      cache: "no-store",
      signal: controller.signal,
    }));
  } finally {
    clearTimeout(timeout);
  }
}

async function checkHealth() {
  try {
    const response = await fetchWithTimeout(API_ROOT + "/health", {}, 1800);
    if (!response.ok) throw new Error("HTTP " + response.status);
    const health = await response.json();
    if (!health.ok || health.runtime_provider !== "RKNN_NPU") {
      throw new Error("服务未报告 RKNN_NPU");
    }
    const wasConnected = state.connected;
    if (!wasConnected) state.connectedAt = Date.now();
    setConnection(true, "板卡已连接");
    byId("runtime").textContent = health.runtime_provider;
    byId("board-model").textContent = health.board_model || "—";
    byId("model-quantization").textContent = health.model_quantization || "—";
    byId("rknn-version").textContent = (health.rknn && health.rknn.api_version) || "—";
    byId("driver-version").textContent = (health.rknn && health.rknn.driver_version) || "—";
    byId("device-model").textContent = modelName(
      health.models && health.models.device && health.models.device.path
    );
    byId("contact-model").textContent = modelName(
      health.models && health.models.contact && health.models.contact.path
    );
    byId("device-sha").textContent = shortHash(
      health.models && health.models.device && health.models.device.sha256
    );
    byId("contact-sha").textContent = shortHash(
      health.models && health.models.contact && health.models.contact.sha256
    );
    if (!wasConnected) log("RK3568 NPU 服务连接成功", "success");
  } catch (error) {
    const wasConnected = state.connected;
    setConnection(false, "板卡未连接");
    if (wasConnected || performance.now() - state.lastErrorLogAt > 10000) {
      const reason = error.name === "AbortError" ? "请求超时" : error.message;
      log("连接失败：" + reason, "error");
      state.lastErrorLogAt = performance.now();
    }
  } finally {
    clearTimeout(state.healthTimer);
    state.healthTimer = setTimeout(checkHealth, state.connected ? 5000 : 2000);
  }
}

function updateConnectionAge() {
  if (!state.connected || !state.connectedAt) {
    byId("connection-age").textContent = "—";
    return;
  }
  const seconds = Math.max(0, Math.floor((Date.now() - state.connectedAt) / 1000));
  const hours = String(Math.floor(seconds / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
  const remainder = String(seconds % 60).padStart(2, "0");
  byId("connection-age").textContent = hours + ":" + minutes + ":" + remainder;
}

async function listCameras() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  const allDevices = await navigator.mediaDevices.enumerateDevices();
  const devices = allDevices.filter(function (item) {
    return item.kind === "videoinput";
  });
  const selected = byId("camera-select").value;
  byId("camera-select").replaceChildren();
  devices.forEach(function (device, index) {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || "摄像头 " + (index + 1);
    byId("camera-select").append(option);
  });
  if (devices.some(function (device) { return device.deviceId === selected; })) {
    byId("camera-select").value = selected;
  }
}

async function startCamera() {
  try {
    const deviceId = byId("camera-select").value;
    const videoConstraints = {
      width: { ideal: CAPTURE_WIDTH },
      height: { ideal: CAPTURE_HEIGHT },
      frameRate: { ideal: 30, max: 30 },
    };
    if (deviceId) videoConstraints.deviceId = { exact: deviceId };
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: videoConstraints,
    });
    byId("camera").srcObject = state.stream;
    await byId("camera").play();
    byId("empty-state").classList.add("hidden");
    await listCameras();
    const label = state.stream.getVideoTracks()[0].label || "默认摄像头";
    log("摄像头已启动：" + label, "success");
    updateControls();
  } catch (error) {
    state.stream = null;
    log("无法打开摄像头：" + error.message, "error");
    updateControls();
  }
}

function stopCamera() {
  stopInference();
  if (state.stream) {
    state.stream.getTracks().forEach(function (track) { track.stop(); });
  }
  state.stream = null;
  byId("camera").srcObject = null;
  byId("empty-state").classList.remove("hidden");
  state.latest = null;
  clearPerformance();
  log("摄像头已停止");
  updateControls();
}

function mirroredBox(box) {
  if (!Array.isArray(box) || box.length !== 4) return null;
  return [CAPTURE_WIDTH - box[2], box[1], CAPTURE_WIDTH - box[0], box[3]];
}

function drawDetection(result) {
  if (!result) return;
  const box = mirroredBox(result.device && result.device.bbox);
  if (box) {
    const x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
    viewContext.save();
    viewContext.strokeStyle = "#35d6e2";
    viewContext.lineWidth = 2;
    viewContext.strokeRect(x1, y1, x2 - x1, y2 - y1);
    viewContext.fillStyle = "#35d6e2";
    viewContext.font = "600 13px Segoe UI, sans-serif";
    viewContext.fillText(
      "设备 " + (Number(result.device.score) * 100).toFixed(1) + "%",
      x1 + 5,
      Math.max(16, y1 - 7)
    );
    viewContext.restore();
  }
  const point = result.contact && result.contact.point;
  if (Array.isArray(point) && point.length === 2) {
    const x = CAPTURE_WIDTH - point[0];
    const y = point[1];
    viewContext.save();
    viewContext.strokeStyle = "#ff6c61";
    viewContext.fillStyle = "#ff6c6133";
    viewContext.lineWidth = 2;
    viewContext.beginPath();
    viewContext.arc(x, y, 14, 0, Math.PI * 2);
    viewContext.fill();
    viewContext.stroke();
    viewContext.beginPath();
    viewContext.moveTo(x - 20, y);
    viewContext.lineTo(x + 20, y);
    viewContext.moveTo(x, y - 20);
    viewContext.lineTo(x, y + 20);
    viewContext.stroke();
    viewContext.restore();
  }
}

function render() {
  if (
    state.stream &&
    byId("camera").readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    viewContext.save();
    viewContext.translate(CAPTURE_WIDTH, 0);
    viewContext.scale(-1, 1);
    viewContext.drawImage(
      byId("camera"), 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT
    );
    viewContext.restore();
    drawDetection(state.latest);
  } else {
    viewContext.fillStyle = "#050b11";
    viewContext.fillRect(0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
  }
  requestAnimationFrame(render);
}

function canvasBlob() {
  return new Promise(function (resolve, reject) {
    captureCanvas.toBlob(function (blob) {
      if (blob) resolve(blob);
      else reject(new Error("JPEG 编码失败"));
    }, "image/jpeg", JPEG_QUALITY);
  });
}

function updateTimings(result, rtt) {
  const timings = result.timings_ms || {};
  const pre = Number(timings.device_preprocess || 0) +
    Number(timings.contact_preprocess || 0);
  const npu = Number(timings.device_inference || 0) +
    Number(timings.contact_inference || 0);
  const post = Number(timings.device_postprocess || 0) +
    Number(timings.contact_postprocess || 0);
  byId("timing-decode").textContent = formatMs(timings.decode);
  byId("timing-pre").textContent = formatMs(pre);
  byId("timing-npu").textContent = formatMs(npu);
  byId("timing-post").textContent = formatMs(post);
  byId("timing-total").textContent = formatMs(timings.total);
  byId("timing-rtt").textContent = formatMs(rtt);
  byId("rtt").textContent = formatMs(rtt);
  byId("board-time").textContent = formatMs(timings.total);
}

function updateFps() {
  state.responseCount += 1;
  const now = performance.now();
  const elapsed = now - state.fpsWindowStarted;
  if (elapsed >= 1000) {
    byId("fps").textContent =
      (state.responseCount * 1000 / elapsed).toFixed(1);
    state.responseCount = 0;
    state.fpsWindowStarted = now;
  }
}

async function inferOneFrame() {
  if (!state.inferenceActive || state.inFlight || !state.stream) return;
  state.inFlight = true;
  try {
    captureContext.drawImage(
      byId("camera"), 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT
    );
    const jpeg = await canvasBlob();
    const frameId = ++state.frameId;
    const captureTime = Date.now();
    const started = performance.now();
    const response = await fetchWithTimeout(
      API_ROOT + "/v1/stream/frame",
      {
        method: "POST",
        headers: {
          "Content-Type": "image/jpeg",
          "X-Frame-Id": String(frameId),
          "X-Capture-Timestamp-Ms": String(captureTime),
        },
        body: jpeg,
      },
      3500
    );
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(
        (payload.error && payload.error.message) || "HTTP " + response.status
      );
    }
    if (payload.result.runtime_provider !== "RKNN_NPU") {
      throw new Error("响应不是 RKNN_NPU 推理结果");
    }
    state.latest = payload.result;
    updateTimings(payload.result, performance.now() - started);
    updateFps();
  } catch (error) {
    if (performance.now() - state.lastErrorLogAt > 3000) {
      const reason = error.name === "AbortError" ? "请求超时" : error.message;
      log("帧推理失败：" + reason, "error");
      state.lastErrorLogAt = performance.now();
    }
  } finally {
    state.inFlight = false;
    if (state.inferenceActive) {
      state.inferTimer = setTimeout(inferOneFrame, 0);
    }
  }
}

function startInference() {
  if (!state.stream || !state.connected || state.inferenceActive) return;
  state.inferenceActive = true;
  state.responseCount = 0;
  state.fpsWindowStarted = performance.now();
  updateControls();
  log("板端流推理已启动", "success");
  inferOneFrame();
}

function stopInference(message) {
  if (!state.inferenceActive) return;
  state.inferenceActive = false;
  clearTimeout(state.inferTimer);
  updateControls();
  log(message || "板端流推理已停止");
}

function clearPerformance() {
  [
    "fps", "rtt", "board-time", "timing-decode", "timing-pre", "timing-npu",
    "timing-post", "timing-total", "timing-rtt",
  ].forEach(function (id) { byId(id).textContent = "—"; });
}

async function resetStream() {
  try {
    const response = await fetchWithTimeout(
      API_ROOT + "/v1/stream/reset", { method: "POST" }, 1800
    );
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(
        (payload.error && payload.error.message) || "HTTP " + response.status
      );
    }
    state.latest = null;
    state.frameId = 0;
    clearPerformance();
    log("板端状态与平滑历史已重置", "success");
  } catch (error) {
    log("重置失败：" + error.message, "error");
  }
}

byId("camera-start").addEventListener("click", startCamera);
byId("camera-stop").addEventListener("click", stopCamera);
byId("inference-start").addEventListener("click", startInference);
byId("inference-toggle").addEventListener("click", function () {
  if (state.inferenceActive) stopInference();
  else startInference();
});
byId("stream-reset").addEventListener("click", resetStream);
byId("clear-log").addEventListener("click", function () {
  byId("event-log").replaceChildren();
});
window.addEventListener("beforeunload", function () {
  if (state.stream) {
    state.stream.getTracks().forEach(function (track) { track.stop(); });
  }
});

if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  log("当前浏览器不支持 getUserMedia", "error");
  byId("camera-start").disabled = true;
} else {
  listCameras().catch(function () {});
}

log("应用就绪");
render();
checkHealth();
setInterval(updateConnectionAge, 1000);
