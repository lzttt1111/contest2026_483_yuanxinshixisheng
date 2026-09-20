#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "medvision/rk3568_pipeline.hpp"

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaximumHeaderBytes = 64U * 1024U;
constexpr std::size_t kMaximumImageBytes = 10U * 1024U * 1024U;
constexpr float kBoxEmaAlpha = 0.45F;
constexpr float kPointEmaAlpha = 0.55F;

std::int64_t monotonic_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             Clock::now().time_since_epoch())
      .count();
}

std::string lower(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
      });
  return value;
}

std::string trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1U);
}

std::string json_escape(const std::string& value) {
  std::ostringstream stream;
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        stream << "\\\"";
        break;
      case '\\':
        stream << "\\\\";
        break;
      case '\b':
        stream << "\\b";
        break;
      case '\f':
        stream << "\\f";
        break;
      case '\n':
        stream << "\\n";
        break;
      case '\r':
        stream << "\\r";
        break;
      case '\t':
        stream << "\\t";
        break;
      default:
        if (character < 0x20U) {
          const char* hex = "0123456789abcdef";
          stream << "\\u00" << hex[(character >> 4U) & 0x0FU]
                 << hex[character & 0x0FU];
        } else {
          stream << static_cast<char>(character);
        }
        break;
    }
  }
  return stream.str();
}

struct HttpRequest {
  std::string method;
  std::string path;
  std::map<std::string, std::string> headers;
  std::vector<std::uint8_t> body;
};

void receive_more(int socket, std::vector<std::uint8_t>& bytes) {
  std::uint8_t buffer[16384];
  for (;;) {
    const ssize_t received = recv(socket, buffer, sizeof(buffer), 0);
    if (received > 0) {
      bytes.insert(bytes.end(), buffer, buffer + received);
      return;
    }
    if (received == 0) {
      throw std::invalid_argument("client closed connection early");
    }
    if (errno != EINTR) {
      throw std::runtime_error(
          std::string("recv failed: ") + std::strerror(errno));
    }
  }
}

HttpRequest read_request(int socket) {
  std::vector<std::uint8_t> bytes;
  const std::string delimiter = "\r\n\r\n";
  std::size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    receive_more(socket, bytes);
    if (bytes.size() > kMaximumHeaderBytes) {
      throw std::invalid_argument("HTTP headers are too large");
    }
    const std::string view(bytes.begin(), bytes.end());
    header_end = view.find(delimiter);
  }
  const std::string header_text(
      bytes.begin(),
      bytes.begin() + static_cast<std::ptrdiff_t>(header_end));
  std::istringstream header_stream(header_text);
  std::string request_line;
  if (!std::getline(header_stream, request_line)) {
    throw std::invalid_argument("missing HTTP request line");
  }
  request_line = trim(request_line);
  std::istringstream request_line_stream(request_line);
  HttpRequest request;
  std::string version;
  if (!(request_line_stream >> request.method >> request.path >> version) ||
      version.find("HTTP/") != 0U) {
    throw std::invalid_argument("invalid HTTP request line");
  }
  const auto query = request.path.find('?');
  if (query != std::string::npos) {
    request.path.resize(query);
  }
  std::string line;
  while (std::getline(header_stream, line)) {
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      throw std::invalid_argument("invalid HTTP header");
    }
    request.headers[lower(trim(line.substr(0, colon)))] =
        trim(line.substr(colon + 1U));
  }
  std::size_t content_length = 0U;
  const auto length = request.headers.find("content-length");
  if (length != request.headers.end()) {
    std::size_t consumed = 0U;
    try {
      content_length =
          static_cast<std::size_t>(std::stoull(length->second, &consumed));
    } catch (const std::exception&) {
      throw std::invalid_argument("invalid Content-Length");
    }
    if (consumed != length->second.size()) {
      throw std::invalid_argument("invalid Content-Length");
    }
  }
  if (content_length > kMaximumImageBytes) {
    throw std::invalid_argument("request body exceeds 10 MiB");
  }
  const std::size_t body_start = header_end + delimiter.size();
  while (bytes.size() < body_start + content_length) {
    receive_more(socket, bytes);
  }
  request.body.assign(
      bytes.begin() + static_cast<std::ptrdiff_t>(body_start),
      bytes.begin() +
          static_cast<std::ptrdiff_t>(body_start + content_length));
  return request;
}

void send_all(int socket, const std::string& bytes) {
  std::size_t sent = 0U;
  while (sent < bytes.size()) {
    const ssize_t result =
        send(socket, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
    if (result > 0) {
      sent += static_cast<std::size_t>(result);
      continue;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    throw std::runtime_error(
        std::string("send failed: ") + std::strerror(errno));
  }
}

std::string reason_phrase(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 204:
      return "No Content";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 413:
      return "Payload Too Large";
    case 500:
      return "Internal Server Error";
    default:
      return "Error";
  }
}

std::string cors_origin(const HttpRequest& request) {
  const auto found = request.headers.find("origin");
  if (found == request.headers.end()) {
    return "";
  }
  if (found->second == "http://127.0.0.1:8090" ||
      found->second == "http://localhost:8090") {
    return found->second;
  }
  return "";
}

void send_response(
    int socket,
    const HttpRequest& request,
    int status,
    const std::string& body) {
  std::ostringstream stream;
  stream << "HTTP/1.1 " << status << " " << reason_phrase(status) << "\r\n"
         << "Content-Type: application/json; charset=utf-8\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Connection: close\r\n"
         << "Cache-Control: no-store\r\n";
  const std::string origin = cors_origin(request);
  if (!origin.empty()) {
    stream << "Access-Control-Allow-Origin: " << origin << "\r\n"
           << "Vary: Origin\r\n";
  }
  stream << "\r\n" << body;
  send_all(socket, stream.str());
}

void send_options(int socket, const HttpRequest& request) {
  std::ostringstream stream;
  stream << "HTTP/1.1 204 No Content\r\n"
         << "Content-Length: 0\r\n"
         << "Connection: close\r\n"
         << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
         << "Access-Control-Allow-Headers: Content-Type, X-Frame-Id, "
            "X-Capture-Timestamp-Ms, X-Source-Timestamp-Ms\r\n";
  const std::string origin = cors_origin(request);
  if (!origin.empty()) {
    stream << "Access-Control-Allow-Origin: " << origin << "\r\n"
           << "Vary: Origin\r\n";
  }
  stream << "\r\n";
  send_all(socket, stream.str());
}

std::string error_json(const std::string& code, const std::string& message) {
  return "{\"ok\":false,\"error\":{\"code\":\"" +
         json_escape(code) + "\",\"message\":\"" + json_escape(message) +
         "\"}}";
}

std::uint64_t unsigned_header(
    const HttpRequest& request,
    const std::string& name,
    bool required) {
  const auto found = request.headers.find(lower(name));
  if (found == request.headers.end() || found->second.empty()) {
    if (required) {
      throw std::invalid_argument("missing " + name);
    }
    return 0U;
  }
  std::size_t consumed = 0U;
  std::uint64_t result = 0U;
  try {
    result = std::stoull(found->second, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid " + name);
  }
  if (consumed != found->second.size()) {
    throw std::invalid_argument("invalid " + name);
  }
  return result;
}

std::int64_t signed_header(
    const HttpRequest& request,
    const std::string& name,
    bool required) {
  const auto found = request.headers.find(lower(name));
  if (found == request.headers.end() || found->second.empty()) {
    if (required) {
      throw std::invalid_argument("missing " + name);
    }
    return 0;
  }
  std::size_t consumed = 0U;
  std::int64_t result = 0;
  try {
    result = std::stoll(found->second, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid " + name);
  }
  if (consumed != found->second.size()) {
    throw std::invalid_argument("invalid " + name);
  }
  return result;
}

void require_jpeg(const HttpRequest& request) {
  if (request.body.empty()) {
    throw std::invalid_argument("request body must contain JPEG bytes");
  }
  const auto content_type = request.headers.find("content-type");
  if (content_type == request.headers.end() ||
      lower(content_type->second).find("image/jpeg") == std::string::npos) {
    throw std::invalid_argument("Content-Type must be image/jpeg");
  }
}

medvision::BoardBox ema_box(
    const medvision::BoardBox& previous,
    const medvision::BoardBox& current,
    float alpha) {
  return {
      previous.x1 * (1.0F - alpha) + current.x1 * alpha,
      previous.y1 * (1.0F - alpha) + current.y1 * alpha,
      previous.x2 * (1.0F - alpha) + current.x2 * alpha,
      previous.y2 * (1.0F - alpha) + current.y2 * alpha,
  };
}

medvision::BoardPoint ema_point(
    const medvision::BoardPoint& previous,
    const medvision::BoardPoint& current,
    float alpha) {
  return {
      previous.x * (1.0F - alpha) + current.x * alpha,
      previous.y * (1.0F - alpha) + current.y * alpha,
  };
}

class StreamState {
 public:
  StreamState(
      std::int64_t device_stale_ms,
      std::int64_t contact_stale_ms,
      std::uint64_t contact_interval)
      : device_stale_ms_(device_stale_ms),
        contact_stale_ms_(contact_stale_ms),
        contact_interval_(contact_interval) {}

  void reset() {
    last_frame_id_ = 0U;
    has_frame_ = false;
    valid_device_frames_ = 0U;
    stable_device_ = medvision::BoardDetection{};
    stable_contact_ = medvision::BoardContact{};
    last_device_ms_ = 0;
    last_contact_ms_ = 0;
  }

  bool contact_due() const {
    return valid_device_frames_ % contact_interval_ == 0U;
  }

  medvision::BoardFrameResult update(
      const medvision::BoardFrameResult& raw,
      std::int64_t now_ms) {
    if (has_frame_ && raw.frame_id <= last_frame_id_) {
      throw std::invalid_argument("X-Frame-Id must increase strictly");
    }
    has_frame_ = true;
    last_frame_id_ = raw.frame_id;
    if (raw.device.valid) {
      if (stable_device_.valid) {
        stable_device_.bbox =
            ema_box(stable_device_.bbox, raw.device.bbox, kBoxEmaAlpha);
        stable_device_.score = raw.device.score;
      } else {
        stable_device_ = raw.device;
      }
      stable_device_.valid = true;
      last_device_ms_ = now_ms;
      ++valid_device_frames_;
    } else if (
        stable_device_.valid &&
        now_ms - last_device_ms_ > device_stale_ms_) {
      stable_device_ = medvision::BoardDetection{};
    }

    if (!stable_device_.valid) {
      stable_contact_ = medvision::BoardContact{};
      last_contact_ms_ = 0;
    } else if (raw.contact_executed && raw.contact.has_point) {
      if (stable_contact_.has_point) {
        const auto smoothed =
            ema_point(stable_contact_.point, raw.contact.point, kPointEmaAlpha);
        stable_contact_ = raw.contact;
        stable_contact_.point = smoothed;
      } else {
        stable_contact_ = raw.contact;
      }
      last_contact_ms_ = now_ms;
    } else if (
        stable_contact_.has_point &&
        now_ms - last_contact_ms_ > contact_stale_ms_) {
      stable_contact_ = medvision::BoardContact{};
    }

    medvision::BoardFrameResult result = raw;
    result.device = stable_device_;
    result.contact = stable_contact_;
    return result;
  }

  std::uint64_t valid_device_frames() const { return valid_device_frames_; }

 private:
  std::uint64_t last_frame_id_ = 0U;
  bool has_frame_ = false;
  std::uint64_t valid_device_frames_ = 0U;
  medvision::BoardDetection stable_device_;
  medvision::BoardContact stable_contact_;
  std::int64_t last_device_ms_ = 0;
  std::int64_t last_contact_ms_ = 0;
  const std::int64_t device_stale_ms_;
  const std::int64_t contact_stale_ms_;
  const std::uint64_t contact_interval_;
};

struct Arguments {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string device_model;
  std::string contact_model;
  std::string device_sha256;
  std::string contact_sha256;
  std::string quantization = "unknown";
  std::string board_model = "ATK-DLRK3568";
  float device_confidence = 0.25F;
  float nms_iou = 0.65F;
  std::int64_t device_stale_ms = 250;
  std::int64_t contact_stale_ms = 150;
  std::uint64_t contact_interval = 2U;
};

Arguments parse_arguments(int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + argument);
    }
    const std::string value = argv[++index];
    if (argument == "--host") {
      result.host = value;
    } else if (argument == "--port") {
      result.port = std::stoi(value);
    } else if (argument == "--device-model") {
      result.device_model = value;
    } else if (argument == "--contact-model") {
      result.contact_model = value;
    } else if (argument == "--device-sha256") {
      result.device_sha256 = value;
    } else if (argument == "--contact-sha256") {
      result.contact_sha256 = value;
    } else if (argument == "--quantization") {
      result.quantization = value;
    } else if (argument == "--board-model") {
      result.board_model = value;
    } else if (argument == "--device-confidence") {
      result.device_confidence = std::stof(value);
    } else if (argument == "--nms-iou") {
      result.nms_iou = std::stof(value);
    } else if (argument == "--device-stale-ms") {
      result.device_stale_ms = std::stoll(value);
    } else if (argument == "--contact-stale-ms") {
      result.contact_stale_ms = std::stoll(value);
    } else if (argument == "--contact-interval") {
      result.contact_interval = std::stoull(value);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (result.device_model.empty() || result.contact_model.empty()) {
    throw std::invalid_argument(
        "--device-model and --contact-model are required");
  }
  if (result.port <= 0 || result.port > 65535) {
    throw std::invalid_argument("port must be between 1 and 65535");
  }
  if (result.device_confidence < 0.0F || result.device_confidence > 1.0F) {
    throw std::invalid_argument("device confidence must be between 0 and 1");
  }
  if (result.nms_iou < 0.0F || result.nms_iou > 1.0F) {
    throw std::invalid_argument("NMS IoU must be between 0 and 1");
  }
  if (result.device_stale_ms <= 0 || result.contact_stale_ms <= 0) {
    throw std::invalid_argument("stale limits must be positive");
  }
  if (result.contact_interval == 0U) {
    throw std::invalid_argument("contact interval must be positive");
  }
  return result;
}

std::string health_json(
    const Arguments& arguments,
    const medvision::Rk3568Pipeline& pipeline) {
  std::ostringstream stream;
  stream << "{\"ok\":true,\"service\":\"medvision_rk3568\""
         << ",\"service_version\":\"RK3568-DEPLOY-V1\""
         << ",\"board_model\":\"" << json_escape(arguments.board_model)
         << "\",\"runtime_provider\":\"RKNN_NPU\""
         << ",\"model_quantization\":\""
         << json_escape(arguments.quantization)
         << "\",\"rknn\":{\"api_version\":\""
         << json_escape(pipeline.sdk_api_version())
         << "\",\"driver_version\":\""
         << json_escape(pipeline.driver_version())
         << "\",\"npu_status\":\"loaded\"}"
         << ",\"models\":{\"device\":{\"path\":\""
         << json_escape(arguments.device_model) << "\",\"sha256\":\""
         << json_escape(arguments.device_sha256)
         << "\",\"loaded\":true},\"contact\":{\"path\":\""
         << json_escape(arguments.contact_model) << "\",\"sha256\":\""
         << json_escape(arguments.contact_sha256)
         << "\",\"loaded\":true}}}";
  return stream.str();
}

std::string ok_result(const std::string& result) {
  return "{\"ok\":true,\"result\":" + result + "}";
}

std::string stream_result(
    const medvision::BoardFrameResult& result,
    const Arguments& arguments,
    std::uint64_t valid_device_frames) {
  std::ostringstream stream_fields;
  stream_fields << ",\"stream\":{\"valid_device_frames\":"
                << valid_device_frames
                << ",\"device_stale_after_ms\":" << arguments.device_stale_ms
                << ",\"contact_stale_after_ms\":" << arguments.contact_stale_ms
                << ",\"contact_interval_valid_frames\":"
                << arguments.contact_interval
                << ",\"smoothing\":\"EMA\"}}";
  std::string json = medvision::frame_result_json(
      result, arguments.board_model, arguments.quantization);
  if (json.empty() || json.back() != '}') {
    throw std::runtime_error("internal JSON serialization error");
  }
  json.pop_back();
  json += stream_fields.str();
  return ok_result(json);
}

void handle_request(
    int socket,
    const HttpRequest& request,
    const Arguments& arguments,
    medvision::Rk3568Pipeline& pipeline,
    StreamState& stream_state) {
  if (request.method == "OPTIONS") {
    send_options(socket, request);
    return;
  }
  if (request.method == "GET" && request.path == "/health") {
    send_response(socket, request, 200, health_json(arguments, pipeline));
    return;
  }
  if (request.method == "POST" && request.path == "/v1/stream/reset") {
    stream_state.reset();
    send_response(socket, request, 200, "{\"ok\":true,\"reset\":true}");
    return;
  }
  if (request.method == "POST" && request.path == "/v1/infer/image") {
    require_jpeg(request);
    const auto frame_id = unsigned_header(request, "X-Frame-Id", false);
    const auto timestamp =
        signed_header(request, "X-Source-Timestamp-Ms", false);
    const auto result =
        pipeline.infer(request.body, frame_id, timestamp, true);
    send_response(
        socket,
        request,
        200,
        ok_result(medvision::frame_result_json(
            result, arguments.board_model, arguments.quantization)));
    return;
  }
  if (request.method == "POST" && request.path == "/v1/stream/frame") {
    require_jpeg(request);
    const auto frame_id = unsigned_header(request, "X-Frame-Id", true);
    const auto timestamp =
        signed_header(request, "X-Capture-Timestamp-Ms", true);
    const bool infer_contact = stream_state.contact_due();
    const auto raw =
        pipeline.infer(request.body, frame_id, timestamp, infer_contact);
    const auto result = stream_state.update(raw, monotonic_ms());
    send_response(
        socket,
        request,
        200,
        stream_result(
            result, arguments, stream_state.valid_device_frames()));
    return;
  }
  const bool known_path =
      request.path == "/health" || request.path == "/v1/infer/image" ||
      request.path == "/v1/stream/frame" ||
      request.path == "/v1/stream/reset";
  send_response(
      socket,
      request,
      known_path ? 405 : 404,
      error_json(
          known_path ? "method_not_allowed" : "not_found",
          known_path ? "HTTP method is not allowed for this endpoint"
                     : "endpoint does not exist"));
}

int create_listener(const std::string& host, int port) {
  const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) {
    throw std::runtime_error(
        std::string("socket failed: ") + std::strerror(errno));
  }
  int reuse = 1;
  if (setsockopt(
          socket_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
    close(socket_fd);
    throw std::runtime_error(
        std::string("setsockopt failed: ") + std::strerror(errno));
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    close(socket_fd);
    throw std::invalid_argument("host must be an IPv4 address");
  }
  if (bind(
          socket_fd,
          reinterpret_cast<const sockaddr*>(&address),
          sizeof(address)) != 0) {
    const std::string message =
        std::string("bind failed: ") + std::strerror(errno);
    close(socket_fd);
    throw std::runtime_error(message);
  }
  if (listen(socket_fd, 4) != 0) {
    const std::string message =
        std::string("listen failed: ") + std::strerror(errno);
    close(socket_fd);
    throw std::runtime_error(message);
  }
  return socket_fd;
}

}  // namespace

int main(int argc, char** argv) {
  signal(SIGPIPE, SIG_IGN);
  Arguments arguments;
  try {
    arguments = parse_arguments(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "ARGUMENT_ERROR=" << error.what() << '\n';
    return EXIT_FAILURE;
  }

  medvision::Rk3568PipelineConfig pipeline_config;
  pipeline_config.device_model_path = arguments.device_model;
  pipeline_config.contact_model_path = arguments.contact_model;
  pipeline_config.detection_confidence = arguments.device_confidence;
  pipeline_config.nms_iou = arguments.nms_iou;
  std::unique_ptr<medvision::Rk3568Pipeline> pipeline;
  try {
    pipeline.reset(new medvision::Rk3568Pipeline(pipeline_config));
  } catch (const std::exception& error) {
    std::cerr << "MODEL_LOAD=FAIL error=" << error.what() << '\n';
    return EXIT_FAILURE;
  }

  int listener = -1;
  try {
    listener = create_listener(arguments.host, arguments.port);
  } catch (const std::exception& error) {
    std::cerr << "HTTP_LISTEN=FAIL error=" << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "MEDVISION_RK3568_URL=http://" << arguments.host << ":"
            << arguments.port << '\n'
            << "RUNTIME_PROVIDER=RKNN_NPU API="
            << pipeline->sdk_api_version() << " DRIVER="
            << pipeline->driver_version() << '\n';
  std::cout.flush();

  StreamState stream_state(
      arguments.device_stale_ms,
      arguments.contact_stale_ms,
      arguments.contact_interval);
  for (;;) {
    const int client = accept(listener, nullptr, nullptr);
    if (client < 0) {
      if (errno == EINTR) {
        continue;
      }
      std::cerr << "ACCEPT_ERROR=" << std::strerror(errno) << '\n';
      continue;
    }
    HttpRequest request;
    bool request_parsed = false;
    try {
      request = read_request(client);
      request_parsed = true;
      handle_request(
          client, request, arguments, *pipeline, stream_state);
    } catch (const std::invalid_argument& error) {
      if (request_parsed) {
        try {
          send_response(
              client,
              request,
              400,
              error_json("invalid_request", error.what()));
        } catch (const std::exception&) {
        }
      }
    } catch (const std::exception& error) {
      std::cerr << "REQUEST_ERROR=" << error.what() << '\n';
      if (request_parsed) {
        try {
          send_response(
              client,
              request,
              500,
              error_json("inference_failed", error.what()));
        } catch (const std::exception&) {
        }
      }
    }
    close(client);
  }
}
