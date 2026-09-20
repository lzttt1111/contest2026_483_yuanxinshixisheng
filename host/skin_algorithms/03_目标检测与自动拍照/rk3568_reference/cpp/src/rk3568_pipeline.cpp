#include "medvision/rk3568_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <im2d.h>
#include <turbojpeg.h>

#include "medvision/rknn_backend.hpp"

namespace medvision {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kDeviceSize = 640;
constexpr int kContactSize = 256;
constexpr int kHeatmapSize = 64;

double elapsed_ms(const Clock::time_point& started) {
  return std::chrono::duration<double, std::milli>(Clock::now() - started)
      .count();
}

float clamp_float(float value, float low, float high) {
  return std::max(low, std::min(value, high));
}

float sigmoid(float value) {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-value));
  }
  const float exponent = std::exp(value);
  return exponent / (1.0F + exponent);
}

struct BgrImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;
};

class TurboJpegHandle {
 public:
  TurboJpegHandle() : handle_(tjInitDecompress()) {
    if (handle_ == nullptr) {
      throw std::runtime_error("tjInitDecompress failed");
    }
  }

  ~TurboJpegHandle() {
    if (handle_ != nullptr) {
      tjDestroy(handle_);
    }
  }

  TurboJpegHandle(const TurboJpegHandle&) = delete;
  TurboJpegHandle& operator=(const TurboJpegHandle&) = delete;

  tjhandle get() const { return handle_; }

 private:
  tjhandle handle_ = nullptr;
};

BgrImage decode_jpeg(const std::vector<std::uint8_t>& jpeg) {
  if (jpeg.empty()) {
    throw std::invalid_argument("JPEG body is empty");
  }
  TurboJpegHandle decoder;
  int width = 0;
  int height = 0;
  int subsampling = 0;
  int colorspace = 0;
  if (tjDecompressHeader3(
          decoder.get(),
          jpeg.data(),
          static_cast<unsigned long>(jpeg.size()),
          &width,
          &height,
          &subsampling,
          &colorspace) != 0) {
    throw std::invalid_argument(
        std::string("invalid JPEG header: ") + tjGetErrorStr2(decoder.get()));
  }
  if (width <= 0 || height <= 0 || width > 8192 || height > 8192) {
    throw std::invalid_argument("JPEG dimensions are outside the safe range");
  }
  const std::size_t pixels =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (pixels > std::numeric_limits<std::size_t>::max() / 3U) {
    throw std::invalid_argument("JPEG dimensions overflow");
  }
  BgrImage image;
  image.width = width;
  image.height = height;
  image.pixels.resize(pixels * 3U);
  if (tjDecompress2(
          decoder.get(),
          jpeg.data(),
          static_cast<unsigned long>(jpeg.size()),
          image.pixels.data(),
          width,
          width * 3,
          height,
          TJPF_BGR,
          TJFLAG_FASTDCT) != 0) {
    throw std::invalid_argument(
        std::string("JPEG decode failed: ") + tjGetErrorStr2(decoder.get()));
  }
  return image;
}

std::vector<std::uint8_t> resize_region(
    const BgrImage& source,
    int source_x1,
    int source_y1,
    int source_x2,
    int source_y2,
    int target_width,
    int target_height,
    bool output_rgb) {
  if (source_x1 < 0 || source_y1 < 0 || source_x2 > source.width ||
      source_y2 > source.height || source_x2 <= source_x1 ||
      source_y2 <= source_y1 || target_width <= 0 || target_height <= 0) {
    throw std::invalid_argument("invalid resize region");
  }
  const int region_width = source_x2 - source_x1;
  const int region_height = source_y2 - source_y1;
  std::vector<std::uint8_t> output(
      static_cast<std::size_t>(target_width) *
      static_cast<std::size_t>(target_height) * 3U);

  auto* source_address =
      const_cast<std::uint8_t*>(source.pixels.data()) +
      (static_cast<std::size_t>(source_y1) * source.width + source_x1) * 3U;
  const rga_buffer_t source_buffer = wrapbuffer_virtualaddr_t(
      source_address,
      region_width,
      region_height,
      source.width,
      region_height,
      RK_FORMAT_BGR_888);
  const rga_buffer_t target_buffer = wrapbuffer_virtualaddr_t(
      output.data(),
      target_width,
      target_height,
      target_width,
      target_height,
      output_rgb ? RK_FORMAT_RGB_888 : RK_FORMAT_BGR_888);
  const im_rect source_rect{};
  const im_rect target_rect{};
  const im_rect pattern_rect{};
  const rga_buffer_t pattern_buffer{};
  if (imcheck_t(
          source_buffer,
          target_buffer,
          pattern_buffer,
          source_rect,
          target_rect,
          pattern_rect,
          0) ==
          IM_STATUS_NOERROR &&
      imresize_t(source_buffer, target_buffer, 0.0, 0.0, 0, 1) ==
          IM_STATUS_SUCCESS) {
    return output;
  }

  for (int target_y = 0; target_y < target_height; ++target_y) {
    const float source_y =
        static_cast<float>(source_y1) +
        (static_cast<float>(target_y) + 0.5F) *
            static_cast<float>(region_height) /
            static_cast<float>(target_height) -
        0.5F;
    const int y0 = std::max(
        source_y1,
        std::min(source_y2 - 1, static_cast<int>(std::floor(source_y))));
    const int y1 = std::min(source_y2 - 1, y0 + 1);
    const float weight_y = clamp_float(source_y - std::floor(source_y), 0.0F, 1.0F);
    for (int target_x = 0; target_x < target_width; ++target_x) {
      const float source_x =
          static_cast<float>(source_x1) +
          (static_cast<float>(target_x) + 0.5F) *
              static_cast<float>(region_width) /
              static_cast<float>(target_width) -
          0.5F;
      const int x0 = std::max(
          source_x1,
          std::min(source_x2 - 1, static_cast<int>(std::floor(source_x))));
      const int x1 = std::min(source_x2 - 1, x0 + 1);
      const float weight_x =
          clamp_float(source_x - std::floor(source_x), 0.0F, 1.0F);
      for (int channel = 0; channel < 3; ++channel) {
        const int source_channel = output_rgb ? 2 - channel : channel;
        const auto value_at = [&](int x, int y) {
          return static_cast<float>(
              source.pixels[
                  (static_cast<std::size_t>(y) * source.width + x) * 3U +
                  static_cast<std::size_t>(source_channel)]);
        };
        const float top =
            value_at(x0, y0) * (1.0F - weight_x) +
            value_at(x1, y0) * weight_x;
        const float bottom =
            value_at(x0, y1) * (1.0F - weight_x) +
            value_at(x1, y1) * weight_x;
        const float value = top * (1.0F - weight_y) + bottom * weight_y;
        output[
            (static_cast<std::size_t>(target_y) * target_width + target_x) *
                3U +
            static_cast<std::size_t>(channel)] =
            static_cast<std::uint8_t>(
                clamp_float(std::floor(value + 0.5F), 0.0F, 255.0F));
      }
    }
  }
  return output;
}

struct DeviceInput {
  std::vector<std::uint8_t> bytes;
  float ratio = 1.0F;
};

DeviceInput prepare_device_input(const BgrImage& image) {
  DeviceInput result;
  result.ratio = std::min(
      static_cast<float>(kDeviceSize) / static_cast<float>(image.height),
      static_cast<float>(kDeviceSize) / static_cast<float>(image.width));
  const int resized_width = std::max(
      1,
      static_cast<int>(static_cast<float>(image.width) * result.ratio));
  const int resized_height = std::max(
      1,
      static_cast<int>(static_cast<float>(image.height) * result.ratio));
  const auto resized = resize_region(
      image,
      0,
      0,
      image.width,
      image.height,
      resized_width,
      resized_height,
      false);
  result.bytes.assign(
      static_cast<std::size_t>(kDeviceSize) * kDeviceSize * 3U,
      static_cast<std::uint8_t>(114));
  for (int y = 0; y < resized_height; ++y) {
    const auto source_offset =
        static_cast<std::size_t>(y) * resized_width * 3U;
    const auto target_offset =
        static_cast<std::size_t>(y) * kDeviceSize * 3U;
    std::copy(
        resized.begin() + static_cast<std::ptrdiff_t>(source_offset),
        resized.begin() +
            static_cast<std::ptrdiff_t>(
                source_offset + static_cast<std::size_t>(resized_width) * 3U),
        result.bytes.begin() + static_cast<std::ptrdiff_t>(target_offset));
  }
  return result;
}

float box_area(const BoardBox& box) {
  return std::max(0.0F, box.x2 - box.x1) *
         std::max(0.0F, box.y2 - box.y1);
}

float box_iou(const BoardBox& left, const BoardBox& right) {
  const float intersection_width =
      std::max(0.0F, std::min(left.x2, right.x2) - std::max(left.x1, right.x1));
  const float intersection_height =
      std::max(0.0F, std::min(left.y2, right.y2) - std::max(left.y1, right.y1));
  const float intersection = intersection_width * intersection_height;
  const float union_area = box_area(left) + box_area(right) - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

float output_value(
    const RknnOutput& output,
    std::size_t channel,
    std::size_t y,
    std::size_t x,
    std::size_t channels,
    std::size_t height,
    std::size_t width) {
  if (output.dims.size() != 4U) {
    throw std::runtime_error("YOLOX output must have four dimensions");
  }
  if (output.dims[1] == channels && output.dims[2] == height &&
      output.dims[3] == width) {
    return output.values[(channel * height + y) * width + x];
  }
  if (output.dims[1] == height && output.dims[2] == width &&
      output.dims[3] == channels) {
    return output.values[(y * width + x) * channels + channel];
  }
  throw std::runtime_error("unsupported YOLOX output layout");
}

BoardDetection decode_device(
    const std::vector<RknnOutput>& outputs,
    float ratio,
    int image_width,
    int image_height,
    float confidence,
    float nms_iou) {
  std::vector<BoardDetection> candidates;
  for (const auto& output : outputs) {
    if (output.dims.size() != 4U || output.values.empty()) {
      throw std::runtime_error("unexpected YOLOX output");
    }
    std::size_t height = 0;
    std::size_t width = 0;
    if (output.dims[1] == 6U) {
      height = output.dims[2];
      width = output.dims[3];
    } else if (output.dims[3] == 6U) {
      height = output.dims[1];
      width = output.dims[2];
    } else {
      throw std::runtime_error("YOLOX output does not have six channels");
    }
    if (height == 0U || width == 0U || height != width ||
        kDeviceSize % static_cast<int>(height) != 0) {
      throw std::runtime_error("unexpected YOLOX feature map size");
    }
    const float stride =
        static_cast<float>(kDeviceSize) / static_cast<float>(height);
    for (std::size_t y = 0; y < height; ++y) {
      for (std::size_t x = 0; x < width; ++x) {
        const float score =
            sigmoid(output_value(output, 4, y, x, 6, height, width)) *
            sigmoid(output_value(output, 5, y, x, 6, height, width));
        if (score < confidence) {
          continue;
        }
        const float center_x =
            (output_value(output, 0, y, x, 6, height, width) +
             static_cast<float>(x)) *
            stride;
        const float center_y =
            (output_value(output, 1, y, x, 6, height, width) +
             static_cast<float>(y)) *
            stride;
        const float box_width =
            std::exp(clamp_float(
                output_value(output, 2, y, x, 6, height, width),
                -20.0F,
                20.0F)) *
            stride;
        const float box_height =
            std::exp(clamp_float(
                output_value(output, 3, y, x, 6, height, width),
                -20.0F,
                20.0F)) *
            stride;
        BoardDetection candidate;
        candidate.valid = true;
        candidate.score = score;
        candidate.bbox = {
            clamp_float(
                (center_x - box_width * 0.5F) / ratio,
                0.0F,
                static_cast<float>(image_width - 1)),
            clamp_float(
                (center_y - box_height * 0.5F) / ratio,
                0.0F,
                static_cast<float>(image_height - 1)),
            clamp_float(
                (center_x + box_width * 0.5F) / ratio,
                0.0F,
                static_cast<float>(image_width - 1)),
            clamp_float(
                (center_y + box_height * 0.5F) / ratio,
                0.0F,
                static_cast<float>(image_height - 1)),
        };
        candidates.push_back(candidate);
      }
    }
  }
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const BoardDetection& left, const BoardDetection& right) {
        return left.score > right.score;
      });
  std::vector<BoardDetection> kept;
  for (const auto& candidate : candidates) {
    bool suppressed = false;
    for (const auto& existing : kept) {
      if (box_iou(candidate.bbox, existing.bbox) > nms_iou) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(candidate);
    }
  }
  return kept.empty() ? BoardDetection{} : kept.front();
}

BoardBox expanded_roi(
    const BoardBox& box,
    int image_width,
    int image_height,
    float expansion) {
  const float width = std::max(1.0F, box.x2 - box.x1);
  const float height = std::max(1.0F, box.y2 - box.y1);
  return {
      std::max(0.0F, std::floor(box.x1 - width * expansion)),
      std::max(0.0F, std::floor(box.y1 - height * expansion)),
      std::min(
          static_cast<float>(image_width),
          std::ceil(box.x2 + width * expansion)),
      std::min(
          static_cast<float>(image_height),
          std::ceil(box.y2 + height * expansion)),
  };
}

std::vector<float> softmax(const std::vector<float>& values) {
  if (values.empty()) {
    throw std::runtime_error("softmax input is empty");
  }
  const float maximum = *std::max_element(values.begin(), values.end());
  std::vector<float> result(values.size());
  double denominator = 0.0;
  for (std::size_t index = 0; index < values.size(); ++index) {
    result[index] =
        static_cast<float>(std::exp(static_cast<double>(values[index] - maximum)));
    denominator += result[index];
  }
  if (!(denominator > 0.0) || !std::isfinite(denominator)) {
    throw std::runtime_error("softmax denominator is not finite");
  }
  for (auto& value : result) {
    value = static_cast<float>(static_cast<double>(value) / denominator);
  }
  return result;
}

BoardContact decode_contact(
    const std::vector<RknnOutput>& outputs,
    const BoardBox& roi) {
  const RknnOutput* heatmap = nullptr;
  const RknnOutput* visibility = nullptr;
  for (const auto& output : outputs) {
    if (output.values.size() ==
        static_cast<std::size_t>(kHeatmapSize * kHeatmapSize)) {
      heatmap = &output;
    } else if (output.values.size() == 3U) {
      visibility = &output;
    }
  }
  if (heatmap == nullptr || visibility == nullptr) {
    throw std::runtime_error("unexpected contact model outputs");
  }
  BoardContact result;
  result.has_roi = true;
  result.roi = roi;
  const auto visibility_probabilities = softmax(visibility->values);
  const auto best_visibility = static_cast<std::size_t>(std::distance(
      visibility_probabilities.begin(),
      std::max_element(
          visibility_probabilities.begin(), visibility_probabilities.end())));
  result.confidence = visibility_probabilities[best_visibility];
  if (best_visibility == 0U) {
    result.visibility = BoardVisibility::kVisible;
  } else if (best_visibility == 1U) {
    result.visibility = BoardVisibility::kOccludedInferable;
  } else {
    result.visibility = BoardVisibility::kNotAnnotatable;
  }

  const auto heatmap_probabilities = softmax(heatmap->values);
  double entropy = 0.0;
  for (const float probability : heatmap_probabilities) {
    if (probability > 0.0F) {
      entropy -=
          static_cast<double>(probability) * std::log(probability);
    }
  }
  result.heatmap_confidence = clamp_float(
      static_cast<float>(
          1.0 - entropy / std::log(static_cast<double>(heatmap->values.size()))),
      0.0F,
      1.0F);
  if (result.visibility == BoardVisibility::kNotAnnotatable) {
    return result;
  }

  const auto peak = std::max_element(heatmap->values.begin(), heatmap->values.end());
  const std::size_t peak_index =
      static_cast<std::size_t>(std::distance(heatmap->values.begin(), peak));
  const std::size_t peak_x = peak_index % kHeatmapSize;
  const std::size_t peak_y = peak_index / kHeatmapSize;
  constexpr std::size_t radius = 3;
  const std::size_t start_x = peak_x > radius ? peak_x - radius : 0U;
  const std::size_t end_x =
      std::min<std::size_t>(kHeatmapSize, peak_x + radius + 1U);
  const std::size_t start_y = peak_y > radius ? peak_y - radius : 0U;
  const std::size_t end_y =
      std::min<std::size_t>(kHeatmapSize, peak_y + radius + 1U);
  double weight_sum = 0.0;
  double weighted_x = 0.0;
  double weighted_y = 0.0;
  for (std::size_t y = start_y; y < end_y; ++y) {
    for (std::size_t x = start_x; x < end_x; ++x) {
      const double weight = std::exp(
          static_cast<double>(heatmap->values[y * kHeatmapSize + x] - *peak));
      weight_sum += weight;
      weighted_x += weight * static_cast<double>(x);
      weighted_y += weight * static_cast<double>(y);
    }
  }
  if (!(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
    throw std::runtime_error("contact heatmap neighborhood is not finite");
  }
  result.has_point = true;
  result.point = {
      static_cast<float>(
          static_cast<double>(roi.x1) +
          weighted_x / weight_sum / 63.0 *
              static_cast<double>(roi.x2 - roi.x1)),
      static_cast<float>(
          static_cast<double>(roi.y1) +
          weighted_y / weight_sum / 63.0 *
              static_cast<double>(roi.y2 - roi.y1)),
  };
  return result;
}

std::string number(double value) {
  if (!std::isfinite(value)) {
    return "null";
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << value;
  return stream.str();
}

std::string box_json(const BoardBox& box) {
  return "[" + number(box.x1) + "," + number(box.y1) + "," +
         number(box.x2) + "," + number(box.y2) + "]";
}

}  // namespace

struct Rk3568Pipeline::Impl {
  explicit Impl(const Rk3568PipelineConfig& value)
      : config(value),
        device(new RknnModel(config.device_model_path)),
        contact(new RknnModel(config.contact_model_path)) {}

  Rk3568PipelineConfig config;
  std::unique_ptr<RknnModel> device;
  std::unique_ptr<RknnModel> contact;
  std::mutex mutex;
};

Rk3568Pipeline::Rk3568Pipeline(const Rk3568PipelineConfig& config)
    : impl_(new Impl(config)) {}

Rk3568Pipeline::~Rk3568Pipeline() = default;

BoardFrameResult Rk3568Pipeline::infer(
    const std::vector<std::uint8_t>& jpeg,
    std::uint64_t frame_id,
    std::int64_t timestamp_ms,
    bool infer_contact) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto total_started = Clock::now();
  BoardFrameResult result;
  result.frame_id = frame_id;
  result.timestamp_ms = timestamp_ms;

  auto started = Clock::now();
  const BgrImage image = decode_jpeg(jpeg);
  result.image_width = image.width;
  result.image_height = image.height;
  result.timings.decode_ms = elapsed_ms(started);

  started = Clock::now();
  const DeviceInput input = prepare_device_input(image);
  result.timings.device_preprocess_ms = elapsed_ms(started);
  started = Clock::now();
  const auto device_outputs = impl_->device->run_u8_nhwc(input.bytes);
  result.timings.device_inference_ms = elapsed_ms(started);
  started = Clock::now();
  result.device = decode_device(
      device_outputs,
      input.ratio,
      image.width,
      image.height,
      impl_->config.detection_confidence,
      impl_->config.nms_iou);
  result.timings.device_postprocess_ms = elapsed_ms(started);

  if (infer_contact && result.device.valid) {
    result.contact_executed = true;
    started = Clock::now();
    const BoardBox roi = expanded_roi(
        result.device.bbox,
        image.width,
        image.height,
        impl_->config.contact_roi_expansion);
    const int x1 = static_cast<int>(roi.x1);
    const int y1 = static_cast<int>(roi.y1);
    const int x2 = static_cast<int>(roi.x2);
    const int y2 = static_cast<int>(roi.y2);
    const auto contact_input = resize_region(
        image, x1, y1, x2, y2, kContactSize, kContactSize, true);
    result.timings.contact_preprocess_ms = elapsed_ms(started);
    started = Clock::now();
    const auto contact_outputs = impl_->contact->run_u8_nhwc(contact_input);
    result.timings.contact_inference_ms = elapsed_ms(started);
    started = Clock::now();
    result.contact = decode_contact(contact_outputs, roi);
    result.timings.contact_postprocess_ms = elapsed_ms(started);
  }
  result.timings.total_ms = elapsed_ms(total_started);
  return result;
}

std::string Rk3568Pipeline::sdk_api_version() const {
  return impl_->device->sdk_version().api_version;
}

std::string Rk3568Pipeline::driver_version() const {
  return impl_->device->sdk_version().drv_version;
}

const char* to_string(BoardVisibility visibility) {
  switch (visibility) {
    case BoardVisibility::kVisible:
      return "visible";
    case BoardVisibility::kOccludedInferable:
      return "occluded_inferable";
    case BoardVisibility::kNotAnnotatable:
      return "not_annotatable";
  }
  return "not_annotatable";
}

std::string frame_result_json(
    const BoardFrameResult& result,
    const std::string& board_model,
    const std::string& quantization) {
  std::ostringstream stream;
  stream << "{\"frame_id\":" << result.frame_id
         << ",\"timestamp_ms\":" << result.timestamp_ms
         << ",\"image_width\":" << result.image_width
         << ",\"image_height\":" << result.image_height
         << ",\"device\":";
  if (result.device.valid) {
    stream << "{\"bbox\":" << box_json(result.device.bbox)
           << ",\"score\":" << number(result.device.score) << "}";
  } else {
    stream << "null";
  }
  stream << ",\"contact\":{\"point\":";
  if (result.contact.has_point) {
    stream << "[" << number(result.contact.point.x) << ","
           << number(result.contact.point.y) << "]";
  } else {
    stream << "null";
  }
  stream << ",\"visibility\":\"" << to_string(result.contact.visibility)
         << "\",\"confidence\":" << number(result.contact.confidence)
         << ",\"heatmap_confidence\":"
         << number(result.contact.heatmap_confidence)
         << ",\"contact_region\":\"unknown\",\"roi\":";
  if (result.contact.has_roi) {
    stream << box_json(result.contact.roi);
  } else {
    stream << "null";
  }
  stream << "},\"contact_executed\":"
         << (result.contact_executed ? "true" : "false")
         << ",\"runtime_provider\":\"RKNN_NPU\""
         << ",\"board_model\":\"" << board_model << "\""
         << ",\"model_quantization\":\"" << quantization << "\""
         << ",\"timings_ms\":{"
         << "\"decode\":" << number(result.timings.decode_ms)
         << ",\"device_preprocess\":"
         << number(result.timings.device_preprocess_ms)
         << ",\"device_inference\":"
         << number(result.timings.device_inference_ms)
         << ",\"device_postprocess\":"
         << number(result.timings.device_postprocess_ms)
         << ",\"contact_preprocess\":"
         << number(result.timings.contact_preprocess_ms)
         << ",\"contact_inference\":"
         << number(result.timings.contact_inference_ms)
         << ",\"contact_postprocess\":"
         << number(result.timings.contact_postprocess_ms)
         << ",\"total\":" << number(result.timings.total_ms) << "}}";
  return stream.str();
}

}  // namespace medvision
