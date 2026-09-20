#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace medvision {

struct BoardPoint {
  float x = 0.0F;
  float y = 0.0F;
};

struct BoardBox {
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
};

struct BoardDetection {
  bool valid = false;
  BoardBox bbox;
  float score = 0.0F;
};

enum class BoardVisibility {
  kVisible,
  kOccludedInferable,
  kNotAnnotatable,
};

struct BoardContact {
  bool has_point = false;
  BoardPoint point;
  BoardVisibility visibility = BoardVisibility::kNotAnnotatable;
  float confidence = 0.0F;
  float heatmap_confidence = 0.0F;
  bool has_roi = false;
  BoardBox roi;
};

struct BoardTimings {
  double decode_ms = 0.0;
  double device_preprocess_ms = 0.0;
  double device_inference_ms = 0.0;
  double device_postprocess_ms = 0.0;
  double contact_preprocess_ms = 0.0;
  double contact_inference_ms = 0.0;
  double contact_postprocess_ms = 0.0;
  double total_ms = 0.0;
};

struct BoardFrameResult {
  std::uint64_t frame_id = 0;
  std::int64_t timestamp_ms = 0;
  int image_width = 0;
  int image_height = 0;
  BoardDetection device;
  BoardContact contact;
  bool contact_executed = false;
  BoardTimings timings;
};

struct Rk3568PipelineConfig {
  std::string device_model_path;
  std::string contact_model_path;
  float detection_confidence = 0.25F;
  float nms_iou = 0.65F;
  float contact_roi_expansion = 0.20F;
};

class Rk3568Pipeline {
 public:
  explicit Rk3568Pipeline(const Rk3568PipelineConfig& config);
  ~Rk3568Pipeline();

  Rk3568Pipeline(const Rk3568Pipeline&) = delete;
  Rk3568Pipeline& operator=(const Rk3568Pipeline&) = delete;

  BoardFrameResult infer(
      const std::vector<std::uint8_t>& jpeg,
      std::uint64_t frame_id,
      std::int64_t timestamp_ms,
      bool infer_contact);

  std::string sdk_api_version() const;
  std::string driver_version() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

const char* to_string(BoardVisibility visibility);
std::string frame_result_json(
    const BoardFrameResult& result,
    const std::string& board_model,
    const std::string& quantization);

}  // namespace medvision
