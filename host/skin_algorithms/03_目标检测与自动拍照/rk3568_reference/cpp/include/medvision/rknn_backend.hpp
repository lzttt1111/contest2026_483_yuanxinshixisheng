#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <rknn_api.h>

namespace medvision {

struct RknnOutput {
  std::string name;
  std::vector<std::uint32_t> dims;
  std::vector<float> values;
};

class RknnModel {
 public:
  explicit RknnModel(const std::string& model_path);
  ~RknnModel();

  RknnModel(const RknnModel&) = delete;
  RknnModel& operator=(const RknnModel&) = delete;

  const rknn_sdk_version& sdk_version() const { return sdk_version_; }
  const std::vector<rknn_tensor_attr>& input_attrs() const {
    return input_attrs_;
  }
  const std::vector<rknn_tensor_attr>& output_attrs() const {
    return output_attrs_;
  }

  std::vector<RknnOutput> run_u8_nhwc(
      const std::vector<std::uint8_t>& input);

 private:
  void query_model();
  void reset();

  rknn_context context_ = 0;
  rknn_sdk_version sdk_version_{};
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
};

std::string describe_tensor(const rknn_tensor_attr& attr);

}  // namespace medvision
