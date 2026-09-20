#include "medvision/rknn_backend.hpp"

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace medvision {
namespace {

void check_rknn(int result, const char* operation) {
  if (result != RKNN_SUCC) {
    std::ostringstream stream;
    stream << operation << " failed with RKNN error " << result;
    throw std::runtime_error(stream.str());
  }
}

std::vector<std::uint32_t> tensor_dims(const rknn_tensor_attr& attr) {
  return std::vector<std::uint32_t>(attr.dims, attr.dims + attr.n_dims);
}

}  // namespace

RknnModel::RknnModel(const std::string& model_path) {
  const int result =
      rknn_init(&context_, const_cast<char*>(model_path.c_str()), 0, 0, nullptr);
  if (result != RKNN_SUCC) {
    context_ = 0;
    check_rknn(result, "rknn_init");
  }
  try {
    query_model();
  } catch (...) {
    reset();
    throw;
  }
}

RknnModel::~RknnModel() { reset(); }

void RknnModel::reset() {
  if (context_ != 0) {
    rknn_destroy(context_);
    context_ = 0;
  }
}

void RknnModel::query_model() {
  check_rknn(
      rknn_query(
          context_,
          RKNN_QUERY_SDK_VERSION,
          &sdk_version_,
          sizeof(sdk_version_)),
      "rknn_query(RKNN_QUERY_SDK_VERSION)");

  rknn_input_output_num counts{};
  check_rknn(
      rknn_query(
          context_,
          RKNN_QUERY_IN_OUT_NUM,
          &counts,
          sizeof(counts)),
      "rknn_query(RKNN_QUERY_IN_OUT_NUM)");
  if (counts.n_input != 1) {
    std::ostringstream stream;
    stream << "only one input is supported, model has " << counts.n_input;
    throw std::runtime_error(stream.str());
  }

  input_attrs_.resize(counts.n_input);
  for (std::uint32_t index = 0; index < counts.n_input; ++index) {
    std::memset(&input_attrs_[index], 0, sizeof(rknn_tensor_attr));
    input_attrs_[index].index = index;
    check_rknn(
        rknn_query(
            context_,
            RKNN_QUERY_INPUT_ATTR,
            &input_attrs_[index],
            sizeof(rknn_tensor_attr)),
        "rknn_query(RKNN_QUERY_INPUT_ATTR)");
  }

  output_attrs_.resize(counts.n_output);
  for (std::uint32_t index = 0; index < counts.n_output; ++index) {
    std::memset(&output_attrs_[index], 0, sizeof(rknn_tensor_attr));
    output_attrs_[index].index = index;
    check_rknn(
        rknn_query(
            context_,
            RKNN_QUERY_OUTPUT_ATTR,
            &output_attrs_[index],
            sizeof(rknn_tensor_attr)),
        "rknn_query(RKNN_QUERY_OUTPUT_ATTR)");
  }
}

std::vector<RknnOutput> RknnModel::run_u8_nhwc(
    const std::vector<std::uint8_t>& input) {
  const auto expected = input_attrs_.front().n_elems;
  if (input.size() != expected) {
    std::ostringstream stream;
    stream << "input size mismatch: expected " << expected << " bytes, got "
           << input.size();
    throw std::runtime_error(stream.str());
  }

  rknn_input input_desc{};
  input_desc.index = 0;
  input_desc.buf = const_cast<std::uint8_t*>(input.data());
  input_desc.size = static_cast<std::uint32_t>(input.size());
  input_desc.pass_through = 0;
  input_desc.type = RKNN_TENSOR_UINT8;
  input_desc.fmt = RKNN_TENSOR_NHWC;
  check_rknn(rknn_inputs_set(context_, 1, &input_desc), "rknn_inputs_set");
  check_rknn(rknn_run(context_, nullptr), "rknn_run");

  std::vector<rknn_output> raw_outputs(output_attrs_.size());
  for (std::size_t index = 0; index < raw_outputs.size(); ++index) {
    raw_outputs[index].want_float = 1;
    raw_outputs[index].is_prealloc = 0;
    raw_outputs[index].index = static_cast<std::uint32_t>(index);
  }
  check_rknn(
      rknn_outputs_get(
          context_,
          static_cast<std::uint32_t>(raw_outputs.size()),
          raw_outputs.data(),
          nullptr),
      "rknn_outputs_get");

  std::vector<RknnOutput> outputs;
  outputs.reserve(raw_outputs.size());
  try {
    for (std::size_t index = 0; index < raw_outputs.size(); ++index) {
      RknnOutput output;
      output.name = output_attrs_[index].name;
      output.dims = tensor_dims(output_attrs_[index]);
      const auto* begin = static_cast<const float*>(raw_outputs[index].buf);
      output.values.assign(begin, begin + output_attrs_[index].n_elems);
      outputs.push_back(std::move(output));
    }
  } catch (...) {
    rknn_outputs_release(
        context_,
        static_cast<std::uint32_t>(raw_outputs.size()),
        raw_outputs.data());
    throw;
  }
  check_rknn(
      rknn_outputs_release(
          context_,
          static_cast<std::uint32_t>(raw_outputs.size()),
          raw_outputs.data()),
      "rknn_outputs_release");
  return outputs;
}

std::string describe_tensor(const rknn_tensor_attr& attr) {
  std::ostringstream stream;
  stream << "index=" << attr.index << " name=" << attr.name << " dims=[";
  for (std::uint32_t index = 0; index < attr.n_dims; ++index) {
    if (index != 0) {
      stream << ",";
    }
    stream << attr.dims[index];
  }
  stream << "] elems=" << attr.n_elems << " bytes=" << attr.size
         << " fmt=" << get_format_string(attr.fmt)
         << " type=" << get_type_string(attr.type)
         << " qnt=" << get_qnt_type_string(attr.qnt_type)
         << " zp=" << attr.zp << " scale=" << attr.scale;
  return stream.str();
}

}  // namespace medvision
