#include "medvision/rknn_backend.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> read_binary(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open input: " + path);
  }
  const auto size = stream.tellg();
  if (size < 0) {
    throw std::runtime_error("cannot stat input: " + path);
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  if (!bytes.empty() &&
      !stream.read(
          reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("cannot read input: " + path);
  }
  return bytes;
}

void write_floats(const std::string& path, const std::vector<float>& values) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open output: " + path);
  }
  if (!values.empty()) {
    stream.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
  }
  if (!stream) {
    throw std::runtime_error("cannot write output: " + path);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: rknn_probe MODEL.rknn INPUT_U8_NHWC.bin OUTPUT_PREFIX\n";
    return 2;
  }
  try {
    medvision::RknnModel model(argv[1]);
    std::cout << "RKNN_API=" << model.sdk_version().api_version << "\n";
    std::cout << "RKNN_DRIVER=" << model.sdk_version().drv_version << "\n";
    for (const auto& attr : model.input_attrs()) {
      std::cout << "INPUT " << medvision::describe_tensor(attr) << "\n";
    }
    for (const auto& attr : model.output_attrs()) {
      std::cout << "OUTPUT " << medvision::describe_tensor(attr) << "\n";
    }

    const auto input = read_binary(argv[2]);
    const auto outputs = model.run_u8_nhwc(input);
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const std::string path =
          std::string(argv[3]) + ".output" + std::to_string(index) + ".f32";
      write_floats(path, outputs[index].values);
      std::cout << "OUTPUT_FILE index=" << index << " name="
                << outputs[index].name << " elems=" << outputs[index].values.size()
                << " path=" << path << "\n";
    }
    std::cout << "RKNN_PROBE=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "RKNN_PROBE=FAIL error=" << error.what() << "\n";
    return 1;
  }
}
