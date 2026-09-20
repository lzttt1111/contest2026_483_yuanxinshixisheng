#include "medvision/rknn_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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

double percentile95(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto position = static_cast<std::size_t>(
      std::ceil(0.95 * static_cast<double>(values.size())) - 1.0);
  return values[std::min(position, values.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: rknn_batch_probe MODEL.rknn INPUT_LIST.txt\n";
    std::cerr << "list format: INPUT_U8_NHWC.bin OUTPUT_PREFIX\n";
    return 2;
  }
  try {
    medvision::RknnModel model(argv[1]);
    std::ifstream list(argv[2]);
    if (!list) {
      throw std::runtime_error(std::string("cannot open list: ") + argv[2]);
    }

    std::vector<double> durations_ms;
    std::string input_path;
    std::string output_prefix;
    std::size_t frame_index = 0;
    while (list >> input_path >> output_prefix) {
      const auto input = read_binary(input_path);
      const auto started = std::chrono::steady_clock::now();
      const auto outputs = model.run_u8_nhwc(input);
      const auto finished = std::chrono::steady_clock::now();
      const double duration_ms =
          std::chrono::duration<double, std::milli>(finished - started).count();
      durations_ms.push_back(duration_ms);

      for (std::size_t output_index = 0; output_index < outputs.size();
           ++output_index) {
        const std::string path =
            output_prefix + ".output" + std::to_string(output_index) + ".f32";
        write_floats(path, outputs[output_index].values);
      }
      std::cout << std::fixed << std::setprecision(3)
                << "FRAME index=" << frame_index
                << " inference_ms=" << duration_ms
                << " outputs=" << outputs.size() << "\n";
      ++frame_index;
    }
    if (durations_ms.empty()) {
      throw std::runtime_error("input list is empty");
    }
    const double mean_ms =
        std::accumulate(durations_ms.begin(), durations_ms.end(), 0.0)
        / static_cast<double>(durations_ms.size());
    std::cout << std::fixed << std::setprecision(3)
              << "RKNN_BATCH_PROBE=PASS frames=" << durations_ms.size()
              << " mean_ms=" << mean_ms
              << " p95_ms=" << percentile95(durations_ms) << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "RKNN_BATCH_PROBE=FAIL error=" << error.what() << "\n";
    return 1;
  }
}
