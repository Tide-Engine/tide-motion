// Tide Motion research probe. No ARDY source is incorporated.
// Loads the released F32 safetensors container and evaluates ONE learned affine
// projection with synthetic input. This is NOT a denoiser or generation backend.
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
int main(int argc, char** argv) try {
    require(argc == 2, "usage: ardy-native-probe denoiser.safetensors (verify SHA-256 externally first)");
    static_assert(std::endian::native == std::endian::little);
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    require(bool(file), "cannot open checkpoint");
    const auto size = static_cast<uint64_t>(file.tellg());
    require(size >= 8 && size <= 1024ull*1024*1024, "checkpoint size outside probe bounds");
    file.seekg(0);
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), 8);
    require(header_size > 0 && header_size <= 1024*1024 && header_size <= size-8, "invalid header length");
    std::string header(static_cast<size_t>(header_size), ' ');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    require(bool(file), "truncated header");
    const auto tensors = json::parse(header);
    require(tensors.is_object(), "header must be an object");
    const auto data_start = 8 + header_size;
    std::vector<std::pair<uint64_t,uint64_t>> spans;
    uint64_t elements = 0;
    for (const auto& [name, tensor] : tensors.items()) {
        if (name == "__metadata__") continue;
        require(tensor.at("dtype") == "F32", "probe only accepts F32 tensors");
        const auto shape = tensor.at("shape").get<std::vector<uint64_t>>();
        uint64_t count = 1;
        for (auto dim : shape) {
            require(dim > 0 && dim <= (size-data_start)/4 && count <= (size-data_start)/4/dim, "invalid tensor shape");
            count *= dim;
        }
        const auto offsets = tensor.at("data_offsets").get<std::vector<uint64_t>>();
        require(offsets.size() == 2 && offsets[0] <= offsets[1] && offsets[1] <= size-data_start, "tensor outside payload");
        require(offsets[1]-offsets[0] == count*4, "tensor shape/byte mismatch");
        spans.emplace_back(offsets[0], offsets[1]);
        elements += count;
    }
    std::sort(spans.begin(), spans.end());
    uint64_t end = 0;
    for (const auto& span : spans) {
        require(span.first == end, "overlapping or gapped tensor data");
        end = span.second;
    }
    require(end == size-data_start, "trailing or missing tensor data");
    auto read = [&](const std::string& name, const std::vector<uint64_t>& shape) {
        const auto& t = tensors.at(name);
        require(t.at("shape").get<std::vector<uint64_t>>() == shape, "projection shape differs from pinned SOMA checkpoint");
        auto offsets = t.at("data_offsets").get<std::vector<uint64_t>>();
        std::vector<float> result(static_cast<size_t>((offsets[1]-offsets[0])/4));
        file.seekg(static_cast<std::streamoff>(data_start + offsets[0]));
        file.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()*4));
        require(bool(file), "truncated projection");
        for (float v : result) require(std::isfinite(v), "nonfinite projection weight");
        return result;
    };
    const auto weights = read("denoiser.global_root_hybrid_proj.weight", {1024,148});
    const auto bias = read("denoiser.global_root_hybrid_proj.bias", {1024});
    // A real input projection: four explicit 5D roots + a 128D body token.
    // Separate double accumulation checks scalar F32 arithmetic, NOT PyTorch parity.
    double checksum = 0, max_error = 0;
    for (size_t row = 0; row < 1024; ++row) {
        float y = bias[row];
        double reference = bias[row];
        for (size_t col = 0; col < 148; ++col) {
            float x = static_cast<float>(static_cast<int>(col%17)-8)/16;
            y += weights[row*148+col]*x;
            reference += static_cast<double>(weights[row*148+col])*x;
        }
        require(std::isfinite(y), "nonfinite projection output");
        max_error = std::max(max_error, std::abs(y-reference));
        require(std::abs(y-reference) <= 1e-4*(1+std::abs(reference)), "projection arithmetic check failed");
        checksum += y;
    }
    std::cout << std::setprecision(12) << "F32 tensors=" << spans.size() << " elements=" << elements
              << " bytes=" << size << "\nprojection=1024x148 checksum=" << checksum
              << " max_abs_vs_double=" << max_error
              << "\nONE affine operator only; no native generation or framework parity validated.\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
