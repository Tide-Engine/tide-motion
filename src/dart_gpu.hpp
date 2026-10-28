#pragma once
#include "motion.hpp"
#include <stdexcept>
namespace motion {
struct DartGpuFailure : std::runtime_error {using std::runtime_error::runtime_error;};
// Plain values only across the separate ORT 1.24.4 DirectML API boundary.
struct DartTensor {std::vector<int64_t> shape;std::vector<float> f;std::vector<int64_t> i;};
using DartFeed=std::map<std::string,DartTensor>;
class DartGpu {
    struct Impl;std::unique_ptr<Impl> impl_;
public:
    DartGpu(const fs::path&,unsigned threads);
    ~DartGpu();
    std::vector<float> run(const std::string& network,DartFeed,const char* output);
    uint64_t peak_memory_bytes() const;
};
bool dart_gpu_compiled();
}
