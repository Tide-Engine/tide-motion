#pragma once
#include "motion.hpp"
#include <stdexcept>
namespace motion {
struct ArdyGpuFailure : std::runtime_error {using std::runtime_error::runtime_error;};
// Values are copied across the isolated GPU runtime boundary. No OrtValue,
// allocator or API table crosses between ORT CPU 1.31.1 and DirectML ORT 1.24.4.
struct ArdyTensor {
    std::vector<int64_t> shape;std::vector<float> f;std::vector<int64_t> i;
    static ArdyTensor floats(std::vector<int64_t> s,std::vector<float> data){return {std::move(s),std::move(data),{}};}
    static ArdyTensor ints(std::vector<int64_t> s,std::vector<int64_t> data){return {std::move(s),{},std::move(data)};}
};
using ArdyFeed=std::map<std::string,ArdyTensor>;
class ArdyGpu {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit ArdyGpu(const fs::path& model_directory,unsigned threads);
    ~ArdyGpu();
    std::vector<float> run(const std::string& network,ArdyFeed);
    uint64_t peak_memory_bytes() const;
};
bool ardy_gpu_compiled();
}
