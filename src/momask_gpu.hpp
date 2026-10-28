#pragma once
#include "motion.hpp"
#include <stdexcept>
namespace motion {
struct MomaskGpuFailure : std::runtime_error {using std::runtime_error::runtime_error;};
// Numeric storage only crosses the isolated ORT 1.24.4 / CPU ORT 1.31.1 boundary.
// Types match ONNX float=1, int64=7 and bool=9; no runtime-owned pointers cross.
struct MomaskTensor {int type=1;std::vector<int64_t> shape;std::vector<unsigned char> bytes;};
class MomaskGpu {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    explicit MomaskGpu(const fs::path&,unsigned threads);
    ~MomaskGpu();
    MomaskTensor run(const std::string&,std::vector<MomaskTensor>);
    uint64_t peak_memory_bytes() const;
};
}
