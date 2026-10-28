#pragma once
#include <memory>
#include <string>
#include <vector>
const char* momask_help();
class MomaskContext {
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend int momask_main(const std::vector<std::string>&,MomaskContext*,bool,bool);
public:
    MomaskContext();
    ~MomaskContext();
};
// Weights are supplied manually; no checkpoint-commercial-rights assertion.
int momask_main(const std::vector<std::string>& args,MomaskContext* context=nullptr,
                bool preflight_only=false,bool defer_source=false);
