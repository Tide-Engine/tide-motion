#pragma once
#include "motion.hpp"
#include <functional>
namespace motion {
struct ArdyPrompt { std::string text; size_t frames=0; };
struct ArdyOptions {
    unsigned steps=10,threads=8,history_frames=240;
    uint64_t seed=0;
    float text_guidance=2,constraint_guidance=2;
    std::span<const float> reference_noise{}; // fixed-noise parity fixtures, no CLI option
    std::span<const float> reference_history{}; // isolates host parity from FK/smoothing rounding
};
class ArdyModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit ArdyModel(const fs::path& directory,unsigned threads=8,const std::string& device="cpu");
    ~ArdyModel();
    std::string device() const;
    Clip generate(const std::vector<ArdyPrompt>&,const ArdyOptions&,
                  const MotionConditions& = {},const Clip* history=nullptr);
    // Developer-only fixed input tests; never uses Python at runtime.
    void verify_reference(const fs::path& fixture_directory);
    void write_text_reference(const std::string& prompt,const fs::path&);
};
std::vector<float> encode_ardy_history(const Clip&);
void check_ardy_models(const fs::path&);
}
