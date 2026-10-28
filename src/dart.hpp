#pragma once
#include "motion.hpp"
#include <nlohmann/json.hpp>
#include <optional>
namespace motion {
// Bounded, offline DART authoring. Fixed labels match the released policy's skills.
struct DartSegment { std::string text; unsigned chunks=0; std::optional<Vec3> goal; };
struct DartPlan { uint32_t seed=1; unsigned warmup=8; std::vector<DartSegment> segments; nlohmann::json recipe; };
DartPlan load_dart_plan(const fs::path&);
struct DartResult { Clip clip; nlohmann::json metrics; };
class DartModel {
    struct Impl; std::unique_ptr<Impl> impl_;
public:
    explicit DartModel(const fs::path& directory,unsigned threads=0);
    ~DartModel();
    DartResult generate(const DartPlan&);
    // Development verification against independently exported PyTorch tensors.
    double verify_reference(const fs::path&);
};
}
