#pragma once
#include "motion.hpp"
#include <nlohmann/json.hpp>
#include <optional>
namespace motion {
// Bounded native authoring; text is not limited to the RL policy's training labels.
struct DartSegment { std::string text; unsigned chunks=0; std::optional<Vec3> goal;float guidance=5; };
struct DartHistory {std::array<std::array<float,276>,2> rows;Transform world;std::vector<Joint> reference;};
struct DartConstraint {size_t frame=0,joint=0;Vec3 position;std::array<bool,3> axes{true,true,true};float weight=1;};
struct DartSdf {fs::path file;std::array<size_t,3> dimensions{};Vec3 origin,spacing;std::vector<float> values;float margin=.03f,weight=1;};
struct DartOptimization {
    unsigned iterations=32;float rate=.01f,jerk_weight=0,floor_weight=0;bool unit_gradient=true,anneal=true,check_gradient=false;
    std::vector<DartConstraint> constraints;fs::path target_file,target_map_file;long target_frame=-1;
    std::vector<size_t> target_joints;float target_weight=1;std::optional<DartHistory> target_history;std::optional<Clip> target_clip;
    std::shared_ptr<DartSdf> scene;bool soma_joints=true;
};
struct DartPlan {
    uint32_t seed=1;unsigned warmup=8;std::vector<DartSegment> segments;nlohmann::json recipe;
    float guidance=5,eta=0,noise_scale=1,ground_height=0;bool zero_noise=false,fix_floor=false;
    std::string sampler="ddim",policy_mode="sample",seed_mode="history";
    std::vector<int> timesteps{0,1,2,3,4,5,6,7,8,9};
    fs::path history_file,source_file,source_map_file;long source_frame=-1;std::optional<DartHistory> history;
    std::optional<Clip> source;
    std::shared_ptr<DartOptimization> optimization;
};
DartPlan load_dart_plan(const fs::path&,bool defer_missing_sources=false);
struct DartResult { Clip clip; nlohmann::json metrics,history,initial_history; };
class DartModel {
    struct Impl; std::unique_ptr<Impl> impl_;
public:
    explicit DartModel(const fs::path& directory,unsigned threads=0,const std::string& device="auto");
    ~DartModel();
    DartResult generate(const DartPlan&);
    DartResult optimize(const DartPlan&);
    // Independent streams share policy/CFG/decoder calls. Maximum eight jobs.
    std::vector<DartResult> generate(std::span<const DartPlan>);
    std::string device() const;
    // Development verification against independently exported PyTorch tensors.
    double verify_reference(const fs::path&);
    double verify_batch_reference(const fs::path& directory);
    double verify_gradient_reference(const fs::path&);
};
}
