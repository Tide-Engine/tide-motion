#pragma once
#include "motion.hpp"
#include <nlohmann/json.hpp>
namespace motion {
struct PoseSequence { Clip start; std::vector<Clip> targets; std::vector<int> tokens; nlohmann::json recipe; };
PoseSequence load_pose_sequence(const fs::path&);
// Append a bridge once, omitting its duplicate first endpoint.
void append_pose_bridge(Clip&, const Clip&);
Clip generate_pose_sequence(const PoseSequence&, BricksModel&);
}
