#pragma once
#include "motion.hpp"
#include <nlohmann/json.hpp>
namespace motion {
// early: frames before a guide target's arrival at which the next target takes over
// (upstream transition_early_by_frames); the guide pose is not reached exactly.
struct PoseTarget { Clip window; BricksRequest request; int early=0; };
struct PoseSequence { Clip start; std::vector<PoseTarget> targets; nlohmann::json recipe; };
struct AuthoredSequence { Clip clip; FootContacts contacts; nlohmann::json recipe; };
PoseSequence load_pose_sequence(const fs::path&);
// Append a bridge once, omitting its duplicate first endpoint.
void append_pose_bridge(Clip&, const Clip&);
// recipe is the plan's with model-chosen tokens and timing filled in.
AuthoredSequence generate_pose_sequence(const PoseSequence&, BricksModel&);
}
