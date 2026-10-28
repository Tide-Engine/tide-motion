#pragma once
// Rules that guide Kimodo's sampling (generate --guide): a hand held on an object,
// hands held where a pose has them on the chest, and the body kept out of itself by
// capsules. At each denoising step the clean prediction is decoded, its upper body is
// turned as little as it can be to keep the rules (a small least-squares solve per
// frame), and the prediction moves toward that pose before the DDIM update, so the
// remaining steps continue from it. Ported from tide-prototypes tools/motion/solve.py.
// Rules name SOMA joints, or with a "character" (a rig and its retarget mapping) that
// character's bones, measured on it as retargeting would pose it; the solve always
// turns SOMA joints.
#include "motion.hpp"
#include <nlohmann/json.hpp>
#include <optional>

namespace motion {
// A rigid frame on a body bone (a held object), unscaled.
struct GuideObject { std::string name; size_t bone=0; Transform offset; };
// hand held at target in the object's frame from start to end (inclusive), turned about
// axis (object frame) by up to around radians to fit, eased in and out over ease frames.
struct GuideGrip {
    std::string name; size_t hand=0, object=0; Transform target; Vec3 axis{0,1,0};
    float around=0; bool orientation=true; long start=0, end=-1; float ease=0;
};
// Hands where pose has them relative to `relative`, and arm joints (SOMA) drawn to its turns.
struct GuideHold {
    long start=0, end=-1; float ease=0; size_t relative=3;
    std::vector<size_t> hands; std::vector<Transform> placed;
    std::vector<size_t> arms; std::vector<Quat> turns;
};
// A capsule from one joint to another, or from start to end in one joint's frame (a
// sphere when they are the same point; a blade with radius 0).
struct GuideCapsule { std::string name; size_t from=0; int to=-1; Vec3 start, end; float radius=0; };
struct GuidePair { size_t a=0, b=0; float slack=0; };
struct GuideKeep { size_t bone=0; };
struct GuideWeights {
    float smooth=0.3f;                  // per degree of change from the frame before
    float grip_place=3, grip_turn=0.3f; // per cm / degree off a grip
    float hold_place=3, hold_turn=0.3f, hold_arm=0.15f;
    float keep_place=0.5f, keep_turn=0.3f;
    float inside=3;                     // per cm of capsule overlap
    float slack=0.005f;                 // metres of overlap before it counts
};
struct Guide {
    std::vector<size_t> joints; std::vector<float> stiffness; // turned SOMA joints, cost per degree
    std::optional<RetargetProxy> character; std::vector<std::string> bones; // the rules' skeleton
    std::vector<GuideObject> objects; std::vector<GuideGrip> grips; std::vector<GuideHold> holds;
    std::vector<GuideCapsule> capsules; std::vector<GuidePair> pairs; std::vector<GuideKeep> keeps;
    GuideWeights weights;
    float strength=1, from=0, until=1; unsigned iterations=6;
    nlohmann::json effective; // as resolved, for the recipe
    bool empty() const { return grips.empty()&&holds.empty()&&pairs.empty(); }
};
// A version-1 guide (inline or from a file); relative paths resolve against base.
Guide load_guide(const nlohmann::json& spec, const fs::path& base);
// Resolves negative frames and checks ranges against the output's frame count.
void resolve_guide(Guide&, size_t frames);
// Guides one clean prediction: rows of kimodo_features unnormalized features.
// Returns milliseconds spent.
double guide_clean(const Guide&, std::span<float> features, size_t frames);
// Whether the guide acts at a denoising step (step counts down from steps-1 to 0).
bool guide_step(const Guide&, unsigned step, unsigned steps);
// Each rule measured on a finished SOMA clip, for the recipe.
nlohmann::json guide_report(const Guide&, const Clip&);
}
