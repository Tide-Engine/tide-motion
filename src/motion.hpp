#pragma once
#include <array>
#include <filesystem>
#include <memory>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace motion {
namespace fs = std::filesystem;
fs::path utf8path(std::string_view);
struct Vec3 { float x=0, y=0, z=0; };
struct Quat { float x=0, y=0, z=0, w=1; };
struct Transform { Vec3 p; Quat q; };
Vec3 operator+(Vec3, Vec3);
Vec3 operator-(Vec3, Vec3);
Vec3 operator*(Vec3, float);
float dot(Vec3, Vec3);
Vec3 cross(Vec3, Vec3);
Vec3 normalized(Vec3);
Quat normalized(Quat);
Quat inverse(Quat);
Quat operator*(Quat, Quat);
Vec3 rotate(Quat, Vec3);
Quat slerp(Quat, Quat, float);
Quat yaw(float);
Quat rotation6(Vec3, Vec3);
Transform operator*(Transform, Transform);
Transform inverse(Transform);
Transform blend(Transform, Transform, float);
struct Joint { std::string name; int parent=-1; Transform rest; float scale=1; };
struct Clip {
    std::string name="motion";
    float fps=30;
    std::vector<Joint> joints;
    std::vector<Transform> poses;
    size_t motion_root=0;
    size_t frames() const;
    float duration() const;
    std::span<Transform> frame(size_t);
    std::span<const Transform> frame(size_t) const;
    void validate() const;
};
std::vector<Joint> soma();
void require_soma(const Clip&);
std::vector<Transform> world_pose(const Clip&, std::span<const Transform>);
std::vector<Transform> sample(const Clip&, float);
Clip load_glb(const fs::path&, const std::string& animation="", bool canonical=true);
Clip load_rig_glb(const fs::path&);
// .fbx through ufbx with Tide's import conventions (Y-up metres); used by the GLB loaders.
bool is_fbx(const fs::path&);
Clip read_fbx(const fs::path&, const std::string& animation, bool rest_only);
void save_glb(const Clip&, const fs::path&);
struct RetargetMapping {
    std::array<std::string,30> bones;
    std::map<std::string,Quat> reference_rotations;
    float root_scale=0, heading_degrees=0, ground_offset=0;
};
struct RetargetResult { Clip clip; float scale=1, heading_degrees=0; bool discarded_translations=false; };
RetargetMapping load_mapping(const fs::path&);
RetargetResult retarget(const Clip& source, const Clip& target, const RetargetMapping&, bool to_soma=false);
// Kimodo SOMA-30 motion representation rows (upstream kimodo_motionrep.py): smooth root
// xyz, heading cos/sin, root-relative joint positions, global cont6d rotations,
// velocities and foot contacts. Constraints fill the first three blocks.
inline constexpr size_t kimodo_features=369;
std::vector<float> encode_kimodo(const Clip& soma_rig, std::span<const Transform> pose);
Clip decode_kimodo(std::span<const float> raw);
float kimodo_heading(const Clip& soma_rig, std::span<const Transform> pose);
unsigned parse_effectors(const std::string&);
std::string effector_list(unsigned);
// A SOMA pose required at an output frame: the whole body, or with effector bits
// (1 left foot, 2 right foot, 4 left hand, 8 right hand) only those chains.
struct PoseConstraint {
    std::string kind; size_t frame=0; std::string file; long source_frame=0;
    std::vector<Transform> pose; unsigned effectors=0; bool anchored=true;
};
struct PathConstraint { size_t frame=0; float x=0, z=0; bool has_heading=false; float heading=0; };
struct MotionConditions {
    std::vector<PoseConstraint> poses; std::vector<PathConstraint> path;
    bool empty() const { return poses.empty()&&path.empty(); }
};
// world pins every pose to its ground position in its file; free leaves later poses
// to float (root height, facing and body kept); automatic is world when output frame 0
// or a path fixes the world, otherwise free. Returns whether the output should then be
// moved so its last pose lands where it is in its file.
enum class Placement { automatic, world, free };
bool place_constraints(MotionConditions&, Placement);
void shift_to_last_pose(Clip& result, const MotionConditions&);
// Makes full-body poses exact, fading the residual out over nearby frames.
void correct_poses(Clip& result, const MotionConditions&);
struct KimodoCondition { std::vector<float> observed, mask; float first_heading=0; Vec3 origin; };
KimodoCondition build_condition(const MotionConditions&, size_t frames);
struct ConstraintError { float root=0, position_mean=0, position_max=0, rotation_mean=0, rotation_max=0; };
ConstraintError constraint_error(const Clip& result, const PoseConstraint&);
float path_error(const Clip& result, const PathConstraint&);
struct BricksInputs {
    std::array<float,40> root{};
    std::array<float,32> local_root{};
    std::array<float,2136> pose{};
    std::array<int32_t,8> has_root{}, has_local_root{}, has_pose{};
    std::array<int32_t,11> allowed{};
    Transform alignment;
};
BricksInputs encode_bricks(const Clip&, const std::array<std::vector<Transform>,8>&, int tokens=0);
Clip decode_bricks(const Clip&, const BricksInputs&, std::span<const float>, int tokens,
    std::span<const Transform> start, std::span<const Transform> end);
Clip connect(const Clip&, const Clip&, const fs::path& model, int tokens=0);
class BricksModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit BricksModel(const fs::path&);
    ~BricksModel();
    Clip generate(const Clip&, const Clip&, int tokens=0);
};
Clip join(const Clip&, const Clip& bridge, const Clip&);
// A procedural SOMA-30 preview body (skinned GLB, no third-party content).
void save_mannequin(const fs::path&);
// Runs and waits; a nonempty output path receives the child's stdout and stderr.
int run_process(const std::vector<std::string>& args, const fs::path& output={});
fs::path model_directory();
void setup_models(const fs::path&, const std::string& selection);
void check_models(const fs::path&, const std::string& selection);
std::string sha256_file(const fs::path&);
// skeleton draws lines; mannequin a procedural SOMA body (the clip must be SOMA-30);
// character a skinned GLB/FBX whose rig the clip already animates (retargeted).
enum class PreviewBody { skeleton, mannequin, character };
struct PreviewOptions {
    std::string tide="tide"; fs::path capture, gif; int frames=0;
    PreviewBody body=PreviewBody::mannequin; fs::path character, texture;
    float facing=0; // radians, the first frame's heading (0 faces +Z)
};
void preview(const Clip&, const fs::path& source, const PreviewOptions&);
}
