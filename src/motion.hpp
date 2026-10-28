#pragma once
#include <array>
#include <filesystem>
#include <iosfwd>
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
// SOMA-77 presentation skeleton (upstream somaskel77): the 30 predicted joints plus
// fingers, toe ends and the head end. Expansion keeps the 30 local rotations and gives
// the other joints relaxed-hands rotations (or those of `others`, a SOMA-77 clip with
// as many frames); reduction takes the 30 back exactly. as_soma77 reorders by name.
std::vector<Joint> soma77();
Clip as_soma77(const Clip&);
bool is_soma77(const Clip&);
Clip soma77_to_30(const Clip&);
Clip soma30_to_77(const Clip&, const Clip* others=nullptr);
Clip mirror_soma77(const Clip&);
std::vector<Transform> world_pose(const Clip&, std::span<const Transform>);
std::vector<Transform> sample(const Clip&, float);
Clip load_glb(const fs::path&, const std::string& animation="", bool canonical=true);
// Explicit transform-track selection plus ancestors. Render-only descendants
// are excluded, so their static nonuniform mesh scale is not a rig restriction.
// No nodes are identified automatically; unrelated channels are not imported.
Clip load_glb_nodes(const fs::path&,const std::vector<std::string>& nodes);
Clip load_rig_glb(const fs::path&, const std::vector<std::string>& extra_nodes={});
// .fbx through ufbx with Tide's import conventions (Y-up metres); used by the GLB loaders.
bool is_fbx(const fs::path&);
Clip read_fbx(const fs::path&, const std::string& animation, bool rest_only, const std::vector<std::string>& extra_nodes={});
// Every named node of a GLB or FBX file (FBX as Tide imports it), bones with no skin
// weights included; preview binds a prop to one of them by name.
std::vector<std::string> node_names(const fs::path&);
std::vector<std::string> fbx_node_names(const fs::path&);
void save_glb(const Clip&, const fs::path&);
struct RetargetMapping {
    std::array<std::string,30> bones;
    std::map<std::string,Quat> reference_rotations;
    float root_scale=0, heading_degrees=0, ground_offset=0;
};
struct FootContact { unsigned foot=0; size_t start=0,end=0; };
struct CleanupPlan {
    std::vector<FootContact> contacts;
    std::vector<size_t> protected_frames;
    unsigned fade_frames=3;
    float max_correction=0.1f;
    float max_joint_correction=30, max_joint_step_increase=5, max_ankle_step_increase=0.005f;
};
struct ContactReport {
    size_t samples=0; bool skipped=false; std::string skip_reason;
    float drift_before=0,drift_after=0,residual_max=0,rotation_max=0;
    float joint_step_before=0,joint_step_after=0,ankle_step_before=0,ankle_step_after=0;
    float attempted_rotation_max=0,joint_step_increase=0,ankle_step_increase=0;
};
CleanupPlan load_cleanup_plan(const fs::path&);
// Explicit contact intervals only. Keeps all translations, root and world foot orientation.
// Custom rigs require a mapping and direct thigh/calf/foot chains; helper chains refused.
std::vector<ContactReport> cleanup_contacts(Clip&,const CleanupPlan&,const RetargetMapping* =nullptr);
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
// interpolated: a dense-path sample between waypoints (it neither places nor turns poses).
struct PathConstraint { size_t frame=0; float x=0, z=0; bool has_heading=false; float heading=0; bool interpolated=false; };
struct MotionConditions {
    std::vector<PoseConstraint> poses; std::vector<PathConstraint> path;
    bool empty() const { return poses.empty()&&path.empty(); }
};
// Colocated paths first place/turn poses, overriding their source ground location.
// world pins every pose to its ground position in its file; free leaves later poses
// to float (root height, facing and body kept); automatic is world when output frame 0
// or a path fixes the world, otherwise free. Returns whether the output should then be
// moved so its last pose lands where it is in its file.
enum class Placement { automatic, world, free };
bool place_constraints(MotionConditions&, Placement);
void shift_to_last_pose(Clip& result, const MotionConditions&);
// Turns a pose constraint by `turn` radians of yaw about the vertical through `pivot`,
// then moves it by `move` on the ground; its body-local rotations are unchanged.
void turn_pose(PoseConstraint&, Vec3 pivot, float turn, Vec3 move);
// --follow ends: keyframes and effectors take the fraction of the end pose's placement
// (the turn and move --end-at or a final path point gave it) that their frame lies
// between the start and end poses, about the end pose's original hips. `source` is
// the end pose before placement. Returns the turn applied to each such pose (radians).
std::vector<float> follow_ends(MotionConditions&, const PoseConstraint& source);
// Makes full-body poses exact, fading the residual out over up to `frames` frames each
// side (0 sets only the constrained frame).
void correct_poses(Clip& result, const MotionConditions&, int frames=10);
// Native port of upstream Kimodo MotionCorrection (post_process_motion): root height
// and ground warps, full-body rotation warps, two-bone effector IK and foot/toe contact
// IK from the model's own contacts ([frames, 4]: LeftFoot, LeftToeBase, RightFoot,
// RightToeBase). An opt-in alternative to correct_poses; SOMA-30 only.
struct KimodoCorrectionReport {
    std::array<size_t,4> contact_frames{}; size_t effector_frames=0, ankle_frames=0, toe_frames=0; double milliseconds=0;
};
// trace (developer parity checks) receives upstream binding-layout inputs and the result.
KimodoCorrectionReport correct_kimodo(Clip&, const MotionConditions&, std::span<const float> contacts, float root_margin=0.04f, float threshold=0.5f, std::ostream* trace=nullptr);
// Intervals [start, end) where a decoded contact channel exceeds 0.5 (upstream's threshold).
std::vector<std::pair<size_t,size_t>> contact_intervals(std::span<const float> contacts, size_t channel);
struct KimodoCondition { std::vector<float> observed, mask; float first_heading=0; Vec3 origin; };
KimodoCondition build_condition(const MotionConditions&, size_t frames);
struct ConstraintError { float root=0, position_mean=0, position_max=0, rotation_mean=0, rotation_max=0; std::string rotation_max_joint; };
ConstraintError constraint_error(const Clip& result, const PoseConstraint&);
float path_error(const Clip& result, const PathConstraint&);
// Upstream smooth_root.py smooth_signal on ground x/z (multigrid ADMM within `margin`
// metres of the input, minimizing acceleration; pos_weight 0, relaxation 1.8, 500
// iterations per level), including its final half-frame averaging. Fewer than five
// samples are returned unchanged (upstream's coarsest system would be singular).
std::vector<std::array<double,2>> smooth_root_path(std::span<const std::array<double,2>>, double margin=0.06);
// The output's smooth root on the ground as the motion representation defines it.
std::vector<std::array<double,2>> smooth_root_of(const Clip&);
// Upstream's dense 2D root path: waypoints (plus frame 0 at `start` when none is
// there) linearly interpolated to every frame of their span, then smoothed when
// there are at least three. Waypoints keep their yaw; added samples are interpolated.
void densify_path(MotionConditions&, Vec3 start);
struct BricksInputs {
    std::array<float,40> root{};
    std::array<float,32> local_root{};
    std::array<float,2136> pose{};
    std::array<int32_t,8> has_root{}, has_local_root{}, has_pose{};
    std::array<int32_t,11> allowed{};
    Transform alignment;
};
// Upstream MotionBricksInferenceConfig/num_poses_to_use: the model chooses a token
// count in min..max (6..16, default 8..14). target_poses 1..4 of the target window
// condition the model: 1 sets a stop pose, 2 adds velocity, 3-4 acceleration.
struct BricksRequest { int min_tokens=8, max_tokens=14, target_poses=4; };
// 0 is the model default 8..14; N fixes N tokens.
BricksRequest bricks_tokens(int tokens);
// "N" or "MIN..MAX".
BricksRequest parse_bricks_tokens(const std::string&);
BricksInputs encode_bricks(const Clip&, const std::array<std::vector<Transform>,8>&, const BricksRequest&);
BricksInputs encode_bricks(const Clip&, const std::array<std::vector<Transform>,8>&, int tokens=0);
// How far the raw prediction missed the start and end seams before endpoint
// correction: hips position deltas and the largest joint rotation change (degrees).
struct SeamCorrection { std::array<Vec3,2> root{}; std::array<float,2> joint_degrees{}; std::array<size_t,2> joint{}; };
Clip decode_bricks(const Clip&, const BricksInputs&, std::span<const float>, int tokens,
    std::span<const Transform> start, std::span<const Transform> end, SeamCorrection* correction=nullptr);
// Predicted contacts per trimmed bridge frame: left heel, left toe, right heel,
// right toe, each clamped to 0..1 (upstream decode_motionbricks_output_kernel).
using FootContacts = std::vector<std::array<float,4>>;
FootContacts decode_bricks_contacts(std::span<const float>, int tokens);
struct Bridge { Clip clip; FootContacts contacts; int tokens=0; SeamCorrection correction; };
// Heel runs at or above threshold lasting at least three frames, as explicit cleanup
// intervals (the ankle is the cleanup anchor; toe-only contact lets the ankle rise).
std::vector<FootContact> planted_contacts(const FootContacts&, float threshold=0.5f, size_t offset=0);
Clip connect(const Clip&, const Clip&, const fs::path& model, int tokens=0);
class BricksModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    // threads 0 uses half the hardware threads.
    explicit BricksModel(const fs::path&, unsigned threads=0);
    ~BricksModel();
    Clip generate(const Clip&, const Clip&, int tokens=0);
    Bridge generate(const Clip&, const Clip&, const BricksRequest&);
    // Independent requests in one ONNX run along the batch axis (upstream N); results
    // keep request order. Each request is encoded, trimmed and corrected separately.
    struct Request { const Clip* from; const Clip* to; BricksRequest options; };
    std::vector<Bridge> generate(std::span<const Request>);
};
// Largest number of requests Tide sends in one MotionBricks run.
inline constexpr size_t bricks_batch_limit=32;
Clip join(const Clip&, const Clip& bridge, const Clip&);
// Maps inherited pins through the same endpoint omission as join(); always pins bridge ends.
std::vector<size_t> connected_protected_frames(size_t source_frames,size_t bridge_frames,size_t target_frames,
    std::span<const size_t> source_pins,std::span<const size_t> target_pins,bool joined);
// Variants of CLIP.glb are CLIP.seedN.glb; their contact sheet CLIP.variants.png/.gif.
fs::path variant_path(const fs::path& output, uint64_t seed);
fs::path variants_path(const fs::path& output, const std::string& extension);
// Left and right swapped and reflected across the first frame's sagittal plane (SOMA-30).
void mirror_soma(Clip&);
// Removes horizontal root travel (a one-second moving average of the root's ground path),
// keeping height, facing and sway; frame 0 stays where it was.
void remove_root_travel(Clip&);
struct Footfall { unsigned foot=0; size_t lift=0,landing=0; float distance=0,height=0; };
// Completed travelling footfalls on SOMA, not an action classifier.
std::vector<Footfall> footfalls(const Clip&);
struct RootTravel { float displacement=0, distance=0, peak_speed=0; };
RootTravel root_travel(const Clip&);
// Frames that show a clip at a glance: first, last and its strongest moments, in order.
std::vector<size_t> key_frames(const Clip&, size_t count);
// A procedural SOMA-30 preview body (skinned GLB, no third-party content).
void save_mannequin(const fs::path&, bool arms_only=false);
// Runs and waits; a nonempty output path receives the child's stdout and stderr.
int run_process(const std::vector<std::string>& args, const fs::path& output={});
fs::path model_directory();
void setup_models(const fs::path&, const std::string& selection);
void check_models(const fs::path&, const std::string& selection);
std::string sha256_file(const fs::path&);
// skeleton draws lines; mannequin a procedural SOMA body (the clip must be SOMA-30);
// character a skinned GLB/FBX whose rig the clip already animates (retargeted).
enum class PreviewBody { skeleton, mannequin, character };
struct PreviewProp {
    fs::path mesh; std::string bone; Vec3 offset{}, rotation{};
};
struct PreviewCamera {
    bool enabled=false;
    std::string bone="Head", mode="bone";
    Vec3 offset{0,0.08f,0.12f}, rotation{};
    float fov=75, near_plane=0.01f;
};
struct EquipmentResult;
struct PreviewOptions {
    std::string tide="tide"; fs::path capture, gif; int frames=0;
    PreviewBody body=PreviewBody::mannequin; fs::path character, texture;
    float facing=0; // radians, the first frame's heading (0 faces +Z)
    // A mesh drawn rigidly on a bone of the body (its model matrix without scale),
    // then offset (metres) and turned (Euler degrees, as Tide's quaternion.Euler) in it.
    std::vector<PreviewProp> props;
    PreviewCamera camera;
    const EquipmentResult* equipment=nullptr; // transient authored world tracks, never serialized
};
// Setup paths are relative to the JSON file; serialized paths are absolute.
PreviewOptions load_preview_setup(const fs::path&);
void save_preview_setup(const PreviewOptions&, const fs::path&);
void validate_preview_setup(const PreviewOptions&);
// Bind the drawn rig's unweighted socket nodes and preserve their animated ancestors.
Clip preview_rig(const Clip&, const PreviewOptions&);
struct MeshBounds { Vec3 low{}, high{}; };
MeshBounds mesh_bounds(const fs::path&);
MeshBounds fbx_mesh_bounds(const fs::path&);
Transform preview_offset(Vec3 metres, Vec3 euler_degrees);
struct PreviewExtent { float low=0, high=0, reach=0, ground=0; };
PreviewExtent preview_extent(const Clip&, const PreviewOptions&);
void preview(const Clip&, const fs::path& source, const PreviewOptions&);
// One image (and optionally a GIF) of several SOMA clips on the mannequin, a labelled
// cell each: key poses in profile and from the front (the GIF plays them together).
struct VariantSheet { std::string tide="tide"; fs::path png, gif; };
void preview_variants(const std::vector<Clip>&, const std::vector<std::string>& labels, const VariantSheet&);
}
