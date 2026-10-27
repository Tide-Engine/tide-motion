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
void save_glb(const Clip&, const fs::path&);
struct RetargetMapping {
    std::array<std::string,30> bones;
    std::map<std::string,Quat> reference_rotations;
    float root_scale=0, heading_degrees=0, ground_offset=0;
};
struct RetargetResult { Clip clip; float scale=1, heading_degrees=0; bool discarded_translations=false; };
RetargetMapping load_mapping(const fs::path&);
RetargetResult retarget(const Clip& source, const Clip& target, const RetargetMapping&, bool to_soma=false);
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
int run_process(const std::vector<std::string>& args);
fs::path model_directory();
void setup_models(const fs::path&, const std::string& selection);
void check_models(const fs::path&, const std::string& selection);
std::string sha256_file(const fs::path&);
void preview(const Clip&, const fs::path& source, const std::string& tide,
    const fs::path& capture, int frames);
}
