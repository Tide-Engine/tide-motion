#pragma once
#include "equipment.hpp"
#include <array>
#include <span>

namespace motion {
inline constexpr size_t dex2hoi_bones=65;
inline constexpr size_t dex2hoi_slots=2;
inline constexpr size_t dex2hoi_slot_features=18;
inline constexpr size_t dex2hoi_features=dex2hoi_bones*6+3+dex2hoi_slots*dex2hoi_slot_features;
inline constexpr size_t dex2hoi_left_wrist=10,dex2hoi_right_wrist=34;

// Explicit checkpoint-order mapping onto a supplied compatible rest rig. Indices
// 0, 10 and 34 are hips, left wrist and right wrist. No bone names are inferred.
// Rig rest rotations define the Blender pose bases; a name map alone does not
// establish that another character has the checkpoint's trained bone frames.
struct Dex2HoiRig {
    std::vector<Joint> joints;
    std::array<size_t,dex2hoi_bones> bones{};
    void validate() const;
};
struct Dex2HoiFormat {
    bool root_displacement=true;
    bool object_displacement=true;
    bool x0_as_condition=true;
    bool object_relative_to_root=false;
};
struct Dex2HoiNormalizer {
    std::array<float,dex2hoi_features> mean{};
    std::array<float,dex2hoi_features> stddev=[] {
        std::array<float,dex2hoi_features> values{};values.fill(1);return values;
    }();
    float epsilon=1e-5f;
    void validate() const;
    void normalize(std::span<float>) const;
    void denormalize(std::span<float>) const;
};
struct Dex2HoiObject {
    std::string name;
    fs::path mesh;
    bool present=true;
    // Duplicate conditioning slots can suppress one exported track explicitly.
    bool export_track=true;
    std::vector<Transform> poses;
    // Optional supplied local surface points, at most 256. With points, the
    // prefix weights use explicit feature-order fingertip positions. Without
    // points they use upstream's wrist-centre proximity branch.
    std::vector<Vec3> surface_points;
};
struct Dex2HoiOrigins {
    Vec3 root{};
    std::array<Vec3,dex2hoi_slots> objects{};
};
struct Dex2HoiHistory {
    // Frame-major [frames, 429]. Runtime tensor layout conversion is separate.
    std::vector<float> features;
    Dex2HoiOrigins origins;
    // Initial root pose delta, the source's tfms_root_global convention.
    Transform initial_root{};
};
// Rows of the rotation matrix, unlike motion::rotation6's column convention.
std::array<float,6> dex2hoi_rotation6(Quat);
Quat dex2hoi_rotation6(std::span<const float,6>);
// TransformEmbedding input: spatial63(translation), rotation6 rows, Euler XYZ.
std::array<float,72> dex2hoi_initial_features(Transform);
std::array<float,18> dex2hoi_encode_object18(Transform object,Transform left_wrist,
    Transform right_wrist,std::span<const Vec3> left_tips={},
    std::span<const Vec3> right_tips={},std::span<const Vec3> surface_world={});
Transform dex2hoi_decode_object18(std::span<const float,18>,Transform left_wrist,
    Transform right_wrist,bool free_translation_only=false);
Dex2HoiHistory encode_dex2hoi_history(const Dex2HoiRig&,const Clip&,
    const std::array<Dex2HoiObject,dex2hoi_slots>&,
    const Dex2HoiFormat& ={},const Dex2HoiNormalizer& ={});
// Rest bases and all unmapped joints come from rig. prefix_frames selects the
// upstream GT-prefix free-translation override; subsequent frames use all three
// learned mixture branches. This is representation decoding, not generation.
EquipmentResult decode_dex2hoi_motion(const Dex2HoiRig&,std::span<const float> features,
    const Dex2HoiOrigins&,const std::array<Dex2HoiObject,dex2hoi_slots>&,
    size_t prefix_frames=0,const Dex2HoiFormat& ={},const Dex2HoiNormalizer& ={},
    float fps=30);
}
