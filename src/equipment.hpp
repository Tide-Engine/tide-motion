#pragma once
#include "motion.hpp"
#include <nlohmann/json.hpp>

namespace motion {
struct EquipmentKey {
    size_t frame=0;
    Transform local{};
    Vec3 mesh_scale{1,1,1};
};
struct EquipmentSegment {
    size_t start=0,end=0;
    std::string space="world",anchor,interpolation="linear";
    std::vector<EquipmentKey> keys;
};
struct EquipmentObject {
    std::string name;
    fs::path mesh;
    std::vector<EquipmentSegment> segments;
};
struct EquipmentGrip {
    std::string name,object,bone;
    std::vector<std::string> chain;
    size_t start=0,end=0;
    Transform target{},hand_offset{};
    bool orientation=true;
    float position_tolerance=0.005f,angle_tolerance=5;
    float max_correction=0.25f,max_rotation=90,max_step_increase=30;
};
struct EquipmentString {
    std::string name,object,top_anchor,bottom_anchor,nock_anchor;
    Vec3 top{},bottom{};
    std::vector<EquipmentKey> nock;
    float thickness=0.004f;
};
struct EquipmentPlan {
    std::vector<EquipmentObject> objects;
    std::vector<EquipmentGrip> grips;
    std::vector<EquipmentString> strings;
    std::vector<size_t> protected_frames;
    nlohmann::json events=nlohmann::json::array(),recipe;
};
// Every object is a world-space root track; scale applies only to its mesh.
// String pieces use a generated centred unit box with length along +Y.
struct EquipmentTrack {
    std::string name;
    fs::path mesh;
    bool builtin_rod=false;
    std::vector<Transform> poses;
    std::vector<Vec3> scales;
};
struct EquipmentResult {
    Clip body;
    std::vector<EquipmentTrack> objects;
    nlohmann::json report,events;
};
EquipmentPlan load_equipment_plan(const fs::path&);
EquipmentResult author_equipment(const Clip&, const EquipmentPlan&, bool correct_grips=false);
void save_equipment_glb(const EquipmentResult&, const fs::path&);
// Standalone parser, also used by the main CLI; no model cache/session required.
int equipment_command(const std::vector<std::string>&);
const char* equipment_help();
}
