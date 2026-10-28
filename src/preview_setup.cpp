#include "motion.hpp"
#include "equipment.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numbers>
#include <stdexcept>

namespace motion {
namespace {
using nlohmann::json;
void keys(const json& j,std::initializer_list<const char*> allowed) {
    if(!j.is_object())throw std::runtime_error("Preview setup entries must be objects");
    for(const auto& [key,value]:j.items()) {
        (void)value;
        if(std::none_of(allowed.begin(),allowed.end(),[&](const char* a){return key==a;}))
            throw std::runtime_error("Unknown preview setup field: "+key);
    }
}
Vec3 vector(const json& j) {
    if(!j.is_array()||j.size()!=3)throw std::runtime_error("Preview vectors need three numbers");
    return {j[0].get<float>(),j[1].get<float>(),j[2].get<float>()};
}
json array(Vec3 v) { return json::array({v.x,v.y,v.z}); }
void finite(Vec3 v,float limit) {
    for(float x:{v.x,v.y,v.z})if(!std::isfinite(x)||std::abs(x)>limit)throw std::runtime_error("Preview vector is nonfinite or out of range");
}
size_t bone(const Clip& c,const std::string& name) {
    size_t found=c.joints.size();
    for(size_t j=0;j<c.joints.size();++j)if(c.joints[j].name==name) {
        if(found!=c.joints.size())throw std::runtime_error("Ambiguous preview bone: "+name);
        found=j;
    }
    if(found==c.joints.size())throw std::runtime_error("Preview bone not found: "+name);
    return found;
}
}
Transform preview_offset(Vec3 p,Vec3 r) {
    // Tide quaternion.Euler applies Z, then X, then Y (Unity's convention).
    constexpr float half=std::numbers::pi_v<float>/360;
    Quat x{std::sin(r.x*half),0,0,std::cos(r.x*half)};
    Quat y{0,std::sin(r.y*half),0,std::cos(r.y*half)};
    Quat z{0,0,std::sin(r.z*half),std::cos(r.z*half)};
    return {p,normalized(y*x*z)};
}
void validate_preview_setup(const PreviewOptions& o) {
    if(o.props.size()>32)throw std::runtime_error("Preview supports at most 32 props");
    for(const auto& p:o.props) {
        if(p.mesh.empty()||p.bone.empty())throw std::runtime_error("Every preview prop needs mesh and bone");
        finite(p.offset,100);finite(p.rotation,3600);
    }
    const auto& c=o.camera;finite(c.offset,100);finite(c.rotation,3600);
    if(c.bone.empty()||(c.mode!="bone"&&c.mode!="root"))throw std::runtime_error("Camera needs a bone and mode bone or root");
    if(!std::isfinite(c.fov)||c.fov<10||c.fov>150||!std::isfinite(c.near_plane)||c.near_plane<0.001f||c.near_plane>1)
        throw std::runtime_error("Camera fov must be 10..150 degrees and near must be 0.001..1 metres");
    if(c.enabled&&o.body==PreviewBody::skeleton)throw std::runtime_error("First-person preview needs --mannequin or --character");
}
PreviewOptions load_preview_setup(const fs::path& path) {
    if(fs::file_size(path)>1024*1024)throw std::runtime_error("Preview setup exceeds 1 MiB");
    std::ifstream in(path);json j;in>>j;keys(j,{"version","props","camera"});
    if(j.at("version")!=1)throw std::runtime_error("Preview setup version must be 1");
    PreviewOptions o;
    if(j.contains("props")) {
        if(!j["props"].is_array()||j["props"].size()>32)throw std::runtime_error("Preview props must be an array of at most 32 attachments");
        for(const auto& v:j["props"]) {
            keys(v,{"mesh","bone","offset","rotation"});PreviewProp p;
            p.mesh=fs::absolute(path.parent_path()/utf8path(v.at("mesh").get<std::string>())).lexically_normal();
            p.bone=v.at("bone").get<std::string>();
            if(v.contains("offset"))p.offset=vector(v["offset"]);
            if(v.contains("rotation"))p.rotation=vector(v["rotation"]);
            o.props.push_back(p);
        }
    }
    if(j.contains("camera")) {
        const auto& v=j["camera"];keys(v,{"enabled","bone","mode","offset","rotation","fov","near"});
        auto& c=o.camera;c.enabled=v.value("enabled",true);c.bone=v.value("bone",c.bone);c.mode=v.value("mode",c.mode);
        c.fov=v.value("fov",c.fov);c.near_plane=v.value("near",c.near_plane);
        if(v.contains("offset"))c.offset=vector(v["offset"]);
        if(v.contains("rotation"))c.rotation=vector(v["rotation"]);
    }
    validate_preview_setup(o);return o;
}
void save_preview_setup(const PreviewOptions& o,const fs::path& path) {
    validate_preview_setup(o);
    if(fs::exists(path))throw std::runtime_error("Preview setup already exists: "+path.string());
    const auto& c=o.camera;
    json j={{"version",1},{"props",json::array()},{"camera",{{"enabled",c.enabled},{"bone",c.bone},{"mode",c.mode},
        {"offset",array(c.offset)},{"rotation",array(c.rotation)},{"fov",c.fov},{"near",c.near_plane}}}};
    for(const auto& p:o.props)j["props"].push_back({{"mesh",fs::absolute(p.mesh).generic_string()},{"bone",p.bone},{"offset",array(p.offset)},{"rotation",array(p.rotation)}});
    std::ofstream out(path);out<<j.dump(2)<<'\n';if(!out)throw std::runtime_error("Cannot write preview setup: "+path.string());
}
Clip preview_rig(const Clip& c,const PreviewOptions& o) {
    if(o.body!=PreviewBody::character)return c;
    std::vector<std::string> extra;
    for(const auto& p:o.props)extra.push_back(p.bone);
    if(o.camera.enabled)extra.push_back(o.camera.bone);
    std::sort(extra.begin(),extra.end());extra.erase(std::unique(extra.begin(),extra.end()),extra.end());
    auto rig=load_rig_glb(o.character,extra);rig.name=c.name;rig.fps=c.fps;
    rig.motion_root=bone(rig,c.joints[c.motion_root].name);
    std::vector<size_t> mapping;
    for(const auto& joint:rig.joints) {
        size_t index=c.joints.size();
        for(size_t j=0;j<c.joints.size();++j)if(c.joints[j].name==joint.name) {
            if(index!=c.joints.size())throw std::runtime_error("Ambiguous animated bone: "+joint.name);
            index=j;
        }
        mapping.push_back(index);
    }
    rig.poses.clear();rig.poses.reserve(c.frames()*rig.joints.size());
    for(size_t f=0;f<c.frames();++f)for(size_t j=0;j<rig.joints.size();++j)
        rig.poses.push_back(mapping[j]<c.joints.size()?c.frame(f)[mapping[j]]:rig.joints[j].rest);
    rig.validate();return rig;
}
PreviewExtent preview_extent(const Clip& c,const PreviewOptions& o) {
    c.validate();validate_preview_setup(o);
    auto rig=preview_rig(c,o);PreviewExtent e;bool first=true;std::vector<float> lows;
    std::vector<std::pair<size_t,float>> spheres;
    std::vector<std::pair<size_t,float>> equipment_spheres;
    for(const auto& p:o.props) {
        auto b=mesh_bounds(p.mesh);float radius=0;
        for(float x:{b.low.x,b.high.x})for(float y:{b.low.y,b.high.y})for(float z:{b.low.z,b.high.z})
            radius=std::max(radius,std::sqrt(x*x+y*y+z*z));
        // A sphere around the socket covers any Euler orientation of mesh + offset.
        radius+=std::sqrt(dot(p.offset,p.offset));spheres.emplace_back(bone(rig,p.bone),radius);
    }
    if(o.equipment)for(size_t i=0;i<o.equipment->objects.size();++i) {
        const auto& object=o.equipment->objects[i];if(object.mesh.empty()&&!object.builtin_rod)continue;
        float radius=0.866026f;
        if(!object.builtin_rod) {
            auto b=mesh_bounds(object.mesh);radius=0;
            for(float x:{b.low.x,b.high.x})for(float y:{b.low.y,b.high.y})for(float z:{b.low.z,b.high.z})radius=std::max(radius,std::sqrt(x*x+y*y+z*z));
        }
        equipment_spheres.emplace_back(i,radius);
    }
    for(size_t f=0;f<rig.frames();++f) {
        auto pose=world_pose(rig,rig.frame(f));const auto centre=pose[rig.motion_root].p;
        float low=pose.front().p.y;
        for(const auto& p:pose) {
            if(first){e.low=e.high=p.p.y;first=false;}
            low=std::min(low,p.p.y);e.low=std::min(e.low,p.p.y);e.high=std::max(e.high,p.p.y);
            e.reach=std::max(e.reach,std::hypot(p.p.x-centre.x,p.p.z-centre.z));
        }
        lows.push_back(low);
        for(auto [index,bound]:equipment_spheres) {
            const auto& object=o.equipment->objects[index];auto pos=object.poses.at(f).p,scale=object.scales.at(f);
            float radius=bound*std::max({scale.x,scale.y,scale.z});
            e.low=std::min(e.low,pos.y-radius);e.high=std::max(e.high,pos.y+radius);
            e.reach=std::max(e.reach,std::hypot(pos.x-centre.x,pos.z-centre.z)+radius);
        }
        for(auto [j,r]:spheres) {
            const auto p=pose[j].p;e.low=std::min(e.low,p.y-r);e.high=std::max(e.high,p.y+r);
            e.reach=std::max(e.reach,std::hypot(p.x-centre.x,p.z-centre.z)+r);
        }
    }
    std::nth_element(lows.begin(),lows.begin()+static_cast<std::ptrdiff_t>(lows.size()/2),lows.end());
    e.ground=lows[lows.size()/2];return e;
}
}
