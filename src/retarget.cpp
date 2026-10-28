#include "motion.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace motion {
namespace {
constexpr float pi=3.14159265358979323846f;
constexpr std::array<int,30> primary{1,2,3,4,5,6,-1,-1,-1,-1,11,12,13,15,-1,-1,17,18,19,21,-1,-1,23,24,25,-1,27,28,29,-1};
constexpr std::array<size_t,15> required{0,3,6,11,12,13,17,18,19,22,23,24,26,27,28};
std::vector<Transform> rest_pose(const Clip& rig) {
    std::vector<Transform> out;
    for(const auto& joint:rig.joints)out.push_back(joint.rest);
    return out;
}
int find(const Clip& rig,const std::string& name) {
    int result=-1;
    for(size_t i=0;i<rig.joints.size();++i)if(rig.joints[i].name==name) {
        if(result>=0)throw std::runtime_error("Ambiguous rig bone name: "+name);
        result=static_cast<int>(i);
    }
    if(result<0)throw std::runtime_error("Mapped bone not found in rig: "+name);
    return result;
}
Quat from_to(Vec3 a,Vec3 b) {
    a=normalized(a);b=normalized(b);
    float d=std::clamp(dot(a,b),-1.0f,1.0f);
    if(d< -0.999999f) {
        Vec3 axis=normalized(cross(a,std::abs(a.x)<0.8f?Vec3{1,0,0}:Vec3{0,1,0}));
        return {axis.x,axis.y,axis.z,0};
    }
    auto axis=cross(a,b);
    return normalized({axis.x,axis.y,axis.z,1+d});
}
float length(Vec3 value) { return std::sqrt(dot(value,value)); }
float scalar(const nlohmann::json& value,const char* name,float low,float high) {
    if(!value.is_number())throw std::runtime_error(std::string(name)+" must be numeric");
    double n=value.get<double>();
    if(!std::isfinite(n)||n<low||n>high)throw std::runtime_error(std::string(name)+" is out of range");
    return static_cast<float>(n);
}
}

RetargetMapping load_mapping(const fs::path& path) {
    if(fs::file_size(path)>65536)throw std::runtime_error("Bone mapping exceeds 64 KiB");
    std::ifstream input(path);nlohmann::json doc;input>>doc;
    if(!doc.is_object())throw std::runtime_error("Bone mapping must be an object");
    const std::set<std::string> fields{"bones","referenceRotations","rootScale","headingDegrees","groundOffset"};
    for(const auto& item:doc.items())if(!fields.contains(item.key()))throw std::runtime_error("Unknown bone mapping field: "+item.key());
    auto& bones=doc.at("bones");
    if(!bones.is_object())throw std::runtime_error("bones must map SOMA bone names to character bone names");
    RetargetMapping out;auto canonical=soma();std::set<std::string> assigned;
    for(const auto& item:bones.items()) {
        auto at=std::find_if(canonical.begin(),canonical.end(),[&](const Joint& j){return j.name==item.key();});
        if(at==canonical.end())throw std::runtime_error("Unknown SOMA bone in mapping: "+item.key());
        std::string name=item.value().get<std::string>();
        if(name.empty()||name.size()>256||!assigned.insert(name).second)throw std::runtime_error("Bone names must be nonempty and mapped only once: "+name);
        out.bones[static_cast<size_t>(at-canonical.begin())]=name;
    }
    for(size_t index:required)if(out.bones[index].empty())throw std::runtime_error("Bone mapping is missing required SOMA bone: "+canonical[index].name);
    if(doc.contains("rootScale")) {
        if(doc["rootScale"]!="auto")out.root_scale=scalar(doc["rootScale"],"rootScale",1e-6f,1e6f);
    }
    if(doc.contains("headingDegrees"))out.heading_degrees=scalar(doc["headingDegrees"],"headingDegrees",-360,360);
    if(doc.contains("groundOffset"))out.ground_offset=scalar(doc["groundOffset"],"groundOffset",-1000,1000);
    if(doc.contains("referenceRotations")) {
        const auto& rotations=doc["referenceRotations"];
        if(!rotations.is_object()||rotations.size()>1024)throw std::runtime_error("referenceRotations must contain target bone names and local XYZW quaternions");
        for(const auto& item:rotations.items()) {
            const auto& q=item.value();
            if(!q.is_array()||q.size()!=4)throw std::runtime_error("Reference rotation requires four XYZW values: "+item.key());
            out.reference_rotations[item.key()]=normalized({scalar(q[0],"quaternion",-1e6f,1e6f),scalar(q[1],"quaternion",-1e6f,1e6f),scalar(q[2],"quaternion",-1e6f,1e6f),scalar(q[3],"quaternion",-1e6f,1e6f)});
        }
    }
    return out;
}

namespace {
// The mapping's reference poses: SOMA's rest and the character's rest with mapped
// segments aimed along SOMA's, the heading between them and the bone correspondence.
struct Calibration {
    std::array<int,30> custom_index; std::vector<int> semantic;
    std::vector<Transform> soma_ref,custom_ref,soma_world,custom_world; Quat alignment; float heading=0;
};
Calibration calibrate(const Clip& canonical,const Clip& custom,const RetargetMapping& map) {
    require_soma(canonical);
    std::set<std::string> names;
    for(const auto& joint:custom.joints)if(joint.name.empty()||!names.insert(joint.name).second)
        throw std::runtime_error("Export unique, nonempty names for rig bones and their ancestors before retargeting");
    std::array<int,30> custom_index;custom_index.fill(-1);
    std::vector<int> semantic(custom.joints.size(),-1);
    for(size_t s:required)if(map.bones[s].empty())throw std::runtime_error("Bone mapping is missing required SOMA bone: "+canonical.joints[s].name);
    for(size_t s=0;s<30;++s)if(!map.bones[s].empty()) {
        int index=find(custom,map.bones[s]);
        if(semantic[static_cast<size_t>(index)]>=0)throw std::runtime_error("A character bone cannot represent multiple SOMA bones");
        custom_index[s]=index;semantic[static_cast<size_t>(index)]=static_cast<int>(s);
    }
    for(size_t s=0;s<30;++s)if(custom_index[s]>=0) {
        int source_parent=canonical.joints[s].parent;
        while(source_parent>=0&&custom_index[static_cast<size_t>(source_parent)]<0)source_parent=canonical.joints[static_cast<size_t>(source_parent)].parent;
        int parent=custom.joints[static_cast<size_t>(custom_index[s])].parent;
        while(parent>=0&&semantic[static_cast<size_t>(parent)]<0)parent=custom.joints[static_cast<size_t>(parent)].parent;
        if((parent<0?-1:semantic[static_cast<size_t>(parent)])!=source_parent)
            throw std::runtime_error("Mapped hierarchy differs at "+canonical.joints[s].name+"; map anatomical ancestors in their original order");
    }
    auto soma_ref=rest_pose(canonical),custom_ref=rest_pose(custom);
    auto soma_world=world_pose(canonical,soma_ref);
    std::set<size_t> overridden;
    for(const auto& [name,rotation]:map.reference_rotations) {
        auto index=static_cast<size_t>(find(custom,name));
        custom_ref[index].q=normalized(rotation);overridden.insert(index);
    }
    auto custom_world=world_pose(custom,custom_ref);
    auto at=[&](size_t s)->size_t{return static_cast<size_t>(custom_index[s]);};
    // Hip separation establishes character heading without assuming bone axes.
    Vec3 source_right=soma_world[22].p-soma_world[26].p;
    Vec3 target_right=custom_world[at(22)].p-custom_world[at(26)].p;
    source_right.y=0;target_right.y=0;
    (void)normalized(source_right);(void)normalized(target_right);
    float heading=std::atan2(source_right.z,source_right.x)-std::atan2(target_right.z,target_right.x)+map.heading_degrees*pi/180;
    Quat alignment=yaw(heading);
    // Aim each mapped segment at its corresponding SOMA reference direction.
    // This calibrates A/T poses while preserving the character's bone axes and
    // lengths. Explicit local reference rotations override automatic aiming.
    for(size_t j=0;j<custom.joints.size();++j) {
        int s=semantic[j];
        if(s<0||overridden.contains(j))continue;
        int child=primary[static_cast<size_t>(s)];
        while(child>=0&&custom_index[static_cast<size_t>(child)]<0)child=primary[static_cast<size_t>(child)];
        if(child<0)continue;
        custom_world=world_pose(custom,custom_ref);
        Vec3 current=custom_world[at(static_cast<size_t>(child))].p-custom_world[j].p;
        Vec3 desired=rotate(alignment,soma_world[static_cast<size_t>(child)].p-soma_world[static_cast<size_t>(s)].p);
        Quat world=normalized(from_to(current,desired)*custom_world[j].q);
        int parent=custom.joints[j].parent;
        custom_ref[j].q=parent<0?world:normalized(inverse(custom_world[static_cast<size_t>(parent)].q)*world);
    }
    custom_world=world_pose(custom,custom_ref);
    return {custom_index,semantic,soma_ref,custom_ref,soma_world,custom_world,alignment,heading};
}
}
RetargetProxy retarget_proxy(const Clip& target,const RetargetMapping& map) {
    target.validate();
    Clip canonical;canonical.joints=soma();for(const auto& joint:canonical.joints)canonical.poses.push_back(joint.rest);
    auto c=calibrate(canonical,target,map);
    RetargetProxy proxy;proxy.joints=target.joints;proxy.local=c.custom_ref;proxy.source=c.semantic;
    proxy.turn.resize(target.joints.size());proxy.scales.resize(target.joints.size());
    for(size_t j=0;j<target.joints.size();++j) {
        proxy.scales[j]=target.joints[j].scale*(target.joints[j].parent<0?1.0f:proxy.scales[static_cast<size_t>(target.joints[j].parent)]);
        if(c.semantic[j]>=0)proxy.turn[j]=normalized(inverse(c.soma_world[static_cast<size_t>(c.semantic[j])].q)*inverse(c.alignment)*c.custom_world[j].q);
    }
    return proxy;
}
void proxy_world(const RetargetProxy& proxy,std::span<const Transform> soma_world,std::vector<Transform>& out) {
    out.resize(proxy.joints.size());
    for(size_t j=0;j<proxy.joints.size();++j) {
        const int parent=proxy.joints[j].parent;
        Transform t=proxy.local[j];
        if(parent>=0) {const auto& p=out[static_cast<size_t>(parent)];t.p=p.p+rotate(p.q,t.p*proxy.scales[static_cast<size_t>(parent)]);t.q=normalized(p.q*t.q);}
        if(proxy.source[j]>=0)t.q=normalized(soma_world[static_cast<size_t>(proxy.source[j])].q*proxy.turn[j]);
        out[j]=t;
    }
}
RetargetResult retarget(const Clip& source,const Clip& target,const RetargetMapping& map,bool to_soma) {
    source.validate();target.validate();
    const auto& custom=to_soma?source:target;
    const auto& canonical=to_soma?target:source;
    auto calibration=calibrate(canonical,custom,map);
    auto& custom_index=calibration.custom_index;auto& semantic=calibration.semantic;
    auto& soma_ref=calibration.soma_ref;auto& custom_ref=calibration.custom_ref;
    auto& soma_world=calibration.soma_world;auto& custom_world=calibration.custom_world;
    const Quat alignment=calibration.alignment;const float heading=calibration.heading;
    auto at=[&](size_t s)->size_t{return static_cast<size_t>(custom_index[s]);};
    float source_legs=0,target_legs=0;
    for(size_t thigh:{size_t{22},size_t{26}}) {
        source_legs+=length(soma_world[thigh+1].p-soma_world[thigh].p)+length(soma_world[thigh+2].p-soma_world[thigh+1].p);
        target_legs+=length(custom_world[at(thigh+1)].p-custom_world[at(thigh)].p)+length(custom_world[at(thigh+2)].p-custom_world[at(thigh+1)].p);
    }
    float scale=map.root_scale>0?map.root_scale:target_legs/source_legs;
    if(!std::isfinite(scale)||scale<1e-6f||scale>1e6f)throw std::runtime_error("Cannot estimate root scale from degenerate legs; fix the rig or set rootScale");
    if(target_legs<1e-6f)throw std::runtime_error("Mapped legs have zero length");
    float soma_floor=std::numeric_limits<float>::infinity(),custom_floor=soma_floor;
    for(size_t s:{size_t{24},size_t{25},size_t{28},size_t{29}})if(custom_index[s]>=0) {
        soma_floor=std::min(soma_floor,soma_world[s].p.y);
        custom_floor=std::min(custom_floor,custom_world[at(s)].p.y);
    }
    auto output_ref=to_soma?soma_ref:custom_ref;
    const auto& source_ref_world=to_soma?custom_world:soma_world;
    const auto& target_ref_world=to_soma?soma_world:custom_world;
    size_t source_root=to_soma?at(0):0,target_root=to_soma?0:at(0);
    Quat basis=to_soma?inverse(alignment):alignment;
    float ratio=to_soma?1/scale:scale;
    float source_floor=to_soma?custom_floor:soma_floor;
    float target_floor=to_soma?soma_floor:custom_floor;
    // groundOffset is a character-space authoring correction and is inverted
    // with the other root placement terms on the trip back to SOMA.
    float output_offset=to_soma?-map.ground_offset/scale:map.ground_offset;
    RetargetResult result;result.scale=ratio;result.heading_degrees=(to_soma?-heading:heading)*180/pi;
    auto& out=result.clip;out.joints=target.joints;out.fps=source.fps;out.name=source.name;out.motion_root=target_root;
    out.poses.reserve(source.frames()*out.joints.size());
    std::vector<float> target_scales(out.joints.size());
    for(size_t j=0;j<out.joints.size();++j) {
        target_scales[j]=out.joints[j].scale;
        if(out.joints[j].parent>=0)target_scales[j]*=target_scales[static_cast<size_t>(out.joints[j].parent)];
    }
    std::vector<bool> root_ancestor(source.joints.size());
    for(int j=static_cast<int>(source_root);j>=0;j=source.joints[static_cast<size_t>(j)].parent)root_ancestor[static_cast<size_t>(j)]=true;
    for(size_t f=0;f<source.frames();++f) {
        for(size_t j=0;j<source.joints.size();++j)if(!root_ancestor[j]) {
            auto difference=source.frame(f)[j].p-source.joints[j].rest.p;
            if(dot(difference,difference)>1e-8f)result.discarded_translations=true;
        }
        auto source_world=world_pose(source,source.frame(f));
        std::vector<Transform> output_world(out.joints.size());
        for(size_t j=0;j<out.joints.size();++j) {
            auto local=output_ref[j];
            int source_joint=to_soma?custom_index[j]:semantic[j];
            int parent=out.joints[j].parent;
            if(source_joint>=0) {
                auto index=static_cast<size_t>(source_joint);
                Quat delta=normalized(source_world[index].q*inverse(source_ref_world[index].q));
                Quat world=normalized(basis*delta*inverse(basis)*target_ref_world[j].q);
                local.q=parent<0?world:normalized(inverse(output_world[static_cast<size_t>(parent)].q)*world);
            }
            if(j==target_root) {
                Vec3 displacement=source_world[source_root].p-source_ref_world[source_root].p+Vec3{0,source_floor,0};
                Vec3 world=target_ref_world[target_root].p+Vec3{0,-target_floor+output_offset,0}+rotate(basis,displacement)*ratio;
                if(parent<0)local.p=world;
                else {
                    auto p=static_cast<size_t>(parent);
                    local.p=rotate(inverse(output_world[p].q),world-output_world[p].p)*(1/target_scales[p]);
                }
            }
            Transform global=local;
            if(parent>=0) {
                auto p=static_cast<size_t>(parent);global.p=global.p*target_scales[p];global=output_world[p]*global;
            }
            output_world[j]=global;out.poses.push_back(local);
        }
    }
    out.validate();if(to_soma)require_soma(out);
    return result;
}
}
