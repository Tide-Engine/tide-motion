#include "dex2hoi_motion.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <set>
#include <stdexcept>

// Independently implemented from the published representation equations. No
// Dex2HOI code, checkpoint, rest mesh or dataset motion is embedded here.
namespace motion {
namespace {
constexpr size_t root_offset=dex2hoi_bones*6;
constexpr size_t object_offset=root_offset+3;
constexpr std::array<size_t,5> left_tips{14,18,22,26,30};
constexpr std::array<size_t,5> right_tips{38,42,46,50,54};
bool finite(Vec3 p) {return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z);}
void require_finite(Transform t) {
    if(!finite(t.p))throw std::runtime_error("Nonfinite Dex2HOI transform");
    (void)normalized(t.q);
}
Vec3 read3(const float* p) {return {p[0],p[1],p[2]};}
void write3(float* p,Vec3 v) {p[0]=v.x;p[1]=v.y;p[2]=v.z;}
float length(Vec3 v) {return std::sqrt(dot(v,v));}
float proximity(float distance) {
    const float value=(0.15f-distance)*80;
    if(value>=0)return 1/(1+std::exp(-value));
    const float e=std::exp(value);return e/(1+e);
}
std::array<float,3> mixture(std::span<const float,18> s) {
    const float maximum=std::max({s[15],s[16],s[17]});
    std::array<float,3> weights{};float sum=0;
    for(size_t j=0;j<3;++j){weights[j]=std::exp(s[15+j]-maximum);sum+=weights[j];}
    for(float& value:weights)value/=sum;
    return weights;
}
Clip rest_clip(const Dex2HoiRig& rig) {
    Clip c;c.joints=rig.joints;c.motion_root=rig.bones[0];
    for(const auto& j:c.joints)c.poses.push_back(j.rest);
    return c;
}
void check_objects(const std::array<Dex2HoiObject,2>& objects,size_t frames,bool history) {
    std::set<std::string> exported;
    for(const auto& o:objects) {
        if(o.surface_points.size()>256)throw std::runtime_error("Dex2HOI supplied contact surface exceeds 256 points");
        for(Vec3 p:o.surface_points)if(!finite(p))throw std::runtime_error("Nonfinite Dex2HOI surface point");
        if(!o.present)continue;
        if(o.name.empty())throw std::runtime_error("Dex2HOI object name is empty");
        if(o.export_track&&!exported.insert(o.name).second)throw std::runtime_error("Duplicate exported Dex2HOI object name");
        if(history) {
            if(o.poses.size()!=frames)throw std::runtime_error("Dex2HOI object history has the wrong frame count");
            for(Transform t:o.poses)require_finite(t);
        }
    }
}
Transform wrist(const std::vector<Transform>& pose,const std::vector<Transform>& rest,size_t j) {
    return {pose[j].p,normalized(pose[j].q*inverse(rest[j].q))};
}
void check_clip_rig(const Dex2HoiRig& rig,const Clip& c) {
    c.validate();
    if(c.joints.size()!=rig.joints.size()||c.motion_root!=rig.bones[0])throw std::runtime_error("Dex2HOI history rig or motion root differs");
    std::set<size_t> mapped(rig.bones.begin(),rig.bones.end());
    for(size_t j=0;j<c.joints.size();++j) {
        const auto& a=c.joints[j];const auto& b=rig.joints[j];
        if(a.name!=b.name||a.parent!=b.parent||length(a.rest.p-b.rest.p)>1e-6f||
            std::abs(dot(rotate(a.rest.q,{1,0,0}),rotate(b.rest.q,{1,0,0}))-1)>1e-5f||
            std::abs(dot(rotate(a.rest.q,{0,1,0}),rotate(b.rest.q,{0,1,0}))-1)>1e-5f||std::abs(a.scale-1)>1e-6f)
            throw std::runtime_error("Dex2HOI history rest rig differs");
        for(size_t f=0;f<c.frames();++f) {
            const auto delta=inverse(b.rest)*c.frame(f)[j];
            if(j!=rig.bones[0]&&length(delta.p)>1e-5f)throw std::runtime_error("Dex2HOI cannot encode animated nonroot translations");
            if(!mapped.contains(j)&&std::abs(normalized(delta.q).w)<1-1e-6f)throw std::runtime_error("Dex2HOI unmapped helper joint must remain at rest");
        }
    }
}
}
void Dex2HoiRig::validate() const {
    auto c=rest_clip(*this);c.validate();
    std::set<size_t> mapped;
    for(size_t j:bones)if(j>=joints.size()||!mapped.insert(j).second)throw std::runtime_error("Dex2HOI requires 65 distinct explicit bone indices");
    if(joints[bones[0]].parent!=-1)throw std::runtime_error("Dex2HOI hips must be a skeleton root");
    std::set<std::string> names;
    for(const auto& j:joints) {
        if(j.name.empty()||!names.insert(j.name).second)throw std::runtime_error("Dex2HOI rig requires unique nonempty names");
        if(std::abs(j.scale-1)>1e-6f)throw std::runtime_error("Bake Dex2HOI rig scales to metres before use");
    }
    for(size_t j:bones) {
        int ancestor=static_cast<int>(j);
        while(ancestor>=0&&static_cast<size_t>(ancestor)!=bones[0])ancestor=joints[static_cast<size_t>(ancestor)].parent;
        if(ancestor<0)throw std::runtime_error("Dex2HOI mapped bones must descend from hips");
    }
}
void Dex2HoiNormalizer::validate() const {
    if(!std::isfinite(epsilon)||epsilon<=0)throw std::runtime_error("Dex2HOI normalization epsilon must be finite and positive");
    for(size_t j=0;j<dex2hoi_features;++j)if(!std::isfinite(mean[j])||!std::isfinite(stddev[j]))throw std::runtime_error("Nonfinite Dex2HOI normalization statistic");
}
void Dex2HoiNormalizer::normalize(std::span<float> values) const {
    validate();if(values.size()%dex2hoi_features)throw std::runtime_error("Dex2HOI feature shape must be [frames,429]");
    for(size_t j=0;j<values.size();++j) {
        const size_t k=j%dex2hoi_features;const float s=stddev[k]<epsilon?1:stddev[k];
        values[j]=(values[j]-mean[k])/s;if(!std::isfinite(values[j]))throw std::runtime_error("Nonfinite Dex2HOI normalized feature");
    }
}
void Dex2HoiNormalizer::denormalize(std::span<float> values) const {
    validate();if(values.size()%dex2hoi_features)throw std::runtime_error("Dex2HOI feature shape must be [frames,429]");
    for(size_t j=0;j<values.size();++j) {
        const size_t k=j%dex2hoi_features;const float s=stddev[k]<epsilon?1:stddev[k];
        values[j]=values[j]*s+mean[k];if(!std::isfinite(values[j]))throw std::runtime_error("Nonfinite Dex2HOI denormalized feature");
    }
}
std::array<float,6> dex2hoi_rotation6(Quat q) {
    q=normalized(q);const auto x=rotate(q,{1,0,0}),y=rotate(q,{0,1,0}),z=rotate(q,{0,0,1});
    return {x.x,y.x,z.x,x.y,y.y,z.y};
}
Quat dex2hoi_rotation6(std::span<const float,6> values) {
    for(float f:values)if(!std::isfinite(f))throw std::runtime_error("Nonfinite Dex2HOI rotation feature");
    // rotation6 constructs a matrix from columns. The model stores its first two
    // rows, so invert that rotation to obtain the published row-basis matrix.
    return inverse(rotation6({values[0],values[1],values[2]},{values[3],values[4],values[5]}));
}
std::array<float,72> dex2hoi_initial_features(Transform transform) {
    require_finite(transform);transform.q=normalized(transform.q);
    std::array<float,72> out{};write3(out.data(),transform.p);size_t at=3;
    for(unsigned band=0;band<10;++band) {
        const auto a=transform.p*static_cast<float>(1U<<band)*std::numbers::pi_v<float>;
        out[at++]=std::sin(a.x);out[at++]=std::sin(a.y);out[at++]=std::sin(a.z);
        out[at++]=std::cos(a.x);out[at++]=std::cos(a.y);out[at++]=std::cos(a.z);
    }
    const auto r=dex2hoi_rotation6(transform.q);std::copy(r.begin(),r.end(),out.begin()+63);
    const auto x=rotate(transform.q,{1,0,0}),y=rotate(transform.q,{0,1,0}),z=rotate(transform.q,{0,0,1});
    out[69]=std::atan2(-z.y,z.z);out[70]=std::asin(std::clamp(z.x,-1.0f,1.0f));out[71]=std::atan2(-y.x,x.x);
    for(float value:out)if(!std::isfinite(value))throw std::runtime_error("Nonfinite Dex2HOI initial-transform feature");
    return out;
}
std::array<float,18> dex2hoi_encode_object18(Transform object,Transform left,Transform right,
    std::span<const Vec3> left_points,std::span<const Vec3> right_points,std::span<const Vec3> surface) {
    require_finite(object);require_finite(left);require_finite(right);
    object.q=normalized(object.q);left.q=normalized(left.q);right.q=normalized(right.q);
    const bool contact=!surface.empty();
    if(contact&&(left_points.empty()||right_points.empty()))throw std::runtime_error("Dex2HOI surface contact requires both fingertip sets");
    auto distance_to_surface=[&](std::span<const Vec3> tips) {
        float best=std::numeric_limits<float>::infinity();
        for(Vec3 tip:tips) {if(!finite(tip))throw std::runtime_error("Nonfinite Dex2HOI fingertip");
            for(Vec3 p:surface){if(!finite(p))throw std::runtime_error("Nonfinite Dex2HOI surface");best=std::min(best,length(tip-p));}}
        return best;
    };
    const float a=proximity(contact?distance_to_surface(left_points):length(object.p-left.p));
    const float b=proximity(contact?distance_to_surface(right_points):length(object.p-right.p));
    const float free=(1-a)*(1-b),sum=std::max(a+b+free,1e-8f);
    std::array<float,18> out{};const auto r=dex2hoi_rotation6(object.q);std::copy(r.begin(),r.end(),out.begin());
    write3(out.data()+6,rotate(inverse(left.q),object.p-left.p));
    write3(out.data()+9,rotate(inverse(right.q),object.p-right.p));
    write3(out.data()+12,object.p);
    out[15]=std::log(std::max(a/sum,1e-8f));out[16]=std::log(std::max(b/sum,1e-8f));out[17]=std::log(std::max(free/sum,1e-8f));
    return out;
}
Transform dex2hoi_decode_object18(std::span<const float,18> values,Transform left,Transform right,bool free_only) {
    require_finite(left);require_finite(right);
    left.q=normalized(left.q);right.q=normalized(right.q);
    for(float f:values)if(!std::isfinite(f))throw std::runtime_error("Nonfinite Dex2HOI object feature");
    Transform out;out.q=dex2hoi_rotation6(values.first<6>());out.p=read3(values.data()+12);
    if(!free_only) {
        const auto weights=mixture(values);
        out.p=(left.p+rotate(left.q,read3(values.data()+6)))*weights[0]+
            (right.p+rotate(right.q,read3(values.data()+9)))*weights[1]+out.p*weights[2];
    }
    require_finite(out);return out;
}
Dex2HoiHistory encode_dex2hoi_history(const Dex2HoiRig& rig,const Clip& body,
    const std::array<Dex2HoiObject,2>& objects,const Dex2HoiFormat& format,const Dex2HoiNormalizer& norm) {
    rig.validate();check_clip_rig(rig,body);check_objects(objects,body.frames(),true);
    if(format.root_displacement&&!format.x0_as_condition)throw std::runtime_error("Dex2HOI dataset root displacement requires external x0 conditioning");
    const auto rest=rest_clip(rig);const auto rest_world=world_pose(rest,rest.frame(0));
    Dex2HoiHistory out;out.features.resize(body.frames()*dex2hoi_features);
    out.initial_root=inverse(rig.joints[rig.bones[0]].rest)*body.frame(0)[rig.bones[0]];out.origins.root=out.initial_root.p;
    for(size_t s=0;s<2;++s)if(objects[s].present) {
        out.origins.objects[s]=objects[s].poses[0].p;
        if(format.object_relative_to_root)out.origins.objects[s]=out.origins.objects[s]-out.origins.root;
    }
    for(size_t f=0;f<body.frames();++f) {
        float* values=out.features.data()+f*dex2hoi_features;const auto pose=body.frame(f);
        const auto world=world_pose(body,pose);
        const auto left=wrist(world,rest_world,rig.bones[dex2hoi_left_wrist]);
        const auto right=wrist(world,rest_world,rig.bones[dex2hoi_right_wrist]);
        for(size_t j=0;j<65;++j) {
            const auto delta=inverse(rig.joints[rig.bones[j]].rest)*pose[rig.bones[j]];
            const auto r=dex2hoi_rotation6(delta.q);std::copy(r.begin(),r.end(),values+j*6);
        }
        const Vec3 root=(inverse(rig.joints[rig.bones[0]].rest)*pose[rig.bones[0]]).p;
        write3(values+root_offset,format.root_displacement?root-out.origins.root:root);
        std::array<Vec3,5> left_fingers{},right_fingers{};
        for(size_t j=0;j<5;++j){left_fingers[j]=world[rig.bones[left_tips[j]]].p;right_fingers[j]=world[rig.bones[right_tips[j]]].p;}
        for(size_t s=0;s<2;++s) {
            if(!objects[s].present)continue;
            const auto object=objects[s].poses[f];std::vector<Vec3> surface;
            for(Vec3 p:objects[s].surface_points)surface.push_back(object.p+rotate(normalized(object.q),p));
            auto slot=dex2hoi_encode_object18(object,left,right,left_fingers,right_fingers,surface);
            Vec3 free=read3(slot.data()+12);
            if(format.object_relative_to_root)free=free-root;
            if(format.object_displacement&&(format.x0_as_condition||f>0))free=free-out.origins.objects[s];
            write3(slot.data()+12,free);std::copy(slot.begin(),slot.end(),values+object_offset+s*18);
        }
    }
    norm.normalize(out.features);return out;
}
EquipmentResult decode_dex2hoi_motion(const Dex2HoiRig& rig,std::span<const float> features,
    const Dex2HoiOrigins& origins,const std::array<Dex2HoiObject,2>& objects,size_t prefix_frames,
    const Dex2HoiFormat& format,const Dex2HoiNormalizer& norm,float fps) {
    rig.validate();
    if(features.empty()||features.size()%dex2hoi_features||features.size()/dex2hoi_features>18000)
        throw std::runtime_error("Dex2HOI feature shape must be [1..18000,429]");
    const size_t frames=features.size()/dex2hoi_features;
    if(prefix_frames>frames)throw std::runtime_error("Dex2HOI prefix exceeds output frame count");
    if(!std::isfinite(fps)||fps<=0)throw std::runtime_error("Invalid Dex2HOI output frame rate");
    if(!finite(origins.root)||!finite(origins.objects[0])||!finite(origins.objects[1]))throw std::runtime_error("Nonfinite Dex2HOI origins");
    check_objects(objects,frames,false);
    std::vector<float> raw(features.begin(),features.end());norm.denormalize(raw);
    EquipmentResult out;out.body=rest_clip(rig);out.body.poses.clear();out.body.fps=fps;out.body.name="dex2hoi";
    out.events=nlohmann::json::array();std::array<int,2> tracks{-1,-1};
    for(size_t s=0;s<2;++s)if(objects[s].present&&objects[s].export_track) {
        tracks[s]=static_cast<int>(out.objects.size());EquipmentTrack t;t.name=objects[s].name;t.mesh=objects[s].mesh;
        out.objects.push_back(std::move(t));
    }
    const auto rest=rest_clip(rig);const auto rest_world=world_pose(rest,rest.frame(0));
    for(size_t f=0;f<frames;++f) {
        float* values=raw.data()+f*dex2hoi_features;const size_t offset=out.body.poses.size();
        for(const auto& joint:rig.joints)out.body.poses.push_back(joint.rest);
        auto pose=std::span<Transform>(out.body.poses.data()+offset,rig.joints.size());
        Vec3 root=read3(values+root_offset);
        if(format.root_displacement)root=root+(format.x0_as_condition?origins.root:read3(raw.data()+root_offset));
        // Legacy frame zero stores an absolute root rather than an offset.
        if(format.root_displacement&&!format.x0_as_condition&&f==0)root=read3(values+root_offset);
        for(size_t j=0;j<65;++j) {
            Transform delta;delta.q=dex2hoi_rotation6(std::span<const float,6>(values+j*6,6));
            if(j==0)delta.p=root;
            pose[rig.bones[j]]=rig.joints[rig.bones[j]].rest*delta;
        }
        const auto world=world_pose(out.body,pose);
        const auto left=wrist(world,rest_world,rig.bones[dex2hoi_left_wrist]);
        const auto right=wrist(world,rest_world,rig.bones[dex2hoi_right_wrist]);
        for(size_t s=0;s<2;++s) {
            if(tracks[s]<0)continue;
            float* slot=values+object_offset+s*18;Vec3 free=read3(slot+12);
            if(format.object_displacement) {
                if(format.x0_as_condition)free=f==0?origins.objects[s]:free+origins.objects[s];
                else if(f>0)free=free+read3(raw.data()+object_offset+s*18+12);
            }
            if(format.object_relative_to_root)free=free+root;
            // Do not mutate raw frame zero: later legacy frames reference it.
            std::array<float,18> adjusted{};std::copy_n(slot,18,adjusted.begin());write3(adjusted.data()+12,free);
            auto& track=out.objects[static_cast<size_t>(tracks[s])];
            track.poses.push_back(dex2hoi_decode_object18(adjusted,left,right,f<prefix_frames));track.scales.push_back({1,1,1});
        }
    }
    out.body.validate();out.report={{"representation","dex2hoi-65x6-root-2x18-v1"},{"frames",frames},{"prefixFrames",prefix_frames},
        {"rootDisplacement",format.root_displacement},{"objectDisplacement",format.object_displacement},
        {"x0AsCondition",format.x0_as_condition},{"objectRelativeToRoot",format.object_relative_to_root}};
    return out;
}
}
