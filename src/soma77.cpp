#include "motion.hpp"
#include "soma77.hpp"
#include <map>
#include <optional>
#include <stdexcept>

// The SOMA-77 presentation skeleton follows NVIDIA Kimodo (Apache-2.0) at
// e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954: SOMASkeleton30.to_SOMASkeleton77 keeps the
// 30 predicted local rotations and gives every other joint its relaxed-hands rest
// rotation; from_SOMASkeleton77 takes the 30 back unchanged (kimodo/skeleton/definitions.py).
namespace motion {
namespace {
// Index of each SOMA-30 joint in the 77-joint order.
std::array<size_t,30> subset() {
    static const std::array<size_t,30> indices=[] {
        std::array<size_t,30> out{};auto rig=soma();
        for(size_t i=0;i<30;++i) {
            size_t found=77;
            for(size_t j=0;j<77;++j)if(soma77_data::names[j]==rig[i].name)found=j;
            if(found==77)throw std::logic_error("SOMA-77 lacks a SOMA-30 joint");
            out[i]=found;
        }
        return out;
    }();
    return indices;
}
// Left and right counterpart in the 77-joint order (a centre joint is its own).
size_t counterpart(size_t j) {
    std::string name(soma77_data::names[j]),other=name;
    if(name.starts_with("Left"))other="Right"+name.substr(4);
    else if(name.starts_with("Right"))other="Left"+name.substr(5);
    for(size_t k=0;k<77;++k)if(soma77_data::names[k]==other)return k;
    throw std::logic_error("SOMA-77 joint without a counterpart: "+name);
}
}
std::vector<Joint> soma77() {
    std::vector<Joint> out;
    for(size_t j=0;j<77;++j) {
        auto o=soma77_data::offsets[j];
        out.push_back({std::string(soma77_data::names[j]),soma77_data::parents[j],{{o[0],o[1],o[2]},{}}});
    }
    return out;
}
Clip as_soma77(const Clip& c) {
    c.validate();
    if(c.joints.size()!=77)throw std::runtime_error("Not a SOMA-77 clip: it has "+std::to_string(c.joints.size())+" joints");
    auto rig=soma77();std::map<std::string,size_t> at;
    for(size_t j=0;j<c.joints.size();++j)if(!at.emplace(c.joints[j].name,j).second)throw std::runtime_error("Duplicate joint name: "+c.joints[j].name);
    std::array<size_t,77> source{};
    for(size_t j=0;j<77;++j) {
        auto found=at.find(rig[j].name);
        if(found==at.end())throw std::runtime_error("Not a SOMA-77 clip: missing "+rig[j].name);
        source[j]=found->second;
    }
    for(size_t j=0;j<77;++j) {
        const auto& joint=c.joints[source[j]];
        int parent=rig[j].parent;
        if((parent<0)!=(joint.parent<0)||(parent>=0&&static_cast<size_t>(joint.parent)!=source[static_cast<size_t>(parent)]))
            throw std::runtime_error("SOMA-77 hierarchy differs at "+rig[j].name);
        if(std::abs(joint.scale-1)>1e-5f)throw std::runtime_error("SOMA-77 joint scale must be one");
        Vec3 d=joint.rest.p-rig[j].rest.p;
        if(j&&dot(d,d)>1e-6f)throw std::runtime_error("SOMA-77 clip has different body proportions; retarget to canonical SOMA first");
    }
    Clip out;out.name=c.name;out.fps=c.fps;out.joints=rig;
    for(size_t f=0;f<c.frames();++f) {
        auto frame=c.frame(f);
        for(size_t j=0;j<77;++j) {
            Transform t=rig[j].rest;t.q=normalized(frame[source[j]].q);
            if(!j)t.p=frame[source[j]].p;
            else {Vec3 d=frame[source[j]].p-c.joints[source[j]].rest.p;if(dot(d,d)>1e-8f)throw std::runtime_error("Animated bone lengths are unsupported in SOMA-77 clips");}
            out.poses.push_back(t);
        }
    }
    return out;
}
bool is_soma77(const Clip& c) {
    if(c.joints.size()!=77)return false;
    try {(void)as_soma77(c);return true;} catch(const std::exception&) {return false;}
}
Clip soma77_to_30(const Clip& c) {
    auto full=as_soma77(c);auto index=subset();
    Clip out;out.name=full.name;out.fps=full.fps;out.joints=soma();
    for(size_t f=0;f<full.frames();++f) {
        auto frame=full.frame(f);
        for(size_t i=0;i<30;++i) {Transform t=out.joints[i].rest;t.q=frame[index[i]].q;if(!i)t.p=frame[0].p;out.poses.push_back(t);}
    }
    require_soma(out);
    return out;
}
Clip soma30_to_77(const Clip& c,const Clip* others) {
    require_soma(c);
    std::optional<Clip> keep;
    if(others) {keep=as_soma77(*others);if(keep->frames()!=c.frames())throw std::runtime_error("SOMA-77 finger source has a different frame count");}
    auto index=subset();std::array<int,77> from;from.fill(-1);
    for(size_t i=0;i<30;++i)from[index[i]]=static_cast<int>(i);
    Clip out;out.name=c.name;out.fps=c.fps;out.joints=soma77();
    for(size_t f=0;f<c.frames();++f) {
        auto frame=c.frame(f);
        for(size_t j=0;j<77;++j) {
            Transform t=out.joints[j].rest;
            if(from[j]>=0)t.q=frame[static_cast<size_t>(from[j])].q;
            else if(keep)t.q=keep->frame(f)[j].q;
            else {auto q=soma77_data::relaxed[j];t.q=normalized({q[0],q[1],q[2],q[3]});}
            if(!j)t.p=frame[0].p;
            out.poses.push_back(t);
        }
    }
    out.validate();
    return out;
}
Clip mirror_soma77(const Clip& c) {
    auto full=as_soma77(c);auto body=soma77_to_30(full);
    mirror_soma(body);
    // The joints outside SOMA-30 take their counterpart's rotation reflected across x
    // (M L M, M the x reflection), as mirror_soma does for the body; rest rotations are
    // identity and the rest offsets are mirror images within 2.5 mm.
    Clip fingers=full;
    for(size_t f=0;f<full.frames();++f)for(size_t j=0;j<77;++j) {
        Quat q=full.frame(f)[counterpart(j)].q;fingers.frame(f)[j].q={q.x,-q.y,-q.z,q.w};
    }
    return soma30_to_77(body,&fingers);
}
}
