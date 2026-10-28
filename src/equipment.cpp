#include "equipment.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>

namespace motion {
namespace {
using nlohmann::json;
float length(Vec3 v){return std::sqrt(dot(v,v));}
float angle(Quat a,Quat b){auto q=normalized(inverse(a)*b);return 2*std::atan2(length({q.x,q.y,q.z}),std::abs(q.w))*57.2957795f;}
void fields(const json& j,std::initializer_list<const char*> allowed) {
    if(!j.is_object())throw std::runtime_error("Equipment entries must be objects");
    for(const auto& [k,v]:j.items()) {
        (void)v;
        if(std::none_of(allowed.begin(),allowed.end(),[&](const char* a){return k==a;}))throw std::runtime_error("Unknown equipment field: "+k);
    }
}
size_t frame(const json& v) {
    if(!v.is_number_integer()||v.get<double>()<0||v.get<double>()>17999)throw std::runtime_error("Equipment frames must be integers in 0..17999");
    return v.get<size_t>();
}
float number(const json& j,float low,float high) {
    if(!j.is_number())throw std::runtime_error("Equipment value must be numeric");
    float n=j.get<float>();if(!std::isfinite(n)||n<low||n>high)throw std::runtime_error("Equipment number out of range");return n;
}
Vec3 vec(const json& j,float limit=10000) {
    if(!j.is_array()||j.size()!=3)throw std::runtime_error("Equipment vector needs three numbers");
    return {number(j[0],-limit,limit),number(j[1],-limit,limit),number(j[2],-limit,limit)};
}
std::string name(const json& v) {
    auto n=v.get<std::string>();if(n.empty()||n.size()>128)throw std::runtime_error("Equipment names need 1..128 bytes");return n;
}
Transform transform(const json& j) {
    return preview_offset(j.contains("position")?vec(j["position"]):Vec3{},j.contains("rotation")?vec(j["rotation"],3600):Vec3{});
}
std::vector<EquipmentKey> keys(const json& list,bool point_only=false) {
    if(!list.is_array()||list.empty()||list.size()>18000)throw std::runtime_error("Equipment keys need 1..18000 entries");
    std::vector<EquipmentKey> out;
    for(const auto& j:list) {
        if(point_only)fields(j,{"frame","position"});else fields(j,{"frame","position","rotation","meshScale"});
        EquipmentKey k;k.frame=frame(j.at("frame"));k.local=transform(j);
        if(j.contains("meshScale")) {
            k.mesh_scale=vec(j["meshScale"],1000);
            for(float s:{k.mesh_scale.x,k.mesh_scale.y,k.mesh_scale.z})if(s<0.000001f)throw std::runtime_error("Equipment meshScale must be positive");
        }
        if(!out.empty()&&k.frame<=out.back().frame)throw std::runtime_error("Equipment keys must be strictly ordered");
        out.push_back(k);
    }
    return out;
}
EquipmentKey at(const std::vector<EquipmentKey>& keys,size_t f,bool step=false) {
    auto hi=std::upper_bound(keys.begin(),keys.end(),f,[](size_t v,const auto& k){return v<k.frame;});
    if(hi==keys.begin())return keys.front();
    if(hi==keys.end()||step)return *std::prev(hi);
    const auto& a=*std::prev(hi);const auto& b=*hi;float t=static_cast<float>(f-a.frame)/static_cast<float>(b.frame-a.frame);
    return {f,blend(a.local,b.local,t),a.mesh_scale*(1-t)+b.mesh_scale*t};
}
size_t joint(const Clip& c,const std::string& n) {
    size_t found=c.joints.size();
    for(size_t j=0;j<c.joints.size();++j)if(c.joints[j].name==n) {
        if(found!=c.joints.size())throw std::runtime_error("Ambiguous equipment bone: "+n);
        found=j;
    }
    if(found==c.joints.size())throw std::runtime_error("Equipment bone not found: "+n);
    return found;
}
Quat aim(Vec3 a,Vec3 b) {
    a=normalized(a);b=normalized(b);float d=dot(a,b);
    if(d< -0.99999f){auto v=normalized(cross(a,std::abs(a.x)<0.8f?Vec3{1,0,0}:Vec3{0,1,0}));return {v.x,v.y,v.z,0};}
    auto v=cross(a,b);return normalized({v.x,v.y,v.z,1+d});
}
json measure(const Clip& c,const EquipmentTrack& o,const EquipmentGrip& g) {
    size_t j=joint(c,g.bone),worst=0;double sum=0,angles=0;float max=0,max_angle=0,drift=0;Vec3 previous{};
    for(size_t f=g.start;f<=g.end;++f) {
        auto hand=world_pose(c,c.frame(f))[j]*g.hand_offset,goal=o.poses[f]*g.target;
        Vec3 error=rotate(inverse(o.poses[f].q),hand.p-goal.p);float e=length(error),a=angle(hand.q,goal.q);
        sum+=e;angles+=a;if(e>max){max=e;worst=f;}max_angle=std::max(max_angle,a);
        if(f>g.start)drift+=length(error-previous);
        previous=error;
    }
    size_t samples=g.end-g.start+1;
    return {{"samples",samples},{"positionMeanMetres",sum/static_cast<double>(samples)},{"positionMaxMetres",max},{"positionWorstFrame",worst},
        {"rotationMeanDegrees",angles/static_cast<double>(samples)},{"rotationMaxDegrees",max_angle},{"relativeDriftMetresPerFrame",samples>1?drift/static_cast<float>(samples-1):0},
        {"satisfied",max<=g.position_tolerance+1e-6f&&(!g.orientation||max_angle<=g.angle_tolerance+0.001f)}};
}
}
EquipmentPlan load_equipment_plan(const fs::path& path) {
    if(fs::file_size(path)>4*1024*1024)throw std::runtime_error("Equipment plan exceeds 4 MiB");
    std::ifstream in(path);json d;in>>d;fields(d,{"version","objects","grips","strings","events","protectedFrames"});
    if(d.at("version")!=1)throw std::runtime_error("Equipment version must be 1");
    EquipmentPlan p;p.recipe=d;std::set<std::string> names;
    if(!d.at("objects").is_array()||d["objects"].empty()||d["objects"].size()>32)throw std::runtime_error("Equipment objects need 1..32 entries");
    for(size_t i=0;i<d["objects"].size();++i) {
        const auto& j=d["objects"][i];fields(j,{"name","mesh","segments"});EquipmentObject o;o.name=name(j.at("name"));
        if(!names.insert(o.name).second)throw std::runtime_error("Duplicate equipment name: "+o.name);
        if(j.contains("mesh")) {
            auto m=utf8path(j["mesh"].get<std::string>());if(m.empty())throw std::runtime_error("Empty equipment mesh path");
            o.mesh=fs::absolute(path.parent_path()/m).lexically_normal();p.recipe["objects"][i]["mesh"]=o.mesh.generic_string();
        }
        if(!j.at("segments").is_array()||j["segments"].empty()||j["segments"].size()>256)throw std::runtime_error("Equipment segments need 1..256 entries");
        for(const auto& s:j["segments"]) {
            fields(s,{"start","end","space","anchor","keys","interpolation"});EquipmentSegment seg;
            seg.start=frame(s.at("start"));seg.end=frame(s.at("end"));seg.space=s.value("space",std::string("world"));seg.anchor=s.value("anchor",std::string());
            seg.interpolation=s.value("interpolation",std::string("linear"));seg.keys=keys(s.at("keys"));
            if(seg.end<seg.start||seg.keys.front().frame!=seg.start||seg.keys.back().frame!=seg.end)throw std::runtime_error("Equipment keys must span the segment endpoints");
            if((o.segments.empty()?seg.start!=0:seg.start!=o.segments.back().end+1))throw std::runtime_error("Equipment segments must cover the clip without gaps or overlaps");
            if(seg.space!="world"&&seg.space!="bone"&&seg.space!="object")throw std::runtime_error("Equipment space must be world, bone or object");
            if((seg.space=="world")!=seg.anchor.empty())throw std::runtime_error("Equipment bone/object space needs anchor; world space must omit it");
            if(seg.interpolation!="linear"&&seg.interpolation!="step")throw std::runtime_error("Equipment interpolation must be linear or step");
            o.segments.push_back(seg);
        }
        p.objects.push_back(o);
    }
    if(d.contains("grips")) {
        if(!d["grips"].is_array()||d["grips"].size()>64)throw std::runtime_error("Equipment grips must be an array of at most 64 intervals");
        std::set<std::string> grip_names;
        for(const auto& j:d["grips"]) {
            fields(j,{"name","object","bone","chain","start","end","position","rotation","handOffset","orientation","positionTolerance","rotationTolerance","maxCorrection","maxRotation","maxStepIncrease"});
            EquipmentGrip g;g.name=name(j.at("name"));g.object=name(j.at("object"));g.bone=name(j.at("bone"));g.start=frame(j.at("start"));g.end=frame(j.at("end"));g.target=transform(j);
            if(!grip_names.insert(g.name).second||!names.contains(g.object)||g.end<g.start)throw std::runtime_error("Invalid grip name, object or interval");
            if(j.contains("chain")){if(!j["chain"].is_array()||j["chain"].size()!=3)throw std::runtime_error("Grip chain needs upper arm, forearm, hand");for(const auto& n:j["chain"])g.chain.push_back(name(n));}
            if(j.contains("handOffset")){fields(j["handOffset"],{"position","rotation"});g.hand_offset=transform(j["handOffset"]);}
            g.orientation=j.value("orientation",true);
            if(j.contains("positionTolerance"))g.position_tolerance=number(j["positionTolerance"],0,0.5f);
            if(j.contains("rotationTolerance"))g.angle_tolerance=number(j["rotationTolerance"],0,180);
            if(j.contains("maxCorrection"))g.max_correction=number(j["maxCorrection"],0,1);
            if(j.contains("maxRotation"))g.max_rotation=number(j["maxRotation"],0,180);
            if(j.contains("maxStepIncrease"))g.max_step_increase=number(j["maxStepIncrease"],0,180);
            p.grips.push_back(g);
        }
    }
    if(d.contains("strings")) {
        if(!d["strings"].is_array()||d["strings"].size()>16)throw std::runtime_error("Equipment strings must be an array of at most 16 entries");
        for(const auto& j:d["strings"]) {
            fields(j,{"name","object","top","bottom","nock","thickness","topAnchor","bottomAnchor","nockAnchor"});EquipmentString s;
            s.name=name(j.at("name"));s.object=name(j.at("object"));s.top=vec(j.at("top"));s.bottom=vec(j.at("bottom"));s.nock=keys(j.at("nock"),true);
            s.top_anchor=j.value("topAnchor",s.object);s.bottom_anchor=j.value("bottomAnchor",s.object);s.nock_anchor=j.value("nockAnchor",s.object);
            if(!names.contains(s.object)||!names.insert(s.name+".top").second||!names.insert(s.name+".bottom").second)throw std::runtime_error("Invalid string object or duplicate part name");
            if(j.contains("thickness"))s.thickness=number(j["thickness"],0.0001f,0.1f);
            p.strings.push_back(s);
        }
    }
    if(d.contains("events")) {
        if(!d["events"].is_array()||d["events"].size()>256)throw std::runtime_error("Equipment events must be an array of at most 256 entries");
        for(const auto& e:d["events"]) {
            fields(e,{"name","frame","object","data"});(void)name(e.at("name"));(void)frame(e.at("frame"));
            if(e.contains("object")&&!names.contains(name(e["object"])))throw std::runtime_error("Event object not found");
        }
        p.events=d["events"];
    }
    if(d.contains("protectedFrames")){if(!d["protectedFrames"].is_array()||d["protectedFrames"].size()>18000)throw std::runtime_error("Invalid equipment protectedFrames");for(const auto& f:d["protectedFrames"])p.protected_frames.push_back(frame(f));}
    return p;
}
EquipmentResult author_equipment(const Clip& source,const EquipmentPlan& p,bool correct) {
    source.validate();if(p.objects.empty()||p.objects.size()>32)throw std::runtime_error("Invalid equipment object count");
    if(p.grips.size()>64||p.strings.size()>16||!p.events.is_array()||p.events.size()>256)throw std::runtime_error("Invalid equipment interval/string/event count");
    auto check_keys=[&](const std::vector<EquipmentKey>& list,size_t start,size_t end) {
        if(list.empty()||list.size()>18000||list.front().frame!=start||list.back().frame!=end)throw std::runtime_error("Equipment keys must span their interval");
        for(size_t i=0;i<list.size();++i) {
            const auto& k=list[i];if(k.frame<start||k.frame>end||(i&&k.frame<=list[i-1].frame))throw std::runtime_error("Invalid equipment key ordering");
            for(float x:{k.local.p.x,k.local.p.y,k.local.p.z,k.mesh_scale.x,k.mesh_scale.y,k.mesh_scale.z})if(!std::isfinite(x))throw std::runtime_error("Nonfinite equipment key");
            if(k.mesh_scale.x<=0||k.mesh_scale.y<=0||k.mesh_scale.z<=0)throw std::runtime_error("Equipment meshScale must be positive");
            (void)normalized(k.local.q);
        }
    };
    EquipmentResult out;out.body=source;out.events=p.events;
    out.objects.reserve(p.objects.size()+p.strings.size()*2);
    std::map<std::string,size_t> objects;std::set<std::string> body_names;
    for(const auto& j:source.joints)if(!j.name.empty()&&!body_names.insert(j.name).second)throw std::runtime_error("Equipment needs unique body bone names");
    for(size_t i=0;i<p.objects.size();++i) {
        const auto& o=p.objects[i];if(o.name.empty()||body_names.contains(o.name)||!objects.emplace(o.name,i).second)throw std::runtime_error("Equipment object names must be unique and distinct from body bones");
        if(o.segments.empty()||o.segments.back().end!=source.frames()-1)throw std::runtime_error("Equipment segments must cover every source frame");
        size_t next=0;
        for(const auto& s:o.segments) {
            if(s.start!=next||s.end<s.start||s.end>=source.frames())throw std::runtime_error("Invalid equipment segment coverage");
            next=s.end+1;
            if(s.space!="world"&&s.space!="bone"&&s.space!="object")throw std::runtime_error("Unknown equipment space");
            if(s.interpolation!="linear"&&s.interpolation!="step")throw std::runtime_error("Unknown equipment interpolation");
            check_keys(s.keys,s.start,s.end);
        }
        out.objects.push_back({o.name,o.mesh,false,std::vector<Transform>(source.frames()),std::vector<Vec3>(source.frames(),{1,1,1})});
    }
    std::vector<std::vector<Transform>> body_world;body_world.reserve(source.frames());
    for(size_t f=0;f<source.frames();++f)body_world.push_back(world_pose(source,source.frame(f)));
    std::vector<unsigned char> state(p.objects.size());
    std::function<void(size_t)> evaluate=[&](size_t index) {
        if(state[index]==2)return;
        if(state[index]==1)throw std::runtime_error("Equipment object anchor cycle");
        state[index]=1;
        for(const auto& seg:p.objects[index].segments) {
            size_t anchor=0;
            if(seg.space=="bone")anchor=joint(source,seg.anchor);
            if(seg.space=="object") {auto found=objects.find(seg.anchor);if(found==objects.end())throw std::runtime_error("Equipment object anchor not found");anchor=found->second;evaluate(anchor);}
            for(size_t f=seg.start;f<=seg.end;++f) {
                auto key=at(seg.keys,f,seg.interpolation=="step");Transform base;
                if(seg.space=="bone")base=body_world[f][anchor];else if(seg.space=="object")base=out.objects[anchor].poses[f];
                out.objects[index].poses[f]=base*key.local;out.objects[index].scales[f]=key.mesh_scale;
            }
        }
        state[index]=2;
    };
    for(size_t i=0;i<p.objects.size();++i)evaluate(i);
    for(const auto& s:p.strings) {
        if(s.nock.empty()||s.nock.front().frame!=0||s.nock.back().frame!=source.frames()-1)throw std::runtime_error("String nock keys must span the clip");
        check_keys(s.nock,0,source.frames()-1);
        auto anchor=[&](const std::string& n)->const EquipmentTrack& {
            auto found=objects.find(n.empty()?s.object:n);if(found==objects.end())throw std::runtime_error("String anchor not found");return out.objects[found->second];
        };
        for(int side=0;side<2;++side) {
            EquipmentTrack rod;rod.name=s.name+(side?".bottom":".top");rod.builtin_rod=true;
            if(body_names.contains(rod.name)||objects.contains(rod.name))throw std::runtime_error("Duplicate string part name");
            for(size_t f=0;f<source.frames();++f) {
                const auto& end=anchor(side?s.bottom_anchor:s.top_anchor).poses[f];const auto& nock=anchor(s.nock_anchor).poses[f];
                Vec3 a=end.p+rotate(end.q,side?s.bottom:s.top),b=nock.p+rotate(nock.q,at(s.nock,f).local.p);
                float len=length(b-a);if(len<0.00001f)throw std::runtime_error("String segment has zero length");
                rod.poses.push_back({(a+b)*0.5f,aim({0,1,0},b-a)});rod.scales.push_back({s.thickness,len,s.thickness});
            }
            out.objects.push_back(std::move(rod));
        }
    }
    for(size_t f:p.protected_frames)if(f>=source.frames())throw std::runtime_error("Protected equipment frame outside clip");
    for(auto& e:out.events) {
        size_t f=frame(e.at("frame"));if(f>=source.frames())throw std::runtime_error("Equipment event outside clip");e["timeSeconds"]=static_cast<double>(f)/source.fps;
        if(e.at("name")=="release") {
            if(!e.contains("object")||!objects.contains(e["object"].get<std::string>()))throw std::runtime_error("Release needs an authored equipment object");
            const auto& segments=p.objects[objects.at(e["object"].get<std::string>())].segments;bool detached=false;
            for(size_t s=1;s<segments.size();++s)if(segments[s].start==f&&segments[s-1].space!="world"&&segments[s].space=="world")detached=true;
            if(!detached)throw std::runtime_error("Release must coincide with an attached-to-world segment transition");
        }
    }
    json reports=json::array();std::vector<std::array<size_t,3>> chains(p.grips.size());
    for(size_t i=0;i<p.grips.size();++i) {
        const auto& g=p.grips[i];if(g.end>=source.frames()||g.start>g.end||!objects.contains(g.object))throw std::runtime_error("Grip outside clip or unknown object");(void)joint(source,g.bone);
        if(!correct)continue;
        if(g.chain.size()!=3||g.chain.back()!=g.bone)throw std::runtime_error("Correction needs an explicit upper-arm/forearm/hand chain ending at bone");
        for(size_t k=0;k<3;++k)chains[i][k]=joint(source,g.chain[k]);
        auto [a,b,e]=chains[i];
        if(source.joints[a].parent<0||a==source.motion_root||source.joints[b].parent!=static_cast<int>(a)||source.joints[e].parent!=static_cast<int>(b))throw std::runtime_error("Grip correction requires direct three-bone chains below the root");
        for(size_t j=0;j<i;++j)if(g.start<=p.grips[j].end&&p.grips[j].start<=g.end) {
            for(size_t n:chains[i])for(size_t m:chains[j])if(n==m)throw std::runtime_error("Overlapping grip corrections share a joint");
            for(size_t n:chains[i])for(int a=source.joints[n].parent;a>=0;a=source.joints[static_cast<size_t>(a)].parent)
                if(std::find(chains[j].begin(),chains[j].end(),static_cast<size_t>(a))!=chains[j].end())throw std::runtime_error("Grip correction chains must be independent");
            for(size_t n:chains[j])for(int a=source.joints[n].parent;a>=0;a=source.joints[static_cast<size_t>(a)].parent)
                if(std::find(chains[i].begin(),chains[i].end(),static_cast<size_t>(a))!=chains[i].end())throw std::runtime_error("Grip correction chains must be independent");
        }
    }
    for(size_t i=0;i<p.grips.size();++i) {
        const auto& g=p.grips[i];const auto& object=out.objects[objects.at(g.object)];auto before=measure(source,object,g);
        std::string status=correct?"accepted":"diagnosed",reason;float max_change=0,max_step=0;Clip candidate=out.body;
        if(correct) {
            auto [a,b,e]=chains[i];
            for(size_t f=g.start;f<=g.end;++f) {
                auto marker=object.poses[f]*g.target,goal=marker*inverse(g.hand_offset);
                if(!g.orientation){goal.q=body_world[f][e].q;goal.p=marker.p-rotate(goal.q,g.hand_offset.p);}
                auto pose=candidate.frame(f);auto w=world_pose(candidate,pose);
                float error=length(goal.p-w[e].p);
                if(error>g.max_correction+1e-6f){reason="correction-bound";break;}
                if(std::find(p.protected_frames.begin(),p.protected_frames.end(),f)!=p.protected_frames.end())continue;
                if(error>1e-7f) {
                    float l1=length(w[b].p-w[a].p),l2=length(w[e].p-w[b].p),d=length(goal.p-w[a].p);
                    if(l1<1e-6f||l2<1e-6f||d<1e-6f){reason="degenerate-chain";break;}
                    if(d>l1+l2+g.position_tolerance||d<std::abs(l1-l2)-g.position_tolerance){reason="unreachable";break;}
                    auto direction=normalized(goal.p-w[a].p);float reach=std::clamp(d,std::abs(l1-l2)+1e-6f,l1+l2-1e-6f);
                    float along=(l1*l1-l2*l2+reach*reach)/(2*reach);Vec3 pole=w[b].p-w[a].p-direction*dot(w[b].p-w[a].p,direction);
                    if(length(pole)<1e-6f){pole=rotate(w[a].q,{0,0,1});pole=pole-direction*dot(pole,direction);}
                    if(length(pole)<1e-6f)pole=cross(direction,std::abs(direction.x)<0.8f?Vec3{1,0,0}:Vec3{0,1,0});
                    auto elbow=w[a].p+direction*along+normalized(pole)*std::sqrt(std::max(0.0f,l1*l1-along*along));
                    pose[a].q=normalized(inverse(w[static_cast<size_t>(source.joints[a].parent)].q)*aim(w[b].p-w[a].p,elbow-w[a].p)*w[a].q);
                    w=world_pose(candidate,pose);pose[b].q=normalized(inverse(w[a].q)*aim(w[e].p-w[b].p,goal.p-w[b].p)*w[b].q);w=world_pose(candidate,pose);
                }
                pose[e].q=normalized(inverse(w[b].q)*(g.orientation?goal.q:body_world[f][e].q));
            }
            for(size_t f=g.start;f<=g.end;++f)for(size_t n:chains[i])max_change=std::max(max_change,angle(source.frame(f)[n].q,candidate.frame(f)[n].q));
            for(size_t f=std::max<size_t>(1,g.start);f<=std::min(source.frames()-1,g.end+1);++f)for(size_t n:chains[i])
                max_step=std::max(max_step,angle(candidate.frame(f-1)[n].q,candidate.frame(f)[n].q)-angle(source.frame(f-1)[n].q,source.frame(f)[n].q));
            if(reason.empty()&&max_change>g.max_rotation+0.001f)reason="rotation-bound";
            if(reason.empty()&&max_step>g.max_step_increase+0.001f)reason="step-increase-bound";
            auto after=measure(candidate,object,g);if(reason.empty()&&!after["satisfied"].get<bool>())reason="residual-or-protected-frame";
            if(reason.empty())out.body=std::move(candidate);else status="rejected";
        }
        reports.push_back({{"name",g.name},{"object",g.object},{"bone",g.bone},{"start",g.start},{"end",g.end},{"status",status},{"reason",reason},{"before",before},
            {"attemptedJointCorrectionDegrees",max_change},{"attemptedStepIncreaseDegrees",max_step}});
    }
    bool satisfied=true;
    for(size_t i=0;i<p.grips.size();++i) {auto final=measure(out.body,out.objects[objects.at(p.grips[i].object)],p.grips[i]);reports[i]["after"]=final;satisfied=satisfied&&final["satisfied"].get<bool>();}
    json seams=json::array();
    for(size_t i=0;i<p.objects.size();++i)for(size_t s=1;s<p.objects[i].segments.size();++s) {size_t f=p.objects[i].segments[s].start;const auto& a=out.objects[i].poses[f-1];const auto& b=out.objects[i].poses[f];
        seams.push_back({{"object",p.objects[i].name},{"frame",f},{"positionStepMetres",length(b.p-a.p)},{"rotationStepDegrees",angle(a.q,b.q)}});}
    out.report={{"version",1},{"frames",source.frames()},{"fps",source.fps},{"correctGrips",correct},{"allGripsSatisfied",satisfied},{"grips",reports},{"anchorTransitions",seams},
        {"objectTrajectoryBasis","source-body-before-grip-correction"},{"collisionChecked",false},{"plan",p.recipe}};
    out.body.validate();return out;
}
}
