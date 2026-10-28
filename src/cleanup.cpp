#include "motion.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>

namespace motion {
namespace {
float len(Vec3 v) {return std::sqrt(dot(v,v));}
float angle(Quat a,Quat b) {
    auto q=normalized(inverse(a)*b);
    return 2*std::atan2(len({q.x,q.y,q.z}),std::abs(q.w))*57.2957795f;
}
// Every affected edge, including the preceding/following sample when present.
std::array<float,6> steps(const Clip& before,const Clip& after,const FootContact& contact,const std::array<size_t,3>& chain) {
    std::array<float,6> out{};
    for(size_t f=std::max<size_t>(1,contact.start);f<=std::min(contact.end+1,before.frames()-1);++f) {
        for(size_t j:chain) {
            float a=angle(before.frame(f-1)[j].q,before.frame(f)[j].q);
            float b=angle(after.frame(f-1)[j].q,after.frame(f)[j].q);
            out[0]=std::max(out[0],a);out[1]=std::max(out[1],b);out[4]=std::max(out[4],b-a);
        }
        auto a=world_pose(before,before.frame(f))[chain[2]].p-world_pose(before,before.frame(f-1))[chain[2]].p;
        auto b=world_pose(after,after.frame(f))[chain[2]].p-world_pose(after,after.frame(f-1))[chain[2]].p;
        out[2]=std::max(out[2],len(a));out[3]=std::max(out[3],len(b));out[5]=std::max(out[5],len(b)-len(a));
    }
    return out;
}
Quat aim(Vec3 a,Vec3 b) {
    a=normalized(a);b=normalized(b);float d=dot(a,b);
    if(d< -0.99999f) {auto v=normalized(cross(a,std::abs(a.x)<0.8f?Vec3{1,0,0}:Vec3{0,1,0}));return {v.x,v.y,v.z,0};}
    auto v=cross(a,b);return normalized({v.x,v.y,v.z,1+d});
}
size_t index(const nlohmann::json& v) {
    if(!v.is_number_integer()||v.get<double>()<0||v.get<double>()>17999)throw std::runtime_error("Contact frame must be an integer in 0..17999");
    return v.get<size_t>();
}
void fields(const nlohmann::json& v,const std::set<std::string>& keys) {
    if(!v.is_object())throw std::runtime_error("Contact plan entries must be objects");
    for(const auto& i:v.items())if(!keys.contains(i.key()))throw std::runtime_error("Unknown contact plan field: "+i.key());
}
}
CleanupPlan load_cleanup_plan(const fs::path& path) {
    if(fs::file_size(path)>65536)throw std::runtime_error("Contact plan exceeds 64 KiB");
    std::ifstream input(path);nlohmann::json d;input>>d;
    fields(d,{"contacts","protectedFrames","fadeFrames","maxCorrection","maxJointCorrectionDegrees","maxJointStepIncreaseDegrees","maxAnkleStepIncrease"});CleanupPlan p;
    if(!d.at("contacts").is_array()||d["contacts"].empty()||d["contacts"].size()>1000)throw std::runtime_error("contacts needs 1..1000 intervals");
    for(const auto& c:d["contacts"]) {
        fields(c,{"foot","start","end"});auto f=c.at("foot").get<std::string>();
        if(f!="left-foot"&&f!="right-foot")throw std::runtime_error("Contact foot must be left-foot or right-foot");
        p.contacts.push_back({f=="left-foot"?0u:1u,index(c.at("start")),index(c.at("end"))});
    }
    if(d.contains("protectedFrames")) {
        if(!d["protectedFrames"].is_array()||d["protectedFrames"].size()>18000)throw std::runtime_error("Invalid protectedFrames");
        for(const auto& v:d["protectedFrames"])p.protected_frames.push_back(index(v));
    }
    if(d.contains("fadeFrames")) {auto n=index(d["fadeFrames"]);if(n>60)throw std::runtime_error("fadeFrames is 0..60");p.fade_frames=static_cast<unsigned>(n);}
    if(d.contains("maxCorrection")) {
        const auto& v=d["maxCorrection"];
        if(!v.is_number()||!std::isfinite(v.get<double>())||v.get<double>()<=0||v.get<double>()>0.5)throw std::runtime_error("maxCorrection is metres in (0,0.5]");
        p.max_correction=v.get<float>();
    }
    auto bound=[&](const char* key,float& value,float maximum) {
        if(!d.contains(key))return;
        const auto& v=d[key];
        if(!v.is_number()||!std::isfinite(v.get<double>())||v.get<double>()<0||v.get<double>()>maximum)throw std::runtime_error(std::string(key)+" is out of range");
        value=v.get<float>();
    };
    bound("maxJointCorrectionDegrees",p.max_joint_correction,180);
    bound("maxJointStepIncreaseDegrees",p.max_joint_step_increase,180);
    bound("maxAnkleStepIncrease",p.max_ankle_step_increase,0.5f);
    // Clip-independent preflight, also used before a batch loads any model.
    for(size_t i=0;i<p.contacts.size();++i) {
        const auto& a=p.contacts[i];if(a.end<a.start+2)throw std::runtime_error("Contact needs at least three frames (inclusive)");
        for(size_t j=0;j<i;++j) {const auto& b=p.contacts[j];if(a.foot==b.foot&&a.start<=b.end&&b.start<=a.end)throw std::runtime_error("Same-foot contact intervals overlap");}
    }
    return p;
}
std::vector<ContactReport> cleanup_contacts(Clip& c,const CleanupPlan& p,const RetargetMapping* mapping) {
    c.validate();
    if(!std::isfinite(p.max_correction)||p.max_correction<=0||p.max_correction>0.5f||p.fade_frames>60||p.contacts.empty())throw std::runtime_error("Invalid cleanup settings");
    for(float value:{p.max_joint_correction,p.max_joint_step_increase})if(!std::isfinite(value)||value<0||value>180)throw std::runtime_error("Invalid joint safeguard");
    if(!std::isfinite(p.max_ankle_step_increase)||p.max_ankle_step_increase<0||p.max_ankle_step_increase>0.5f)throw std::runtime_error("Invalid ankle safeguard");
    auto canonical=soma();std::array<std::array<size_t,3>,2> chains{};
    if(!mapping)require_soma(c);
    for(size_t side=0;side<2;++side)for(size_t k=0;k<3;++k) {
        size_t slot=22+side*4+k;std::string name=mapping?mapping->bones[slot]:canonical[slot].name;
        size_t found=c.joints.size();
        for(size_t j=0;j<c.joints.size();++j)if(c.joints[j].name==name&&!name.empty()) {if(found!=c.joints.size())throw std::runtime_error("Ambiguous contact bone");found=j;}
        if(found==c.joints.size())throw std::runtime_error("Missing contact chain bone: "+name);
        chains[side][k]=found;
    }
    std::set<size_t> used;
    for(auto chain:chains) {
        for(size_t joint:chain)if(!used.insert(joint).second)throw std::runtime_error("Contact chains must be distinct");
        int ancestor=c.joints[chain[0]].parent;
        while(ancestor>=0&&static_cast<size_t>(ancestor)!=c.motion_root)ancestor=c.joints[static_cast<size_t>(ancestor)].parent;
        if(ancestor<0)throw std::runtime_error("Contact chains must descend from the motion root");
        if(c.joints[chain[1]].parent!=static_cast<int>(chain[0])||c.joints[chain[2]].parent!=static_cast<int>(chain[1])||chain[0]==c.motion_root||c.joints[chain[0]].parent<0)
            throw std::runtime_error("Cleanup requires direct thigh/calf/foot chains below the motion root");
    }
    // Distinct joint indices alone do not rule out a malicious/mistaken serial
    // mapping where one leg is below the other; their corrections would interact.
    for(size_t side=0;side<2;++side) {
        int ancestor=c.joints[chains[side][0]].parent;
        while(ancestor>=0) {
            const auto& other=chains[1-side];
            if(std::find(other.begin(),other.end(),static_cast<size_t>(ancestor))!=other.end())throw std::runtime_error("Cleanup requires independent leg chains");
            ancestor=c.joints[static_cast<size_t>(ancestor)].parent;
        }
    }
    for(size_t f:p.protected_frames)if(f>=c.frames())throw std::runtime_error("Protected frame outside clip");
    for(size_t i=0;i<p.contacts.size();++i) {
        auto a=p.contacts[i];if(a.foot>1||a.start>=c.frames()||a.end>=c.frames()||a.end<=a.start||a.end-a.start<2)throw std::runtime_error("Invalid contact interval");
        for(size_t j=0;j<i;++j) {auto b=p.contacts[j];if(a.foot==b.foot&&a.start<=b.end&&b.start<=a.end)throw std::runtime_error("Overlapping contact intervals");}
    }
    const Clip original=c;Clip result=c;std::vector<ContactReport> reports;
    for(auto contact:p.contacts) {
        ContactReport report;auto chain=chains[contact.foot];auto [a,b,e]=chain;
        Vec3 anchor=world_pose(original,original.frame(contact.start))[e].p;
        for(size_t f=contact.start;f<=contact.end;++f)if(len(world_pose(original,original.frame(f))[e].p-anchor)>p.max_correction)report.skipped=true;
        Vec3 prev_before{},prev_after{};
        for(size_t f=contact.start;f<=contact.end;++f) {
            auto before=world_pose(original,original.frame(f));auto pose=result.frame(f);auto w=world_pose(result,pose);
            float weight=1;
            if(p.fade_frames)weight=std::min(1.0f,static_cast<float>(std::min(f-contact.start,contact.end-f))/static_cast<float>(p.fade_frames));
            for(size_t pin:p.protected_frames) {
                size_t distance=f>pin?f-pin:pin-f;
                if(!distance)weight=0;
                else if(p.fade_frames)weight=std::min(weight,static_cast<float>(distance)/static_cast<float>(p.fade_frames));
            }
            weight=0.5f-0.5f*std::cos(3.14159265358979323846f*weight);
            if(!report.skipped&&weight>0&&len(anchor-w[e].p)>1e-7f) {
                Vec3 target=w[e].p+(anchor-w[e].p)*weight;
                float l1=len(w[b].p-w[a].p),l2=len(w[e].p-w[b].p),distance=len(target-w[a].p);
                if(l1<1e-6f||l2<1e-6f||distance<1e-6f)throw std::runtime_error("Degenerate contact chain");
                auto direction=normalized(target-w[a].p);
                float reach=std::clamp(distance,std::abs(l1-l2)+1e-6f,l1+l2-1e-6f);
                float along=(l1*l1-l2*l2+reach*reach)/(2*reach);
                Vec3 pole=(w[b].p-w[a].p)-direction*dot(w[b].p-w[a].p,direction);
                if(len(pole)<1e-6f) {pole=rotate(w[a].q,{0,0,1});pole=pole-direction*dot(pole,direction);}
                if(len(pole)<1e-6f)pole=cross(direction,std::abs(direction.x)<0.8f?Vec3{1,0,0}:Vec3{0,1,0});
                Vec3 knee=w[a].p+direction*along+normalized(pole)*std::sqrt(std::max(0.0f,l1*l1-along*along));
                auto parent=static_cast<size_t>(c.joints[a].parent);
                pose[a].q=normalized(inverse(w[parent].q)*aim(w[b].p-w[a].p,knee-w[a].p)*w[a].q);
                w=world_pose(result,pose);
                pose[b].q=normalized(inverse(w[a].q)*aim(w[e].p-w[b].p,target-w[b].p)*w[b].q);
                w=world_pose(result,pose);
                pose[e].q=normalized(inverse(w[b].q)*before[e].q);
            }
            auto after=world_pose(result,pose);
            if(f>contact.start) {
                auto x=before[e].p-prev_before;auto y=after[e].p-prev_after;x.y=0;y.y=0;
                report.drift_before+=len(x);report.drift_after+=len(y);
            }
            prev_before=before[e].p;prev_after=after[e].p;
            report.residual_max=std::max(report.residual_max,len(after[e].p-anchor));
            for(size_t j:chain)report.rotation_max=std::max(report.rotation_max,angle(original.frame(f)[j].q,pose[j].q));
            ++report.samples;
        }
        report.drift_before/=static_cast<float>(report.samples-1);report.drift_after/=static_cast<float>(report.samples-1);
        if(report.skipped)report.skip_reason="correction-bound";
        report.attempted_rotation_max=report.rotation_max;
        auto candidate=steps(original,result,contact,chain);
        report.joint_step_increase=candidate[4];report.ankle_step_increase=candidate[5];
        if(!report.skipped) {
            if(report.drift_after>report.drift_before+1e-7f)report.skip_reason="drift-increased";
            else if(report.rotation_max>p.max_joint_correction+1e-3f)report.skip_reason="joint-correction-bound";
            else if(candidate[4]>p.max_joint_step_increase+1e-3f)report.skip_reason="joint-step-increased";
            else if(candidate[5]>p.max_ankle_step_increase+1e-6f)report.skip_reason="ankle-step-increased";
        }
        // Revert the whole interval, retaining previously accepted disjoint chains.
        if(!report.skip_reason.empty()) {
            report.skipped=true;report.drift_after=report.drift_before;
            report.rotation_max=0;report.residual_max=0;
            for(size_t f=contact.start;f<=contact.end;++f) {
                for(size_t j:chain)result.frame(f)[j]=original.frame(f)[j];
                report.residual_max=std::max(report.residual_max,len(world_pose(original,original.frame(f))[e].p-anchor));
            }
        }
        reports.push_back(report);
    }
    // Adjacent intervals can share an edge; report its final state after all decisions.
    for(size_t i=0;i<p.contacts.size();++i) {
        auto final=steps(original,result,p.contacts[i],chains[p.contacts[i].foot]);auto& r=reports[i];
        r.joint_step_before=final[0];r.joint_step_after=final[1];r.ankle_step_before=final[2];r.ankle_step_after=final[3];
    }
    result.validate();c=std::move(result);return reports;
}
}
