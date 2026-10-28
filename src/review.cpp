#include "review.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>

namespace motion {
namespace {
using Json=nlohmann::json;
float length(Vec3 v){return std::sqrt(dot(v,v));}
// Closest points of two closed segments, including degenerate segments.
float distance(Vec3 p,Vec3 q,Vec3 a,Vec3 b) {
    Vec3 u=q-p,v=b-a,w=p-a;float uu=dot(u,u),vv=dot(v,v),uv=dot(u,v),uw=dot(u,w),vw=dot(v,w);
    float s=0,t=0;
    if(uu<1e-12f&&vv<1e-12f)return length(w);
    if(uu<1e-12f)t=std::clamp(vw/vv,0.0f,1.0f);
    else if(vv<1e-12f)s=std::clamp(-uw/uu,0.0f,1.0f);
    else {
        float den=uu*vv-uv*uv;
        if(den>1e-12f)s=std::clamp((uv*vw-uw*vv)/den,0.0f,1.0f);
        t=(uv*s+vw)/vv;
        if(t<0){t=0;s=std::clamp(-uw/uu,0.0f,1.0f);}
        else if(t>1){t=1;s=std::clamp((uv-uw)/uu,0.0f,1.0f);}
    }
    return length(w+u*s-v*t);
}
// Display tracks only: round to .01 mm to keep an offline page compact. All
// observations above/below are computed from the original float transforms.
Json xyz(Vec3 p){auto v=[](float x){return std::round(static_cast<double>(x)*100000)/100000;};return Json::array({v(p.x),v(p.y),v(p.z)});}
Vec3 vector3(const Json& a) {
    if(!a.is_array()||a.size()!=3)throw std::runtime_error("Review vector needs three finite metre values");
    Vec3 out;float* fields[]={&out.x,&out.y,&out.z};
    for(size_t i=0;i<3;++i){if(!a[i].is_number()||!std::isfinite(a[i].get<double>())||std::abs(a[i].get<double>())>1000)throw std::runtime_error("Invalid review vector");*fields[i]=a[i].get<float>();}
    return out;
}
}
nlohmann::json review_motion(const Clip& clip,const nlohmann::json& settings) {
    clip.validate();
    if(clip.joints.size()>128)throw std::runtime_error("Review supports at most 128 joints");
    const std::set<std::string> keys={"groundY","rotationStepDegrees","belowGroundMetres","nearGroundMetres","lowFootSpeed","overlapMetres","props"};
    if(!settings.is_object())throw std::runtime_error("Review settings must be an object");
    for(auto it=settings.begin();it!=settings.end();++it)if(!keys.contains(it.key()))throw std::runtime_error("Unknown review setting: "+it.key());
    auto value=[&](const char* key,float def,float lo,float hi){if(!settings.contains(key))return def;const auto& v=settings.at(key);if(!v.is_number()||!std::isfinite(v.get<double>())||v.get<double>()<lo||v.get<double>()>hi)throw std::runtime_error(std::string("Invalid review setting: ")+key);return v.get<float>();};
    float ground=value("groundY",0,-1000,1000),angle=value("rotationStepDegrees",30,0.1f,180),below=value("belowGroundMetres",.02f,0,1),near=value("nearGroundMetres",.06f,0,1),speed=value("lowFootSpeed",.15f,0,10),overlap=value("overlapMetres",.02f,0,1);
    std::map<std::string,size_t> names;Json joints=Json::array();
    for(size_t j=0;j<clip.joints.size();++j){if(!names.emplace(clip.joints[j].name,j).second)throw std::runtime_error("Review needs unique joint names");joints.push_back({{"name",clip.joints[j].name},{"parent",clip.joints[j].parent}});}
    bool canonical=true;try{require_soma(clip);}catch(const std::exception&){canonical=false;}
    std::vector<std::vector<Transform>> world;world.reserve(clip.frames());Json positions=Json::array();
    for(size_t f=0;f<clip.frames();++f){world.push_back(world_pose(clip,clip.frame(f)));Json p=Json::array();for(const auto& t:world.back())p.push_back(xyz(t.p));positions.push_back(std::move(p));}
    Json findings=Json::array();
    auto series=[&](const std::string& rule,const std::vector<std::string>& subjects,const std::string& unit,float threshold,const std::string& method,const std::vector<float>& values){
        for(size_t f=0;f<values.size();){if(values[f]<=threshold){++f;continue;}size_t start=f,peak=f;while(f+1<values.size()&&values[f+1]>threshold){++f;if(values[f]>values[peak])peak=f;}size_t end=f++;
            if(findings.size()>=10000)throw std::runtime_error("Review exceeds 10000 observation intervals");
            std::string id=rule;for(const auto& s:subjects)id+="/"+s;id+="/"+std::to_string(start)+"-"+std::to_string(end);
            findings.push_back({{"id",id},{"rule",rule},{"subjects",subjects},{"startFrame",start},{"endFrame",end},{"peakFrame",peak},{"startSeconds",static_cast<double>(start)/clip.fps},{"endSeconds",static_cast<double>(end)/clip.fps},{"peakValue",values[peak]},{"unit",unit},{"threshold",threshold},{"method",method},{"interpretation","unreviewed"}});
        }
    };
    for(size_t j=0;j<clip.joints.size();++j){std::vector<float> v(clip.frames());for(size_t f=1;f<clip.frames();++f){Quat q=normalized(inverse(clip.frame(f-1)[j].q)*clip.frame(f)[j].q);v[f]=2*std::atan2(length({q.x,q.y,q.z}),std::abs(q.w))*180/std::numbers::pi_v<float>;}
        series("local-rotation-step",{clip.joints[j].name},"degrees/sample",angle,"Shortest local quaternion change from preceding sample; rapid intentional action may qualify",v);}
    Json proxies=Json::array();
    if(canonical){
        proxies.push_back({{"name","torso"},{"from","Hips"},{"to","Chest"},{"radius",.13}});
        size_t hips=names.at("Hips"),chest=names.at("Chest");
        for(const std::string side:{"Left","Right"}){
            auto a=names.at(side+"ForeArm"),b=names.at(side+"Hand");std::vector<float> v(clip.frames());
            for(size_t f=0;f<clip.frames();++f)v[f]=std::max(0.0f,.165f-distance(world[f][a].p,world[f][b].p,world[f][hips].p,world[f][chest].p));
            series("forearm-torso-proxy-overlap",{side+"ForeArm",side+"Hand","Hips","Chest"},"metres",overlap,"SOMA forearm capsule radius .035 m against torso capsule radius .13 m; not skin collision",v);
            proxies.push_back({{"name",side+"ForeArm"},{"from",side+"ForeArm"},{"to",side+"Hand"},{"radius",.035}});
            for(const std::string suffix:{"Foot","ToeBase"}){size_t j=names.at(side+suffix);std::vector<float> depth(clip.frames()),drift(clip.frames());
                for(size_t f=0;f<clip.frames();++f){depth[f]=std::max(0.0f,ground-world[f][j].p.y);if(f&&world[f][j].p.y<=ground+near&&world[f-1][j].p.y<=ground+near){Vec3 d=world[f][j].p-world[f-1][j].p;drift[f]=std::hypot(d.x,d.z)*clip.fps;}}
                series("foot-joint-below-plane",{side+suffix},"metres",below,"Joint centre below configured plane; not sole/mesh penetration",depth);
                series("low-foot-horizontal-motion",{side+suffix},"metres/second",speed,"Adjacent joint centres below configured height; no planted-contact inference",drift);
            }
        }
    }
    Json propTracks=Json::array();
    if(settings.contains("props")){
        const auto& props=settings.at("props");if(!props.is_array()||props.size()>16)throw std::runtime_error("Review props must be an array of at most sixteen explicit capsule proxies");std::set<std::string> propNames;
        for(const auto& p:props){if(!p.is_object())throw std::runtime_error("Invalid review prop");for(auto it=p.begin();it!=p.end();++it)if(it.key()!="name"&&it.key()!="bone"&&it.key()!="start"&&it.key()!="end"&&it.key()!="radius")throw std::runtime_error("Unknown review prop field: "+it.key());
            std::string name=p.at("name").get<std::string>(),bone=p.at("bone").get<std::string>();if(name.empty()||!propNames.insert(name).second||!names.contains(bone))throw std::runtime_error("Review prop needs a unique name and an existing bone");Vec3 start=vector3(p.at("start")),end=vector3(p.at("end"));
            if(!p.at("radius").is_number()||!std::isfinite(p.at("radius").get<double>())||p.at("radius").get<double>()<=0||p.at("radius").get<double>()>2)throw std::runtime_error("Review prop radius must be in (0,2] metres");
            float r=p.at("radius").get<float>();
            Json track=Json::array();std::vector<float> v(clip.frames());for(size_t f=0;f<clip.frames();++f){const auto& t=world[f][names.at(bone)];Vec3 a=t.p+rotate(t.q,start),b=t.p+rotate(t.q,end);track.push_back({xyz(a),xyz(b)});if(canonical)v[f]=std::max(0.0f,r+.13f-distance(a,b,world[f][names.at("Hips")].p,world[f][names.at("Chest")].p));}
            if(canonical)series("prop-torso-proxy-overlap",{name,"Hips","Chest"},"metres",overlap,"Explicit bone-local prop capsule against SOMA torso capsule; sampled frames only, not mesh collision",v);
            propTracks.push_back({{"name",name},{"bone",bone},{"radius",r},{"positions",track}});
        }
    }
    return {{"schemaVersion",1},{"purpose","advisory-observations"},{"frames",clip.frames()},{"fps",clip.fps},{"rig",canonical?"soma30":"custom"},{"settings",{{"groundY",ground},{"rotationStepDegrees",angle},{"belowGroundMetres",below},{"nearGroundMetres",near},{"lowFootSpeed",speed},{"overlapMetres",overlap}}},{"coverage",{{"localRotationSteps","sampled"},{"bodyAndGroundProxies",canonical?"sampled":"not-evaluated: canonical SOMA required"},{"propTorsoProxies",canonical&&!propTracks.empty()?"sampled":"not-evaluated"},{"anatomicalReachability","not-evaluated: no calibrated joint limits or target contract"},{"meshCollision","not-evaluated"},{"continuousCollision","not-evaluated"},{"physicalBalance","not-evaluated"},{"aestheticQuality","not-evaluated"}}},{"observations",findings},{"joints",joints},{"positions",positions},{"bodyProxies",proxies},{"props",propTracks}};
}
std::string review_html(const nlohmann::json& bundle) {
    std::string data=bundle.dump(-1,' ',false,nlohmann::json::error_handler_t::replace),safe;safe.reserve(data.size());
    // JSON lives in a script data element: escape '<' to prevent closing-script injection.
    for(char c:data){if(c=='<')safe+="\\u003c";else safe+=c;}
    std::string html=
#include "review_page.inc"
    ;const std::string marker="__TIDE_REVIEW_DATA__";auto at=html.find(marker);html.replace(at,marker.size(),safe);return html;
}
}
