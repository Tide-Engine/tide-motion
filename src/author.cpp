#include "author.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
namespace motion {
namespace {
using Json=nlohmann::json;
void fields(const Json& item,std::set<std::string> keys) {
    if(!item.is_object())throw std::runtime_error("Pose sequence entries must be objects");
    for(const auto& field:item.items())if(!keys.contains(field.key()))throw std::runtime_error("Unknown pose sequence field: "+field.key());
}
double numeric(const Json& j,double lo,double hi) {
    if(!j.is_number())throw std::runtime_error("Pose sequence values must be numeric");
    double v=j.get<double>();
    if(!std::isfinite(v)||v<lo||v>hi)throw std::runtime_error("Pose sequence value out of range");
    return v;
}
int whole(const Json& j,int lo,int hi) {
    double v=numeric(j,lo,hi);if(v!=std::floor(v))throw std::runtime_error("Pose sequence frame/tokens must be an integer");return static_cast<int>(v);
}
}
PoseSequence load_pose_sequence(const fs::path& path) {
    if(fs::file_size(path)>1024*1024)throw std::runtime_error("Pose sequence exceeds 1 MiB");
    std::ifstream stream(path);Json doc;stream>>doc;
    fields(doc,{"schema","start","targets"});
    if(doc.at("schema")!="tide-motion-pose-sequence-1")throw std::runtime_error("Unknown pose sequence schema");
    const auto& targets=doc.at("targets");
    if(!targets.is_array()||targets.empty()||targets.size()>128)throw std::runtime_error("Pose sequence requires 1..128 targets");
    PoseSequence plan;plan.recipe=doc;size_t end_frame=0;
    auto read=[&](const Json& item,bool start) {
        fields(item,start?std::set<std::string>{"file","clip","frame","position","yaw","hold"}:std::set<std::string>{"file","clip","frame","position","yaw","hold","tokens","startFrame","endFrame","seconds"});
        auto file=utf8path(item.at("file").get<std::string>());
        if(file.empty())throw std::runtime_error("Empty pose sequence source");
        file=fs::absolute(file.is_absolute()?file:path.parent_path()/file).lexically_normal();
        auto source=load_glb(file,item.value("clip",std::string{}),true);require_soma(source);
        int index=whole(item.value("frame",Json(start?-1:0)),-18000,17999);
        if(index<0)index+=static_cast<int>(source.frames());
        if(index<0||index>=static_cast<int>(source.frames()))throw std::runtime_error("Pose sequence source frame outside clip");
        bool hold=item.value("hold",false);
        const auto& p=item.at("position");
        if(!p.is_array()||p.size()!=3)throw std::runtime_error("position requires root X Y Z in metres");
        Vec3 position{static_cast<float>(numeric(p[0],-10000,10000)),static_cast<float>(numeric(p[1],-10000,10000)),static_cast<float>(numeric(p[2],-10000,10000))};
        float degrees=static_cast<float>(numeric(item.at("yaw"),-3600,3600));
        auto anchor=sample(source,static_cast<float>(index)/30);
        Quat turn=yaw(degrees*0.017453292519943295f-kimodo_heading(source,anchor));
        Clip window;window.joints=source.joints;
        for(int f=0;f<4;++f) {
            auto pose=sample(source,static_cast<float>(index+(hold?0:(start?f-3:f)))/30);
            pose[0].p=position+rotate(turn,pose[0].p-anchor[0].p);
            pose[0].q=normalized(turn*pose[0].q);
            window.poses.insert(window.poses.end(),pose.begin(),pose.end());
        }
        window.validate();
        Json effective=item;effective["file"]=file.string();effective["frame"]=index;effective["clip"]=item.value("clip",std::string{});effective["hold"]=hold;
        effective["position"]={position.x,position.y,position.z};effective["yaw"]=degrees;
        return std::pair{window,effective};
    };
    auto [start,effective]=read(doc.at("start"),true);plan.start=std::move(start);plan.recipe["start"]=effective;
    for(size_t i=0;i<targets.size();++i) {
        auto [window,target]=read(targets[i],false);
        int tokens=whole(targets[i].at("tokens"),6,16);
        size_t intervals=static_cast<size_t>(4*tokens-7);
        target["tokens"]=tokens;target["startFrame"]=end_frame;end_frame+=intervals;target["endFrame"]=end_frame;target["seconds"]=static_cast<double>(intervals)/30;
        plan.targets.push_back(std::move(window));plan.tokens.push_back(tokens);plan.recipe["targets"][i]=target;
    }
    if(end_frame+1>18000)throw std::runtime_error("Pose sequence exceeds 18000 frames");
    return plan;
}
void append_pose_bridge(Clip& result,const Clip& bridge) {
    require_soma(result);require_soma(bridge);
    if(bridge.frames()<4||result.frames()+bridge.frames()-1>18000)throw std::runtime_error("Invalid pose sequence bridge length");
    auto a=result.frame(result.frames()-1);auto b=bridge.frame(0);
    for(size_t j=0;j<30;++j)if(a[j].p.x!=b[j].p.x||a[j].p.y!=b[j].p.y||a[j].p.z!=b[j].p.z||a[j].q.x!=b[j].q.x||a[j].q.y!=b[j].q.y||a[j].q.z!=b[j].q.z||a[j].q.w!=b[j].q.w)throw std::runtime_error("Pose sequence bridge seam is not exact");
    result.poses.insert(result.poses.end(),bridge.poses.begin()+30,bridge.poses.end());
}
Clip generate_pose_sequence(const PoseSequence& plan,BricksModel& model) {
    Clip context=plan.start,result;result.joints=context.joints;
    auto first=context.frame(3);result.poses.assign(first.begin(),first.end());
    for(size_t i=0;i<plan.targets.size();++i) {
        auto bridge=model.generate(context,plan.targets[i],plan.tokens[i]);
        // Sampling normalizes quaternion signs/lengths; retain the already exported
        // endpoint bit for bit instead of introducing that numerical residue.
        std::copy(result.frame(result.frames()-1).begin(),result.frame(result.frames()-1).end(),bridge.frame(0).begin());
        append_pose_bridge(result,bridge);
        context=std::move(bridge);
    }
    result.validate();return result;
}
}

