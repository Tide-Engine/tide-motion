// Native host loop adapted from zkf1997/DART at bb67ae1ed6ce051080468bf15bc6e54a6c3f8417.
// Apache-2.0, licenses/DART-SOURCE.txt. Uses no parametric body model or mesh assets.
#include "dart.hpp"
#include "dart_gpu.hpp"
#include "clip_text.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <chrono>
#include <stdexcept>
#include <limits>
namespace motion {
namespace {
using Row=std::array<float,276>;
constexpr float pi=3.14159265358979323846f;
const Quat basis=yaw(pi)*Quat{-std::sqrt(.5f),0,0,std::sqrt(.5f)};
constexpr int parents[]={-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19};
constexpr const char* names[]={"pelvis","left_hip","right_hip","spine1","left_knee","right_knee","spine2","left_ankle","right_ankle","spine3","left_foot","right_foot","neck","left_collar","right_collar","head","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist"};
constexpr int slots[]={0,3,6,9,12,-1,15,-1,-1,-1,13,16,18,20,-1,-1,14,17,19,21,-1,-1,1,4,7,10,2,5,8,11};
Vec3 vector(const Row& r,size_t k) {return {r[k],r[k+1],r[k+2]};}
void put(Row& r,size_t k,Vec3 v) {r[k]=v.x;r[k+1]=v.y;r[k+2]=v.z;}
Quat rotation(const Row& r,size_t k) {return inverse(rotation6(vector(r,k),vector(r,k+3)));} // row-6D
void put_rotation(Row& r,size_t k,Quat q) {
    q=inverse(q);put(r,k,rotate(q,{1,0,0}));put(r,k+3,rotate(q,{0,1,0}));
}
bool policy_skill(const std::string& s){return s=="walk"||s=="run"||s=="hop on left leg";}
void keys(const nlohmann::json& j,std::initializer_list<const char*> allowed) {
    if(!j.is_object())throw std::runtime_error("DART expects JSON objects");
    for(const auto& x:j.items())if(std::none_of(allowed.begin(),allowed.end(),[&](const char* k){return x.key()==k;}))throw std::runtime_error("Unknown DART field: "+x.key());
}
unsigned count(const nlohmann::json& j,unsigned lo,unsigned hi) {
    if(!j.is_number_unsigned()&&!j.is_number_integer())throw std::runtime_error("DART counts must be integers");
    auto n=j.get<int64_t>();if(n<lo||n>hi)throw std::runtime_error("DART count outside supported range");return static_cast<unsigned>(n);
}
float scalar(const nlohmann::json& j,float lo,float hi,const char* name){if(!j.is_number())throw std::runtime_error(std::string("DART ")+name+" must be numeric");double v=j.get<double>();if(!std::isfinite(v)||v<lo||v>hi)throw std::runtime_error(std::string("DART ")+name+" outside supported finite range");return static_cast<float>(v);}
DartHistory read_history(const fs::path& path,bool first=false) {
    if(fs::file_size(path)>64*1024*1024)throw std::runtime_error("DART history recipe exceeds 64 MiB");
    std::ifstream f(path);nlohmann::json doc;f>>doc;const auto& j=doc.contains(first?"dartInitialState":"dartState")?doc.at(first?"dartInitialState":"dartState"):doc;
    if(j.at("version")!=1||j.at("representation")!="dart-smplx-276-bb67ae1"||j.at("normalizerSha256")!="cb442539380d15a5763b799ba063d9b0b6574a751a03ae0c86d4a6f24bff5bd4")throw std::runtime_error("Incompatible DART native history");
    DartHistory h;h.rows=j.at("rows").get<decltype(h.rows)>();auto p=j.at("worldPosition").get<std::array<float,3>>();auto q=j.at("worldRotation").get<std::array<float,4>>();h.world={{p[0],p[1],p[2]},normalized(Quat{q[0],q[1],q[2],q[3]})};
    const auto& rest=j.at("referenceOffsets");if(!rest.is_array()||rest.size()!=22)throw std::runtime_error("DART reference requires 22 fixed joints");
    for(size_t k=0;k<22;++k){auto offset=rest[k].get<std::array<float,3>>();for(float v:offset)if(!std::isfinite(v)||std::abs(v)>10)throw std::runtime_error("Invalid DART reference offsets");h.reference.push_back({names[k],parents[k],{{offset[0],offset[1],offset[2]}, {}}});}
    for(const auto& row:h.rows)for(float v:row)if(!std::isfinite(v)||std::abs(v)>1e6)throw std::runtime_error("Invalid DART native history values");
    for(float v:p)if(!std::isfinite(v)||std::abs(v)>1e6)throw std::runtime_error("Invalid DART history position");
    for(float v:q)if(!std::isfinite(v))throw std::runtime_error("Invalid DART history rotation");
    return h;
}
size_t constraint_joint(const std::string& name,bool canonical=false){auto rig=soma();if(canonical){for(size_t i=0;i<30;++i)if(rig[i].name==name)return i;for(size_t j=0;j<22;++j)if(name==names[j])for(size_t i=0;i<30;++i)if(slots[i]==static_cast<int>(j))return i;}else{for(size_t i=0;i<22;++i)if(name==names[i])return i;for(size_t i=0;i<30;++i)if(rig[i].name==name&&slots[i]>=0)return static_cast<size_t>(slots[i]);}throw std::runtime_error("Unknown DART constraint joint; use jointSpace soma for presentation joints: "+name);}
Vec3 point(const nlohmann::json& j,const char* name){if(!j.is_array()||j.size()!=3)throw std::runtime_error(std::string(name)+" must be [x,y,z]");return {scalar(j[0],-1e6,1e6,name),scalar(j[1],-1e6,1e6,name),scalar(j[2],-1e6,1e6,name)};}
}
DartPlan load_dart_plan(const fs::path& path,bool defer) {
    if(fs::file_size(path)>1024*1024)throw std::runtime_error("DART plan exceeds 1 MiB");
    std::ifstream f(path);nlohmann::json j;f>>j;keys(j,{"version","seed","warmupChunks","segments","guidance","eta","noiseScale","zeroNoise","fixFloor","groundHeight","sampler","policyMode","timesteps","history","historyState","source","sourceFrame","sourceMap","sourceClip","seedMode","optimization"});
    if(j.at("version")!=1)throw std::runtime_error("DART plan version must be 1");
    DartPlan p;if(j.contains("seed"))p.seed=count(j["seed"],0,4294967295U);
    if(j.contains("warmupChunks"))p.warmup=count(j["warmupChunks"],j.contains("history")?0:1,32);
    if(j.contains("guidance"))p.guidance=scalar(j["guidance"],0,20,"guidance");
    if(j.contains("eta"))p.eta=scalar(j["eta"],0,1,"eta");
    if(j.contains("noiseScale"))p.noise_scale=scalar(j["noiseScale"],0,10,"noiseScale");
    if(j.contains("groundHeight"))p.ground_height=scalar(j["groundHeight"],-1e6,1e6,"groundHeight");
    p.zero_noise=j.value("zeroNoise",false);p.fix_floor=j.value("fixFloor",false);p.sampler=j.value("sampler",std::string("ddim"));p.policy_mode=j.value("policyMode",std::string("sample"));
    if(p.sampler!="ddim"&&p.sampler!="ddpm")throw std::runtime_error("DART sampler is ddim or ddpm");
    if(p.policy_mode!="sample"&&p.policy_mode!="mean"&&p.policy_mode!="none")throw std::runtime_error("DART policyMode is sample, mean or none");
    p.seed_mode=j.value("seedMode",std::string("history"));if(p.seed_mode!="history"&&p.seed_mode!="repeat")throw std::runtime_error("DART seedMode is history or repeat");
    if(j.contains("timesteps")){p.timesteps=j["timesteps"].get<std::vector<int>>();if(p.timesteps.empty()||p.timesteps.size()>10||p.timesteps.front()<0||p.timesteps.back()>9||!std::is_sorted(p.timesteps.begin(),p.timesteps.end())||std::adjacent_find(p.timesteps.begin(),p.timesteps.end())!=p.timesteps.end())throw std::runtime_error("DART timesteps must increase uniquely within 0..9");}
    auto resolve=[&](const nlohmann::json& v){auto q=utf8path(v.get<std::string>());if(q.empty())throw std::runtime_error("Empty DART input path");return fs::weakly_canonical(fs::absolute(q.is_absolute()?q:path.parent_path()/q));};
    auto state=j.value("historyState",std::string("last"));if(state!="first"&&state!="last")throw std::runtime_error("historyState is first or last");
    if(j.contains("history")){p.history_file=resolve(j["history"]);if(fs::exists(p.history_file))p.history=read_history(p.history_file,state=="first");else if(!defer)throw std::runtime_error("DART history does not exist");}
    if(j.contains("sourceMap")){p.source_map_file=resolve(j["sourceMap"]);(void)load_mapping(p.source_map_file);if(!j.contains("source"))throw std::runtime_error("sourceMap requires source");}
    if(j.contains("source")){p.source_file=resolve(j["source"]);if(fs::exists(p.source_file)){if(p.source_map_file.empty())p.source=load_glb(p.source_file,j.value("sourceClip",std::string{}));else{Clip target;target.joints=soma();for(const auto& bone:target.joints)target.poses.push_back(bone.rest);p.source=retarget(load_glb(p.source_file,j.value("sourceClip",std::string{}),false),target,load_mapping(p.source_map_file),true).clip;}require_soma(*p.source);if(p.source->frames()<(p.seed_mode=="repeat"?1:2))throw std::runtime_error("DART history source needs two frames; seedMode repeat accepts one pose");}else if(!defer)throw std::runtime_error("DART source does not exist");}
    if(!p.history_file.empty()&&!p.source_file.empty())throw std::runtime_error("Choose DART history or source, not both");
    if(j.contains("sourceFrame")){p.source_frame=static_cast<long>(count(j["sourceFrame"],p.seed_mode=="repeat"?0:1,17999));if(p.source_file.empty())throw std::runtime_error("sourceFrame requires source");}
    if(j.contains("sourceClip")&&p.source_file.empty())throw std::runtime_error("sourceClip requires source");
    if(p.source&&p.source_frame>=static_cast<long>(p.source->frames()))throw std::runtime_error("DART sourceFrame is past the clip");
    const auto& segments=j.at("segments");if(!segments.is_array()||segments.empty()||segments.size()>64)throw std::runtime_error("DART needs 1..64 segments");
    unsigned total=0;
    for(const auto& s:segments) {
        keys(s,{"text","chunks","target","guidance"});DartSegment v;v.text=s.at("text").get<std::string>();if(v.text.size()>65536)throw std::runtime_error("DART text exceeds 64 KiB");v.guidance=s.contains("guidance")?scalar(s["guidance"],0,20,"segment guidance"):p.guidance;v.chunks=count(s.at("chunks"),1,512);total+=v.chunks;
        if(s.contains("target")) {
            const auto& t=s["target"];if(!t.is_array()||t.size()!=2)throw std::runtime_error("DART target is [x,z] in metres");
            double x=t[0].get<double>(),z=t[1].get<double>();
            if(!std::isfinite(x)||!std::isfinite(z)||std::abs(x)>100||std::abs(z)>100)throw std::runtime_error("DART target outside +/-100 metres");
            v.goal=Vec3{static_cast<float>(x),0,static_cast<float>(z)};
        }
        p.segments.push_back(v);
    }
    if(total>2249)throw std::runtime_error("DART plan exceeds 18000-frame budget");
    if(j.contains("optimization")) {
        const auto& o=j["optimization"];keys(o,{"iterations","learningRate","unitGradient","anneal","jerkWeight","floorWeight","constraints","target","targetFrame","targetClip","targetMap","targetJoints","targetWeight","scene","checkGradient","jointSpace"});p.optimization=std::make_shared<DartOptimization>();auto& opt=*p.optimization;auto space=o.value("jointSpace",std::string("soma"));if(space!="soma"&&space!="predicted")throw std::runtime_error("DART optimization jointSpace is soma or predicted");opt.soma_joints=space=="soma";
        if(o.contains("iterations"))opt.iterations=count(o["iterations"],0,10000);
        if(o.contains("learningRate"))opt.rate=scalar(o["learningRate"],1e-7f,10,"learningRate");
        opt.unit_gradient=o.value("unitGradient",true);opt.anneal=o.value("anneal",true);
        opt.check_gradient=o.value("checkGradient",false);
        if(o.contains("jerkWeight"))opt.jerk_weight=scalar(o["jerkWeight"],0,1e4,"jerkWeight");
        if(o.contains("floorWeight"))opt.floor_weight=scalar(o["floorWeight"],0,1e4,"floorWeight");
        if(o.contains("constraints")){if(!o["constraints"].is_array()||o["constraints"].size()>4096)throw std::runtime_error("DART supports up to 4096 optimization constraints");for(const auto& c:o["constraints"]){keys(c,{"frame","joint","position","axes","weight"});DartConstraint v;v.frame=count(c.at("frame"),0,1+8*total);v.joint=constraint_joint(c.at("joint"),opt.soma_joints);v.position=point(c.at("position"),"constraint position");if(c.contains("axes"))v.axes=c["axes"].get<std::array<bool,3>>();if(c.contains("weight"))v.weight=scalar(c["weight"],0,1e4,"constraint weight");opt.constraints.push_back(v);}}
        if(o.contains("targetMap")){opt.target_map_file=resolve(o["targetMap"]);(void)load_mapping(opt.target_map_file);}
        if(o.contains("target")){opt.target_file=resolve(o["target"]);if(fs::exists(opt.target_file)){if(opt.target_file.extension()==".json")opt.target_history=read_history(opt.target_file);else{if(opt.target_map_file.empty())opt.target_clip=load_glb(opt.target_file,o.value("targetClip",std::string{}));else{Clip rig;rig.joints=soma();for(const auto& b:rig.joints)rig.poses.push_back(b.rest);opt.target_clip=retarget(load_glb(opt.target_file,o.value("targetClip",std::string{}),false),rig,load_mapping(opt.target_map_file),true).clip;}require_soma(*opt.target_clip);}}else if(!defer)throw std::runtime_error("DART optimization target does not exist");}
        if((o.contains("targetMap")||o.contains("targetClip")||o.contains("targetFrame")||o.contains("targetJoints"))&&!o.contains("target"))throw std::runtime_error("DART target controls require target");
        if(o.contains("targetFrame"))opt.target_frame=static_cast<long>(count(o["targetFrame"],0,17999));
        if(opt.target_clip&&opt.target_frame>=static_cast<long>(opt.target_clip->frames()))throw std::runtime_error("DART targetFrame is past the target clip");
        if(opt.target_history&&opt.target_frame>1)throw std::runtime_error("DART native targetFrame is 0 or 1");
        if(opt.target_history&&(o.contains("targetClip")||o.contains("targetMap")))throw std::runtime_error("Native DART history targets need no clip selection or mapping");
        if(o.contains("targetJoints")){if(!o["targetJoints"].is_array()||o["targetJoints"].size()>30)throw std::runtime_error("DART targetJoints must be an array of up to 30 names");for(const auto& name:o["targetJoints"])opt.target_joints.push_back(constraint_joint(name,opt.soma_joints));}else for(size_t k=0;k<(opt.soma_joints?30U:22U);++k)opt.target_joints.push_back(k);
        if(o.contains("targetWeight"))opt.target_weight=scalar(o["targetWeight"],0,1e4,"targetWeight");
        if(o.contains("scene")){const auto& scene=o["scene"];keys(scene,{"file","margin","weight"});opt.scene=std::make_shared<DartSdf>();auto& sdf=*opt.scene;sdf.file=resolve(scene.at("file"));if(fs::file_size(sdf.file)>256*1024*1024)throw std::runtime_error("DART SDF exceeds 256 MiB");std::ifstream sf(sdf.file);nlohmann::json grid;sf>>grid;keys(grid,{"version","dimensions","origin","spacing","values"});if(grid.at("version")!=1)throw std::runtime_error("DART SDF version must be 1");sdf.dimensions=grid.at("dimensions").get<std::array<size_t,3>>();size_t cells=1;for(auto n:sdf.dimensions){if(n<2||n>256)throw std::runtime_error("DART SDF dimensions must be 2..256");cells*=n;}sdf.origin=point(grid.at("origin"),"SDF origin");sdf.spacing=point(grid.at("spacing"),"SDF spacing");if(sdf.spacing.x<=0||sdf.spacing.y<=0||sdf.spacing.z<=0)throw std::runtime_error("DART SDF spacing must be positive");sdf.values=grid.at("values").get<std::vector<float>>();if(sdf.values.size()!=cells)throw std::runtime_error("DART SDF cell count mismatch");for(float v:sdf.values)if(!std::isfinite(v))throw std::runtime_error("DART SDF values must be finite");if(scene.contains("margin"))sdf.margin=scalar(scene["margin"],0,10,"SDF margin");if(scene.contains("weight"))sdf.weight=scalar(scene["weight"],0,1e4,"SDF weight");}
    }
    p.recipe=j;p.recipe["seed"]=p.seed;p.recipe["warmupChunks"]=p.warmup;
    if(!p.history_file.empty())p.recipe["history"]=p.history_file.string();
    if(!p.source_file.empty())p.recipe["source"]=p.source_file.string();
    if(!p.source_map_file.empty())p.recipe["sourceMap"]=p.source_map_file.string();
    if(p.optimization){if(!p.optimization->target_file.empty())p.recipe["optimization"]["target"]=p.optimization->target_file.string();if(!p.optimization->target_map_file.empty())p.recipe["optimization"]["targetMap"]=p.optimization->target_map_file.string();if(p.optimization->scene)p.recipe["optimization"]["scene"]["file"]=p.optimization->scene->file.string();}
    return p;
}
struct DartModel::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"tide-dart"};
    std::unique_ptr<Ort::Session> den,dec,policy,text;
    fs::path directory;unsigned threads=4;std::string requested;
    std::unique_ptr<DartGpu> gpu;std::map<std::string,std::unique_ptr<Ort::Session>> batches;
    Row mean{},stddev{};std::array<std::vector<float>,3> embeddings;
    std::unique_ptr<ClipTokenizer> tokenizer;std::map<std::string,std::vector<float>> text_cache;
    Impl(const fs::path& path,unsigned count,const std::string& device):directory(path),threads(count?count:4),requested(device) {
        if(device!="auto"&&device!="cpu"&&device!="gpu")throw std::runtime_error("DART device must be auto, cpu or gpu");
        if(device!="cpu"&&(device=="gpu"||dart_gpu_compiled()))try{gpu=std::make_unique<DartGpu>(directory,threads);}catch(const DartGpuFailure& e){if(device=="gpu")throw;std::cerr<<e.what()<<"; DART falls back to CPU\n";}
        Ort::SessionOptions o;o.SetIntraOpNumThreads(threads?static_cast<int>(threads):4);o.SetInterOpNumThreads(1);
        auto load=[&](const char* name){auto p=directory/name;return std::make_unique<Ort::Session>(env,p.c_str(),o);};
        std::cerr<<"Loading DART native ONNX sessions (motion "<<(gpu?"DirectML GPU":"CPU")<<", text CPU)...\n";
        text=load("text.onnx");
        std::ifstream stats(directory/"stats.pkl",std::ios::binary);
        stats.seekg(344);stats.read(reinterpret_cast<char*>(mean.data()),sizeof(mean));
        stats.seekg(1774);stats.read(reinterpret_cast<char*>(stddev.data()),sizeof(stddev));
        if(!stats)throw std::runtime_error("DART statistics truncated");
        for(size_t k=0;k<276;++k)if(!std::isfinite(mean[k])||!std::isfinite(stddev[k])||stddev[k]<=0)throw std::runtime_error("Invalid DART statistics");
        tokenizer=std::make_unique<ClipTokenizer>(directory/"bpe_simple_vocab_16e6.txt.gz");
        for(size_t a=0;a<3;++a)embeddings[a]=embed(std::array<const char*,3>{"walk","run","hop on left leg"}[a]);
    }
    std::vector<float> embed(const std::string& prompt) {
        if(prompt.empty())return std::vector<float>(512);
        if(auto cached=text_cache.find(prompt);cached!=text_cache.end())return cached->second;
        {
            auto tokens=tokenizer->encode(prompt);
            auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);std::array<int64_t,2> shape{1,77};
            auto input=Ort::Value::CreateTensor<int64_t>(memory,tokens.data(),tokens.size(),shape.data(),shape.size());
            const char* in="tokens";const char* out="embedding";
            auto result=text->Run(Ort::RunOptions{nullptr},&in,&input,1,&out,1);
            if(result[0].GetTensorTypeAndShapeInfo().GetElementCount()!=512)throw std::runtime_error("Invalid DART text output shape");
            const float* data=result[0].GetTensorData<float>();std::vector<float> embedding(data,data+512);
            for(float v:embedding)if(!std::isfinite(v))throw std::runtime_error("Nonfinite CLIP text embedding");
            if(text_cache.size()>=4096)text_cache.clear();
            text_cache[prompt]=embedding;return embedding;
        }
    }
    std::vector<float> execute(const std::string& name,DartFeed feed,const char* output) {
        size_t batch=static_cast<size_t>(feed.begin()->second.shape.at(0));std::vector<float> result;
        if(gpu)result=gpu->run(name,std::move(feed),output);
        else {
            if(!batches.contains(name)) {Ort::SessionOptions o;o.SetIntraOpNumThreads(static_cast<int>(threads));o.SetInterOpNumThreads(1);batches[name]=std::make_unique<Ort::Session>(env,(directory/(name+"-batch.onnx")).c_str(),o);}
            auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);std::vector<const char*> names;std::vector<Ort::Value> values;
            for(auto& [key,t]:feed){names.push_back(key.c_str());if(t.i.empty())values.push_back(Ort::Value::CreateTensor<float>(memory,t.f.data(),t.f.size(),t.shape.data(),t.shape.size()));else values.push_back(Ort::Value::CreateTensor<int64_t>(memory,t.i.data(),t.i.size(),t.shape.data(),t.shape.size()));}
            auto out=batches.at(name)->Run(Ort::RunOptions{nullptr},names.data(),values.data(),values.size(),&output,1);auto n=out[0].GetTensorTypeAndShapeInfo().GetElementCount();const float* ptr=out[0].GetTensorData<float>();result.assign(ptr,ptr+n);
        }
        size_t width=name=="denoiser"?256:name=="decoder"?2208:name=="decoder-grad"||name=="denoiser-grad"?808:name=="transition"?1092:name=="soma-points"?720:name=="transition-grad"||name=="soma-points-grad"?2220:512;
        if(result.size()!=batch*width)throw std::runtime_error("Invalid DART batch output shape");
        for(float v:result)if(!std::isfinite(v))throw std::runtime_error("Nonfinite DART batch output");
        return result;
    }
    std::vector<float> run(Ort::Session& session,std::initializer_list<const char*> names,std::initializer_list<std::span<const float>> buffers,std::initializer_list<std::vector<int64_t>> shapes,const char* output,std::optional<int64_t> time={}) {
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);std::vector<Ort::Value> values;auto shape=shapes.begin();
        for(auto b:buffers) {values.push_back(Ort::Value::CreateTensor<float>(memory,const_cast<float*>(b.data()),b.size(),shape->data(),shape->size()));++shape;}
        std::vector<const char*> inputs(names);
        if(time) {const int64_t one=1;values.insert(values.begin()+1,Ort::Value::CreateTensor<int64_t>(memory,&*time,1,&one,1));}
        auto result=session.Run(Ort::RunOptions{nullptr},inputs.data(),values.data(),values.size(),&output,1);
        auto size=result[0].GetTensorTypeAndShapeInfo().GetElementCount();size_t expected=std::string_view(output)=="future"?2208:std::string_view(output)=="x0"?256:512;
        if(size!=expected)throw std::runtime_error("Invalid DART output shape");
        const auto* data=result[0].GetTensorData<float>();std::vector<float> out(data,data+size);
        for(float x:out)if(!std::isfinite(x))throw std::runtime_error("Nonfinite DART inference output");
        return out;
    }
};
DartModel::DartModel(const fs::path& p,unsigned t,const std::string& d):impl_(std::make_unique<Impl>(p,t,d)){}
std::string DartModel::device() const{return impl_->gpu?"gpu":"cpu";}
DartModel::~DartModel()=default;
double DartModel::verify_gradient_reference(const fs::path& path) {
    std::ifstream f(path);nlohmann::json fixtures;f>>fixtures;double worst=0;
    for(const auto& row:fixtures){DartFeed feed;for(const auto& item:row.at("inputs").items()){const auto& value=item.value();DartTensor t;t.shape=value.at("shape").get<std::vector<int64_t>>();if(value.at("integer").get<bool>())t.i=value.at("data").get<std::vector<int64_t>>();else t.f=value.at("data").get<std::vector<float>>();feed[item.key()]=std::move(t);}
        auto expected=row.at("output").get<std::vector<float>>();auto actual=impl_->execute(row.at("name"),std::move(feed),"output");if(actual.size()!=expected.size())throw std::runtime_error("DART gradient fixture size mismatch");double error=0;
        for(size_t i=0;i<actual.size();++i){auto e=std::abs(double(actual[i])-expected[i]);error=std::max(error,e);if(e>2e-4+2e-4*std::abs(expected[i]))throw std::runtime_error("DART native gradient parity failed");}
        std::cout<<"DART "<<row.at("name")<<" max gradient reference error "<<error<<'\n';worst=std::max(worst,error);
    }return worst;
}
double DartModel::verify_batch_reference(const fs::path& directory) {
    double worst=0;
    for(const auto* name:{"denoiser","decoder","policy"}) {
        std::ifstream file(directory/(std::string(name)+"-batch-reference.json"));nlohmann::json fixtures;file>>fixtures;
        for(const auto& fixture:fixtures) {
            int64_t n=fixture.at("batch");const auto& input=fixture.at("inputs");DartFeed feed;const char* output=nullptr;
            if(std::string_view(name)=="denoiser"){feed={{"noise",{{n,1,256},input[0].get<std::vector<float>>(),{}}},{"timestep",{{n},{},input[1].get<std::vector<int64_t>>()}},{"history",{{n,2,276},input[2].get<std::vector<float>>(),{}}},{"text",{{n,512},input[3].get<std::vector<float>>(),{}}}};output="x0";}
            else if(std::string_view(name)=="decoder"){feed={{"latent",{{n,1,256},input[0].get<std::vector<float>>(),{}}},{"history",{{n,2,276},input[1].get<std::vector<float>>(),{}}}};output="future";}
            else {feed={{"observation",{{n,1069},input[0].get<std::vector<float>>(),{}}}};output="distribution";}
            auto actual=impl_->execute(name,std::move(feed),output);auto expected=fixture.at("output").get<std::vector<float>>();
            if(actual.size()!=expected.size())throw std::runtime_error("DART batch fixture size mismatch");
            double error=0;for(size_t i=0;i<actual.size();++i)error=std::max(error,std::abs(double(actual[i])-expected[i]));
            std::cout<<"DART "<<name<<" batch "<<n<<" reference error "<<error<<'\n';worst=std::max(worst,error);
        }
    }
    if(worst>2e-4)throw std::runtime_error("DART batched native/PyTorch parity failed");
    return worst;
}
double DartModel::verify_reference(const fs::path& path) {
    if(fs::file_size(path)>1024*1024)throw std::runtime_error("DART reference exceeds 1 MiB");
    std::ifstream f(path);nlohmann::json doc;f>>doc;double worst=0;std::set<std::string> checked;
    for(const auto& item:doc) {
        const auto name=item.at("name").get<std::string>();std::vector<std::vector<float>> inputs;
        if(!checked.insert(name).second)throw std::runtime_error("Duplicate DART reference graph");
        for(const auto& input:item.at("inputs"))inputs.push_back(input.get<std::vector<float>>());
        auto expected=item.at("output").get<std::vector<float>>();std::vector<float> output;auto& m=*impl_;
        if(!m.den){Ort::SessionOptions o;o.SetIntraOpNumThreads(4);m.den=std::make_unique<Ort::Session>(m.env,(m.directory/"denoiser.onnx").c_str(),o);m.dec=std::make_unique<Ort::Session>(m.env,(m.directory/"decoder.onnx").c_str(),o);m.policy=std::make_unique<Ort::Session>(m.env,(m.directory/"policy.onnx").c_str(),o);}
        if(name=="denoiser"&&inputs.size()==3&&inputs[0].size()==256&&inputs[1].size()==552&&inputs[2].size()==512)
            output=m.run(*m.den,{"noise","timestep","history","text"},{inputs[0],inputs[1],inputs[2]},{{1,1,256},{1,2,276},{1,512}},"x0",9);
        else if(name=="decoder"&&inputs.size()==2&&inputs[0].size()==256&&inputs[1].size()==552)
            output=m.run(*m.dec,{"latent","history"},{inputs[0],inputs[1]},{{1,1,256},{1,2,276}},"future");
        else if(name=="policy"&&inputs.size()==1&&inputs[0].size()==1069)
            output=m.run(*m.policy,{"observation"},{inputs[0]},{{1,1069}},"distribution");
        else if(name=="text")output=m.embeddings[0];
        else throw std::runtime_error("Invalid DART reference inputs");
        if(output.size()!=expected.size())throw std::runtime_error("Invalid DART reference output size");
        for(size_t k=0;k<output.size();++k){double error=std::abs(static_cast<double>(output[k])-expected[k]);if(!std::isfinite(error))throw std::runtime_error("Invalid DART reference number");worst=std::max(worst,error);}
    }
    if(checked.size()!=4||worst>2e-4)throw std::runtime_error("DART native/PyTorch tensor parity failed");
    return worst;
}
namespace {
struct Stream {
    const DartPlan& plan;Row mean,stddev;std::array<Row,2> history;
    Vec3 delta;Transform world;std::mt19937 rng;std::normal_distribution<float> normal;
    unsigned warmup,index=0,chunks=0;size_t first_frame=1;
    std::vector<std::vector<float>> prompts;std::vector<float> embedding;
    Clip raw;DartResult result;std::vector<float> observation;
    Stream(const DartPlan& p,const Row& m,const Row& s):plan(p),mean(m),stddev(s),history{m,m},delta(vector(m,144)-vector(m,0)),rng(p.seed),warmup(p.warmup) {
        float floor=0;for(size_t j=0;j<22;++j)floor=std::min(floor,history[0][146+3*j]);world.p.z=-floor+p.ground_height;
        raw.fps=30;raw.name="dart";result.metrics["segments"]=nlohmann::json::array();
        if(p.history){history=p.history->rows;world=p.history->world;raw.joints=p.history->reference;warmup=0;if(p.seed_mode=="repeat"){history[0]=history[1];for(auto& h:history){put(h,135,{});put_rotation(h,138,{});for(size_t k=0;k<22;++k)put(h,210+3*k,{});}}emit(history[0]);emit(history[1]);}
    }
    void advance(const std::array<Row,8>& future) {
        history={future[6],future[7]};Vec3 origin=vector(history[0],144),x=vector(history[0],150)-vector(history[0],147);x.z=0;x=normalized(x);
        Quat q=rotation6(x,cross(Vec3{0,0,1},x)),qi=inverse(q);
        for(auto& h:history) {
            put(h,0,rotate(qi,vector(h,0)+delta-origin)-delta);put_rotation(h,3,qi*rotation(h,3));
            put(h,135,rotate(qi,vector(h,135)));put_rotation(h,138,qi*rotation(h,138)*q);
            for(size_t j=0;j<22;++j){put(h,144+3*j,rotate(qi,vector(h,144+3*j)-origin));put(h,210+3*j,rotate(qi,vector(h,210+3*j)));}
        }
        world=world*Transform{origin,q};
    }
    void import_source() {
        Clip reference;reference.joints=raw.joints;for(const auto& j:reference.joints)reference.poses.push_back(j.rest);
        RetargetMapping map;map.root_scale=1;for(size_t s=0;s<30;++s)if(slots[s]>=0)map.bones[s]=names[slots[s]];
        auto converted=retarget(*plan.source,reference,map).clip;size_t last=plan.source_frame<0?converted.frames()-1:static_cast<size_t>(plan.source_frame);
        std::array<Row,3> rows{};
        for(size_t f=0;f<3;++f){size_t sample_index=plan.seed_mode=="repeat"?last:f==0?last-1:f==1?last:std::min(last+1,converted.frames()-1);auto pose=std::vector<Transform>(converted.frame(sample_index).begin(),converted.frame(sample_index).end());pose[0].p.x=plan.source->frame(sample_index)[0].p.x;pose[0].p.z=plan.source->frame(sample_index)[0].p.z;auto joints=world_pose(converted,pose);
            for(size_t k=0;k<22;++k){put(rows[f],144+3*k,rotate(inverse(basis),joints[k].p));put_rotation(rows[f],3+6*k,k?pose[k].q:inverse(basis)*pose[k].q);}put(rows[f],0,vector(rows[f],144)-delta);
        }
        if(last+1==converted.frames()){put(rows[2],0,vector(rows[1],0)*2-vector(rows[0],0));for(size_t k=0;k<22;++k)put(rows[2],144+3*k,vector(rows[1],144+3*k)*2-vector(rows[0],144+3*k));put_rotation(rows[2],3,rotation(rows[1],3)*inverse(rotation(rows[0],3))*rotation(rows[1],3));}
        for(size_t f=0;f<2;++f){put(rows[f],135,vector(rows[f+1],0)-vector(rows[f],0));put_rotation(rows[f],138,rotation(rows[f+1],3)*inverse(rotation(rows[f],3)));for(size_t k=0;k<22;++k)put(rows[f],210+3*k,vector(rows[f+1],144+3*k)-vector(rows[f],144+3*k));}
        world=Transform{};std::array<Row,8> future{};future[6]=rows[0];future[7]=rows[1];advance(future);raw.poses.clear();emit(history[0]);emit(history[1]);
        result.metrics["sourceAdapter"]={{"kind","explicit SOMA-to-DART retarget and FK"},{"sourceFrame",last},{"extrapolatedLastDerivatives",last+1==converted.frames()},{"calibration","fixed native generated reference, no SMPL body assets"}};
    }
    void emit(const Row& r) {
        std::array<Quat,22> global;
        for(size_t j=0;j<22;++j)global[j]=j?global[static_cast<size_t>(parents[j])]*rotation(r,3+6*j):basis*world.q*rotation(r,3);
        if(raw.joints.empty())for(size_t j=0;j<22;++j) {
            Vec3 offset{};if(j)offset=rotate(inverse(global[static_cast<size_t>(parents[j])]),rotate(basis*world.q,vector(r,144+3*j)-vector(r,144+3*static_cast<size_t>(parents[j]))));
            raw.joints.push_back({names[j],parents[j],{offset,{}}});
        }
        for(size_t j=0;j<22;++j) {
            auto pose=raw.joints[j].rest;pose.q=j?rotation(r,3+6*j):global[0];
            if(!j)pose.p=rotate(basis,world.p+rotate(world.q,vector(r,144)));
            raw.poses.push_back(pose);
        }
    }
    nlohmann::json saved_history() const {
        nlohmann::json offsets=nlohmann::json::array();for(const auto& j:raw.joints)offsets.push_back({j.rest.p.x,j.rest.p.y,j.rest.p.z});
        return {{"version",1},{"representation","dart-smplx-276-bb67ae1"},{"normalizerSha256","cb442539380d15a5763b799ba063d9b0b6574a751a03ae0c86d4a6f24bff5bd4"},{"rows",history},{"worldPosition",{world.p.x,world.p.y,world.p.z}},{"worldRotation",{world.q.x,world.q.y,world.q.z,world.q.w}},{"referenceOffsets",offsets}};
    }
    bool prepare() {
        observation.clear();if(warmup)return true;
        if(result.initial_history.is_null())result.initial_history=saved_history();
        while(index<plan.segments.size()) {
            const auto& segment=plan.segments[index];embedding=prompts[index];float error=0;
            Vec3 pelvis=world.p+rotate(world.q,vector(history[1],144)),direction{};
            if(segment.goal){direction=rotate(inverse(basis),*segment.goal)-pelvis;direction.z=0;error=std::sqrt(dot(direction,direction));}
            if(chunks>=segment.chunks||(segment.goal&&error<.3f)) {
                nlohmann::json metric={{"text",segment.text},{"chunks",chunks},{"firstFrame",first_frame},{"lastFrame",raw.frames()-1}};
                if(segment.goal){metric["target"]={segment.goal->x,segment.goal->z};metric["modelGoalErrorMetres"]=error;metric["reached"]=error<.3f;metric["policyTrainingSkill"]=policy_skill(segment.text);}
                result.metrics["segments"].push_back(metric);++index;chunks=0;first_frame=raw.frames()-1;continue;
            }
            if(segment.goal&&plan.policy_mode!="none") {
                direction=direction*(1/std::max(error,1e-12f));Vec3 x=rotate(world.q,vector(history[1],150)-vector(history[1],147));x.z=0;x=normalized(x);Vec3 forward=cross(Vec3{0,0,1},x);
                float angle=std::acos(std::clamp(dot(direction,forward),.5f,1.0f));float side=cross(forward,direction).z;if(side<0)angle=-angle;else if(side==0)angle=0;
                Quat turn{0,0,std::sin(angle/2),std::cos(angle/2)};Vec3 local=rotate(inverse(world.q),rotate(turn,forward));
                observation={local.x,local.y,local.z,std::min(error,5.0f)};observation.insert(observation.end(),embedding.begin(),embedding.end());
                for(const auto& h:history)observation.insert(observation.end(),h.begin(),h.end());
                observation.push_back(plan.ground_height-(world.p+rotate(world.q,vector(history[0],144))).z);
            }
            return true;
        }
        return false;
    }
    void accept(std::span<const float> decoded) {
        std::array<Row,8> future;for(size_t f=0;f<8;++f)for(size_t k=0;k<276;++k)future[f][k]=decoded[f*276+k]*stddev[k]+mean[k];
        if(warmup) {
            advance(future);if(--warmup==0){auto first=world*Transform{vector(history[1],144),{}};world.p.x-=first.p.x;world.p.y-=first.p.y;emit(history[0]);emit(history[1]);if(plan.source)import_source();}
        } else {for(const auto& r:future)emit(r);advance(future);++chunks;}
    }
    DartResult finish() {
        Clip target;target.joints=soma();for(const auto& j:target.joints)target.poses.push_back(j.rest);
        RetargetMapping mapping;for(size_t s=0;s<30;++s)if(slots[s]>=0)mapping.bones[s]=names[slots[s]];
        mapping.root_scale=1;result.clip=retarget(raw,target,mapping,true).clip;
        for(size_t f=0;f<raw.frames();++f){result.clip.frame(f)[0].p.x=raw.frame(f)[0].p.x;result.clip.frame(f)[0].p.z=raw.frame(f)[0].p.z;}
        if(plan.fix_floor)for(size_t f=0;f<result.clip.frames();++f){auto pose=world_pose(result.clip,result.clip.frame(f));float floor=pose[0].p.y;for(const auto& joint:pose)floor=std::min(floor,joint.p.y);result.clip.frame(f)[0].p.y+=plan.ground_height-floor;}
        require_soma(result.clip);result.metrics["initialization"]="normalized model mean, native walk warmup";
        if(plan.history)result.metrics["initialization"]="saved native DART history";
        else if(plan.source)result.metrics["initialization"]="owner source clip adapted through fixed generated DART reference";
        result.history=saved_history();
        result.metrics["sourceReference"]="fixed offsets recovered from first generated pose and predicted joints";return std::move(result);
    }
};
}
DartResult DartModel::generate(const DartPlan& plan){return generate(std::span<const DartPlan>(&plan,1)).front();}
std::vector<DartResult> DartModel::generate(std::span<const DartPlan> plans) {
    if(plans.empty()||plans.size()>8)throw std::runtime_error("DART neural batch needs 1..8 independent plans");
    if(std::any_of(plans.begin(),plans.end(),[](const auto& p){return bool(p.optimization);})) {std::vector<DartResult> results;for(const auto& p:plans)results.push_back(p.optimization?optimize(p):generate(p));return results;}
    auto started=std::chrono::steady_clock::now();auto& m=*impl_;
    try {
        std::vector<Stream> streams;streams.reserve(plans.size());for(const auto& p:plans){streams.emplace_back(p,m.mean,m.stddev);auto& stream=streams.back();stream.embedding=m.embeddings[0];for(const auto& segment:p.segments)stream.prompts.push_back(m.embed(segment.text));}
        const int64_t n=static_cast<int64_t>(plans.size());const size_t size=plans.size();
        std::array<float,10> alpha{};double cumulative=1;
        auto abar=[](double t){double a=std::cos((t+.008)/1.008*3.14159265358979323846/2);return a*a;};
        for(size_t t=0;t<10;++t){cumulative*=1-std::min(1-abar((t+1)/10.0)/abar(t/10.0),.999);alpha[t]=static_cast<float>(cumulative);}
        size_t waves=0,denoiser_calls=0;
        while(true) {
            std::vector<bool> active(size);bool any=false,controlled=false;
            std::vector<float> history(size*552),noise(size*256),observations(size*1069),text(size*1024);
            for(size_t b=0;b<size;++b) {
                auto& s=streams[b];active[b]=s.prepare();if(!active[b])continue;any=true;
                for(size_t f=0;f<2;++f)for(size_t k=0;k<276;++k)history[b*552+f*276+k]=(s.history[f][k]-m.mean[k])/m.stddev[k];
                std::copy(s.embedding.begin(),s.embedding.end(),text.begin()+static_cast<ptrdiff_t>((size+b)*512));
                if(!s.observation.empty()){controlled=true;std::copy(s.observation.begin(),s.observation.end(),observations.begin()+static_cast<ptrdiff_t>(b*1069));}
            }
            if(!any)break;
            ++waves;std::vector<float> distribution;
            if(controlled)distribution=m.execute("policy",{{"observation",{{n,1069},std::move(observations),{}}}},"distribution");
            for(size_t b=0;b<size;++b)if(active[b])for(size_t k=0;k<256;++k){auto& s=streams[b];noise[b*256+k]=s.observation.empty()?(s.plan.zero_noise?0:s.normal(s.rng)):distribution[b*512+k]+(s.plan.policy_mode=="mean"?0:distribution[b*512+256+k]*s.normal(s.rng));noise[b*256+k]*=s.plan.noise_scale;}
            std::vector<float> histories=history;histories.insert(histories.end(),history.begin(),history.end());
            for(int t=9;t>=0;--t) {
                bool participating=false;for(size_t b=0;b<size;++b)if(active[b]&&std::find(streams[b].plan.timesteps.begin(),streams[b].plan.timesteps.end(),t)!=streams[b].plan.timesteps.end())participating=true;
                if(!participating)continue;
                ++denoiser_calls;
                std::vector<float> both=noise;both.insert(both.end(),noise.begin(),noise.end());
                auto output=m.execute("denoiser",{{"noise",{{2*n,1,256},std::move(both),{}}},{"timestep",{{2*n},{},std::vector<int64_t>(2*size,t)}},{"history",{{2*n,2,276},histories,{}}},{"text",{{2*n,512},text,{}}}},"x0");
                for(size_t b=0;b<size;++b)if(active[b]) {
                    auto& s=streams[b];auto step=std::find(s.plan.timesteps.begin(),s.plan.timesteps.end(),t);if(step==s.plan.timesteps.end())continue;
                    bool last=step==s.plan.timesteps.begin();float previous=last?1:alpha[static_cast<size_t>(*(step-1))],current=alpha[static_cast<size_t>(t)];
                    float guidance=s.warmup?s.plan.guidance:s.plan.segments[s.index].guidance;if(!s.warmup&&s.plan.segments[s.index].text.empty())guidance=0;
                    for(size_t k=0;k<256;++k){size_t i=b*256+k;float x0=output[i]+guidance*(output[size*256+i]-output[i]);
                        if(s.plan.sampler=="ddpm") {float beta=1-current/previous;float mean=std::sqrt(previous)*beta/(1-current)*x0+std::sqrt(current/previous)*(1-previous)/(1-current)*noise[i];noise[i]=mean+(last?0:std::sqrt(beta)*s.normal(s.rng));}
                        else if(s.plan.eta>0) {float sigma=s.plan.eta*std::sqrt((1-previous)/(1-current)*(1-current/previous));float eps=(noise[i]-std::sqrt(current)*x0)/std::sqrt(1-current);noise[i]=std::sqrt(previous)*x0+std::sqrt(std::max(0.f,1-previous-sigma*sigma))*eps+(last?0:sigma*s.normal(s.rng));}
                        else {float a=std::sqrt((1-previous)/(1-current));float coefficient=std::sqrt(previous)-a*std::sqrt(current);noise[i]=a*noise[i]+coefficient*x0;}
                    }
                }
            }
            auto decoded=m.execute("decoder",{{"latent",{{n,1,256},std::move(noise),{}}},{"history",{{n,2,276},std::move(history),{}}}},"future");
            for(size_t b=0;b<size;++b)if(active[b])streams[b].accept(std::span<const float>(decoded).subspan(b*2208,2208));
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();std::vector<DartResult> results;
        for(auto& stream:streams){auto result=stream.finish();result.metrics["seconds"]=seconds;result.metrics["neuralBatch"]={{"requests",size},{"primitiveWaves",waves},{"guidanceBranchesPerCall",2*size},{"denoiserCalls",denoiser_calls}};result.metrics["device"]=device();result.metrics["gpuPeakLocalBytes"]=m.gpu?m.gpu->peak_memory_bytes():0;results.push_back(std::move(result));}
        return results;
    }catch(const DartGpuFailure& e) {
        if(m.requested!="auto")throw;
        std::cerr<<e.what()<<"; restarting the entire DART group on CPU from its seeds\n";m.gpu.reset();return generate(plans);
    }
}
#include "dart_optimize.inc"
}
