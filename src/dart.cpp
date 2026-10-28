// Native host loop adapted from zkf1997/DART at bb67ae1ed6ce051080468bf15bc6e54a6c3f8417.
// Apache-2.0, licenses/DART-SOURCE.txt. Uses no parametric body model or mesh assets.
#include "dart.hpp"
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
unsigned action(const std::string& s) {
    if(s=="walk")return 0;
    if(s=="run")return 1;
    if(s=="hop on left leg")return 2;
    throw std::runtime_error("DART text must be walk, run or hop on left leg (released policy skills)");
}
void keys(const nlohmann::json& j,std::initializer_list<const char*> allowed) {
    if(!j.is_object())throw std::runtime_error("DART expects JSON objects");
    for(const auto& x:j.items())if(std::none_of(allowed.begin(),allowed.end(),[&](const char* k){return x.key()==k;}))throw std::runtime_error("Unknown DART field: "+x.key());
}
unsigned count(const nlohmann::json& j,unsigned lo,unsigned hi) {
    if(!j.is_number_unsigned()&&!j.is_number_integer())throw std::runtime_error("DART counts must be integers");
    auto n=j.get<int64_t>();if(n<lo||n>hi)throw std::runtime_error("DART count outside supported range");return static_cast<unsigned>(n);
}
}
DartPlan load_dart_plan(const fs::path& path) {
    if(fs::file_size(path)>1024*1024)throw std::runtime_error("DART plan exceeds 1 MiB");
    std::ifstream f(path);nlohmann::json j;f>>j;keys(j,{"version","seed","warmupChunks","segments"});
    if(j.at("version")!=1)throw std::runtime_error("DART plan version must be 1");
    DartPlan p;if(j.contains("seed"))p.seed=count(j["seed"],0,4294967295U);
    if(j.contains("warmupChunks"))p.warmup=count(j["warmupChunks"],1,32);
    const auto& segments=j.at("segments");if(!segments.is_array()||segments.empty()||segments.size()>64)throw std::runtime_error("DART needs 1..64 segments");
    unsigned total=0;
    for(const auto& s:segments) {
        keys(s,{"text","chunks","target"});DartSegment v;v.text=s.at("text").get<std::string>();(void)action(v.text);v.chunks=count(s.at("chunks"),1,512);total+=v.chunks;
        if(s.contains("target")) {
            const auto& t=s["target"];if(!t.is_array()||t.size()!=2)throw std::runtime_error("DART target is [x,z] in metres");
            double x=t[0].get<double>(),z=t[1].get<double>();
            if(!std::isfinite(x)||!std::isfinite(z)||std::abs(x)>100||std::abs(z)>100)throw std::runtime_error("DART target outside +/-100 metres");
            v.goal=Vec3{static_cast<float>(x),0,static_cast<float>(z)};
        }
        p.segments.push_back(v);
    }
    if(total>2249)throw std::runtime_error("DART plan exceeds 18000-frame budget");
    p.recipe=j;p.recipe["seed"]=p.seed;p.recipe["warmupChunks"]=p.warmup;return p;
}
struct DartModel::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"tide-dart"};
    std::unique_ptr<Ort::Session> den,dec,policy,text;
    Row mean{},stddev{};std::array<std::vector<float>,3> embeddings;
    Impl(const fs::path& directory,unsigned threads) {
        Ort::SessionOptions o;o.SetIntraOpNumThreads(threads?static_cast<int>(threads):4);o.SetInterOpNumThreads(1);
        auto load=[&](const char* name){auto p=directory/name;return std::make_unique<Ort::Session>(env,p.c_str(),o);};
        std::cerr<<"Loading DART native ONNX sessions (CPU)...\n";
        den=load("denoiser.onnx");dec=load("decoder.onnx");policy=load("policy.onnx");text=load("text.onnx");
        std::ifstream stats(directory/"stats.pkl",std::ios::binary);
        stats.seekg(344);stats.read(reinterpret_cast<char*>(mean.data()),sizeof(mean));
        stats.seekg(1774);stats.read(reinterpret_cast<char*>(stddev.data()),sizeof(stddev));
        if(!stats)throw std::runtime_error("DART statistics truncated");
        for(size_t k=0;k<276;++k)if(!std::isfinite(mean[k])||!std::isfinite(stddev[k])||stddev[k]<=0)throw std::runtime_error("Invalid DART statistics");
        constexpr int64_t ids[3][6]={{49406,2374,49407,0,0,0},{49406,1934,49407,0,0,0},{49406,5833,525,1823,4924,49407}};
        for(size_t a=0;a<3;++a) {
            std::array<int64_t,77> tokens{};std::copy_n(ids[a],6,tokens.begin());
            auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);std::array<int64_t,2> shape{1,77};
            auto input=Ort::Value::CreateTensor<int64_t>(memory,tokens.data(),tokens.size(),shape.data(),shape.size());
            const char* in="tokens";const char* out="embedding";
            auto result=text->Run(Ort::RunOptions{nullptr},&in,&input,1,&out,1);
            if(result[0].GetTensorTypeAndShapeInfo().GetElementCount()!=512)throw std::runtime_error("Invalid DART text output shape");
            const float* data=result[0].GetTensorData<float>();embeddings[a].assign(data,data+512);
        }
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
DartModel::DartModel(const fs::path& p,unsigned t):impl_(std::make_unique<Impl>(p,t)){}
DartModel::~DartModel()=default;
double DartModel::verify_reference(const fs::path& path) {
    if(fs::file_size(path)>1024*1024)throw std::runtime_error("DART reference exceeds 1 MiB");
    std::ifstream f(path);nlohmann::json doc;f>>doc;double worst=0;std::set<std::string> checked;
    for(const auto& item:doc) {
        const auto name=item.at("name").get<std::string>();std::vector<std::vector<float>> inputs;
        if(!checked.insert(name).second)throw std::runtime_error("Duplicate DART reference graph");
        for(const auto& input:item.at("inputs"))inputs.push_back(input.get<std::vector<float>>());
        auto expected=item.at("output").get<std::vector<float>>();std::vector<float> output;auto& m=*impl_;
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
DartResult DartModel::generate(const DartPlan& plan) {
    auto started=std::chrono::steady_clock::now();auto& m=*impl_;
    std::array<Row,2> history{m.mean,m.mean};
    Vec3 delta=vector(m.mean,144)-vector(m.mean,0);Transform world;
    float floor=0;for(size_t j=0;j<22;++j)floor=std::min(floor,history[0][146+3*j]);world.p.z=-floor;
    std::mt19937 rng(plan.seed);std::normal_distribution<float> normal;
    std::array<float,10> alpha{};double cumulative=1;
    auto abar=[](double t){double a=std::cos((t+.008)/1.008*3.14159265358979323846/2);return a*a;};
    for(size_t t=0;t<10;++t){cumulative*=1-std::min(1-abar((t+1)/10.0)/abar(t/10.0),.999);alpha[t]=static_cast<float>(cumulative);}
    auto sample=[&](unsigned label,const std::vector<float>* observation=nullptr) {
        std::vector<float> h(552),noise(256),zero(512);
        for(size_t f=0;f<2;++f)for(size_t k=0;k<276;++k)h[f*276+k]=(history[f][k]-m.mean[k])/m.stddev[k];
        std::vector<float> distribution;
        if(observation)distribution=m.run(*m.policy,{"observation"},{*observation},{{1,1069}},"distribution");
        for(size_t k=0;k<256;++k)noise[k]=observation?distribution[k]+distribution[256+k]*normal(rng):normal(rng);
        for(int t=9;t>=0;--t) {
            auto u=m.run(*m.den,{"noise","timestep","history","text"},{noise,h,zero},{{1,1,256},{1,2,276},{1,512}},"x0",t);
            auto c=m.run(*m.den,{"noise","timestep","history","text"},{noise,h,m.embeddings[label]},{{1,1,256},{1,2,276},{1,512}},"x0",t);
            float previous=t?alpha[static_cast<size_t>(t-1)]:1;float a=std::sqrt((1-previous)/(1-alpha[static_cast<size_t>(t)]));float b=std::sqrt(previous)-a*std::sqrt(alpha[static_cast<size_t>(t)]);
            for(size_t k=0;k<256;++k)noise[k]=a*noise[k]+b*(u[k]+5*(c[k]-u[k]));
        }
        auto decoded=m.run(*m.dec,{"latent","history"},{noise,h},{{1,1,256},{1,2,276}},"future");
        if(decoded.size()!=8*276)throw std::runtime_error("Invalid DART decoder shape");
        std::array<Row,8> future;for(size_t f=0;f<8;++f)for(size_t k=0;k<276;++k)future[f][k]=decoded[f*276+k]*m.stddev[k]+m.mean[k];return future;
    };
    auto advance=[&](const std::array<Row,8>& future) {
        history={future[6],future[7]};Vec3 origin=vector(history[0],144),x=vector(history[0],150)-vector(history[0],147);x.z=0;x=normalized(x);
        Quat q=rotation6(x,cross(Vec3{0,0,1},x)),qi=inverse(q);
        for(auto& h:history) {
            put(h,0,rotate(qi,vector(h,0)+delta-origin)-delta);put_rotation(h,3,qi*rotation(h,3));
            put(h,135,rotate(qi,vector(h,135)));put_rotation(h,138,qi*rotation(h,138)*q);
            for(size_t j=0;j<22;++j){put(h,144+3*j,rotate(qi,vector(h,144+3*j)-origin));put(h,210+3*j,rotate(qi,vector(h,210+3*j)));}
        }
        world=world*Transform{origin,q};
    };
    for(unsigned i=0;i<plan.warmup;++i)advance(sample(0));
    auto first=world*Transform{vector(history[1],144),{}};world.p.x-=first.p.x;world.p.y-=first.p.y;
    Clip raw;raw.fps=30;raw.name="dart";
    auto emit=[&](const Row& r) {
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
    };
    emit(history[0]);emit(history[1]);DartResult result;result.metrics["segments"]=nlohmann::json::array();
    for(const auto& segment:plan.segments) {
        const unsigned label=action(segment.text);unsigned used=0;float error=0;bool reached=false;
        size_t start_frame=raw.frames()-1;
        for(;used<segment.chunks;++used) {
            std::vector<float> observation;
            if(segment.goal) {
                Vec3 pelvis=world.p+rotate(world.q,vector(history[1],144));Vec3 target=rotate(inverse(basis),*segment.goal);Vec3 direction=target-pelvis;direction.z=0;error=std::sqrt(dot(direction,direction));
                if(error<.3f){reached=true;break;}
                direction=direction*(1/std::max(error,1e-12f));Vec3 x=rotate(world.q,vector(history[1],150)-vector(history[1],147));x.z=0;x=normalized(x);Vec3 forward=cross(Vec3{0,0,1},x);
                float angle=std::acos(std::clamp(dot(direction,forward),.5f,1.0f));float side=cross(forward,direction).z;if(side<0)angle=-angle;else if(side==0)angle=0;
                Quat turn{0,0,std::sin(angle/2),std::cos(angle/2)};Vec3 local=rotate(inverse(world.q),rotate(turn,forward));
                observation={local.x,local.y,local.z,std::min(error,5.0f)};observation.insert(observation.end(),m.embeddings[label].begin(),m.embeddings[label].end());
                for(const auto& h:history)observation.insert(observation.end(),h.begin(),h.end());
                observation.push_back(-(world.p+rotate(world.q,vector(history[0],144))).z);
            }
            auto future=sample(label,segment.goal?&observation:nullptr);for(const auto& r:future)emit(r);advance(future);
        }
        if(segment.goal){auto p=rotate(basis,world.p+rotate(world.q,vector(history[1],144)));auto d=p-*segment.goal;d.y=0;error=std::sqrt(dot(d,d));reached=error<.3f;}
        nlohmann::json metric={{"text",segment.text},{"chunks",used},{"firstFrame",start_frame},{"lastFrame",raw.frames()-1}};
        if(segment.goal){metric["target"]={segment.goal->x,segment.goal->z};metric["modelGoalErrorMetres"]=error;metric["reached"]=reached;}
        result.metrics["segments"].push_back(metric);
    }
    Clip target;target.joints=soma();for(const auto& j:target.joints)target.poses.push_back(j.rest);
    RetargetMapping mapping;for(size_t s=0;s<30;++s)if(slots[s]>=0)mapping.bones[s]=names[slots[s]];
    // Preserve world-space goal coordinates; body proportions still use fixed SOMA offsets.
    mapping.root_scale=1;
    result.clip=retarget(raw,target,mapping,true).clip;
    // Retarget calibrates rest heading/ground; restore the authored world ground path exactly.
    for(size_t f=0;f<raw.frames();++f) {result.clip.frame(f)[0].p.x=raw.frame(f)[0].p.x;result.clip.frame(f)[0].p.z=raw.frame(f)[0].p.z;}
    require_soma(result.clip);result.metrics["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    result.metrics["initialization"]="normalized model mean, native walk warmup";result.metrics["sourceReference"]="fixed offsets recovered from first generated pose and predicted joints";
    return result;
}
}
