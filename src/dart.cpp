// Native host loop adapted from zkf1997/DART at bb67ae1ed6ce051080468bf15bc6e54a6c3f8417.
// Apache-2.0, licenses/DART-SOURCE.txt. Uses no parametric body model or mesh assets.
#include "dart.hpp"
#include "dart_gpu.hpp"
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
    fs::path directory;unsigned threads=4;std::string requested;
    std::unique_ptr<DartGpu> gpu;std::map<std::string,std::unique_ptr<Ort::Session>> batches;
    Row mean{},stddev{};std::array<std::vector<float>,3> embeddings;
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
    std::vector<float> execute(const std::string& name,DartFeed feed,const char* output) {
        size_t batch=static_cast<size_t>(feed.begin()->second.shape.at(0));std::vector<float> result;
        if(gpu)result=gpu->run(name,std::move(feed),output);
        else {
            if(!batches.contains(name)) {Ort::SessionOptions o;o.SetIntraOpNumThreads(static_cast<int>(threads));o.SetInterOpNumThreads(1);batches[name]=std::make_unique<Ort::Session>(env,(directory/(name+"-batch.onnx")).c_str(),o);}
            auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);std::vector<const char*> names;std::vector<Ort::Value> values;
            for(auto& [key,t]:feed){names.push_back(key.c_str());if(t.i.empty())values.push_back(Ort::Value::CreateTensor<float>(memory,t.f.data(),t.f.size(),t.shape.data(),t.shape.size()));else values.push_back(Ort::Value::CreateTensor<int64_t>(memory,t.i.data(),t.i.size(),t.shape.data(),t.shape.size()));}
            auto out=batches.at(name)->Run(Ort::RunOptions{nullptr},names.data(),values.data(),values.size(),&output,1);auto n=out[0].GetTensorTypeAndShapeInfo().GetElementCount();const float* ptr=out[0].GetTensorData<float>();result.assign(ptr,ptr+n);
        }
        size_t width=name=="denoiser"?256:name=="decoder"?2208:512;
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
    unsigned warmup,index=0,chunks=0,label=0;size_t first_frame=1;
    Clip raw;DartResult result;std::vector<float> observation;
    Stream(const DartPlan& p,const Row& m,const Row& s):plan(p),mean(m),stddev(s),history{m,m},delta(vector(m,144)-vector(m,0)),rng(p.seed),warmup(p.warmup) {
        float floor=0;for(size_t j=0;j<22;++j)floor=std::min(floor,history[0][146+3*j]);world.p.z=-floor;
        raw.fps=30;raw.name="dart";result.metrics["segments"]=nlohmann::json::array();
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
    bool prepare(const std::array<std::vector<float>,3>& embeddings) {
        observation.clear();label=0;if(warmup)return true;
        while(index<plan.segments.size()) {
            const auto& segment=plan.segments[index];label=action(segment.text);float error=0;
            Vec3 pelvis=world.p+rotate(world.q,vector(history[1],144)),direction{};
            if(segment.goal){direction=rotate(inverse(basis),*segment.goal)-pelvis;direction.z=0;error=std::sqrt(dot(direction,direction));}
            if(chunks>=segment.chunks||(segment.goal&&error<.3f)) {
                nlohmann::json metric={{"text",segment.text},{"chunks",chunks},{"firstFrame",first_frame},{"lastFrame",raw.frames()-1}};
                if(segment.goal){metric["target"]={segment.goal->x,segment.goal->z};metric["modelGoalErrorMetres"]=error;metric["reached"]=error<.3f;}
                result.metrics["segments"].push_back(metric);++index;chunks=0;first_frame=raw.frames()-1;continue;
            }
            if(segment.goal) {
                direction=direction*(1/std::max(error,1e-12f));Vec3 x=rotate(world.q,vector(history[1],150)-vector(history[1],147));x.z=0;x=normalized(x);Vec3 forward=cross(Vec3{0,0,1},x);
                float angle=std::acos(std::clamp(dot(direction,forward),.5f,1.0f));float side=cross(forward,direction).z;if(side<0)angle=-angle;else if(side==0)angle=0;
                Quat turn{0,0,std::sin(angle/2),std::cos(angle/2)};Vec3 local=rotate(inverse(world.q),rotate(turn,forward));
                observation={local.x,local.y,local.z,std::min(error,5.0f)};observation.insert(observation.end(),embeddings[label].begin(),embeddings[label].end());
                for(const auto& h:history)observation.insert(observation.end(),h.begin(),h.end());
                observation.push_back(-(world.p+rotate(world.q,vector(history[0],144))).z);
            }
            return true;
        }
        return false;
    }
    void accept(std::span<const float> decoded) {
        std::array<Row,8> future;for(size_t f=0;f<8;++f)for(size_t k=0;k<276;++k)future[f][k]=decoded[f*276+k]*stddev[k]+mean[k];
        if(warmup) {
            advance(future);if(--warmup==0){auto first=world*Transform{vector(history[1],144),{}};world.p.x-=first.p.x;world.p.y-=first.p.y;emit(history[0]);emit(history[1]);}
        } else {for(const auto& r:future)emit(r);advance(future);++chunks;}
    }
    DartResult finish() {
        Clip target;target.joints=soma();for(const auto& j:target.joints)target.poses.push_back(j.rest);
        RetargetMapping mapping;for(size_t s=0;s<30;++s)if(slots[s]>=0)mapping.bones[s]=names[slots[s]];
        mapping.root_scale=1;result.clip=retarget(raw,target,mapping,true).clip;
        for(size_t f=0;f<raw.frames();++f){result.clip.frame(f)[0].p.x=raw.frame(f)[0].p.x;result.clip.frame(f)[0].p.z=raw.frame(f)[0].p.z;}
        require_soma(result.clip);result.metrics["initialization"]="normalized model mean, native walk warmup";
        result.metrics["sourceReference"]="fixed offsets recovered from first generated pose and predicted joints";return std::move(result);
    }
};
}
DartResult DartModel::generate(const DartPlan& plan){return generate(std::span<const DartPlan>(&plan,1)).front();}
std::vector<DartResult> DartModel::generate(std::span<const DartPlan> plans) {
    if(plans.empty()||plans.size()>8)throw std::runtime_error("DART neural batch needs 1..8 independent plans");
    auto started=std::chrono::steady_clock::now();auto& m=*impl_;
    try {
        std::vector<Stream> streams;streams.reserve(plans.size());for(const auto& p:plans)streams.emplace_back(p,m.mean,m.stddev);
        const int64_t n=static_cast<int64_t>(plans.size());const size_t size=plans.size();
        std::array<float,10> alpha{};double cumulative=1;
        auto abar=[](double t){double a=std::cos((t+.008)/1.008*3.14159265358979323846/2);return a*a;};
        for(size_t t=0;t<10;++t){cumulative*=1-std::min(1-abar((t+1)/10.0)/abar(t/10.0),.999);alpha[t]=static_cast<float>(cumulative);}
        size_t waves=0;
        while(true) {
            std::vector<bool> active(size);bool any=false,controlled=false;
            std::vector<float> history(size*552),noise(size*256),observations(size*1069),text(size*1024);
            for(size_t b=0;b<size;++b) {
                auto& s=streams[b];active[b]=s.prepare(m.embeddings);if(!active[b])continue;any=true;
                for(size_t f=0;f<2;++f)for(size_t k=0;k<276;++k)history[b*552+f*276+k]=(s.history[f][k]-m.mean[k])/m.stddev[k];
                std::copy(m.embeddings[s.label].begin(),m.embeddings[s.label].end(),text.begin()+static_cast<ptrdiff_t>((size+b)*512));
                if(!s.observation.empty()){controlled=true;std::copy(s.observation.begin(),s.observation.end(),observations.begin()+static_cast<ptrdiff_t>(b*1069));}
            }
            if(!any)break;
            ++waves;std::vector<float> distribution;
            if(controlled)distribution=m.execute("policy",{{"observation",{{n,1069},std::move(observations),{}}}},"distribution");
            for(size_t b=0;b<size;++b)if(active[b])for(size_t k=0;k<256;++k){auto& s=streams[b];noise[b*256+k]=s.observation.empty()?s.normal(s.rng):distribution[b*512+k]+distribution[b*512+256+k]*s.normal(s.rng);}
            std::vector<float> histories=history;histories.insert(histories.end(),history.begin(),history.end());
            for(int t=9;t>=0;--t) {
                std::vector<float> both=noise;both.insert(both.end(),noise.begin(),noise.end());
                auto output=m.execute("denoiser",{{"noise",{{2*n,1,256},std::move(both),{}}},{"timestep",{{2*n},{},std::vector<int64_t>(2*size,t)}},{"history",{{2*n,2,276},histories,{}}},{"text",{{2*n,512},text,{}}}},"x0");
                float previous=t?alpha[static_cast<size_t>(t-1)]:1;float a=std::sqrt((1-previous)/(1-alpha[static_cast<size_t>(t)]));float b=std::sqrt(previous)-a*std::sqrt(alpha[static_cast<size_t>(t)]);
                for(size_t k=0;k<noise.size();++k)noise[k]=a*noise[k]+b*(output[k]+5*(output[size*256+k]-output[k]));
            }
            auto decoded=m.execute("decoder",{{"latent",{{n,1,256},std::move(noise),{}}},{"history",{{n,2,276},std::move(history),{}}}},"future");
            for(size_t b=0;b<size;++b)if(active[b])streams[b].accept(std::span<const float>(decoded).subspan(b*2208,2208));
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();std::vector<DartResult> results;
        for(auto& stream:streams){auto result=stream.finish();result.metrics["seconds"]=seconds;result.metrics["neuralBatch"]={{"requests",size},{"primitiveWaves",waves},{"guidanceBranchesPerCall",2*size},{"denoiserCalls",10*waves}};result.metrics["device"]=device();result.metrics["gpuPeakLocalBytes"]=m.gpu?m.gpu->peak_memory_bytes():0;results.push_back(std::move(result));}
        return results;
    }catch(const DartGpuFailure& e) {
        if(m.requested!="auto")throw;
        std::cerr<<e.what()<<"; restarting the entire DART group on CPU from its seeds\n";m.gpu.reset();return generate(plans);
    }
}
}
