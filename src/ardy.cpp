// Native host adaptation of NVIDIA ARDY at e706e27acc88070918308a10922b8f7174d0825e.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// Modified for bounded native ORT inference and canonical SOMA GLB assets.
#include "ardy.hpp"
#include "llm_text_encoder.hpp"
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numbers>
#include <optional>
#include <random>
#include <stdexcept>

namespace motion {
namespace {
using json=nlohmann::json;
constexpr size_t N=75,F=300,D=148,R=369;
void need(bool b,const std::string& why) {if(!b)throw std::runtime_error("ARDY: "+why);}
json read_json(const fs::path& p,size_t limit=32*1024*1024) {
    need(fs::file_size(p)<=limit,"JSON exceeds bound");std::ifstream s(p);json j;s>>j;return j;
}
// Dense Cholesky solves the small (<=240) trajectory system without a new runtime dependency.
std::vector<float> smooth_xz(const Clip& c) {
    const size_t count=c.frames();std::vector<double> target(count*2),result(count*2);
    for(size_t i=0;i<count;++i){target[2*i]=c.frame(i)[0].p.x;target[2*i+1]=c.frame(i)[0].p.z;}
    for(size_t d=0;d<2;++d){float mean=0;for(size_t i=0;i<count;++i)mean+=static_cast<float>(target[2*i+d]);mean/=static_cast<float>(count);for(size_t i=0;i<count;++i)result[2*i+d]=mean;}
    size_t level=std::max(1,static_cast<int>(std::floor(std::log2(count)))-4);
    for(size_t stride=size_t(1)<<level;;stride/=2) {
        const size_t n=(count+stride-1)/stride;
        if(n<2)continue;
        std::vector<double> m(n*n),l(n*n),x(n*2),z(n*2),u(n*2);
        for(size_t i=1;i+1<n;++i)for(size_t a=0;a<3;++a)for(size_t b=0;b<3;++b)
            m[(i+a-1)*n+i+b-1]+=(a==1?2.:-1.)*(b==1?2.:-1.);
        double diag=0;for(size_t i=0;i<n;++i)diag=std::max(diag,m[i*n+i]);
        const double rho=.25*std::sqrt(diag);
        if(rho==0){if(stride==1)break;continue;}
        for(size_t i=0;i<n;++i)m[i*n+i]+=rho;
        for(size_t i=0;i<n;++i)for(size_t j=0;j<=i;++j){double v=m[i*n+j];for(size_t k=0;k<j;++k)v-=l[i*n+k]*l[j*n+k];l[i*n+j]=i==j?std::sqrt(v):v/l[j*n+j];}
        for(size_t i=0;i<n;++i)for(size_t d=0;d<2;++d)x[2*i+d]=result[2*i*stride+d];
        for(int iter=0;iter<500;++iter) {
            for(size_t i=0;i<n;++i){
                float delta[2];for(size_t d=0;d<2;++d)delta[d]=static_cast<float>(x[2*i+d])+static_cast<float>(u[2*i+d])-static_cast<float>(target[2*i*stride+d]);
                const float norm=std::sqrt(delta[0]*delta[0]+delta[1]*delta[1]);
                for(size_t d=0;d<2;++d){z[2*i+d]=static_cast<float>(static_cast<float>(target[2*i*stride+d])+static_cast<float>(delta[d]*std::min(1.,.06/std::max(static_cast<double>(norm),1e-30))));u[2*i+d]=static_cast<float>(static_cast<float>(u[2*i+d])+1.8f*(static_cast<float>(x[2*i+d])-static_cast<float>(z[2*i+d])));}
            }
            for(size_t d=0;d<2;++d){
                for(size_t i=0;i<n;++i){double v=static_cast<float>(static_cast<float>(rho)*(static_cast<float>(z[2*i+d])-static_cast<float>(u[2*i+d])));for(size_t k=0;k<i;++k)v-=l[i*n+k]*x[2*k+d];x[2*i+d]=v/l[i*n+i];}
                for(size_t i=n;i-->0;){double v=x[2*i+d];for(size_t k=i+1;k<n;++k)v-=l[k*n+i]*x[2*k+d];x[2*i+d]=v/l[i*n+i];}
                for(size_t i=0;i<n;++i)x[2*i+d]=static_cast<float>(x[2*i+d]);
            }
        }
        for(size_t i=0;i<n;++i)for(size_t d=0;d<2;++d)result[2*i*stride+d]=x[2*i+d];
        if(stride==1){
            // Upstream smooth_signal also executes its interpolation at the final
            // stride (next_stepsize == 0). Preserve that half-sample shift exactly.
            for(size_t d=0;d<2;++d)result[2*(count-1)+d]=static_cast<float>(static_cast<float>(result[2*(count-1)+d])+(static_cast<float>(result[2*(count-1)+d])-static_cast<float>(result[2*(count-2)+d]))/2);
            auto copy=result;for(size_t i=0;i+1<count;++i)for(size_t d=0;d<2;++d)result[2*i+d]=(static_cast<float>(copy[2*i+d])+static_cast<float>(copy[2*(i+1)+d]))/2;
            break;
        }
        for(size_t i=stride/2;i<count;i+=stride)for(size_t d=0;d<2;++d){
            size_t prev=i-stride/2,next=i+stride/2;
            result[2*i+d]=next<count?(static_cast<float>(result[2*prev+d])+static_cast<float>(result[2*next+d]))/2:static_cast<float>(result[2*prev+d])+(static_cast<float>(result[2*prev+d])-static_cast<float>(result[2*(prev-stride)+d]))/2;
        }
    }
    return std::vector<float>(result.begin(),result.end());
}
struct Tensor {
    std::vector<int64_t> shape;std::vector<float> f;std::vector<int64_t> i;
    static Tensor floats(std::vector<int64_t> s,std::vector<float> data){return {std::move(s),std::move(data),{}};}
    static Tensor ints(std::vector<int64_t> s,std::vector<int64_t> data){return {std::move(s),{},std::move(data)};}
};
using Feed=std::map<std::string,Tensor>;
struct ScopedEnv {
    std::string name,old;bool existed;
    ScopedEnv(const char* n,const std::string& value):name(n){const char* p=std::getenv(n);existed=p!=nullptr;if(p)old=p;set(value);}
    void set(const std::string& v) {
#ifdef _WIN32
        _putenv_s(name.c_str(),v.c_str());
#else
        setenv(name.c_str(),v.c_str(),1);
#endif
    }
    ~ScopedEnv(){if(existed)set(old);else {
#ifdef _WIN32
        _putenv_s(name.c_str(),"");
#else
        unsetenv(name.c_str());
#endif
    }}
};
std::vector<float> run(Ort::Session& session,Feed feed) {
    auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::vector<const char*> names;std::vector<Ort::Value> inputs;
    for(auto& [name,t]:feed){names.push_back(name.c_str());if(t.i.empty())inputs.push_back(Ort::Value::CreateTensor<float>(mem,t.f.data(),t.f.size(),t.shape.data(),t.shape.size()));else inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem,t.i.data(),t.i.size(),t.shape.data(),t.shape.size()));}
    const char* out="output";auto values=session.Run(Ort::RunOptions{nullptr},names.data(),inputs.data(),inputs.size(),&out,1);
    auto size=values[0].GetTensorTypeAndShapeInfo().GetElementCount();const auto* p=values[0].GetTensorData<float>();std::vector<float> result(p,p+size);
    for(float v:result)need(std::isfinite(v),"nonfinite model output");return result;
}
}
std::vector<float> encode_ardy_history(const Clip& c) {
    require_soma(c);need(c.frames()>=4&&c.frames()<=240&&c.frames()%4==0,"history needs 4..240 frames, multiple of four");
    auto root=smooth_xz(c);std::vector<float> out;
    for(size_t f=0;f<c.frames();++f){auto row=encode_kimodo(c,c.frame(f));const float dx=row[0]-root[2*f],dz=row[2]-root[2*f+1];row[0]=root[2*f];row[2]=root[2*f+1];for(size_t j=0;j<30;++j){row[5+3*j]+=dx;row[7+3*j]+=dz;}out.insert(out.end(),row.begin(),row.end());}
    // The tokenizer consumes body features, including global velocities and contacts.
    std::vector<std::vector<Transform>> world;for(size_t f=0;f<c.frames();++f)world.push_back(world_pose(c,c.frame(f)));
    constexpr size_t feet[]={24,25,28,29};
    for(size_t f=0;f<c.frames();++f){const size_t a=f+1<c.frames()?f:f-1,b=a+1;
        for(size_t j=0;j<30;++j){const auto v=(world[b][j].p-world[a][j].p)*30;size_t k=f*R+275+3*j;out[k]=v.x;out[k+1]=v.y;out[k+2]=v.z;}
        for(size_t j=0;j<4;++j){auto foot=feet[j];const auto v=(world[b][foot].p-world[a][foot].p)*30;out[f*R+365+j]=(world[f][foot].p.y<.10f&&dot(v,v)<.15f*.15f)?1.f:0.f;}
    }
    return out;
}
void check_ardy_models(const fs::path& dir) {
    need(sha256_file(dir/"manifest.json")=="aac3b966ed1368c4ec126c969f0763d59126b047a1053a2394c20983acb917d2","manifest differs from pinned conversion");
    need(sha256_file(dir/"LICENSE")=="11ccdcb47db87ac30b63765b5e5154403026c4fbb52c19919d99d912ec454e19","model license differs from pinned grant");
    need(fs::exists(dir/"NOTICE"),"model NOTICE is required");
    auto m=read_json(dir/"manifest.json",64*1024);
    need(m.at("schema")=="tide-ardy-1"&&m.at("checkpointRevision")=="e6c6804fb6b55636ad927145d74959fd46882ccd"&&m.at("tokens")==N,"wrong model contract");
    for(const char* name:{"denoiser.onnx","encoder.onnx","decoder.onnx"}){
        auto spec=m.at("files").at(name);need(fs::file_size(dir/name)==spec.at("bytes").get<uint64_t>(),"incomplete model");need(sha256_file(dir/name)==spec.at("sha256").get<std::string>(),"model checksum mismatch");
    }
}
struct ArdyModel::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"tide-ardy"};
    Ort::Session denoiser{nullptr},encoder{nullptr},decoder{nullptr};
    std::unique_ptr<kimodo::detail::llm_text_encoder> text;
    fs::path root;json meta;unsigned thread_count;
    std::map<std::string,std::array<float,4096>> embeddings;
    Impl(const fs::path& dir,unsigned threads):root(dir),meta(read_json(dir/"manifest.json",64*1024)),thread_count(threads?threads:8) {
        check_ardy_models(dir);Ort::SessionOptions opts;opts.SetIntraOpNumThreads(static_cast<int>(threads?threads:8));opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::cerr<<"Loading ARDY SOMA KimodoRep (native ONNX CPU)...\n";
        denoiser=Ort::Session(env,(dir/"denoiser.onnx").c_str(),opts);
        encoder=Ort::Session(env,(dir/"encoder.onnx").c_str(),opts);
        decoder=Ort::Session(env,(dir/"decoder.onnx").c_str(),opts);
    }
    std::array<float,4096> embed(const std::string& prompt) {
        if(embeddings.contains(prompt))return embeddings.at(prompt);
        ScopedEnv backend("KIMODO_BACKEND","cpu"),threads("KIMODO_THREADS",std::to_string(thread_count));
        if(!text){check_models(root.parent_path(),"text");auto loaded=kimodo::detail::llm_text_encoder::load((root.parent_path()/"text/encoder.gguf").string());need(bool(loaded),loaded?"":loaded.error());text=std::move(*loaded);}
        auto v=text->encode(prompt);need(bool(v),v?"":v.error());embeddings[prompt]=*v;return *v;
    }
    void recenter(std::vector<float>& history,size_t count,Vec3& origin) {
        auto mean=meta.at("rootMean").get<std::vector<float>>(),scale=meta.at("rootScale").get<std::vector<float>>();
        auto lm=meta.at("latentMean").get<std::vector<float>>(),ls=meta.at("latentScale").get<std::vector<float>>();
        size_t frame=count*4-1,t=frame/4,k=(frame%4)*5;
        float x=history[t*D+k]*scale[0]+mean[0],z=history[t*D+k+2]*scale[2]+mean[2];
        for(size_t i=0;i<count;++i){for(size_t f=0;f<4;++f){history[i*D+f*5]-=x/scale[0];history[i*D+f*5+2]-=z/scale[2];}
            for(size_t j=0;j<128;++j){float v=history[i*D+20+j]*ls[j]+lm[j];history[i*D+20+j]=(std::nearbyint(std::clamp(v,-1.f,1.f)*32)/32-lm[j])/ls[j];}}
        origin.x+=x;origin.z+=z;
    }
};
ArdyModel::ArdyModel(const fs::path& dir,unsigned threads):impl_(std::make_unique<Impl>(dir,threads)){}
ArdyModel::~ArdyModel()=default;
Clip ArdyModel::generate(const std::vector<ArdyPrompt>& prompts,const ArdyOptions& opt,const MotionConditions& conditions,const Clip* imported) {
    const auto started=std::chrono::steady_clock::now();
    need(!prompts.empty()&&prompts.size()<=16,"needs 1..16 prompts");need(opt.steps>=1&&opt.steps<=10,"steps must be 1..10");need(opt.history_frames>=4&&opt.history_frames<=240&&opt.history_frames%4==0,"history frames must be 4..240, multiple of four");
    size_t total=0;for(auto& p:prompts){need(!p.text.empty()&&p.text.size()<=8192&&p.frames>0&&p.frames<=18000,"invalid prompt or duration");total+=p.frames;}need(total<=18000,"output exceeds 18000 frames");
    auto condition=build_condition(conditions,total);Vec3 origin=condition.origin;
    std::vector<float> history;size_t hn=0;float heading=condition.first_heading;
    auto& m=*impl_;
    if(imported){auto raw=encode_ardy_history(*imported);if(!opt.reference_history.empty()){need(opt.reference_history.size()==raw.size(),"reference history shape");raw.assign(opt.reference_history.begin(),opt.reference_history.end());}std::vector<float> padded(F*R);std::copy(raw.begin(),raw.end(),padded.begin());history=run(m.encoder,{{"motion",Tensor::floats({1,F,R},std::move(padded))},{"length",Tensor::ints({1},{static_cast<int64_t>(imported->frames())})}});hn=imported->frames()/4;history.resize(hn*D);origin={};m.recenter(history,hn,origin);heading=kimodo_heading(*imported,imported->frame(0));}
    std::mt19937 random(static_cast<uint32_t>(opt.seed));std::normal_distribution<float> normal;size_t noise_index=0;
    // Match upstream float32 cosine schedule, timestep respacing and deterministic DDIM.
    std::vector<float> base(10);float product=1;
    for(size_t i=0;i<10;++i){auto abar=[](double t){return std::pow(std::cos((t+.008)/1.008*std::numbers::pi/2),2);};float beta=static_cast<float>(std::min(1-abar(static_cast<double>(i+1)/10)/abar(static_cast<double>(i)/10),.999));product*=1-beta;base[i]=product;}
    std::vector<int> mapped(opt.steps);std::vector<float> alpha(opt.steps);
    for(unsigned i=0;i<opt.steps;++i){mapped[i]=std::min(9,static_cast<int>(std::nearbyint(i*9.0/std::max(1u,opt.steps-1))));alpha[i]=std::max(1e-9f,base[static_cast<size_t>(mapped[i])]);}
    Clip out;out.joints=soma();size_t done=0;
    auto rootMean=m.meta.at("rootMean").get<std::vector<float>>(),rootScale=m.meta.at("rootScale").get<std::vector<float>>();
    while(done<total){
        size_t keep=std::min(hn,static_cast<size_t>(opt.history_frames/4));
        if(keep<hn)history.erase(history.begin(),history.begin()+static_cast<ptrdiff_t>((hn-keep)*D));hn=keep;
        if(hn){float c=history[3]*rootScale[3]+rootMean[3],s=history[4]*rootScale[4]+rootMean[4];heading=std::atan2(s,c);}
        size_t promptIndex=0,until=prompts[0].frames;while(done>=until&&promptIndex+1<prompts.size())until+=prompts[++promptIndex].frames;
        auto embedding=m.embed(prompts[promptIndex].text);
        std::vector<float> x(N*D),obs(F*R),mask(F*R);std::copy(history.begin(),history.end(),x.begin());
        for(size_t i=hn*D;i<(hn+15)*D;++i){if(!opt.reference_noise.empty()){need(noise_index<opt.reference_noise.size(),"reference noise exhausted");x[i]=opt.reference_noise[noise_index++];}else x[i]=normal(random);}
        for(size_t f=hn*4;f<F;++f){size_t source=done+f-hn*4;if(source>=total)break;for(size_t j=0;j<R;++j){float masked=condition.mask[source*R+j];mask[f*R+j]=masked;float v=condition.observed[source*R+j];if(j==0)v-=origin.x-condition.origin.x;if(j==2)v-=origin.z-condition.origin.z;obs[f*R+j]=v;}}
        Feed feed={{"text",Tensor::floats({1,1,4096},{embedding.begin(),embedding.end()})},{"observed",Tensor::floats({1,F,R},obs)},{"mask",Tensor::floats({1,F,R},mask)},
            {"lengths",Tensor::ints({3},{static_cast<int64_t>(hn*4),60,static_cast<int64_t>(F-hn*4-60)})},{"heading",Tensor::floats({1},{heading})},{"guidance",Tensor::floats({2},{opt.text_guidance,opt.constraint_guidance})}};
        for(unsigned step=opt.steps;step-->0;){feed["x"]=Tensor::floats({1,N,D},x);feed["timestep"]=Tensor::ints({1},{mapped[step]});auto clean=run(m.denoiser,feed);float a=alpha[step],prev=step?alpha[step-1]:1;
            for(size_t i=hn*D;i<(hn+15)*D;++i){float eps=(x[i]/std::sqrt(a)-clean[i])/std::sqrt(1/a-1);x[i]=std::sqrt(prev)*clean[i]+std::sqrt(1-prev)*eps;}}
        const size_t count=hn+15;
        // Decode with the same cropped causal context as this chunk; only new frames are exported.
        auto raw=run(m.decoder,{{"x",Tensor::floats({1,N,D},x)},{"length",Tensor::ints({1},{static_cast<int64_t>(count*4)})}});
        auto chunk=decode_kimodo(raw);size_t take=std::min(size_t(60),total-done);
        for(size_t f=0;f<take;++f){auto pose=chunk.frame(hn*4+f);pose[0].p=pose[0].p+origin;out.poses.insert(out.poses.end(),pose.begin(),pose.end());}
        history.assign(x.begin(),x.begin()+static_cast<ptrdiff_t>(count*D));hn=count;m.recenter(history,hn,origin);
        done+=take;std::cerr<<"ARDY generated "<<done<<"/"<<total<<" frames\n";
    }
    out.validate();std::cerr<<"ARDY generation seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<" (includes prompt encoding, excludes model load)\n";return out;
}
void ArdyModel::verify_reference(const fs::path& dir) {
    for(auto name:{"denoiser","encoder","decoder"})for(int history:{0,4,240}){
        auto f=read_json(dir/(std::string(name)+"-"+std::to_string(history)+".json"));Feed feed;
        for(auto& [key,t]:f.at("inputs").items()){auto shape=t.at("shape").get<std::vector<int64_t>>();if(t.at("dtype")=="int64")feed[key]=Tensor::ints(shape,t.at("data").get<std::vector<int64_t>>());else feed[key]=Tensor::floats(shape,t.at("data").get<std::vector<float>>());}
        auto& session=std::string(name)=="denoiser"?impl_->denoiser:std::string(name)=="encoder"?impl_->encoder:impl_->decoder;
        auto actual=run(session,std::move(feed)),expected=f.at("output").at("data").get<std::vector<float>>();need(actual.size()==expected.size(),"reference shape mismatch");double sum=0,max=0;
        for(size_t i=0;i<actual.size();++i){double e=std::abs(actual[i]-expected[i]);max=std::max(max,e);sum+=e;need(e<=.002+.002*std::abs(expected[i]),"reference parity failed");}
        std::cout<<name<<" history="<<history<<" max="<<max<<" mean="<<sum/static_cast<double>(actual.size())<<"\n";
    }
    for(const char* fixture:{"host.json","imported.json"})if(fs::exists(dir/fixture)) {
        auto f=read_json(dir/fixture);
        const auto prompt=f.at("prompt").get<std::string>();
        auto text=f.at("text").get<std::vector<float>>();need(text.size()==4096,"text shape");
        std::array<float,4096> embedding;std::copy(text.begin(),text.end(),embedding.begin());impl_->embeddings[prompt]=embedding;
        auto noise=f.at("noise").get<std::vector<float>>();ArdyOptions opt;opt.reference_noise=noise;
        MotionConditions conditions;std::optional<Clip> imported;
        if(f.contains("glb")){imported=load_glb(utf8path(f.at("glb").get<std::string>()));imported->poses.resize(64*30);}
        else conditions.path.push_back({150,2.0f,1.0f,false,0});
        std::vector<float> reference_history;
        if(f.contains("historyFeatures")){reference_history=f.at("historyFeatures").get<std::vector<float>>();opt.reference_history=reference_history;}
        auto actual=generate({{prompt,imported?121u:183u}},opt,conditions,imported?&*imported:nullptr);
        auto ref=f.at("features").get<std::vector<float>>();auto expected=decode_kimodo(ref);need(actual.frames()==expected.frames(),"host frame count");
        double maxp=0,maxq=0;
        for(size_t i=0;i<actual.poses.size();++i){auto a=actual.poses[i],b=expected.poses[i];auto delta=a.p-b.p;maxp=std::max(maxp,static_cast<double>(std::sqrt(dot(delta,delta))));double q=std::abs(a.q.x*b.q.x+a.q.y*b.q.y+a.q.z*b.q.z+a.q.w*b.q.w);maxq=std::max(maxq,2*std::acos(std::min(1.,q)));}
        std::cout<<fixture<<" frames="<<actual.frames()<<" maxPosition="<<maxp<<" maxRotationRadians="<<maxq<<"\n";
        need(maxp<.003&&maxq<.01,"host-loop decoded parity failed");
        if(imported){opt.reference_history={};auto local=generate({{prompt,121}},opt,{},&*imported);double pmax=0,qmax=0;
            for(size_t i=0;i<local.poses.size();++i){auto a=local.poses[i],b=expected.poses[i];auto d=a.p-b.p;pmax=std::max(pmax,static_cast<double>(std::sqrt(dot(d,d))));double q=std::abs(a.q.x*b.q.x+a.q.y*b.q.y+a.q.z*b.q.z+a.q.w*b.q.w);qmax=std::max(qmax,2*std::acos(std::min(1.,q)));}
            std::cout<<"imported FK/smoothing/FSQ end-to-end maxPosition="<<pmax<<" maxRotationRadians="<<qmax<<"\n";
            // FSQ is discontinuous: sub-millimetre preprocessing roundoff can
            // choose a neighbouring scalar bin. Keep this distinct from fixed-input parity.
            need(pmax<.005&&qmax<.05,"imported pipeline divergence beyond tested quantization bound");
        }
    }
    if(fs::exists(dir/"history.json")) {
        auto f=read_json(dir/"history.json");auto c=load_glb(utf8path(f.at("glb").get<std::string>()));c.poses.resize(64*30);
        auto raw=encode_ardy_history(c),expected=f.at("features").get<std::vector<float>>();need(raw.size()==expected.size(),"history feature shape");
        double max=0,mean=0;size_t worst=0;for(size_t i=0;i<raw.size();++i){double e=std::abs(raw[i]-expected[i]);if(e>max){max=e;worst=i;}mean+=e;}
        std::cout<<"history features max="<<max<<" mean="<<mean/static_cast<double>(raw.size())<<" worst="<<worst/R<<","<<worst%R<<" got="<<raw[worst]<<" expected="<<expected[worst]<<"\n";
        
        need(max<.0002,"history feature parity failed");
        auto rig=soma();auto names=f.at("names").get<std::vector<std::string>>();auto parents=f.at("parents").get<std::vector<int>>();auto neutral=f.at("neutral");
        for(size_t j=0;j<30;++j){need(rig[j].name==names[j]&&rig[j].parent==parents[j],"canonical SOMA hierarchy mismatch");if(j){auto p=static_cast<size_t>(parents[j]);Vec3 offset{neutral[j][0].get<float>()-neutral[p][0].get<float>(),neutral[j][1].get<float>()-neutral[p][1].get<float>(),neutral[j][2].get<float>()-neutral[p][2].get<float>()};auto d=offset-rig[j].rest.p;need(dot(d,d)<1e-10f,"canonical SOMA rest mismatch");}}
    }
}
void ArdyModel::write_text_reference(const std::string& prompt,const fs::path& path) {
    auto embedding=impl_->embed(prompt);json f={{"prompt",prompt},{"text",embedding}};std::ofstream s(path);s<<f.dump();need(bool(s),"write text reference");
}
}
