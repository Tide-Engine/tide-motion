#include "momask.hpp"
#include "humanml.hpp"
#include "momask_gpu.hpp"
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <numbers>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
using namespace motion;
using nlohmann::json;
namespace h=motion::humanml;
struct PathLess {
    bool operator()(const fs::path& a,const fs::path& b) const {
#ifdef _WIN32
        int result=CompareStringOrdinal(a.c_str(),-1,b.c_str(),-1,TRUE);
        if(!result)throw std::runtime_error("Cannot compare MoMask output paths");
        return result==CSTR_LESS_THAN;
#else
        return a<b;
#endif
    }
};
// Exact supported local bundle. This backend never downloads model weights.
constexpr const char* bundle_hash="bb7989e45678b35f795ec3bb4ec8442d78e37a3e7b6eebe093dcf7784490c184";
json read_json(const fs::path& p,size_t limit) {
    if(fs::file_size(p)>limit)throw std::runtime_error("MoMask JSON file exceeds limit");
    std::ifstream in(p);json j;in>>j;return j;
}
std::vector<int64_t> tokenize(std::string text,const json& metadata) {
    if(text.empty()||text.size()>4096)throw std::runtime_error("MoMask prompt must contain 1..4096 ASCII bytes");
    for(char& c:text) {
        auto u=static_cast<unsigned char>(c);
        if(u>=127||(u<32&&c!='\n'&&c!='\r'&&c!='\t')||c=='&')throw std::runtime_error("MoMask supports ASCII prompts without HTML entities only");
        if(c>='A'&&c<='Z')c=static_cast<char>(c-'A'+'a');
    }
    if(text.find("<|")!=std::string::npos)throw std::runtime_error("MoMask prompt cannot contain tokenizer special tokens");
    std::map<std::string,int64_t> vocab;int64_t id=0;
    for(const auto& word:metadata.at("vocabulary"))vocab.emplace(word.get<std::string>(),id++);
    std::map<std::pair<std::string,std::string>,size_t> ranks;size_t rank=0;
    for(const auto& pair:metadata.at("merges"))ranks.emplace(std::make_pair(pair[0].get<std::string>(),pair[1].get<std::string>()),rank++);
    std::regex pattern("'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\\sA-Za-z0-9]+");
    std::vector<int64_t> result{49406};
    for(std::sregex_iterator i(text.begin(),text.end(),pattern),end;i!=end;++i) {
        std::string word=i->str();std::vector<std::string> pieces;
        for(char c:word)pieces.emplace_back(1,c);
        pieces.back()+="</w>";
        while(pieces.size()>1) {
            size_t best=ranks.size()+1;std::pair<std::string,std::string> selected;
            for(size_t p=0;p+1<pieces.size();++p) {auto it=ranks.find({pieces[p],pieces[p+1]});if(it!=ranks.end()&&it->second<best){best=it->second;selected=it->first;}}
            if(best>ranks.size())break;
            std::vector<std::string> merged;
            for(size_t p=0;p<pieces.size();++p) {if(p+1<pieces.size()&&pieces[p]==selected.first&&pieces[p+1]==selected.second){merged.push_back(pieces[p]+pieces[p+1]);++p;}else merged.push_back(pieces[p]);}
            pieces=std::move(merged);
        }
        for(const auto& piece:pieces)result.push_back(vocab.at(piece));
    }
    if(result.size()==1)throw std::runtime_error("MoMask prompt is empty after tokenization");
    result.push_back(49407);if(result.size()>77){result.resize(77);result.back()=49407;}result.resize(77,0);return result;
}
struct Random {
    uint64_t state;
    double uniform() {state+=0x9e3779b97f4a7c15ULL;uint64_t z=state;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;z^=z>>31;return std::min((static_cast<double>(z>>11)+.5)/9007199254740992.0,1.0-1.0/9007199254740992.0);}
};
struct Job {
    std::string prompt;fs::path output,source;double duration=2,guidance=4,residual_guidance=5,temperature=1,topkr=.9;
    uint32_t seed=0;size_t steps=18,begin=0,end=0;std::optional<Clip> input,full_input;
    unsigned threads=4,variants=1,variant_index=0;bool in_place=false,variants_gif=false,rig_explicit=false;
    std::string device="cpu",rig="soma30",tide="tide";fs::path variant_base;
    std::string sampling="categorical";unsigned residual_layers=5;
    std::optional<double> residual_temperature,residual_topkr;
    bool unconditional=false,continuation=false;size_t target_tokens=0;uint64_t stream=0;
    std::vector<std::pair<std::string,double>> segments;
    fs::path history;unsigned history_frames=119,chunk_frames=89;std::optional<Clip> history_clip;
    bool append_history=false;std::optional<Clip> history77;
    std::vector<std::pair<size_t,size_t>> ranges;std::vector<bool> editable_tokens;
    struct Observation {size_t token=0;Clip clip;};std::vector<Observation> observations;
    bool preserve_context_residuals=false,preserve_root=false;unsigned context_overlap_tokens=1;
    std::string edit_mode="continuous";
    struct PoseSpec {std::string kind,reference;long frame=0;unsigned effectors=0;};
    std::vector<PoseSpec> pose_specs;std::vector<std::pair<PathConstraint,long>> path_specs;
    MotionConditions conditions;fs::path pose_map,contacts_out;
    std::string placement="auto",follow="world",correction="tide";
    bool dense_path=false,shift_last=false,fit_poses=false,loop=false,attempts_explicit=false;int correction_frames=10,wanted_footsteps=-1;unsigned attempts=8;
    double path_margin=.04;size_t context_frames=0;
};
Clip preserved_source77(const fs::path& path) {
    auto raw=load_glb(path,"",false);auto out=as_soma77(raw); // validates the explicit canonical names/hierarchy
    for(size_t j=0;j<out.joints.size();++j) {
        auto found=std::find_if(raw.joints.begin(),raw.joints.end(),[&](const Joint& joint){return joint.name==out.joints[j].name;});
        size_t index=static_cast<size_t>(found-raw.joints.begin());out.joints[j].rest=found->rest;out.joints[j].scale=found->scale;
        for(size_t f=0;f<out.frames();++f)out.frame(f)[j]=raw.frame(f)[index];
    }
    return out;
}
Clip export_clip(const Job& job,const Clip& clip) {
    if(job.rig=="soma30")return clip;
    auto out=soma30_to_77(clip);auto body=soma();
    if(job.append_history&&job.history77) {
        out.joints=job.history77->joints;
        for(size_t f=0;f<out.frames();++f)for(size_t j=0;j<out.joints.size();++j) {
            bool extra=std::none_of(body.begin(),body.end(),[&](const Joint& joint){return joint.name==out.joints[j].name;});
            if(f<job.history77->frames()||extra)out.frame(f)[j]=job.history77->frame(std::min(f,job.history77->frames()-1))[j];
            else if(j)out.frame(f)[j].p=job.history77->frame(job.history77->frames()-1)[j].p;
        }
        out.validate();return out;
    }
    for(size_t j=0;j<out.joints.size();++j) {
        auto found=std::find_if(body.begin(),body.end(),[&](const Joint& joint){return joint.name==out.joints[j].name;});
        if(job.full_input) {
            // Keep the exact source's tolerated rest offsets and every extra track,
            // rather than reducing then recreating them through a presentation rig.
            out.joints[j].rest=job.full_input->joints[j].rest;
            out.joints[j].scale=job.full_input->joints[j].scale;
            for(size_t f=0;f<out.frames();++f) {
                bool edited=f>job.begin&&f<job.end-1;
                if(!job.ranges.empty()){edited=false;for(auto [begin,end]:job.ranges)edited|=f>begin&&f+1<end;}
                if(found==body.end()||!edited)out.frame(f)[j]=job.full_input->frame(f)[j];
                else if(j)out.frame(f)[j].p=job.full_input->frame(f)[j].p;
            }
        }
    }
    out.validate();return out;
}
json verify_bundle(const fs::path& root) {
    if(!fs::is_regular_file(root/"bundle.json"))throw std::runtime_error("MoMask weights are not installed; use tide-motion models import-momask BUNDLE_DIRECTORY (prepared ONNX bundle)");
    if(sha256_file(root/"bundle.json")!=bundle_hash)throw std::runtime_error("MoMask bundle manifest does not match the supported prepared ONNX bundle");
    auto manifest=read_json(root/"bundle.json",65536);
    for(const auto& [name,pin]:manifest.at("files").items()) {
        if(fs::path(name).filename()!=fs::path(name))throw std::runtime_error("Invalid MoMask bundle file");
        auto p=root/name;if(!fs::is_regular_file(p)||fs::file_size(p)!=pin.at("size").get<uint64_t>()||sha256_file(p)!=pin.at("sha256").get<std::string>())throw std::runtime_error("MoMask model checksum mismatch: "+name);
    }
    return manifest;
}
struct Model {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"tide-momask"};
    Ort::SessionOptions options;
    Ort::MemoryInfo memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::map<std::string,std::unique_ptr<Ort::Session>> sessions;
    std::unique_ptr<MomaskGpu> gpu;Ort::AllocatorWithDefaultOptions allocator;
    json metadata;h::Statistics stats;std::vector<float> embeddings;
    explicit Model(const fs::path& root,unsigned threads,const std::string& device) {
        verify_bundle(root);
        metadata=read_json(root/"metadata.json",8*1024*1024);
        if(metadata.at("vocabulary").size()!=49408||metadata.at("mean").size()!=263||metadata.at("std").size()!=263)throw std::runtime_error("Invalid MoMask metadata");
        for(size_t i=0;i<263;++i){stats.mean[i]=metadata["mean"][i];stats.deviation[i]=metadata["std"][i];}stats.validate();
        embeddings.resize(5*513*512);std::ifstream in(root/"res_embeddings.bin",std::ios::binary);
        if(!in.read(reinterpret_cast<char*>(embeddings.data()),static_cast<std::streamsize>(embeddings.size()*sizeof(float))))throw std::runtime_error("Cannot read residual embeddings");
        options.SetIntraOpNumThreads(static_cast<int>(threads));options.SetInterOpNumThreads(1);
        if(device!="cpu")try {gpu=std::make_unique<MomaskGpu>(root,threads);}
        catch(const MomaskGpuFailure& e){if(device=="gpu")throw;std::cerr<<e.what()<<"; MoMask auto falls back to CPU.\n";}
        for(const auto* name:{"text","masked_step","residual_step","rvq_encoder","rvq_decoder"})if(!gpu||std::string_view(name)=="text")sessions[name]=std::make_unique<Ort::Session>(env,(root/(std::string(name)+".onnx")).c_str(),options);
        std::cerr<<"Loaded MoMask bundle once (native ORT "<<OrtGetApiBase()->GetVersionString()<<", "<<(gpu?"DirectML motion / CPU text":"CPU")<<"). Checkpoint commercial scope remains unresolved.\n";
    }
    template<class T> Ort::Value tensor(T* data,size_t count,std::initializer_list<int64_t> shape) {
        return Ort::Value::CreateTensor<T>(memory,data,count,shape.begin(),shape.size());
    }
    Ort::Value run(const std::string& name,std::vector<Ort::Value> values) {
        if(gpu&&name!="text") {
            std::vector<MomaskTensor> feed;
            for(const auto& value:values) {
                auto info=value.GetTensorTypeAndShapeInfo();int type=static_cast<int>(info.GetElementType());
                if(type!=1&&type!=7&&type!=9)throw std::runtime_error("Unexpected MoMask GPU input type");
                size_t bytes=info.GetElementCount()*(type==1?4:type==7?8:1);
                MomaskTensor t{type,info.GetShape(),std::vector<unsigned char>(bytes)};std::memcpy(t.bytes.data(),value.GetTensorRawData(),bytes);feed.push_back(std::move(t));
            }
            auto result=gpu->run(name,std::move(feed));
            auto output=Ort::Value::CreateTensor(allocator,result.shape.data(),result.shape.size(),static_cast<ONNXTensorElementDataType>(result.type));
            std::memcpy(output.GetTensorMutableRawData(),result.bytes.data(),result.bytes.size());return output;
        }
        static const char* names[]={"input0","input1","input2","input3"};const char* output="output";
        auto out=sessions.at(name)->Run(Ort::RunOptions{nullptr},names,values.data(),values.size(),&output,1);
        return std::move(out[0]);
    }
    std::vector<float> floats(const std::string& name,std::vector<Ort::Value> values,size_t expected) {
        auto result=run(name,std::move(values));auto info=result.GetTensorTypeAndShapeInfo();
        if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||info.GetElementCount()!=expected)throw std::runtime_error("Unexpected MoMask tensor shape/type");
        auto p=result.GetTensorData<float>();std::vector<float> out(p,p+expected);
        for(float f:out)if(!std::isfinite(f))throw std::runtime_error("Nonfinite MoMask model output");
        return out;
    }
    std::vector<float> text(const std::string& prompt,std::vector<int64_t>& tokens) {
        tokens=tokenize(prompt,metadata);std::vector<Ort::Value> args;args.push_back(tensor(tokens.data(),77,{1,77}));return floats("text",std::move(args),512);
    }
    std::vector<float> logits(bool residual,std::vector<int64_t>& ids,std::vector<float>& history,int64_t layer,std::vector<float>& text_vector,size_t n) {
        std::array<bool,49> padding{};std::vector<Ort::Value> args;
        // Upstream residual prepends zeros_like(mask[:, :2]); its logical
        // one-token case needs one masked padding slot to create both prefixes.
        size_t length=residual?std::max(size_t{2},n):n;std::vector<float> padded;
        if(length!=n){padded=history;padded.resize(length*512);padding[1]=true;}
        if(residual){auto& data=padded.empty()?history:padded;args.push_back(tensor(data.data(),data.size(),{1,static_cast<int64_t>(length),512}));args.push_back(tensor(&layer,1,{1}));}
        else args.push_back(tensor(ids.data(),ids.size(),{1,static_cast<int64_t>(n)}));
        args.push_back(tensor(text_vector.data(),512,{1,512}));args.push_back(tensor(padding.data(),length,{1,static_cast<int64_t>(length)}));
        auto result=floats(residual?"residual_step":"masked_step",std::move(args),length*(residual?513:512));
        if(length!=n){std::vector<float> first(513);for(size_t c=0;c<513;++c)first[c]=result[c*length];return first;}
        return result;
    }
    std::vector<h::Row> generate(const Job& job,json& recipe) {
        h::Encoded encoded;size_t n=static_cast<size_t>(std::floor(job.duration*5));
        std::vector<int64_t> source_tokens;
        auto encode=[&](const h::Encoded& features) {
            size_t source_n=features.rows.size()/4;auto normalized=h::normalize(features.rows,stats);
            std::vector<Ort::Value> args;args.push_back(tensor(normalized[0].data(),normalized.size()*263,{1,static_cast<int64_t>(normalized.size()),263}));
            auto result=run("rvq_encoder",std::move(args));auto info=result.GetTensorTypeAndShapeInfo();
            if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64||info.GetElementCount()!=source_n*6)throw std::runtime_error("Unexpected MoMask encoder result");
            const auto* p=result.GetTensorData<int64_t>();std::vector<int64_t> ids(p,p+source_n*6);for(auto id:ids)if(id<0||id>=512)throw std::runtime_error("Invalid MoMask encoder token ID");return ids;
        };
        if(job.input) {
            if(job.observations.empty()) {
                encoded=h::from_soma(*job.input);n=job.target_tokens?job.target_tokens:encoded.rows.size()/4;source_tokens=encode(encoded);
            } else {
                n=job.target_tokens;encoded.origin=job.input->frame(0)[0].p;encoded.origin.y=0;encoded.heading=kimodo_heading(*job.input,job.input->frame(0));h::Reference reference{encoded.origin,encoded.heading};source_tokens.assign(n*6,512);
                for(const auto& block:job.observations){auto ids=encode(h::from_soma(block.clip,&reference));if(block.token*6+ids.size()>source_tokens.size())throw std::runtime_error("MoMask observation exceeds token window");std::copy(ids.begin(),ids.end(),source_tokens.begin()+static_cast<std::ptrdiff_t>(block.token*6));}
            }
        }
        if(job.target_tokens)n=job.target_tokens;
        if(n<1||n>49)throw std::runtime_error("MoMask duration exceeds 49 tokens");
        std::vector<int64_t> text_tokens;auto cond=text(job.prompt,text_tokens);std::vector<float> zero(512),history(n*512,0);
        if(job.unconditional)cond=zero;
        std::vector<int64_t> ids(n,512);std::vector<double> scores(n,0);std::vector<bool> editable(n,true);
        if(job.input)for(size_t t=0;t<n;++t){
            bool known=t*6<source_tokens.size();if(known)ids[t]=source_tokens[t*6];
            size_t prefix=source_tokens.size()/6,overlap=std::min(static_cast<size_t>(job.context_overlap_tokens),prefix?prefix-1:0);
            editable[t]=job.continuation?t>=prefix-overlap:(t*6<job.end&&(t+1)*6>job.begin);
            if(!job.editable_tokens.empty()){if(job.editable_tokens.size()!=n)throw std::runtime_error("MoMask token mask size mismatch");editable[t]=job.editable_tokens[t];}
            if(!editable[t]&&ids[t]>=512)throw std::runtime_error("MoMask attempted to freeze a token without an observation");
            scores[t]=editable[t]?0:1e5;
        }
        size_t editable_count=static_cast<size_t>(std::count(editable.begin(),editable.end(),true));
        if(job.input){recipe["sourceTokens"]=source_tokens;recipe["editableTokens"]=json::array();for(bool value:editable)recipe["editableTokens"].push_back(value);}
        if(!editable_count&&!job.residual_layers)throw std::runtime_error("MoMask has no generative token freedom; release a pose anchor or enable residual layers");
        recipe["editableBaseTokens"]=editable_count;
        Random rng{static_cast<uint64_t>(job.seed)+0x9e3779b97f4a7c15ULL*job.stream};
        recipe["rngStream"]=job.stream;
        auto sample=[&](const std::vector<float>& logits,size_t t,size_t vocab,bool gumbel,bool residual=false) {
            std::vector<size_t> sorted(vocab);std::iota(sorted.begin(),sorted.end(),0);
            std::stable_sort(sorted.begin(),sorted.end(),[&](size_t a,size_t b){return logits[a*n+t]>logits[b*n+t];});
            if(!residual&&job.sampling=="greedy")return static_cast<int64_t>(sorted[0]);
            double temperature=residual?job.residual_temperature.value_or(job.temperature):job.temperature;
            double topkr=residual?job.residual_topkr.value_or(job.topkr):job.topkr;
            size_t k=std::max(size_t{1},static_cast<size_t>(std::ceil((1-topkr)*static_cast<double>(vocab))));
            double largest=logits[sorted[0]*n+t]/temperature,sum=0;
            if(gumbel) {double best=-1e300;size_t selected=0;for(size_t c=0;c<vocab;++c){double noise=-std::log(-std::log(rng.uniform()));if(std::find(sorted.begin(),sorted.begin()+static_cast<std::ptrdiff_t>(k),c)==sorted.begin()+static_cast<std::ptrdiff_t>(k))continue;double v=logits[c*n+t]/temperature+noise;if(v>best){best=v;selected=c;}}return static_cast<int64_t>(selected);}
            std::vector<double> probabilities(vocab);for(size_t i=0;i<k;++i){auto c=sorted[i];probabilities[c]=std::exp(logits[c*n+t]/temperature-largest);sum+=probabilities[c];}
            double choice=rng.uniform()*sum;for(size_t c=0;c<vocab;++c){choice-=probabilities[c];if(choice<0)return static_cast<int64_t>(c);}return static_cast<int64_t>(sorted[k-1]);
        };
        for(size_t iteration=0;editable_count&&iteration<job.steps;++iteration) {
            size_t count=h::mask_count(editable_count,iteration,job.steps);std::vector<size_t> sorted(n);std::iota(sorted.begin(),sorted.end(),0);
            std::stable_sort(sorted.begin(),sorted.end(),[&](size_t a,size_t b){return scores[a]<scores[b];});std::vector<bool> masked(n,false);
            for(size_t k=0;k<count;++k){ids[sorted[k]]=512;masked[sorted[k]]=true;}
            auto conditional=logits(false,ids,history,0,cond,n),unconditional=logits(false,ids,history,0,zero,n);
            for(size_t i=0;i<conditional.size();++i)conditional[i]=static_cast<float>(unconditional[i]+job.guidance*(conditional[i]-unconditional[i]));
            for(size_t t=0;t<n;++t) {
                int64_t token=sample(conditional,t,512,job.sampling=="gumbel");
                if(masked[t]) {ids[t]=token;double high=-1e300,total=0;for(size_t c=0;c<512;++c)high=std::max(high,static_cast<double>(conditional[c*n+t]));for(size_t c=0;c<512;++c)total+=std::exp(conditional[c*n+t]-high);scores[t]=std::exp(conditional[static_cast<size_t>(token)*n+t]-high)/total;}
                else scores[t]=1e5;
            }
        }
        std::vector<int64_t> all(n*6,-1);for(size_t t=0;t<n;++t)all[t*6]=ids[t];recipe["baseTokens"]=ids;
        for(int64_t layer=1;layer<=static_cast<int64_t>(job.residual_layers);++layer) {
            for(size_t t=0;t<n;++t)for(size_t c=0;c<512;++c)history[t*512+c]+=embeddings[(static_cast<size_t>(layer-1)*513+static_cast<size_t>(ids[t]))*512+c];
            auto conditional=logits(true,ids,history,layer,cond,n),unconditional=logits(true,ids,history,layer,zero,n);
            for(size_t i=0;i<conditional.size();++i)conditional[i]=static_cast<float>(unconditional[i]+job.residual_guidance*(conditional[i]-unconditional[i]));
            for(size_t t=0;t<n;++t){ids[t]=sample(conditional,t,513,true,true);if(job.preserve_context_residuals&&!editable[t]&&t*6+static_cast<size_t>(layer)<source_tokens.size())ids[t]=source_tokens[t*6+static_cast<size_t>(layer)];all[t*6+static_cast<size_t>(layer)]=ids[t]==512?-1:ids[t];}
        }
        recipe["tokens"]=all;recipe["textTokens"]=text_tokens;recipe["rng"]="splitmix64-open53-v1";
        std::vector<Ort::Value> args;args.push_back(tensor(all.data(),all.size(),{1,static_cast<int64_t>(n),6}));auto raw=floats("rvq_decoder",std::move(args),n*4*263);
        std::vector<h::Row> rows(n*4);std::memcpy(rows.data(),raw.data(),raw.size()*sizeof(float));rows=h::normalize(rows,stats,true);
        if(job.input){recipe["sourceOrigin"]={encoded.origin.x,encoded.origin.y,encoded.origin.z};recipe["sourceHeading"]=encoded.heading;}
        return rows;
    }
};
struct Rendered {Clip clip;std::vector<float> contacts;};
Clip slice(const Clip& clip,size_t begin,size_t end) {
    if(begin>=end||end>clip.frames())throw std::runtime_error("Invalid MoMask clip slice");
    Clip out=clip;out.poses.assign(clip.poses.begin()+static_cast<std::ptrdiff_t>(begin*30),clip.poses.begin()+static_cast<std::ptrdiff_t>(end*30));return out;
}
Job anchor_window(const Job& original,json& recipe) {
    Job job=original;bool anchors=false;
    for(const auto& pose:job.conditions.poses)if(!pose.effectors)anchors=true;
    if(!anchors)return job;
    size_t n=job.target_tokens?job.target_tokens:job.input?(job.input->frames()+6)/6:static_cast<size_t>(std::floor(job.duration*5));
    size_t frames=n*6-1,context=job.continuation?(job.context_frames?job.context_frames:job.input->frames()):0;
    std::vector<bool> editable(n,true);
    if(!job.editable_tokens.empty())editable=job.editable_tokens;
    else if(job.input)for(size_t t=0;t<n;++t){size_t count=(context+6)/6,overlap=std::min(static_cast<size_t>(job.context_overlap_tokens),count?count-1:0);editable[t]=job.continuation?t>=count-overlap:(t*6<job.end&&(t+1)*6>job.begin);}
    if(job.input&&job.observations.empty())job.observations.push_back({0,*job.input});
    recipe["tokenAnchors"]=json::array();
    for(const auto& pose:job.conditions.poses)if(!pose.effectors&&pose.frame<frames) {
        size_t token=std::min(n-1,pose.frame/6);if(!editable[token])continue;
        Clip observation;observation.joints=soma();for(size_t f=0;f<5;++f)observation.poses.insert(observation.poses.end(),pose.pose.begin(),pose.pose.end());
        if(!job.input)job.input=observation;
        job.observations.push_back({token,std::move(observation)});
        editable[token]=false;recipe["tokenAnchors"].push_back({{"frame",pose.frame},{"token",token},{"mode","whole-body RVQ context; spatial placement is postprocessing"}});
    }
    job.target_tokens=n;job.editable_tokens=std::move(editable);job.context_frames=context;
    return job;
}
Rendered sample_window(Model& model,const Job& requested,json& recipe) {
    Job controlled=anchor_window(requested,recipe);const auto& job=controlled;
    auto rows=model.generate(job,recipe);recipe["sourceFrames20Hz"]=rows.size();auto clip=h::to_soma(rows);
    if(job.input) {
        float heading=recipe["sourceHeading"];auto origin=recipe["sourceOrigin"].get<std::vector<float>>();Quat alignment=yaw(heading);
        for(size_t f=0;f<clip.frames();++f){auto& root=clip.frame(f)[0];root.p=rotate(alignment,root.p)+Vec3{origin[0],origin[1],origin[2]};root.q=normalized(alignment*root.q);}
        if(job.continuation) {
            size_t context=job.context_frames?job.context_frames:job.input->frames();size_t boundary=context-1;if(clip.frames()<=boundary)throw std::runtime_error("MoMask continuation window is too short");
            Vec3 drift=job.input->frame(boundary)[0].p-clip.frame(boundary)[0].p;
            for(size_t f=0;f<clip.frames();++f)clip.frame(f)[0].p=clip.frame(f)[0].p+drift;
            MotionConditions constraints;PoseConstraint p;p.frame=boundary;p.pose.assign(job.input->frame(boundary).begin(),job.input->frame(boundary).end());constraints.poses.push_back(std::move(p));
            recipe["boundaryModelErrorMetres"]=constraint_error(clip,constraints.poses[0]).position_max;
            correct_poses(clip,constraints,job.correction=="none"?0:job.correction_frames);
            std::copy_n(job.input->poses.begin(),context*30,clip.poses.begin());
            recipe["boundaryCorrection"]="constant root translation; configurable pose fade; context restored";
        } else if(requested.input&&!requested.source.empty()) {
            const auto& source=*requested.input;
            if(clip.frames()<source.frames())throw std::runtime_error("MoMask encoded window does not cover source timeline");
            clip.poses.resize(source.frames()*30);
            MotionConditions constraints;
            for(size_t f:{job.begin,job.end-1}){PoseConstraint p;p.frame=f;p.pose.assign(source.frame(f).begin(),source.frame(f).end());constraints.poses.push_back(std::move(p));}
            recipe["uncorrectedEndpointRootError"]=json::array();for(const auto& p:constraints.poses)recipe["uncorrectedEndpointRootError"].push_back(constraint_error(clip,p).root);
            Vec3 first=source.frame(job.begin)[0].p-clip.frame(job.begin)[0].p,last=source.frame(job.end-1)[0].p-clip.frame(job.end-1)[0].p;
            for(size_t f=job.begin;f<job.end;++f){float t=static_cast<float>(f-job.begin)/static_cast<float>(job.end-job.begin-1);clip.frame(f)[0].p=clip.frame(f)[0].p+first*(1-t)+last*t;}
            correct_poses(clip,constraints,job.correction=="none"?0:job.correction_frames);clip=h::splice(source,clip,job.begin,job.end);
            recipe["rootEndpointDriftCorrection"]="linear over edited interval";recipe["endpointCorrectionFrames"]=job.correction=="none"?0:job.correction_frames;
        }
    }
    Rendered out;out.clip=std::move(clip);out.contacts.resize(out.clip.frames()*4);
    for(size_t f=0;f<out.clip.frames();++f)for(size_t k=0;k<4;++k)out.contacts[f*4+k]=std::clamp(rows[std::min(rows.size()-1,f*2/3)][259+k],0.0f,1.0f);
    if(requested.input&&!requested.source.empty())for(size_t f=0;f<out.clip.frames();++f)if(f<=requested.begin||f+1>=requested.end)std::fill_n(out.contacts.begin()+static_cast<std::ptrdiff_t>(f*4),4,0.0f);
    return out;
}
Rendered render_job(Model& model,const Job& job,json& recipe) {
    if(job.input) {
        if(job.input->frames()<=293&&job.ranges.empty())return sample_window(model,job,recipe);
        Rendered result;result.clip=*job.input;result.contacts.resize(result.clip.frames()*4);
        auto ranges=job.ranges;if(ranges.empty())ranges.push_back({job.begin,job.end});
        recipe["windows"]=json::array();recipe["ranges"]=ranges;uint64_t stream=0;
        if(job.edit_mode=="continuous") {
            for(auto [begin,end]:ranges)for(size_t cursor=begin+1;cursor<end-1;) {
                size_t available=std::min(cursor,static_cast<size_t>(job.history_frames));
                size_t context=available<5?available:(available+1)/6*6-1;
                size_t offset=cursor-context,capacity=293-context;
                bool final=end-cursor<=capacity;
                size_t count=final?end-cursor:std::min(static_cast<size_t>(job.chunk_frames),capacity);
                if(!final&&end-cursor-count<12)count=end-cursor-12;
                Job window=job;window.full_input.reset();window.ranges.clear();window.stream=stream++;
                window.conditions.poses.clear();window.begin=context-1;window.context_frames=context;
                size_t limit=final?std::min(result.clip.frames(),offset+293):offset+293;
                window.target_tokens=(limit-offset+6)/6;
                if(final) {
                    window.input=slice(result.clip,offset,limit);window.end=end-offset;
                    size_t prefix=(context+6)/6,overlap=std::min(static_cast<size_t>(job.context_overlap_tokens),prefix?prefix-1:0);
                    size_t frozen=prefix-overlap,suffix=(window.end-1)/6;
                    window.editable_tokens.assign(window.target_tokens,true);
                    for(size_t t=0;t<frozen;++t)window.editable_tokens[t]=false;
                    for(size_t t=suffix;t<window.target_tokens;++t)window.editable_tokens[t]=false;
                    if(frozen)window.observations.push_back({0,slice(*window.input,0,frozen*6-1)});
                    window.observations.push_back({suffix,slice(*window.input,suffix*6,window.input->frames())});
                } else {
                    window.input=slice(result.clip,offset,cursor);window.continuation=true;window.end=293;
                }
                for(auto pose:job.conditions.poses)if(pose.frame>=cursor&&pose.frame<limit){pose.frame-=offset;window.conditions.poses.push_back(std::move(pose));}
                json part={{"offset",offset},{"begin",cursor},{"end",cursor+count},{"contextFrames",context},{"final",final},{"lookaheadFrames",limit-cursor-count}};
                auto generated=sample_window(model,window,part);
                std::copy_n(generated.clip.poses.begin()+static_cast<std::ptrdiff_t>(context*30),count*30,result.clip.poses.begin()+static_cast<std::ptrdiff_t>(cursor*30));
                std::copy_n(generated.contacts.begin()+static_cast<std::ptrdiff_t>(context*4),count*4,result.contacts.begin()+static_cast<std::ptrdiff_t>(cursor*4));
                recipe["windows"].push_back(std::move(part));cursor+=count;
            }
            recipe["windowedEditing"]="generated prefix continuation; final observed source suffix; exact outside/end poses";return result;
        }
        for(auto [begin,end]:ranges)for(size_t cursor=begin;cursor<end;) {
            size_t offset=cursor>60?cursor-60:0,stop=std::min(end,offset+233);
            if(end>stop&&end-stop<12)stop=end-12;
            size_t limit=std::min(result.clip.frames(),offset+293);
            Job window=job;window.input=slice(result.clip,offset,limit);window.full_input.reset();window.ranges.clear();window.begin=cursor-offset;window.end=stop-offset;window.stream=stream++;
            window.conditions.poses.clear();for(auto pose:job.conditions.poses)if(pose.frame>=cursor&&pose.frame<stop){pose.frame-=offset;window.conditions.poses.push_back(std::move(pose));}
            json part={{"offset",offset},{"begin",cursor},{"end",stop}};auto generated=sample_window(model,window,part);
            std::copy(generated.clip.poses.begin()+static_cast<std::ptrdiff_t>(window.begin*30),generated.clip.poses.begin()+static_cast<std::ptrdiff_t>(window.end*30),result.clip.poses.begin()+static_cast<std::ptrdiff_t>(cursor*30));
            std::copy(generated.contacts.begin()+static_cast<std::ptrdiff_t>(window.begin*4),generated.contacts.begin()+static_cast<std::ptrdiff_t>(window.end*4),result.contacts.begin()+static_cast<std::ptrdiff_t>(cursor*4));
            recipe["windows"].push_back(std::move(part));cursor=stop;
        }
        recipe["windowedEditing"]="bounded source-context windows; source poses retained at internal boundaries";return result;
    }
    if(job.duration<=9.8&&job.segments.empty()&&!job.history_clip)return sample_window(model,job,recipe);
    auto segments=job.segments;if(segments.empty())segments.push_back({job.prompt,job.duration});
    Clip accumulated;accumulated.joints=soma();size_t history_size=0;
    if(job.history_clip){accumulated=*job.history_clip;history_size=accumulated.frames();}
    std::vector<float> contacts(history_size*4);recipe["windows"]=json::array();uint64_t stream=0;
    for(const auto& [prompt,duration]:segments) {
        size_t remaining=static_cast<size_t>(std::lround(duration*30));
        while(remaining) {
            Job window=job;window.prompt=prompt;window.segments.clear();window.history_clip.reset();window.stream=stream++;
            size_t context=0;
            if(accumulated.frames()) {
                size_t available=std::min(accumulated.frames(),static_cast<size_t>(job.history_frames));
                context=std::max(size_t{5},(available+1)/6*6-1);context=std::min(context,accumulated.frames());
                window.input=slice(accumulated,accumulated.frames()-context,accumulated.frames());window.continuation=true;
                window.context_frames=context;
            }
            size_t count=std::min({remaining,293-context,static_cast<size_t>(job.chunk_frames)});
            if(remaining>count&&remaining-count<12&&count>=24)count-=12-(remaining-count);
            size_t frames=context+count;
            window.target_tokens=std::min(size_t{49},(frames+6)/6);
            window.conditions.poses.clear();
            long offset=static_cast<long>(accumulated.frames()-(job.append_history?0:history_size))-static_cast<long>(context);
            for(auto pose:job.conditions.poses){long local=static_cast<long>(pose.frame)-offset;if(local>=static_cast<long>(context)&&local<static_cast<long>(frames)){pose.frame=static_cast<size_t>(local);window.conditions.poses.push_back(std::move(pose));}}
            json part={{"prompt",prompt},{"contextFrames",context},{"newFrames",count},{"outputBegin",accumulated.frames()-(job.append_history?0:history_size)}};
            auto generated=sample_window(model,window,part);
            if(generated.clip.frames()<frames)throw std::runtime_error("MoMask window does not cover requested continuation");
            for(size_t f=context;f<frames;++f){auto pose=generated.clip.frame(f);for(size_t j=0;j<30;++j){auto transform=pose[j];if(j)transform.p=accumulated.joints[j].rest.p;accumulated.poses.push_back(transform);}}
            contacts.insert(contacts.end(),generated.contacts.begin()+static_cast<std::ptrdiff_t>(context*4),generated.contacts.begin()+static_cast<std::ptrdiff_t>(frames*4));
            recipe["windows"].push_back(std::move(part));remaining-=count;
        }
    }
    size_t begin=job.append_history?0:history_size;
    Rendered out;out.clip=slice(accumulated,begin,accumulated.frames());out.contacts.assign(contacts.begin()+static_cast<std::ptrdiff_t>(begin*4),contacts.end());
    recipe["generatedBegin"]=job.append_history?history_size:0;
    recipe["windowedGeneration"]="RVQ prefix-token context; independent window RNG streams; no overlap blending";return out;
}
double number(const std::string& s) {size_t used=0;double v=std::stod(s,&used);if(used!=s.size()||!std::isfinite(v))throw std::runtime_error("Invalid MoMask number: "+s);return v;}
size_t integer(const std::string& s) {double v=number(s);if(v<0||v>4294967295.0||std::floor(v)!=v)throw std::runtime_error("Invalid MoMask integer: "+s);return static_cast<size_t>(v);}
long signed_frame(const std::string& s) {double v=number(s);if(v<-18000||v>17999||std::floor(v)!=v)throw std::runtime_error("MoMask frame must be an integer in -18000..17999");return static_cast<long>(v);}
size_t output_frames(const Job& j) {
    if(j.input)return j.input->frames();
    size_t prefix=j.append_history&&j.history_clip?j.history_clip->frames():0;
    if(!j.segments.empty()){size_t n=prefix;for(const auto& part:j.segments)n+=static_cast<size_t>(std::lround(part.second*30));return n;}
    if(j.duration>9.8||!j.history.empty())return prefix+static_cast<size_t>(std::lround(j.duration*30));
    return static_cast<size_t>(std::floor(j.duration*5))*6-1;
}
std::pair<std::string,std::optional<long>> pose_reference(const std::string& value) {
    auto at=value.rfind('@');if(at==std::string::npos)return {value,std::nullopt};auto part=value.substr(at+1);
    if(part=="first")return {value.substr(0,at),0};
    if(part=="last")return {value.substr(0,at),-1};
    if(!part.empty()&&part.find_first_not_of("-0123456789")==std::string::npos)return {value.substr(0,at),signed_frame(part)};
    return {value,std::nullopt};
}
bool edited_frame(const Job& j,size_t f) {
    if(j.append_history&&j.history_clip)return f>=j.history_clip->frames();
    if(!j.input)return true;
    if(j.ranges.empty())return f>j.begin&&f+1<j.end;
    for(auto [begin,end]:j.ranges)if(f>begin&&f+1<end)return true;
    return false;
}
Clip read_pose_source(const fs::path& file,const std::optional<RetargetMapping>& mapping) {
    try {return load_glb(file);}
    catch(const std::exception&){if(!mapping)throw;Clip target;target.joints=soma();for(const auto& joint:target.joints)target.poses.push_back(joint.rest);return retarget(load_glb(file,"",false),target,*mapping,true).clip;}
}
void prepare_conditions(Job& j,bool defer) {
    size_t frames=output_frames(j);auto resolve=[&](long f){long n=f<0?static_cast<long>(frames)+f:f;if(n<0||n>=static_cast<long>(frames))throw std::runtime_error("MoMask constraint frame exceeds output timeline");return static_cast<size_t>(n);};
    j.conditions={};std::optional<RetargetMapping> mapping;if(!j.pose_map.empty())mapping=load_mapping(j.pose_map);
    for(const auto& spec:j.pose_specs) {
        size_t frame=j.append_history&&j.history_clip&&spec.kind=="start-pose"?j.history_clip->frames():resolve(spec.frame);if(!edited_frame(j,frame))throw std::runtime_error("MoMask edit targets must be strictly inside selected ranges; boundary/outside poses are preserved");
        auto [file,wanted]=pose_reference(spec.reference);if(file.empty())throw std::runtime_error("MoMask pose source is empty");
        auto ext=utf8path(file).extension();if(ext!=".glb"&&ext!=".fbx")throw std::runtime_error("MoMask pose source must be GLB or explicitly mapped FBX");
        if(defer&&!fs::exists(utf8path(file)))continue;
        auto source=read_pose_source(utf8path(file),mapping);
        long index=wanted.value_or(spec.kind=="start-pose"?-1:0);if(index<0)index+=static_cast<long>(source.frames());
        if(index<0||index>=static_cast<long>(source.frames()))throw std::runtime_error("MoMask pose source frame exceeds clip");
        PoseConstraint pose;pose.kind=spec.kind;pose.frame=frame;pose.file=file;pose.source_frame=index;pose.effectors=spec.effectors;auto p=source.frame(static_cast<size_t>(index));pose.pose.assign(p.begin(),p.end());auto rig=soma();for(size_t joint=1;joint<30;++joint)pose.pose[joint].p=rig[joint].rest.p;j.conditions.poses.push_back(std::move(pose));
    }
    for(auto [point,frame]:j.path_specs){point.frame=resolve(frame);if(!edited_frame(j,point.frame))throw std::runtime_error("MoMask path target lies outside editable interior");j.conditions.path.push_back(point);}
    auto end=std::find_if(j.conditions.poses.begin(),j.conditions.poses.end(),[](const PoseConstraint& p){return p.kind=="end-pose";});
    std::optional<PoseConstraint> original_end;if(end!=j.conditions.poses.end())original_end=*end;
    auto placement=j.placement=="auto"?Placement::automatic:j.placement=="world"?Placement::world:Placement::free;
    j.shift_last=place_constraints(j.conditions,placement);
    if(j.follow=="ends") {if(!original_end&&!defer)throw std::runtime_error("MoMask follow ends needs an end pose");if(original_end)follow_ends(j.conditions,*original_end);}
    if(j.fit_poses&&std::none_of(j.pose_specs.begin(),j.pose_specs.end(),[](const auto& p){return p.effectors==0;}))throw std::runtime_error("MoMask fit-poses requires a full-body pose");
    if(j.loop&&std::any_of(j.conditions.poses.begin(),j.conditions.poses.end(),[&](const auto& p){return p.frame+1==frames;}))throw std::runtime_error("MoMask loop supplies the final pose; omit targets at the last frame");
    std::set<size_t> pose_frames,path_frames;for(const auto& p:j.conditions.poses)if(!pose_frames.insert(p.frame).second)throw std::runtime_error("MoMask accepts one pose/effector source per frame; combine effector parts into one target");
    for(const auto& p:j.conditions.path)if(!path_frames.insert(p.frame).second)throw std::runtime_error("MoMask path frames must be distinct");
}
void apply_authoring(Rendered& rendered,const Job& job,json& recipe) {
    auto& clip=rendered.clip;auto conditions=job.conditions;
    if(job.shift_last)shift_to_last_pose(clip,conditions);
    if(job.loop) {PoseConstraint pose;pose.kind="loop-close";pose.frame=clip.frames()-1;pose.pose.assign(clip.frame(0).begin(),clip.frame(0).end());conditions.poses.push_back(std::move(pose));}
    if(job.input&&(!conditions.empty()||job.correction=="motion"||job.correction=="kimodo")) {
        auto ranges=job.ranges;if(ranges.empty())ranges.push_back({job.begin,job.end});
        for(auto [begin,end]:ranges)for(size_t f:{begin,end-1}){PoseConstraint p;p.kind="edit-boundary";p.frame=f;p.pose.assign(job.input->frame(f).begin(),job.input->frame(f).end());conditions.poses.push_back(std::move(p));}
    }
    if(job.append_history&&job.history_clip&&!conditions.empty()){PoseConstraint p;p.kind="history-boundary";p.frame=job.history_clip->frames()-1;p.pose.assign(job.history_clip->frame(p.frame).begin(),job.history_clip->frame(p.frame).end());conditions.poses.push_back(std::move(p));}
    auto measure=[&](){json result=json::array();for(const auto& pose:conditions.poses){auto e=constraint_error(clip,pose);json rotations=json::array();for(const auto& joint:pose.pose)rotations.push_back({joint.q.x,joint.q.y,joint.q.z,joint.q.w});result.push_back({{"targetRotations",rotations},{"kind",pose.kind},{"frame",pose.frame},{"effectors",pose.effectors},{"file",pose.file},{"sourceFrame",pose.source_frame},{"anchored",pose.anchored},{"targetRoot",{pose.pose[0].p.x,pose.pose[0].p.y,pose.pose[0].p.z}},{"rootMetres",e.root},{"positionMeanMetres",e.position_mean},{"positionMaxMetres",e.position_max},{"rotationMeanDegrees",e.rotation_mean},{"rotationMaxDegrees",e.rotation_max}});}return result;};
    if(!conditions.empty())recipe["constraintErrorBeforePostprocessing"]=measure();
    if(job.correction!="none") {
        std::vector<std::pair<size_t,Vec3>> deltas;
        for(const auto& pose:conditions.poses)if(!pose.effectors&&pose.anchored){Vec3 delta=pose.pose[0].p-clip.frame(pose.frame)[0].p;delta.y=0;deltas.push_back({pose.frame,delta});}
        if(!deltas.empty()) {
            std::sort(deltas.begin(),deltas.end(),[](const auto& a,const auto& b){return a.first<b.first;});
            for(size_t f=0;f<clip.frames();++f){auto next=std::upper_bound(deltas.begin(),deltas.end(),f,[](size_t frame,const auto& d){return frame<d.first;});Vec3 delta=next==deltas.begin()?deltas.front().second:next==deltas.end()?deltas.back().second:std::prev(next)->second+(next->second-std::prev(next)->second)*(static_cast<float>(f-std::prev(next)->first)/static_cast<float>(next->first-std::prev(next)->first));clip.frame(f)[0].p=clip.frame(f)[0].p+delta;}
            recipe["rootAnchorWarp"]="piecewise linear residual between anchored full-body poses; constant outside; explicit postprocessing";
        }
    }
    std::vector<std::pair<size_t,float>> heading;
    for(const auto& point:conditions.path)if(point.has_heading){float turn=point.heading*std::numbers::pi_v<float>/180-kimodo_heading(clip,clip.frame(point.frame));while(turn>std::numbers::pi_v<float>)turn-=2*std::numbers::pi_v<float>;while(turn<-std::numbers::pi_v<float>)turn+=2*std::numbers::pi_v<float>;heading.push_back({point.frame,turn});}
    if(!heading.empty()) {
        std::sort(heading.begin(),heading.end());for(size_t i=1;i<heading.size();++i){while(heading[i].second-heading[i-1].second>std::numbers::pi_v<float>)heading[i].second-=2*std::numbers::pi_v<float>;while(heading[i].second-heading[i-1].second<-std::numbers::pi_v<float>)heading[i].second+=2*std::numbers::pi_v<float>;}
        for(size_t f=0;f<clip.frames();++f){auto next=std::upper_bound(heading.begin(),heading.end(),f,[](size_t frame,const auto& point){return frame<point.first;});float turn=next==heading.begin()?heading.front().second:next==heading.end()?heading.back().second:std::prev(next)->second+(next->second-std::prev(next)->second)*static_cast<float>(f-std::prev(next)->first)/static_cast<float>(next->first-std::prev(next)->first);clip.frame(f)[0].q=normalized(yaw(turn)*clip.frame(f)[0].q);}
        recipe["pathHeadingMode"]="interpolated root-yaw postprocessing before IK";
    }
    auto requested_path=conditions.path;
    if(job.dense_path)densify_path(conditions,clip.frame(0)[0].p);
    if(job.input)std::erase_if(conditions.path,[&](const PathConstraint& point){return !edited_frame(job,point.frame);});
    if(job.append_history)std::erase_if(conditions.path,[&](const PathConstraint& point){return !edited_frame(job,point.frame);});
    bool spatial=!conditions.path.empty()||std::any_of(conditions.poses.begin(),conditions.poses.end(),[](const auto& p){return p.effectors!=0;});
    if(job.correction=="motion"||job.correction=="kimodo"||spatial) {
        auto report=correct_kimodo(clip,conditions,rendered.contacts,static_cast<float>(job.path_margin));
        recipe["motionCorrection"]={{"mode","native MotionCorrection postprocessing; not learned spatial conditioning"},{"contactSource","MoMask decoded HumanML channels 259..262"},{"rootMarginMetres",job.path_margin},{"effectorFrames",report.effector_frames},{"ankleFrames",report.ankle_frames},{"toeFrames",report.toe_frames},{"milliseconds",report.milliseconds}};
    } else if(job.correction=="tide")correct_poses(clip,conditions,job.correction_frames);
    if(job.input)for(size_t f=0;f<clip.frames();++f)if(!edited_frame(job,f))std::copy(job.input->frame(f).begin(),job.input->frame(f).end(),clip.frame(f).begin());
    if(job.preserve_root&&job.input){for(size_t f=0;f<clip.frames();++f)clip.frame(f)[0]=job.input->frame(f)[0];recipe["rootPreservation"]="exact source root transforms; body-only edit";}
    if(job.append_history&&job.history_clip)std::copy(job.history_clip->poses.begin(),job.history_clip->poses.end(),clip.poses.begin());
    auto path_record=[&](const PathConstraint& p){json item={{"frame",p.frame},{"x",p.x},{"z",p.z},{"metres",path_error(clip,p)}};if(p.has_heading){item["headingDegrees"]=p.heading;item["headingErrorDegrees"]=std::remainder(kimodo_heading(clip,clip.frame(p.frame))*180/std::numbers::pi_v<float>-p.heading,360.0f);}return item;};
    if(!conditions.path.empty()){recipe["pathErrors"]=json::array();for(const auto& point:conditions.path)recipe["pathErrors"].push_back(path_record(point));}
    if(!requested_path.empty()){recipe["requestedPathErrors"]=json::array();for(const auto& p:requested_path)recipe["requestedPathErrors"].push_back(path_record(p));}
    if(job.in_place)remove_root_travel(clip);
    recipe["correction"]=job.correction;recipe["correctionFrames"]=job.correction_frames;recipe["loop"]=job.loop;
    if(job.loop&&job.in_place){MotionConditions closure;PoseConstraint p;p.frame=clip.frames()-1;p.pose.assign(clip.frame(0).begin(),clip.frame(0).end());closure.poses.push_back(std::move(p));correct_poses(clip,closure,job.correction_frames);}
    if(!conditions.empty()){recipe["constraintErrorAfterPostprocessing"]=measure();recipe["constraintMode"]="whole-body RVQ token context plus explicit pose/spatial postprocessing";}
    recipe["footContacts"]=json::array();for(size_t k=0;k<4;++k)recipe["footContacts"].push_back(contact_intervals(rendered.contacts,k));
    std::set<size_t> protected_frames;for(const auto& pose:conditions.poses)protected_frames.insert(pose.frame);for(const auto& point:requested_path)protected_frames.insert(point.frame);
    if(job.input||job.append_history)for(size_t f=0;f<clip.frames();++f)if(!edited_frame(job,f))protected_frames.insert(f);
    if(!protected_frames.empty())recipe["protectedFrames"]=protected_frames;
}
Rendered candidate(Model& model,const Job& job,json& recipe) {
    auto generated=render_job(model,job,recipe);
    if(job.loop) {
        json initial=recipe;Job closed=job;closed.loop=false;
        auto first=std::find_if(closed.conditions.poses.begin(),closed.conditions.poses.end(),[](const PoseConstraint& p){return p.frame==0&&!p.effectors;});
        PoseConstraint target;
        if(first!=closed.conditions.poses.end())target=*first;
        else {target.kind="loop-start-context";target.frame=0;target.pose.assign(generated.clip.frame(0).begin(),generated.clip.frame(0).end());closed.conditions.poses.push_back(target);}
        target.kind="loop-end-context";target.frame=generated.clip.frames()-1;target.anchored=false;closed.conditions.poses.push_back(target);
        generated=render_job(model,closed,recipe);recipe["loopInitialPass"]=std::move(initial);
        apply_authoring(generated,closed,recipe);
        MotionConditions close;target.anchored=true;target.pose.assign(generated.clip.frame(0).begin(),generated.clip.frame(0).end());close.poses.push_back(target);correct_poses(generated.clip,close,job.correction_frames);
        auto error=constraint_error(generated.clip,target);recipe["loop"]=true;recipe["loopEndpointError"]={{"positionMetres",error.position_max},{"rotationDegrees",error.rotation_max}};
    } else apply_authoring(generated,job,recipe);
    return generated;
}
Rendered search_candidates(Model& model,const Job& job,json& recipe) {
    if(!job.fit_poses&&job.wanted_footsteps<0)return candidate(model,job,recipe);
    json records=json::array(),original=recipe,best_recipe;std::optional<Rendered> best;double score=1e300;
    for(unsigned attempt=0;attempt<job.attempts;++attempt) {
        Job test=job;test.seed+=attempt;json item=original;item["seed"]=test.seed;
        auto generated=candidate(model,test,item);size_t steps=footfalls(generated.clip).size();double pose_error=0;
        if(item.contains("constraintErrorBeforePostprocessing"))for(const auto& p:item["constraintErrorBeforePostprocessing"])if(p.at("effectors")==0)pose_error=std::max(pose_error,p.at("rotationMeanDegrees").get<double>());
        records.push_back({{"seed",test.seed},{"footfalls",steps},{"poseRotationMeanDegrees",pose_error}});
        if(job.wanted_footsteps>=0&&steps!=static_cast<size_t>(job.wanted_footsteps))continue;
        if(!best||pose_error<score){score=pose_error;best=std::move(generated);best_recipe=std::move(item);}
        if(!job.fit_poses)break;
    }
    if(!best)throw std::runtime_error("No MoMask candidate matched requested footsteps; no output saved: "+records.dump());
    recipe=std::move(best_recipe);recipe["selection"]={{"firstSeed",job.seed},{"attemptLimit",job.attempts},{"attempted",records},{"rankBy",job.fit_poses?"worst full-body mean rotation before postprocessing":"first matching geometric travelling-footfall count"}};
    if(job.wanted_footsteps>=0)recipe["selection"]["requestedFootsteps"]=job.wanted_footsteps;
    return std::move(*best);
}
void preflight(Job& j,bool defer_source=false) {
    if(j.append_history&&j.in_place)throw std::runtime_error("MoMask extend cannot remove travel from its preserved history");
    if(j.placement!="auto"&&j.placement!="world"&&j.placement!="free")throw std::runtime_error("MoMask placement is auto, world or free");
    if(j.follow!="world"&&j.follow!="ends")throw std::runtime_error("MoMask follow is world or ends");
    if(j.correction!="tide"&&j.correction!="none"&&j.correction!="motion"&&j.correction!="kimodo")throw std::runtime_error("MoMask correction is tide, motion (kimodo alias) or none");
    if(j.correction_frames<0||j.correction_frames>60||!std::isfinite(j.path_margin)||j.path_margin<0||j.path_margin>1)throw std::runtime_error("Invalid MoMask correction frames/path margin");
    if(j.dense_path&&j.path_specs.empty())throw std::runtime_error("MoMask dense-path needs path waypoints");
    if(j.correction=="none"&&(!j.path_specs.empty()||std::any_of(j.pose_specs.begin(),j.pose_specs.end(),[](const auto& p){return p.effectors!=0;})))throw std::runtime_error("MoMask paths/effectors require explicit postprocessing; correction none cannot satisfy them");
    if(j.loop&&(!j.in_place||!j.source.empty()||j.correction=="none"||std::any_of(j.pose_specs.begin(),j.pose_specs.end(),[](const auto& p){return p.kind=="end-pose";})))throw std::runtime_error("MoMask loop requires corrected in-place generation and supplies its own end pose");
    if(j.in_place&&!j.path_specs.empty())throw std::runtime_error("MoMask in-place and path travel are mutually exclusive");
    if(j.attempts<1||j.attempts>64||j.wanted_footsteps>64)throw std::runtime_error("MoMask attempts must be 1..64 and footsteps 0..64");
    bool selecting=j.fit_poses||j.wanted_footsteps>=0;
    if(j.attempts_explicit&&!selecting)throw std::runtime_error("MoMask attempts requires fit-poses or footsteps");
    if(selecting&&(j.variants>1||static_cast<uint64_t>(j.seed)+j.attempts-1>4294967295ULL))throw std::runtime_error("MoMask candidate selection needs one output and an unoverflowed seed range");
    if(j.sampling!="categorical"&&j.sampling!="gumbel"&&j.sampling!="greedy")throw std::runtime_error("MoMask sampling is categorical, gumbel or greedy");
    if(j.residual_layers>5)throw std::runtime_error("MoMask residual-layers must be 0..5");
    if(j.residual_temperature&&(!std::isfinite(*j.residual_temperature)||*j.residual_temperature<=0||*j.residual_temperature>10))throw std::runtime_error("Invalid MoMask residual temperature");
    if(j.residual_topkr&&(!std::isfinite(*j.residual_topkr)||*j.residual_topkr<0||*j.residual_topkr>=1))throw std::runtime_error("Invalid MoMask residual topkr");
    if(j.history_frames<12||j.history_frames>239)throw std::runtime_error("MoMask history-frames must be 12..239");
    if(j.chunk_frames<12||j.chunk_frames>293)throw std::runtime_error("MoMask chunk-frames must be 12..293");
    if(j.context_overlap_tokens>4)throw std::runtime_error("MoMask context-overlap-tokens must be 0..4");
    if(j.edit_mode!="continuous"&&j.edit_mode!="anchored")throw std::runtime_error("MoMask edit-mode is continuous or anchored");
    if(j.preserve_root&&j.source.empty())throw std::runtime_error("MoMask preserve-root applies to inpaint only");
    if(j.preserve_root&&(!j.pose_specs.empty()||!j.path_specs.empty()))throw std::runtime_error("MoMask preserve-root supports body-only edits; omit pose/path targets");
    if(j.threads<1||j.threads>1024)throw std::runtime_error("MoMask threads must be 1..1024");
    if(j.device!="cpu"&&j.device!="auto"&&j.device!="gpu")throw std::runtime_error("MoMask device is cpu, auto or gpu");
    if(j.rig!="soma30"&&j.rig!="soma77")throw std::runtime_error("MoMask rig is soma30 or soma77");
    if(j.variants<1||j.variants>64||static_cast<uint64_t>(j.seed)-j.variant_index+j.variants-1>4294967295ULL)throw std::runtime_error("MoMask variants must be 1..64 with seed range <=4294967295");
    if(j.variants_gif&&j.variants<2)throw std::runtime_error("MoMask variants-gif requires variants >1");
    if(j.output.empty()||j.output.extension()!=".glb"||fs::exists(j.output)||fs::exists(j.output.string()+".json")||fs::exists(j.output.string()+".part")||fs::exists(j.output.string()+".json.part"))throw std::runtime_error("MoMask needs new .glb, recipe and partial paths");
    if(j.prompt.empty()||j.prompt.find_first_not_of(" \t\n\r")==std::string::npos||j.prompt.size()>4096||!std::isfinite(j.duration)||j.duration<.2||j.duration>600||j.steps<2||j.steps>1000||!std::isfinite(j.guidance)||j.guidance<0||j.guidance>10||!std::isfinite(j.residual_guidance)||j.residual_guidance<0||j.residual_guidance>10||!std::isfinite(j.temperature)||j.temperature<=0||j.temperature>10||!std::isfinite(j.topkr)||j.topkr<0||j.topkr>=1)throw std::runtime_error("Invalid MoMask generation controls");
    for(unsigned char c:j.prompt)if(c>=127||(c<32&&c!='\n'&&c!='\r'&&c!='\t')||c=='&')throw std::runtime_error("MoMask requires ASCII prompts without HTML entities");
    if(j.prompt.find("<|")!=std::string::npos)throw std::runtime_error("MoMask special tokens refused");
    double total=0;if(j.segments.size()>16)throw std::runtime_error("MoMask allows at most 16 prompt segments");
    for(const auto& [prompt,duration]:j.segments) {
        if(prompt.empty()||prompt.find_first_not_of(" \t\r\n")==std::string::npos||prompt.size()>4096||prompt.find("<|")!=std::string::npos||!std::isfinite(duration)||duration<.2||duration>600)throw std::runtime_error("Invalid MoMask prompt segment");
        for(unsigned char c:prompt)if(c>=127||(c<32&&c!='\n'&&c!='\r'&&c!='\t')||c=='&')throw std::runtime_error("MoMask requires ASCII segment prompts without HTML entities");
        total+=duration;
    }
    if(total>600)throw std::runtime_error("MoMask total timeline exceeds 600 seconds");
    auto parent=j.output.parent_path();while(!parent.empty()&&!fs::exists(parent))parent=parent.parent_path();
    if(!parent.empty()&&!fs::is_directory(parent))throw std::runtime_error("MoMask output parent is not a directory");
    if(!j.contacts_out.empty()&&(j.variants>1||j.contacts_out.extension()!=".json"||fs::exists(j.contacts_out)||fs::exists(j.contacts_out.string()+".part")))throw std::runtime_error("MoMask contacts-out needs a new .json path and one output");
    if(!j.contacts_out.empty()){auto parent=j.contacts_out.parent_path();while(!parent.empty()&&!fs::exists(parent))parent=parent.parent_path();if(!parent.empty()&&!fs::is_directory(parent))throw std::runtime_error("MoMask contact plan parent is not a directory");}
    if(j.fit_poses&&std::none_of(j.pose_specs.begin(),j.pose_specs.end(),[](const auto& p){return p.effectors==0;}))throw std::runtime_error("MoMask fit-poses requires a full-body pose");
    if(j.follow=="ends"&&std::none_of(j.pose_specs.begin(),j.pose_specs.end(),[](const auto& p){return p.kind=="end-pose";}))throw std::runtime_error("MoMask follow ends needs an end pose");
    // An earlier batch output may defer timeline geometry, but existing pose
    // assets/mappings must still be validated before the first inference job.
    if(defer_source) {
        std::optional<RetargetMapping> mapping;if(!j.pose_map.empty())mapping=load_mapping(j.pose_map);
        for(const auto& spec:j.pose_specs) {
            auto [file,index]=pose_reference(spec.reference);auto path=utf8path(file);
            if(path.extension()!=".glb"&&path.extension()!=".fbx")throw std::runtime_error("MoMask pose source must be GLB or explicitly mapped FBX");
            if(!fs::exists(path))continue;
            auto clip=read_pose_source(path,mapping);long f=index.value_or(spec.kind=="start-pose"?-1:0);if(f<0)f+=static_cast<long>(clip.frames());
            if(f<0||f>=static_cast<long>(clip.frames()))throw std::runtime_error("MoMask pose source frame exceeds clip");
        }
    }
    if(!j.source.empty()) {
        if(j.source.extension()!=".glb")throw std::runtime_error("MoMask inpaint source must be GLB");
        if(!j.history.empty()||!j.segments.empty())throw std::runtime_error("MoMask inpaint does not accept history or prompt segments");
        if(j.in_place)throw std::runtime_error("MoMask inpaint refuses --in-place because it would change outside-range poses");
        auto ranges=j.ranges;if(ranges.empty())ranges.push_back({j.begin,j.end});size_t last=0;
        for(auto [begin,end]:ranges){if(begin>=end||end>18000||end-begin<12||begin<last)throw std::runtime_error("MoMask edit ranges must be ordered, disjoint, at least 12 frames, within 0..18000");last=end;}
        if(defer_source&&!fs::exists(j.source))return;
        auto rig=load_rig_glb(j.source);
        if(is_soma77(rig)) {
            if(j.rig_explicit&&j.rig!="soma77")throw std::runtime_error("MoMask inpaint must preserve its SOMA-77 source rig");
            j.full_input=preserved_source77(j.source);j.input=soma77_to_30(*j.full_input);j.rig="soma77";
        } else {
            if(rig.joints.size()!=30)throw std::runtime_error("MoMask inpaint requires SOMA-30 or SOMA-77; custom extra joints would not be preserved");
            if(j.rig=="soma77")throw std::runtime_error("MoMask inpaint preserves its source rig; use --rig soma77 during generation first");
            j.input=load_glb(j.source);
        }
        require_soma(*j.input);
        if(j.input->frames()>18000)throw std::runtime_error("MoMask source exceeds 18000 frames");
        for(auto [begin,end]:ranges){(void)begin;if(end>j.input->frames())throw std::runtime_error("MoMask edit range exceeds source clip");}
    }
    if(!j.history.empty()) {
        if(j.history.extension()!=".glb")throw std::runtime_error("MoMask history source must be GLB");
        if(defer_source&&!fs::exists(j.history))return;
        if(j.append_history&&is_soma77(load_rig_glb(j.history))) {
            if(j.rig_explicit&&j.rig!="soma77")throw std::runtime_error("MoMask extend must preserve its source rig");
            j.history77=preserved_source77(j.history);j.history_clip=soma77_to_30(*j.history77);j.rig="soma77";
        } else {if(j.append_history&&j.rig!="soma30")throw std::runtime_error("MoMask extend must preserve its source rig");j.history_clip=load_glb(j.history);}
        require_soma(*j.history_clip);
        if(j.history_clip->frames()<12||j.history_clip->frames()>18000)throw std::runtime_error("MoMask history must contain 12..18000 frames");
    }
    if(output_frames(j)>18000)throw std::runtime_error("MoMask output including history exceeds 18000 frames");
    prepare_conditions(j,defer_source);
    if(j.residual_layers==0&&output_frames(j)<=293&&!j.history_clip){json proof;auto controlled=anchor_window(j,proof);if(!controlled.editable_tokens.empty()&&std::none_of(controlled.editable_tokens.begin(),controlled.editable_tokens.end(),[](bool v){return v;}))throw std::runtime_error("MoMask has no generative token freedom; release an anchor or enable residual layers");}
}
}
struct MomaskContext::Impl {std::unique_ptr<Model> model;motion::fs::path root;unsigned threads=0;std::string device;};
MomaskContext::MomaskContext():impl(std::make_unique<Impl>()) {}
MomaskContext::~MomaskContext()=default;
const char* momask_help() {return R"(MoMask: native offline generation, source-context continuation and inpainting.
  tide-motion generate "PROMPT" --backend momask -o OUT.glb [options]
  tide-motion extend "PROMPT" --source SOMA.glb --duration SECONDS -o LONGER.glb
  tide-motion inpaint "PROMPT" --source SOMA.glb --begin F --end F -o EDIT.glb
  tide-motion inpaint "PROMPT" --source SOMA.glb --range BEGIN END [--range BEGIN END]... -o EDIT.glb
  tide-motion models import-momask PREPARED_ONNX_DIRECTORY [--models CACHE]
  tide-motion models check momask [--models CACHE]

Generation: --duration .2..600 (2), --then "PROMPT" --duration SECONDS (up to 16
segments / 600 seconds). Short plain generation retains native token rounding;
long generation, timelines and history use bounded native RVQ-context windows.
--history FILE.glb conditions new-only output; extend appends while preserving
its complete source prefix. Output max 18000 canonical 30 Hz frames.
--history-frames 12..239 (119), --chunk-frames 12..293 (89 new frames/window),
--context-overlap-tokens 0..4 (1), --preserve-context-residuals.
Inpaint sources up to 18000 frames; ordered disjoint ranges cover at least 12
frames each. Outside poses and range endpoints stay fixed, including SOMA-77
extra tracks. --edit-mode continuous|anchored (continuous) chooses generated
prefix context with a final source suffix or source anchors at each window.
--preserve-root keeps exact source root transforms on body-only inpainting.

Sampling: --seed 0..4294967295, --steps 2..1000 (18), --guidance 0..10 (4),
--residual-guidance 0..10 (generate 5 / inpaint 2), --temperature (0,10] (1),
--topkr [0,1) (.9), --sampling categorical|gumbel|greedy, --residual-layers 0..5 (5),
--residual-temperature (0,10], --residual-topkr [0,1), --unconditional.

Pose context: --start-pose FILE[@F], --end-pose FILE[@F],
--keyframe FRAME FILE[@F], --pose-map BONES.json, --placement auto|world|free,
--follow world|ends. Whole-body poses provide RVQ token context; exact pose,
world placement and spatial targets use explicitly recorded postprocessing.
--effector PARTS FRAME FILE[@F], --path X,Z[,YAW]@FRAME, --end-at X,Z[,YAW],
--dense-path, --path-margin 0..1 (.04). Paths target SOMA hips, not a learned
Kimodo constraint input. Effectors/paths use existing native MotionCorrection.
One pose/effector source per frame; combine parts (right-hand,left-hand,etc.).
--correction tide|motion|none (tide; kimodo is a motion alias),
--correction-frames 0..60 (10). None leaves pose anchors approximate; source edit
boundaries/history alignment are always retained. --contacts-out PLAN.json
exports qualifying model-predicted heel intervals for optional existing cleanup.

Candidates: --variants 1..64, --variants-gif, --tide PATH;
--fit-poses / --footsteps 0..64 with --attempts 1..64 (8), separate from variants.
Fit ranks pre-correction pose rotation error; footsteps filters geometric
travelling footfalls, not an action classifier. --in-place is generation-only.
--loop requires --in-place: native two-pass endpoint-token context plus explicit
closure correction; no universal contact/velocity continuity guarantee.

Runtime: --device cpu|auto|gpu (cpu), --threads 1..1024 (4), --rig soma30|soma77.
Windows GPU uses native DirectML motion networks; CLIP text stays CPU. Auto
falls back/restarts on CPU; gpu is strict. Batch reuses sessions. Inpaint/extend
keep the source rig. New SOMA-77 generations add static relaxed extra joints;
extensions retain the last extra-joint pose in the tail, with no predicted fingers.
ASCII prompts only. Models live in CACHE/momask; --models/TIDE_MOTION_MODELS/OS
cache chooses CACHE. No model downloads, Python runtime or hosted service.
Weights remain manually supplied, with unresolved checkpoint commercial scope.
See notes/momask-capabilities.md for tested controls, measurements and limits.
)";}
int momask_main(const std::vector<std::string>& args,MomaskContext* context,bool preflight_only,bool defer_source) {
    using namespace motion;
    if(args.size()<3||args[2]=="--help"||args[2]=="help") {
        std::cout<<momask_help();return 0;
    }
    fs::path root;std::vector<std::string> positional;std::map<std::string,std::string> options;
    std::vector<std::pair<std::string,double>> segments;std::vector<std::pair<size_t,size_t>> ranges;
    std::vector<Job::PoseSpec> pose_specs;std::vector<std::pair<PathConstraint,long>> path_specs;
    for(size_t i=3;i<args.size();++i) {
        if(args[i]=="--then") {
            if(i+1==args.size()||positional.size()!=1)throw std::runtime_error("MoMask --then needs a preceding prompt and another prompt");
            if(segments.empty())segments.push_back({positional[0],options.contains("--duration")?number(options.at("--duration")):2});
            segments.push_back({args[++i],2});
        } else if(args[i]=="--duration"&&!segments.empty()) {
            if(i+1==args.size())throw std::runtime_error("MoMask missing segment duration");
            segments.back().second=number(args[++i]);
        } else if(args[i]=="--range") {
            if(i+2>=args.size())throw std::runtime_error("MoMask --range needs BEGIN END");
            size_t begin=integer(args[++i]),end=integer(args[++i]);ranges.push_back({begin,end});
        } else if(args[i]=="--start-pose"||args[i]=="--end-pose") {
            auto kind=args[i];if(i+1==args.size())throw std::runtime_error("MoMask missing pose source");pose_specs.push_back({kind.substr(2),args[++i],kind=="--start-pose"?0:-1,0});
        } else if(args[i]=="--keyframe"||args[i]=="--effector") {
            bool effector=args[i]=="--effector";if(i+(effector?3:2)>=args.size())throw std::runtime_error("MoMask incomplete pose constraint");unsigned bits=effector?parse_effectors(args[++i]):0;long frame=signed_frame(args[++i]);pose_specs.push_back({effector?"effector":"keyframe",args[++i],frame,bits});
        } else if(args[i]=="--path"||args[i]=="--end-at") {
            bool last=args[i]=="--end-at";if(i+1==args.size())throw std::runtime_error("MoMask missing path point");auto text=args[++i];auto at=text.rfind('@');
            if(last){text+="@-1";at=text.rfind('@');}
            if(at==std::string::npos)throw std::runtime_error("MoMask path uses X,Z[,YAW]@FRAME");
            long frame=signed_frame(text.substr(at+1));text.resize(at);
            std::vector<double> values;size_t begin=0;while(true){auto comma=text.find(',',begin);values.push_back(number(text.substr(begin,comma-begin)));if(comma==std::string::npos)break;begin=comma+1;}
            if(values.size()<2||values.size()>3||std::abs(values[0])>10000||std::abs(values[1])>10000||(values.size()==3&&std::abs(values[2])>3600))throw std::runtime_error("MoMask path point is out of range");
            PathConstraint p;p.x=static_cast<float>(values[0]);p.z=static_cast<float>(values[1]);if(values.size()==3){p.has_heading=true;p.heading=static_cast<float>(values[2]);}path_specs.push_back({p,frame});
        } else if(args[i]=="--in-place"||args[i]=="--variants-gif"||args[i]=="--unconditional"||args[i]=="--dense-path"||args[i]=="--fit-poses"||args[i]=="--loop"||args[i]=="--no-correction"||args[i]=="--preserve-root"||args[i]=="--preserve-context-residuals") {if(options.contains(args[i]))throw std::runtime_error("Duplicate MoMask flag");options[args[i]]="true";}
        else if(args[i].starts_with("--")||args[i]=="-o") {if(i+1==args.size()||options.contains(args[i]))throw std::runtime_error("Missing or duplicate MoMask option");auto key=args[i];options[key]=args[++i];}
        else positional.push_back(args[i]);
    }
    const std::set<std::string> allowed={"--models","-o","--source","--begin","--end","--duration","--seed","--steps","--guidance","--residual-guidance","--temperature","--topkr","--threads","--device","--rig","--in-place","--variants","--variants-gif","--tide","--sampling","--residual-layers","--residual-temperature","--residual-topkr","--unconditional","--history","--history-frames","--chunk-frames","--pose-map","--placement","--follow","--correction","--correction-frames","--dense-path","--path-margin","--fit-poses","--footsteps","--attempts","--loop","--no-correction","--contacts-out","--context-overlap-tokens","--preserve-context-residuals","--preserve-root","--edit-mode"};
    for(const auto& [key,value]:options){(void)value;if(!allowed.contains(key))throw std::runtime_error("Unknown MoMask option: "+key);}
    if(!options.contains("--models"))throw std::runtime_error("MoMask needs a model bundle path");
    root=utf8path(options.at("--models"));
    if(args[2]=="check") {
        if(!positional.empty()||options.size()!=1)throw std::runtime_error("MoMask check accepts --models only");
        verify_bundle(root);std::cout<<"MoMask weights verified in "<<root.string()<<"\n";return 0;
    }
    if(args[2]=="install") {
        if(positional.size()!=1||options.size()!=1)throw std::runtime_error("MoMask install requires one private bundle and --models destination");
        fs::path source=utf8path(positional[0]);if(fs::exists(root))throw std::runtime_error("MoMask install destination already exists");auto manifest=verify_bundle(source);
        fs::path stage=root;stage+=".part";
        if(!stage.parent_path().empty())fs::create_directories(stage.parent_path());
        if(!fs::create_directory(stage))throw std::runtime_error("MoMask install staging path exists");
        try {for(const auto& [name,pin]:manifest["files"].items()){(void)pin;fs::copy_file(source/name,stage/name);}fs::copy_file(source/"bundle.json",stage/"bundle.json");verify_bundle(stage);fs::rename(stage,root);}catch(...){throw std::runtime_error("MoMask install failed; inspect staging directory: "+stage.string());}
        std::cout<<"Imported verified MoMask weights into "<<root.string()<<". Checkpoint commercial scope remains unresolved.\n";return 0;
    }
    std::vector<Job> jobs;
    if(args[2]=="batch") {
        if(positional.size()!=1||options.size()!=1)throw std::runtime_error("MoMask batch accepts a manifest and --models only");
        fs::path file=utf8path(positional[0]);auto manifest=read_json(file,1024*1024);
        if(manifest.at("schema")!=1||!manifest.at("jobs").is_array()||manifest["jobs"].empty()||manifest["jobs"].size()>64)throw std::runtime_error("Invalid MoMask batch schema");
        const std::set<std::string> fields={"prompt","output","source","begin","end","duration","seed","steps","guidance","residualGuidance","temperature","topkr","threads","device","rig","inPlace","variants","variantsGif","tide"};
        for(const auto& value:manifest["jobs"]) {
            for(const auto& [key,v]:value.items()){(void)v;if(!fields.contains(key))throw std::runtime_error("Unknown MoMask batch field: "+key);}
            Job j;j.prompt=value.at("prompt");j.output=file.parent_path()/utf8path(value.at("output").get<std::string>());
            if(value.contains("source"))j.source=file.parent_path()/utf8path(value.at("source").get<std::string>());
            if(!j.source.empty()&&value.contains("duration"))throw std::runtime_error("MoMask inpaint duration is fixed by its source timeline");
            auto get=[&](const char* key,double fallback){return value.contains(key)?number(value[key].dump()):fallback;};
            j.duration=get("duration",2);j.guidance=get("guidance",4);j.residual_guidance=get("residualGuidance",j.source.empty()?5:2);j.temperature=get("temperature",1);j.topkr=get("topkr",.9);
            j.seed=static_cast<uint32_t>(integer(value.value("seed",json(0)).dump()));j.steps=integer(value.value("steps",json(18)).dump());j.begin=integer(value.value("begin",json(0)).dump());j.end=integer(value.value("end",json(0)).dump());jobs.push_back(std::move(j));
            auto& added=jobs.back();added.threads=static_cast<unsigned>(integer(value.value("threads",json(4)).dump()));added.variants=static_cast<unsigned>(integer(value.value("variants",json(1)).dump()));
            added.device=value.value("device",std::string("cpu"));added.rig=value.value("rig",std::string("soma30"));added.rig_explicit=value.contains("rig");
            added.in_place=value.value("inPlace",false);added.variants_gif=value.value("variantsGif",false);added.tide=value.value("tide",std::string("tide"));
        }
    } else if(args[2]=="generate"||args[2]=="inpaint"||args[2]=="extend") {
        if(positional.size()!=1||!options.contains("-o"))throw std::runtime_error("MoMask requires one prompt and -o output.glb");
        Job j;j.prompt=positional[0];j.output=utf8path(options.at("-o"));
        j.pose_specs=pose_specs;j.path_specs=path_specs;
        j.segments=segments;j.ranges=ranges;
        if(!ranges.empty()&&(args[2]!="inpaint"||options.contains("--begin")||options.contains("--end")))throw std::runtime_error("MoMask --range replaces --begin/--end for inpaint only");
        if(!ranges.empty()){j.begin=ranges.front().first;j.end=ranges.back().second;}
        if(options.contains("--source"))j.source=utf8path(options.at("--source"));
        if(args[2]=="extend") {if(j.source.empty()||options.contains("--history")||options.contains("--begin")||options.contains("--end"))throw std::runtime_error("MoMask extend needs source and duration, without edit ranges/history option");j.history=j.source;j.source.clear();j.append_history=true;}
        if(args[2]=="inpaint"&&j.source.empty())throw std::runtime_error("MoMask inpaint requires --source");
        if(args[2]=="inpaint"&&options.contains("--duration"))throw std::runtime_error("MoMask inpaint duration is fixed by its source timeline");
        if(args[2]=="generate"&&(options.contains("--source")||options.contains("--begin")||options.contains("--end")))throw std::runtime_error("MoMask generate does not accept edit options");
        auto get=[&](const char* key,double fallback){return options.contains(key)?number(options.at(key)):fallback;};
        j.duration=get("--duration",2);j.guidance=get("--guidance",4);j.residual_guidance=get("--residual-guidance",j.source.empty()?5:2);j.temperature=get("--temperature",1);j.topkr=get("--topkr",.9);
        if(options.contains("--seed"))j.seed=static_cast<uint32_t>(integer(options.at("--seed")));
        if(options.contains("--steps"))j.steps=integer(options.at("--steps"));
        if(options.contains("--begin"))j.begin=integer(options.at("--begin"));
        if(options.contains("--end"))j.end=integer(options.at("--end"));
        if(options.contains("--threads"))j.threads=static_cast<unsigned>(integer(options.at("--threads")));
        if(options.contains("--variants"))j.variants=static_cast<unsigned>(integer(options.at("--variants")));
        if(options.contains("--device"))j.device=options.at("--device");
        if(options.contains("--rig")){j.rig=options.at("--rig");j.rig_explicit=true;}
        if(options.contains("--tide"))j.tide=options.at("--tide");
        j.in_place=options.contains("--in-place");j.variants_gif=options.contains("--variants-gif");
        j.unconditional=options.contains("--unconditional");
        if(options.contains("--sampling"))j.sampling=options.at("--sampling");
        if(options.contains("--residual-layers"))j.residual_layers=static_cast<unsigned>(integer(options.at("--residual-layers")));
        if(options.contains("--residual-temperature"))j.residual_temperature=number(options.at("--residual-temperature"));
        if(options.contains("--residual-topkr"))j.residual_topkr=number(options.at("--residual-topkr"));
        if(options.contains("--history"))j.history=utf8path(options.at("--history"));
        if(options.contains("--history-frames"))j.history_frames=static_cast<unsigned>(integer(options.at("--history-frames")));
        if(options.contains("--chunk-frames"))j.chunk_frames=static_cast<unsigned>(integer(options.at("--chunk-frames")));
        j.preserve_context_residuals=options.contains("--preserve-context-residuals");j.preserve_root=options.contains("--preserve-root");
        if(options.contains("--context-overlap-tokens"))j.context_overlap_tokens=static_cast<unsigned>(integer(options.at("--context-overlap-tokens")));
        if(options.contains("--edit-mode"))j.edit_mode=options.at("--edit-mode");
        if(options.contains("--pose-map"))j.pose_map=utf8path(options.at("--pose-map"));
        if(options.contains("--contacts-out"))j.contacts_out=utf8path(options.at("--contacts-out"));
        if(options.contains("--placement"))j.placement=options.at("--placement");
        if(options.contains("--follow"))j.follow=options.at("--follow");
        if(options.contains("--correction"))j.correction=options.at("--correction");
        if(options.contains("--no-correction")){if(options.contains("--correction"))throw std::runtime_error("Use correction none or no-correction, not both");j.correction="none";}
        if(options.contains("--correction-frames"))j.correction_frames=static_cast<int>(integer(options.at("--correction-frames")));
        if(options.contains("--path-margin"))j.path_margin=number(options.at("--path-margin"));
        if(options.contains("--attempts")){j.attempts=static_cast<unsigned>(integer(options.at("--attempts")));j.attempts_explicit=true;}
        if(options.contains("--footsteps")){auto steps=integer(options.at("--footsteps"));if(steps>64)throw std::runtime_error("MoMask footsteps must be 0..64");j.wanted_footsteps=static_cast<int>(steps);}
        j.fit_poses=options.contains("--fit-poses");j.dense_path=options.contains("--dense-path");j.loop=options.contains("--loop");
        jobs.push_back(std::move(j));
    }else throw std::runtime_error("Unknown MoMask action");
    std::map<fs::path,VariantSheet,PathLess> sheets;
    std::vector<Job> expanded;
    for(auto& j:jobs) {
        if(j.variants<1||j.variants>64||static_cast<uint64_t>(j.seed)+j.variants-1>4294967295ULL)throw std::runtime_error("MoMask variants must be 1..64 with seed range <=4294967295");
        if(j.variants>1) {
            VariantSheet sheet;sheet.tide=j.tide;sheet.png=variants_path(j.output,".png");if(j.variants_gif)sheet.gif=variants_path(j.output,".gif");
            if(!sheets.emplace(j.output,sheet).second)throw std::runtime_error("MoMask duplicate variants group");
        }
        for(unsigned v=0;v<j.variants;++v){Job candidate=j;candidate.seed+=v;candidate.variant_index=v;if(j.variants>1){candidate.variant_base=j.output;candidate.output=variant_path(j.output,candidate.seed);}expanded.push_back(std::move(candidate));}
    }
    jobs=std::move(expanded);
    std::set<fs::path,PathLess> outputs;
    for(auto& j:jobs){preflight(j,defer_source);for(auto p:{j.output,fs::path(j.output.string()+".json"),fs::path(j.output.string()+".part"),fs::path(j.output.string()+".json.part"),j.contacts_out})if(!p.empty()&&!outputs.insert(fs::weakly_canonical(fs::absolute(p))).second)throw std::runtime_error("MoMask batch output collision");}
    for(const auto& [base,sheet]:sheets){(void)base;for(auto p:{sheet.png,sheet.gif})if(!p.empty()&&(fs::exists(p)||fs::exists(p.string()+".part")||!outputs.insert(fs::weakly_canonical(fs::absolute(p))).second))throw std::runtime_error("MoMask contact sheet needs new distinct paths");}
    if(preflight_only)return 0;
    MomaskContext owned;if(!context)context=&owned;
    auto canonical=fs::weakly_canonical(fs::absolute(root));
    std::map<fs::path,std::vector<Clip>,PathLess> made;std::map<fs::path,std::vector<std::string>,PathLess> labels;
    for(const auto& job:jobs) {
        if(!context->impl->model||context->impl->root!=canonical||context->impl->threads!=job.threads||context->impl->device!=job.device) {
            context->impl->model.reset();context->impl->model=std::make_unique<Model>(root,job.threads,job.device);
            context->impl->root=canonical;context->impl->threads=job.threads;context->impl->device=job.device;
        }
        auto start=std::chrono::steady_clock::now();json recipe={{"tool","tide-motion"},{"command",job.append_history?"extend":job.input?"inpaint":"generate"},{"backend","momask"},{"checkpointRights","unresolved; manually supplied weights"},{"bundleSha256",bundle_hash},{"upstreamRevision","94a6636c9c463b7a9414c3401a6f1b67e6c51824"},{"checkpointMirrorRevision","75d34d1ab9d705d895ff4897c454fdc35775dfaa"},{"clipRevision","d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"},{"representationVersion",h::representation_version},{"prompt",job.prompt},{"seed",job.seed},{"steps",job.steps},{"guidance",job.guidance},{"residualGuidance",job.residual_guidance},{"temperature",job.temperature},{"topkr",job.topkr}};
        Rendered generated;
        json base_recipe=recipe;
        try {generated=search_candidates(*context->impl->model,job,recipe);}
        catch(const MomaskGpuFailure& e) {
            if(job.device!="auto")throw;
            std::cerr<<e.what()<<"; restarting this MoMask job from its seed on CPU.\n";
            recipe=base_recipe;
            context->impl->model.reset();context->impl->model=std::make_unique<Model>(root,job.threads,"cpu");
            generated=search_candidates(*context->impl->model,job,recipe);
        }
        auto& model=*context->impl->model;
        recipe["device"]=model.gpu?"gpu":"cpu";recipe["accelerator"]=model.gpu?"directml":"cpu";recipe["threads"]=job.threads;recipe["textDevice"]="cpu";
        if(model.gpu)recipe["sampledGpuLocalPeakBytes"]=model.gpu->peak_memory_bytes();
        if(!job.input){double requested=job.duration;if(!job.segments.empty()){requested=0;for(const auto& segment:job.segments)requested+=segment.second;}recipe["requestedDuration"]=requested;}
        if(job.input){recipe["source"]=fs::absolute(job.source).string();recipe["sourceSha256"]=sha256_file(job.source);recipe["begin"]=job.begin;recipe["end"]=job.end;}
        if(job.history_clip){recipe["history"]=fs::absolute(job.history).string();recipe["historySha256"]=sha256_file(job.history);recipe["historyFrames"]=job.history_frames;}
        recipe["contextOverlapTokens"]=job.context_overlap_tokens;recipe["preserveContextResiduals"]=job.preserve_context_residuals;recipe["preserveRoot"]=job.preserve_root;recipe["editMode"]=job.edit_mode;
        recipe["placement"]=job.placement;recipe["follow"]=job.follow;recipe["densePath"]=job.dense_path;recipe["pathMargin"]=job.path_margin;
        recipe["poseSources"]=json::array();for(const auto& pose:job.conditions.poses)if(!pose.file.empty())recipe["poseSources"].push_back({{"file",fs::absolute(utf8path(pose.file)).string()},{"sha256",sha256_file(utf8path(pose.file))},{"frame",pose.source_frame},{"outputFrame",pose.frame},{"effectors",pose.effectors}});
        if(!job.pose_map.empty())recipe["poseMap"]={{"file",fs::absolute(job.pose_map).string()},{"sha256",sha256_file(job.pose_map)}};
        recipe["chunkFrames"]=job.chunk_frames;recipe["historyContextLimit"]=job.history_frames;
        recipe["sampling"]=job.sampling;recipe["residualLayers"]=job.residual_layers;recipe["unconditional"]=job.unconditional;
        recipe["residualTemperature"]=job.residual_temperature.value_or(job.temperature);recipe["residualTopkr"]=job.residual_topkr.value_or(job.topkr);
        auto clip=std::move(generated.clip);
        require_soma(clip);
        clip.name=job.output.stem().string();recipe["frames"]=clip.frames();recipe["rig"]=job.rig;recipe["inPlace"]=job.in_place;
        if(job.history77){recipe["preservedHistoryExtraJoints"]=47;recipe["newExtraJoints"]="last source extra poses held; not model predictions";}
        else if(job.full_input)recipe["preservedSourceExtraJoints"]=47;
        else if(job.rig=="soma77")recipe["extraJoints"]="static relaxed pose; not model predictions";
        if(job.variants>1)recipe["variants"]={{"count",job.variants},{"firstSeed",job.seed-job.variant_index},{"sheet",sheets.at(job.variant_base).png.filename().string()}};
        recipe["nativeSeconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(!job.output.parent_path().empty())fs::create_directories(job.output.parent_path());
        save_glb(export_clip(job,clip),job.output);
        if(!job.contacts_out.empty()) {
            FootContacts contacts(clip.frames());for(size_t f=0;f<clip.frames();++f)std::copy_n(generated.contacts.begin()+static_cast<std::ptrdiff_t>(f*4),4,contacts[f].begin());
            auto planted=planted_contacts(contacts);json intervals=json::array();for(const auto& contact:planted)intervals.push_back({{"foot",contact.foot==0?"left":"right"},{"start",contact.start},{"end",contact.end}});
            recipe["contactPlan"]={{"source","MoMask decoded contact channels; not geometric contact detection"},{"intervals",intervals.size()}};
            if(!planted.empty()){if(!job.contacts_out.parent_path().empty())fs::create_directories(job.contacts_out.parent_path());std::ofstream plan(job.contacts_out);plan<<json{{"contacts",intervals}}.dump(2);if(!plan)throw std::runtime_error("Cannot write MoMask contact plan");recipe["contactPlan"]["file"]=job.contacts_out.string();}
            else recipe["contactPlan"]["file"]=nullptr;
        }
        std::ofstream out(job.output.string()+".json");out<<recipe.dump(2);if(!out)throw std::runtime_error("Cannot write MoMask recipe");
        if(job.variants>1){made[job.variant_base].push_back(clip);labels[job.variant_base].push_back("seed "+std::to_string(job.seed));}
        std::cout<<"MoMask saved "<<job.output.string()<<" ("<<clip.frames()<<" frames, "<<recipe["nativeSeconds"]<<" s native inference/conversion).\n";
    }
    for(const auto& [base,sheet]:sheets)try {preview_variants(made.at(base),labels.at(base),sheet);std::cout<<"Saved "<<sheet.png.string()<<"\n";}
    catch(const std::exception& e){std::cerr<<"Contact sheet skipped: "<<e.what()<<"\n";}
    return 0;
}
