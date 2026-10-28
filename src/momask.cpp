#include "momask.hpp"
#include "humanml.hpp"
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
    uint32_t seed=0;size_t steps=18,begin=0,end=0;std::optional<Clip> input;
};
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
    json metadata;h::Statistics stats;std::vector<float> embeddings;
    explicit Model(const fs::path& root) {
        verify_bundle(root);
        metadata=read_json(root/"metadata.json",8*1024*1024);
        if(metadata.at("vocabulary").size()!=49408||metadata.at("mean").size()!=263||metadata.at("std").size()!=263)throw std::runtime_error("Invalid MoMask metadata");
        for(size_t i=0;i<263;++i){stats.mean[i]=metadata["mean"][i];stats.deviation[i]=metadata["std"][i];}stats.validate();
        embeddings.resize(5*513*512);std::ifstream in(root/"res_embeddings.bin",std::ios::binary);
        if(!in.read(reinterpret_cast<char*>(embeddings.data()),static_cast<std::streamsize>(embeddings.size()*sizeof(float))))throw std::runtime_error("Cannot read residual embeddings");
        options.SetIntraOpNumThreads(4);options.SetInterOpNumThreads(1);
        for(const auto* name:{"text","masked_step","residual_step","rvq_encoder","rvq_decoder"})sessions[name]=std::make_unique<Ort::Session>(env,(root/(std::string(name)+".onnx")).c_str(),options);
        std::cerr<<"Loaded MoMask bundle once (native ORT "<<OrtGetApiBase()->GetVersionString()<<", CPU). Checkpoint commercial scope remains unresolved.\n";
    }
    template<class T> Ort::Value tensor(T* data,size_t count,std::initializer_list<int64_t> shape) {
        return Ort::Value::CreateTensor<T>(memory,data,count,shape.begin(),shape.size());
    }
    Ort::Value run(const std::string& name,std::vector<Ort::Value> values) {
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
        if(residual){args.push_back(tensor(history.data(),history.size(),{1,static_cast<int64_t>(n),512}));args.push_back(tensor(&layer,1,{1}));}
        else args.push_back(tensor(ids.data(),ids.size(),{1,static_cast<int64_t>(n)}));
        args.push_back(tensor(text_vector.data(),512,{1,512}));args.push_back(tensor(padding.data(),n,{1,static_cast<int64_t>(n)}));
        return floats(residual?"residual_step":"masked_step",std::move(args),n*(residual?513:512));
    }
    std::vector<h::Row> generate(const Job& job,json& recipe) {
        h::Encoded encoded;size_t n=static_cast<size_t>(std::floor(job.duration*5));
        std::vector<int64_t> source_tokens;
        if(job.input) {
            encoded=h::from_soma(*job.input);n=encoded.rows.size()/4;auto normalized=h::normalize(encoded.rows,stats);
            std::vector<Ort::Value> args;args.push_back(tensor(normalized[0].data(),normalized.size()*263,{1,static_cast<int64_t>(normalized.size()),263}));
            auto result=run("rvq_encoder",std::move(args));auto info=result.GetTensorTypeAndShapeInfo();
            if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64||info.GetElementCount()!=n*6)throw std::runtime_error("Unexpected MoMask encoder result");
            const auto* p=result.GetTensorData<int64_t>();source_tokens.assign(p,p+n*6);
        }
        if(n<1||n>49)throw std::runtime_error("MoMask duration exceeds 49 tokens");
        std::vector<int64_t> text_tokens;auto cond=text(job.prompt,text_tokens);std::vector<float> zero(512),history(n*512,0);
        std::vector<int64_t> ids(n,512);std::vector<double> scores(n,0);std::vector<bool> editable(n,true);
        if(job.input)for(size_t t=0;t<n;++t){ids[t]=source_tokens[t*6];editable[t]=(t*6<job.end&&(t+1)*6>job.begin);scores[t]=editable[t]?0:1e5;}
        size_t editable_count=static_cast<size_t>(std::count(editable.begin(),editable.end(),true));
        if(editable_count==0)throw std::runtime_error("MoMask edit mask covers no tokens");
        Random rng{job.seed};
        auto sample=[&](const std::vector<float>& logits,size_t t,size_t vocab,bool gumbel) {
            std::vector<size_t> sorted(vocab);std::iota(sorted.begin(),sorted.end(),0);
            std::stable_sort(sorted.begin(),sorted.end(),[&](size_t a,size_t b){return logits[a*n+t]>logits[b*n+t];});
            size_t k=std::max(size_t{1},static_cast<size_t>(std::ceil((1-job.topkr)*static_cast<double>(vocab))));
            double largest=logits[sorted[0]*n+t]/job.temperature,sum=0;
            if(gumbel) {double best=-1e300;size_t selected=0;for(size_t c=0;c<vocab;++c){double noise=-std::log(-std::log(rng.uniform()));if(std::find(sorted.begin(),sorted.begin()+static_cast<std::ptrdiff_t>(k),c)==sorted.begin()+static_cast<std::ptrdiff_t>(k))continue;double v=logits[c*n+t]/job.temperature+noise;if(v>best){best=v;selected=c;}}return static_cast<int64_t>(selected);}
            std::vector<double> probabilities(vocab);for(size_t i=0;i<k;++i){auto c=sorted[i];probabilities[c]=std::exp(logits[c*n+t]/job.temperature-largest);sum+=probabilities[c];}
            double choice=rng.uniform()*sum;for(size_t c=0;c<vocab;++c){choice-=probabilities[c];if(choice<0)return static_cast<int64_t>(c);}return static_cast<int64_t>(sorted[k-1]);
        };
        for(size_t iteration=0;iteration<job.steps;++iteration) {
            size_t count=h::mask_count(editable_count,iteration,job.steps);std::vector<size_t> sorted(n);std::iota(sorted.begin(),sorted.end(),0);
            std::stable_sort(sorted.begin(),sorted.end(),[&](size_t a,size_t b){return scores[a]<scores[b];});std::vector<bool> masked(n,false);
            for(size_t k=0;k<count;++k){ids[sorted[k]]=512;masked[sorted[k]]=true;}
            auto conditional=logits(false,ids,history,0,cond,n),unconditional=logits(false,ids,history,0,zero,n);
            for(size_t i=0;i<conditional.size();++i)conditional[i]=static_cast<float>(unconditional[i]+job.guidance*(conditional[i]-unconditional[i]));
            for(size_t t=0;t<n;++t) {
                int64_t token=sample(conditional,t,512,false);
                if(masked[t]) {ids[t]=token;double high=-1e300,total=0;for(size_t c=0;c<512;++c)high=std::max(high,static_cast<double>(conditional[c*n+t]));for(size_t c=0;c<512;++c)total+=std::exp(conditional[c*n+t]-high);scores[t]=std::exp(conditional[static_cast<size_t>(token)*n+t]-high)/total;}
                else scores[t]=1e5;
            }
        }
        std::vector<int64_t> all(n*6);for(size_t t=0;t<n;++t)all[t*6]=ids[t];recipe["baseTokens"]=ids;
        for(int64_t layer=1;layer<6;++layer) {
            for(size_t t=0;t<n;++t)for(size_t c=0;c<512;++c)history[t*512+c]+=embeddings[(static_cast<size_t>(layer-1)*513+static_cast<size_t>(ids[t]))*512+c];
            auto conditional=logits(true,ids,history,layer,cond,n),unconditional=logits(true,ids,history,layer,zero,n);
            for(size_t i=0;i<conditional.size();++i)conditional[i]=static_cast<float>(unconditional[i]+job.residual_guidance*(conditional[i]-unconditional[i]));
            for(size_t t=0;t<n;++t){ids[t]=sample(conditional,t,513,true);all[t*6+static_cast<size_t>(layer)]=ids[t]==512?-1:ids[t];}
        }
        recipe["tokens"]=all;recipe["textTokens"]=text_tokens;recipe["rng"]="splitmix64-open53-v1";
        std::vector<Ort::Value> args;args.push_back(tensor(all.data(),all.size(),{1,static_cast<int64_t>(n),6}));auto raw=floats("rvq_decoder",std::move(args),n*4*263);
        std::vector<h::Row> rows(n*4);std::memcpy(rows.data(),raw.data(),raw.size()*sizeof(float));rows=h::normalize(rows,stats,true);
        if(job.input){recipe["sourceOrigin"]={encoded.origin.x,encoded.origin.y,encoded.origin.z};recipe["sourceHeading"]=encoded.heading;}
        return rows;
    }
};
double number(const std::string& s) {size_t used=0;double v=std::stod(s,&used);if(used!=s.size()||!std::isfinite(v))throw std::runtime_error("Invalid MoMask number: "+s);return v;}
size_t integer(const std::string& s) {double v=number(s);if(v<0||v>4294967295.0||std::floor(v)!=v)throw std::runtime_error("Invalid MoMask integer: "+s);return static_cast<size_t>(v);}
void preflight(Job& j,bool defer_source=false) {
    if(j.output.empty()||j.output.extension()!=".glb"||fs::exists(j.output)||fs::exists(j.output.string()+".json")||fs::exists(j.output.string()+".part")||fs::exists(j.output.string()+".json.part"))throw std::runtime_error("MoMask needs new .glb, recipe and partial paths");
    if(j.prompt.empty()||j.prompt.find_first_not_of(" \t\n\r")==std::string::npos||j.prompt.size()>4096||!std::isfinite(j.duration)||j.duration<.2||j.duration>9.8||j.steps<2||j.steps>1000||!std::isfinite(j.guidance)||j.guidance<0||j.guidance>10||!std::isfinite(j.residual_guidance)||j.residual_guidance<0||j.residual_guidance>10||!std::isfinite(j.temperature)||j.temperature<=0||j.temperature>10||!std::isfinite(j.topkr)||j.topkr<0||j.topkr>=1)throw std::runtime_error("Invalid MoMask generation controls");
    for(unsigned char c:j.prompt)if(c>=127||(c<32&&c!='\n'&&c!='\r'&&c!='\t')||c=='&')throw std::runtime_error("MoMask requires ASCII prompts without HTML entities");
    if(j.prompt.find("<|")!=std::string::npos)throw std::runtime_error("MoMask special tokens refused");
    auto parent=j.output.parent_path();while(!parent.empty()&&!fs::exists(parent))parent=parent.parent_path();
    if(!parent.empty()&&!fs::is_directory(parent))throw std::runtime_error("MoMask output parent is not a directory");
    if(!j.source.empty()) {
        if(j.begin>=j.end||j.end>293||j.end-j.begin<12)throw std::runtime_error("MoMask edit range [begin,end) must cover at least 12 source frames, within 0..293");
        if(defer_source&&!fs::exists(j.source))return;
        if(load_rig_glb(j.source).joints.size()!=30)throw std::runtime_error("MoMask inpaint requires a 30-joint SOMA source; extra joints would not be preserved");
        j.input=load_glb(j.source);require_soma(*j.input);h::from_soma(*j.input);
        if(j.end>j.input->frames())throw std::runtime_error("MoMask edit range exceeds source clip");
    }
}
}
struct MomaskContext::Impl {std::unique_ptr<Model> model;motion::fs::path root;};
MomaskContext::MomaskContext():impl(std::make_unique<Impl>()) {}
MomaskContext::~MomaskContext()=default;
int momask_main(const std::vector<std::string>& args,MomaskContext* context,bool preflight_only,bool defer_source) {
    using namespace motion;
    if(args.size()<3||args[2]=="--help"||args[2]=="help") {
        std::cout<<"MoMask: native CPU text-to-motion and temporal inpainting. Weights are supplied manually.\n"
          "  tide-motion models import-momask BUNDLE_DIRECTORY [--models CACHE]\n"
          "  tide-motion models check momask [--models CACHE]\n"
          "  tide-motion generate PROMPT --backend momask -o OUT.glb [--duration .2..9.8] [--seed N] [--steps 2..1000]\n"
          "  tide-motion inpaint PROMPT --source SOMA.glb --begin F --end F -o OUT.glb\n"
          "Prepared weights go in CACHE/momask; CACHE follows --models, TIDE_MOTION_MODELS or the OS cache.\n"
          "Controls: --guidance 0..10 (4), --residual-guidance 0..10 (generate 5, inpaint 2), --temperature (0,10] (1), --topkr [0,1) (.9).\n"
          "ASCII prompts only. Output SOMA-30 GLB. Edits preserve outside [begin,end); endpoints are corrected inside the range.\n"
          "Raw .tar/.pt files are not the prepared ONNX bundle. No automatic MoMask downloads or commercial checkpoint clearance.\n";return 0;
    }
    fs::path root;std::vector<std::string> positional;std::map<std::string,std::string> options;
    for(size_t i=3;i<args.size();++i) {
        if(args[i].starts_with("--")||args[i]=="-o") {if(i+1==args.size()||options.contains(args[i]))throw std::runtime_error("Missing or duplicate MoMask option");auto key=args[i];options[key]=args[++i];}
        else positional.push_back(args[i]);
    }
    const std::set<std::string> allowed={"--models","-o","--source","--begin","--end","--duration","--seed","--steps","--guidance","--residual-guidance","--temperature","--topkr"};
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
        const std::set<std::string> fields={"prompt","output","source","begin","end","duration","seed","steps","guidance","residualGuidance","temperature","topkr"};
        for(const auto& value:manifest["jobs"]) {
            for(const auto& [key,v]:value.items()){(void)v;if(!fields.contains(key))throw std::runtime_error("Unknown MoMask batch field: "+key);}
            Job j;j.prompt=value.at("prompt");j.output=file.parent_path()/utf8path(value.at("output").get<std::string>());
            if(value.contains("source"))j.source=file.parent_path()/utf8path(value.at("source").get<std::string>());
            if(!j.source.empty()&&value.contains("duration"))throw std::runtime_error("MoMask inpaint duration is fixed by its source timeline");
            auto get=[&](const char* key,double fallback){return value.contains(key)?number(value[key].dump()):fallback;};
            j.duration=get("duration",2);j.guidance=get("guidance",4);j.residual_guidance=get("residualGuidance",j.source.empty()?5:2);j.temperature=get("temperature",1);j.topkr=get("topkr",.9);
            j.seed=static_cast<uint32_t>(integer(value.value("seed",json(0)).dump()));j.steps=integer(value.value("steps",json(18)).dump());j.begin=integer(value.value("begin",json(0)).dump());j.end=integer(value.value("end",json(0)).dump());jobs.push_back(std::move(j));
        }
    } else if(args[2]=="generate"||args[2]=="inpaint") {
        if(positional.size()!=1||!options.contains("-o"))throw std::runtime_error("MoMask requires one prompt and -o output.glb");
        Job j;j.prompt=positional[0];j.output=utf8path(options.at("-o"));
        if(options.contains("--source"))j.source=utf8path(options.at("--source"));
        if(args[2]=="inpaint"&&j.source.empty())throw std::runtime_error("MoMask inpaint requires --source");
        if(args[2]=="inpaint"&&options.contains("--duration"))throw std::runtime_error("MoMask inpaint duration is fixed by its source timeline");
        if(args[2]=="generate"&&(options.contains("--source")||options.contains("--begin")||options.contains("--end")))throw std::runtime_error("MoMask generate does not accept edit options");
        auto get=[&](const char* key,double fallback){return options.contains(key)?number(options.at(key)):fallback;};
        j.duration=get("--duration",2);j.guidance=get("--guidance",4);j.residual_guidance=get("--residual-guidance",j.source.empty()?5:2);j.temperature=get("--temperature",1);j.topkr=get("--topkr",.9);
        if(options.contains("--seed"))j.seed=static_cast<uint32_t>(integer(options.at("--seed")));
        if(options.contains("--steps"))j.steps=integer(options.at("--steps"));
        if(options.contains("--begin"))j.begin=integer(options.at("--begin"));
        if(options.contains("--end"))j.end=integer(options.at("--end"));
        jobs.push_back(std::move(j));
    }else throw std::runtime_error("Unknown MoMask action");
    std::set<fs::path,PathLess> outputs;
    for(auto& j:jobs){preflight(j,defer_source);for(auto p:{j.output,fs::path(j.output.string()+".json"),fs::path(j.output.string()+".part"),fs::path(j.output.string()+".json.part")})if(!outputs.insert(fs::weakly_canonical(fs::absolute(p))).second)throw std::runtime_error("MoMask batch output collision");}
    if(preflight_only)return 0;
    MomaskContext owned;if(!context)context=&owned;
    auto canonical=fs::weakly_canonical(fs::absolute(root));
    if(!context->impl->model||context->impl->root!=canonical){context->impl->model=std::make_unique<Model>(root);context->impl->root=canonical;}
    auto& model=*context->impl->model;
    for(const auto& job:jobs) {
        auto start=std::chrono::steady_clock::now();json recipe={{"tool","tide-motion"},{"command",job.input?"inpaint":"generate"},{"backend","momask"},{"checkpointRights","unresolved; manually supplied weights"},{"bundleSha256",bundle_hash},{"upstreamRevision","94a6636c9c463b7a9414c3401a6f1b67e6c51824"},{"checkpointMirrorRevision","75d34d1ab9d705d895ff4897c454fdc35775dfaa"},{"clipRevision","d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"},{"representationVersion",1},{"prompt",job.prompt},{"seed",job.seed},{"steps",job.steps},{"guidance",job.guidance},{"residualGuidance",job.residual_guidance},{"temperature",job.temperature},{"topkr",job.topkr}};
        auto rows=model.generate(job,recipe);recipe["sourceFrames20Hz"]=rows.size();if(!job.input)recipe["requestedDuration"]=job.duration;auto clip=h::to_soma(rows);
        if(job.input) {
            if(clip.frames()<job.input->frames())throw std::runtime_error("MoMask encoded window does not cover source timeline");
            clip.poses.resize(job.input->frames()*30); // padded encoding covers the source duration
            float heading=recipe["sourceHeading"];auto origin=recipe["sourceOrigin"].get<std::vector<float>>();Quat alignment=yaw(heading);
            for(size_t f=0;f<clip.frames();++f){auto& root_pose=clip.frame(f)[0];root_pose.p=rotate(alignment,root_pose.p)+Vec3{origin[0],origin[1],origin[2]};root_pose.q=normalized(alignment*root_pose.q);}
            MotionConditions constraints;
            for(size_t frame:{job.begin,job.end-1}){PoseConstraint p;p.frame=frame;p.pose.assign(job.input->frame(frame).begin(),job.input->frame(frame).end());p.anchored=true;constraints.poses.push_back(std::move(p));}
            recipe["uncorrectedEndpointRootError"]=json::array();for(const auto& p:constraints.poses)recipe["uncorrectedEndpointRootError"].push_back(constraint_error(clip,p).root);
            // Learned features integrate root velocities over the complete window.
            // Distribute endpoint drift across the edited interval, then correct
            // boundary body poses. No original pose outside the range is changed.
            Vec3 first=job.input->frame(job.begin)[0].p-clip.frame(job.begin)[0].p;
            Vec3 last=job.input->frame(job.end-1)[0].p-clip.frame(job.end-1)[0].p;
            for(size_t f=job.begin;f<job.end;++f){float t=static_cast<float>(f-job.begin)/static_cast<float>(job.end-job.begin-1);clip.frame(f)[0].p=clip.frame(f)[0].p+first*(1-t)+last*t;}
            recipe["rootEndpointDriftCorrection"]="linear over edited interval";
            correct_poses(clip,constraints,10);clip=h::splice(*job.input,clip,job.begin,job.end);
            recipe["source"]=fs::absolute(job.source).string();recipe["sourceSha256"]=sha256_file(job.source);recipe["begin"]=job.begin;recipe["end"]=job.end;recipe["endpointCorrectionFrames"]=10;
        }
        require_soma(clip);clip.name=job.output.stem().string();recipe["frames"]=clip.frames();recipe["nativeSeconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(!job.output.parent_path().empty())fs::create_directories(job.output.parent_path());
        save_glb(clip,job.output);std::ofstream out(job.output.string()+".json");out<<recipe.dump(2);if(!out)throw std::runtime_error("Cannot write MoMask recipe");
        std::cout<<"MoMask saved "<<job.output.string()<<" ("<<clip.frames()<<" frames, "<<recipe["nativeSeconds"]<<" s native inference/conversion).\n";
    }
    return 0;
}
