#include "dex2hoi.hpp"
#include "dex2hoi_bundle.hpp"
#include "dex2hoi_motion.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <numbers>
#include <regex>
#include <set>
#include <stdexcept>

namespace motion::dex2hoi {
namespace {
using nlohmann::json;
constexpr size_t feature_count=429,prediction_frames=60,prefix_frames=10;
constexpr size_t object_values=2*512*512;
void require(bool ok,const std::string& message) {if(!ok)throw std::runtime_error("Dex2HOI: "+message);}
json read_json(const fs::path& path,size_t limit=32*1024*1024) {
    require(fs::is_regular_file(path)&&fs::file_size(path)<=limit,"missing or oversized JSON: "+path.string());
    std::ifstream input(path);json result;input>>result;return result;
}
void finite(std::span<const float> values,const char* label) {
    for(float f:values)require(std::isfinite(f),std::string("nonfinite ")+label);
}
std::vector<float> vector(const json& value,size_t count,const char* label) {
    require(value.is_array()&&value.size()==count,std::string("wrong ")+label+" size");
    auto result=value.get<std::vector<float>>();finite(result,label);return result;
}
void new_file(const fs::path& path) {
    require(!path.empty()&&!fs::exists(path)&&!fs::exists(path.string()+".part"),"output or partial exists: "+path.string());
}
void write_json(const fs::path& path,const json& value) {
    new_file(path);if(path.has_parent_path())fs::create_directories(path.parent_path());
    fs::path part=path.string()+".part";
#ifdef _WIN32
    FILE* file=_wfopen(part.c_str(),L"wbx");
#else
    FILE* file=std::fopen(part.c_str(),"wbx");
#endif
    require(file!=nullptr,"cannot create JSON partial exclusively");
    try {
        auto text=value.dump(2)+"\n";
        bool ok=std::fwrite(text.data(),1,text.size(),file)==text.size();
        if(std::fclose(file))ok=false;
        file=nullptr;require(ok,"cannot write JSON partial");
        fs::create_hard_link(part,path);std::error_code ignored;fs::remove(part,ignored);
    }catch(...){if(file)std::fclose(file);std::error_code ignored;fs::remove(part,ignored);throw;}
}
unsigned integer(const std::string& text,unsigned maximum,const char* label) {
    require(!text.empty()&&text.find_first_not_of("0123456789")==std::string::npos,std::string("invalid ")+label);
    auto result=std::stoull(text);require(result<=maximum,std::string(label)+" exceeds bound");return static_cast<unsigned>(result);
}
float number(const std::string& text,const char* label) {
    size_t used=0;float result=std::stof(text,&used);require(used==text.size()&&std::isfinite(result),std::string("invalid ")+label);return result;
}
std::vector<int64_t> tokenize(std::string text,const json& metadata) {
    require(!text.empty()&&text.size()<=4096,"prompt must contain 1..4096 ASCII bytes");
    for(char& c:text) {
        auto u=static_cast<unsigned char>(c);
        require(u<127&&(u>=32||c=='\n'||c=='\r'||c=='\t')&&c!='&',"prompt supports ASCII without HTML entities only");
        if(c>='A'&&c<='Z')c=static_cast<char>(c-'A'+'a');
    }
    require(text.find("<|")==std::string::npos,"prompt cannot contain tokenizer special tokens");
    require(metadata.at("vocabulary").size()==49408,"invalid CLIP vocabulary");
    std::map<std::string,int64_t> vocabulary;int64_t id=0;
    for(const auto& word:metadata.at("vocabulary"))vocabulary.emplace(word.get<std::string>(),id++);
    std::map<std::pair<std::string,std::string>,size_t> ranks;size_t rank=0;
    for(const auto& pair:metadata.at("merges")) {
        require(pair.is_array()&&pair.size()==2,"invalid CLIP merge");
        ranks.emplace(std::make_pair(pair[0].get<std::string>(),pair[1].get<std::string>()),rank++);
    }
    std::regex pattern("'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\\sA-Za-z0-9]+");
    std::vector<int64_t> result{49406};
    for(std::sregex_iterator i(text.begin(),text.end(),pattern),end;i!=end;++i) {
        std::string word=i->str();std::vector<std::string> pieces;
        for(char c:word)pieces.emplace_back(1,c);
        pieces.back()+="</w>";
        while(pieces.size()>1) {
            size_t best=ranks.size()+1;std::pair<std::string,std::string> selected;
            for(size_t p=0;p+1<pieces.size();++p) {
                auto it=ranks.find({pieces[p],pieces[p+1]});
                if(it!=ranks.end()&&it->second<best){best=it->second;selected=it->first;}
            }
            if(best>ranks.size())break;
            std::vector<std::string> merged;
            for(size_t p=0;p<pieces.size();++p) {
                if(p+1<pieces.size()&&pieces[p]==selected.first&&pieces[p+1]==selected.second){merged.push_back(pieces[p]+pieces[p+1]);++p;}
                else merged.push_back(pieces[p]);
            }
            pieces=std::move(merged);
        }
        for(const auto& piece:pieces)result.push_back(vocabulary.at(piece));
    }
    require(result.size()>1,"prompt is empty after tokenization");result.push_back(49407);
    if(result.size()>77){result.resize(77);result.back()=49407;}result.resize(77,0);return result;
}
struct Random {
    uint64_t state;
    double uniform() {
        state+=0x9e3779b97f4a7c15ULL;uint64_t z=state;
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;z^=z>>31;
        return std::min((static_cast<double>(z>>11)+.5)/9007199254740992.0,1.0-1.0/9007199254740992.0);
    }
    void normal(std::span<float> output) {
        for(size_t i=0;i<output.size();i+=2) {
            double radius=std::sqrt(-2*std::log(uniform())),angle=2*std::numbers::pi*uniform();
            output[i]=static_cast<float>(radius*std::cos(angle));
            if(i+1<output.size())output[i+1]=static_cast<float>(radius*std::sin(angle));
        }
    }
};
struct TensorSpec {const char* name;ONNXTensorElementDataType type;std::vector<int64_t> shape;};
void graph_contract(Ort::Session& session,std::initializer_list<TensorSpec> inputs,const TensorSpec& output) {
    Ort::AllocatorWithDefaultOptions allocator;
    require(session.GetInputCount()==inputs.size()&&session.GetOutputCount()==1,"unexpected graph input/output count");
    size_t index=0;
    for(const auto& expected:inputs) {
        auto name=session.GetInputNameAllocated(index,allocator);auto info=session.GetInputTypeInfo(index);auto type=info.GetTensorTypeAndShapeInfo();
        require(std::string(name.get())==expected.name&&type.GetElementType()==expected.type&&type.GetShape()==expected.shape,"unexpected graph input contract: "+std::string(expected.name));++index;
    }
    auto name=session.GetOutputNameAllocated(0,allocator);auto info=session.GetOutputTypeInfo(0);auto type=info.GetTensorTypeAndShapeInfo();
    require(std::string(name.get())==output.name&&type.GetElementType()==output.type&&type.GetShape()==output.shape,"unexpected graph output contract: "+std::string(output.name));
}
}

std::vector<int64_t> sampling_timesteps(unsigned steps) {
    require(steps>=2&&steps<=500,"steps must be 2..500");std::vector<int64_t> result;result.reserve(steps);
    for(unsigned i=0;i<steps;++i)result.push_back(static_cast<int64_t>(std::llround(static_cast<double>(i)*499/static_cast<double>(steps-1))));
    return result;
}
std::array<double,500> cosine_alphas() {
    std::array<double,500> result{};double product=1;
    auto alpha=[](double time){return std::pow(std::cos((time+.008)/1.008*std::numbers::pi/2),2);};
    for(size_t i=0;i<result.size();++i) {
        double beta=std::min(1-alpha(static_cast<double>(i+1)/500)/alpha(static_cast<double>(i)/500),.999);
        product*=1-beta;result[i]=product;
    }
    return result;
}
std::vector<float> ddim_step(std::span<const float> x,std::span<const float> x0,double alpha,double previous_alpha) {
    require(x.size()==x0.size()&&alpha>0&&alpha<1&&previous_alpha>=alpha&&previous_alpha<=1,"invalid DDIM state");
    finite(x,"DDIM input");finite(x0,"DDIM prediction");std::vector<float> result(x.size());
    // Extracted diffusion arrays and coefficients are float32 in the reference.
    float reciprocal=static_cast<float>(std::sqrt(1/alpha));
    float reciprocal_noise=static_cast<float>(std::sqrt(1/alpha-1));
    float alpha_previous=static_cast<float>(previous_alpha);
    float signal=std::sqrt(alpha_previous),noise=std::sqrt(1-alpha_previous);
    for(size_t i=0;i<x.size();++i)result[i]=x0[i]*signal+noise*((reciprocal*x[i]-x0[i])/reciprocal_noise);
    finite(result,"DDIM result");return result;
}

struct Context::Impl {
    Bundle bundle;
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING,"tide-dex2hoi"};
    Ort::SessionOptions options;
    Ort::MemoryInfo memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::unique_ptr<Ort::Session> clip,object,denoiser;
    json tokenizer;
    explicit Impl(const fs::path& root,unsigned threads):bundle(inspect_bundle(root)) {
        require(threads>=1&&threads<=64,"threads must be 1..64");
        options.SetIntraOpNumThreads(static_cast<int>(threads));options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        clip=std::make_unique<Ort::Session>(environment,(root/"clip.onnx").c_str(),options);
        object=std::make_unique<Ort::Session>(environment,(root/"object.onnx").c_str(),options);
        denoiser=std::make_unique<Ort::Session>(environment,(root/"denoiser.onnx").c_str(),options);
        const auto f=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,i=ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,b=ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        graph_contract(*clip,{{"tokens",i,{1,77}}},{"text",f,{1,512}});
        graph_contract(*object,{{"points",f,{2,128,3}}},{"objects",f,{2,512,512}});
        graph_contract(*denoiser,{{"x",f,{1,429,1,60}},{"timestep",i,{1}},{"prefix",f,{1,429,1,10}},
            {"text",f,{1,512}},{"objects",f,{2,512,512}},{"object_mask",b,{2}},
            {"init_features",f,{1,72}},{"obj_x0",f,{1,2,3}},{"text_scale",f,{1}}},
            {"prediction",f,{1,429,1,60}});
        tokenizer=read_json(root/"tokenizer.json",16*1024*1024);
    }
    template<class T> Ort::Value tensor(const T* values,size_t count,std::initializer_list<int64_t> shape) {
        return Ort::Value::CreateTensor<T>(memory,const_cast<T*>(values),count,shape.begin(),shape.size());
    }
    std::vector<float> run(Ort::Session& session,std::initializer_list<const char*> names,std::vector<Ort::Value> input,const char* output,size_t count) {
        auto values=session.Run(Ort::RunOptions{nullptr},names.begin(),input.data(),input.size(),&output,1);
        auto type=values.front().GetTensorTypeAndShapeInfo();
        require(type.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT&&type.GetElementCount()==count,"invalid graph result");
        const float* data=values.front().GetTensorData<float>();std::vector<float> result(data,data+count);finite(result,"network output");return result;
    }
};
Context::Context(const fs::path& root,unsigned threads):impl(std::make_unique<Impl>(root,threads)) {}
Context::~Context()=default;
const json& Context::manifest() const {return impl->bundle.manifest;}
const std::string& Context::manifest_sha256() const {return impl->bundle.manifest_sha256;}
std::vector<float> Context::text(const std::string& prompt,std::vector<int64_t>* output_tokens) {
    auto tokens=tokenize(prompt,impl->tokenizer);if(output_tokens)*output_tokens=tokens;
    std::vector<Ort::Value> input;input.push_back(impl->tensor(tokens.data(),tokens.size(),{1,77}));
    return impl->run(*impl->clip,{"tokens"},std::move(input),"text",512);
}
std::vector<float> Context::objects(std::span<const float> points) {
    require(points.size()==2*128*3,"object point tensors require two clouds of 128 xyz points");finite(points,"object points");
    std::vector<Ort::Value> input;input.push_back(impl->tensor(points.data(),points.size(),{2,128,3}));
    return impl->run(*impl->object,{"points"},std::move(input),"objects",object_values);
}
std::vector<float> Context::denoise(std::span<const float> x,int64_t timestep,const Conditions& condition,float text_scale) {
    require(x.size()==feature_count*prediction_frames&&condition.prefix.size()==feature_count*prefix_frames&&
        condition.text.size()==512&&condition.objects.size()==object_values,"invalid denoiser tensor count");
    require(timestep>=0&&timestep<500&&(text_scale==0||text_scale==1),"invalid denoiser timestep/text mask");
    finite(x,"denoiser x");finite(condition.prefix,"prefix");finite(condition.text,"text");finite(condition.objects,"object embeddings");
    finite(condition.init_features,"initial features");finite(condition.root_x0,"root x0");finite(condition.obj_x0,"object x0");
    std::vector<Ort::Value> input;
    input.push_back(impl->tensor(x.data(),x.size(),{1,429,1,60}));input.push_back(impl->tensor(&timestep,1,{1}));
    input.push_back(impl->tensor(condition.prefix.data(),condition.prefix.size(),{1,429,1,10}));
    input.push_back(impl->tensor(condition.text.data(),512,{1,512}));input.push_back(impl->tensor(condition.objects.data(),object_values,{2,512,512}));
    input.push_back(impl->tensor(condition.object_mask.data(),2,{2}));input.push_back(impl->tensor(condition.init_features.data(),72,{1,72}));
    input.push_back(impl->tensor(condition.obj_x0.data(),6,{1,2,3}));
    input.push_back(impl->tensor(&text_scale,1,{1}));
    return impl->run(*impl->denoiser,{"x","timestep","prefix","text","objects","object_mask","init_features","obj_x0","text_scale"},std::move(input),"prediction",feature_count*prediction_frames);
}
std::vector<float> Context::sample(const Conditions& condition,const SamplingOptions& options,json& recipe) {
    require(std::isfinite(options.guidance)&&options.guidance>=0&&options.guidance<=10,"guidance must be 0..10");
    auto timesteps=sampling_timesteps(options.steps);auto alphas=cosine_alphas();std::vector<float> x(feature_count*prediction_frames);Random random{options.seed};random.normal(x);
    recipe["rng"]="splitmix64-open53-boxmuller-v1";recipe["seed"]=options.seed;recipe["samplingTimesteps"]=timesteps;
    recipe["sampler"]="ddim-x0-cosine500-eta0-v1";recipe["guidance"]=options.guidance;
    for(size_t iteration=timesteps.size();iteration-->0;) {
        int64_t timestep=timesteps[iteration];std::vector<float> prediction;
        if(options.guidance==0)prediction=denoise(x,timestep,condition,0);
        else {
            prediction=denoise(x,timestep,condition,1);
            if(options.guidance!=1) {
                auto unconditional=denoise(x,timestep,condition,0);
                for(size_t i=0;i<prediction.size();++i)prediction[i]=unconditional[i]+options.guidance*(prediction[i]-unconditional[i]);
            }
        }
        double previous=iteration?alphas[static_cast<size_t>(timesteps[iteration-1])]:1;
        x=ddim_step(x,prediction,alphas[static_cast<size_t>(timestep)],previous);
        if(iteration%25==0||iteration+1==timesteps.size())std::cerr<<"Dex2HOI DDIM "<<(timesteps.size()-iteration)<<"/"<<timesteps.size()<<"\n";
    }
    return x;
}

const char* help(){return R"(tide-motion models import-dex2hoi PREPARED-BUNDLE [--models CACHE]
tide-motion models check dex2hoi [--models CACHE]
tide-motion dex2hoi generate --input OWN-HISTORY.json --prompt TEXT -o NEW.glb
    [--bundle BUNDLE]
    [--seed N] [--steps 2..500] [--guidance 0..10] [--threads 1..64]
tide-motion dex2hoi denoise --bundle BUNDLE --input TENSORS.json -o NEW.json [--threads N]
tide-motion dex2hoi install PREPARED-BUNDLE --bundle NEW-PRIVATE-DESTINATION
tide-motion dex2hoi check --bundle BUNDLE

Native CPU ONNX inference with manually supplied weights. Generation uses one
10-frame history plus 60 predicted frames at 30 Hz. The input names an explicit
65-bone owner rig mapping, history and two object slots with local point samples.
No dataset lookup, Python runtime or automatic model download is performed.
The default sampler uses every original cosine diffusion timestep (500), DDIM
eta=0 and text-only CFG 2.5. Fewer steps are an explicit experimental respacing.
`denoise` is a developer tensor/parity path, not an animation quality check.
Weight import/check never establishes a license grant. See notes/dex2hoi-native.md.
)";}

namespace {
void keys(const json& value,std::initializer_list<const char*> allowed,const char* label) {
    require(value.is_object(),std::string(label)+" must be an object");
    for(auto it=value.begin();it!=value.end();++it)
        require(std::find_if(allowed.begin(),allowed.end(),[&](const char* key){return it.key()==key;})!=allowed.end(),std::string("unknown ")+label+" key: "+it.key());
}
template<size_t N> std::array<float,N> array(const json& value,const char* label) {
    auto values=vector(value,N,label);std::array<float,N> result{};std::copy(values.begin(),values.end(),result.begin());return result;
}
fs::path relative_file(const fs::path& source,const json& value) {
    auto path=utf8path(value.get<std::string>());require(!path.empty(),"empty input path");
    return fs::absolute(path.is_absolute()?path:source.parent_path()/path);
}
Transform transform(const json& value) {
    keys(value,{"position","rotation"},"pose");auto p=array<3>(value.at("position"),"pose position");auto q=array<4>(value.at("rotation"),"pose rotation");
    double norm=0;for(float f:q)norm+=f*f;require(std::abs(norm-1)<1e-3,"pose quaternion must be unit xyzw");
    return {{p[0],p[1],p[2]},normalized(Quat{q[0],q[1],q[2],q[3]})};
}
std::vector<float> points(const json& value) {
    require(value.is_array()&&value.size()==128,"object points must contain 128 local xyz samples");std::vector<float> result;result.reserve(128*3);
    for(const auto& point:value){auto xyz=vector(point,3,"object point");result.insert(result.end(),xyz.begin(),xyz.end());}return result;
}
Conditions tensor_conditions(Context& context,const json& input,std::vector<int64_t>& tokens) {
    Conditions condition;condition.prefix=vector(input.at("prefix"),429*10,"prefix");
    if(input.contains("prompt")){require(!input.contains("text"),"provide prompt or text tensor, not both");condition.text=context.text(input.at("prompt").get<std::string>(),&tokens);}
    else condition.text=vector(input.at("text"),512,"text");
    if(input.contains("points")){require(!input.contains("objects"),"provide points or object embedding tensor, not both");condition.objects=context.objects(vector(input.at("points"),2*128*3,"points"));}
    else condition.objects=vector(input.at("objects"),object_values,"objects");
    const auto& mask=input.at("objectMask");require(mask.is_array()&&mask.size()==2&&mask[0].is_boolean()&&mask[1].is_boolean(),"objectMask needs two booleans");
    condition.object_mask={mask[0].get<bool>(),mask[1].get<bool>()};
    condition.init_features=array<72>(input.at("initFeatures"),"initFeatures");if(input.contains("rootX0"))condition.root_x0=array<3>(input.at("rootX0"),"rootX0");
    condition.obj_x0=array<6>(input.at("objectX0"),"objectX0");return condition;
}
struct GenerationInput {
    Dex2HoiRig rig;Clip history;std::array<Dex2HoiObject,2> objects;std::vector<float> points;
    fs::path history_path;json manifest;
};
GenerationInput generation_input(const fs::path& source) {
    GenerationInput result;result.manifest=read_json(source);const auto& input=result.manifest;
    keys(input,{"version","history","bones","objects"},"history manifest");require(input.at("version")==1,"history manifest version must be 1");
    result.history_path=relative_file(source,input.at("history"));result.history=load_glb(result.history_path,"",false);
    require(result.history.frames()==10&&std::abs(result.history.fps-30)<1e-4,"history must have exactly 10 frames at 30 Hz");
    result.rig.joints=result.history.joints;
    const auto& bones=input.at("bones");require(bones.is_array()&&bones.size()==65,"bones must explicitly map 65 checkpoint-order entries");
    for(size_t i=0;i<65;++i) {
        auto name=bones[i].get<std::string>();auto found=std::find_if(result.rig.joints.begin(),result.rig.joints.end(),[&](const Joint& joint){return joint.name==name;});
        require(found!=result.rig.joints.end(),"mapped bone absent from history: "+name);result.rig.bones[i]=static_cast<size_t>(found-result.rig.joints.begin());
    }
    result.rig.validate();const auto& objects=input.at("objects");require(objects.is_array()&&objects.size()==2,"two explicit object slots required");
    for(size_t slot=0;slot<2;++slot) {
        const auto& spec=objects[slot];keys(spec,{"name","mesh","present","exportTrack","points","poses"},"object");auto& object=result.objects[slot];
        object.name=spec.at("name").get<std::string>();object.present=spec.value("present",true);object.export_track=spec.value("exportTrack",true);
        if(spec.contains("mesh")){object.mesh=relative_file(source,spec.at("mesh"));require(fs::is_regular_file(object.mesh),"object mesh does not exist");}
        auto cloud=points(spec.at("points"));result.points.insert(result.points.end(),cloud.begin(),cloud.end());
        for(size_t i=0;i<128;++i)object.surface_points.push_back({cloud[3*i],cloud[3*i+1],cloud[3*i+2]});
        const auto& poses=spec.at("poses");require(poses.is_array()&&poses.size()==10,"object poses must contain exactly ten world transforms");
        for(const auto& pose:poses)object.poses.push_back(transform(pose));
    }
    return result;
}
Dex2HoiNormalizer normalizer(const fs::path& bundle) {
    auto input=read_json(bundle/"normalization.json",128*1024);Dex2HoiNormalizer result;
    result.mean=array<429>(input.at("mean"),"normalization mean");result.stddev=array<429>(input.at("std"),"normalization std");
    result.epsilon=input.value("eps",1e-5f);result.validate();return result;
}
}

int command(const std::vector<std::string>& args) {
    if(args.size()<3||args[2]=="help"||std::find(args.begin()+2,args.end(),"--help")!=args.end()){std::cout<<help();return 0;}
    const auto& action=args[2];
    if(action=="install") {
        require(args.size()==6&&args[4]=="--bundle","install needs PREPARED-BUNDLE --bundle NEW-DESTINATION");
        auto bundle=import_bundle(utf8path(args[3]),utf8path(args[5]));
        std::cout<<"Dex2HOI private bundle installed: "<<bundle.root.string()<<"\nSHA256 "<<bundle.manifest_sha256<<"\n";return 0;
    }
    if(action=="check") {
        require(args.size()==5&&args[3]=="--bundle","check needs --bundle BUNDLE");
        auto bundle=inspect_bundle(utf8path(args[4]));std::cout<<"Dex2HOI manual bundle verified: "<<bundle.root.string()<<"\nSHA256 "<<bundle.manifest_sha256<<"\n";return 0;
    }
    require(action=="generate"||action=="denoise","action must be install, check, generate or denoise");
    std::map<std::string,std::string> options;
    for(size_t i=3;i<args.size();++i) {
        auto name=args[i];if(name=="--output")name="-o";
        require(name=="--bundle"||name=="--input"||name=="--prompt"||name=="--seed"||name=="--steps"||name=="--guidance"||name=="--threads"||name=="-o","unknown option: "+name);
        require(!options.contains(name)&&i+1<args.size(),"missing or duplicate option: "+name);options[name]=args[++i];
    }
    for(const char* required:{"--input","-o"})require(options.contains(required),"missing option: "+std::string(required));
    if(!options.contains("--bundle"))options["--bundle"]=(model_directory()/"dex2hoi").string();
    fs::path bundle=utf8path(options.at("--bundle")),source=utf8path(options.at("--input")),output=utf8path(options.at("-o"));new_file(output);
    unsigned threads=options.contains("--threads")?integer(options.at("--threads"),64,"threads"):4;
    require(threads>0,"threads must be 1..64");
    if(action=="denoise") {
        require(!options.contains("--prompt")&&!options.contains("--seed")&&!options.contains("--steps")&&!options.contains("--guidance"),"denoise uses its explicit input tensors only");
        require(output.extension()==".json","denoise output needs .json");auto input=read_json(source,64*1024*1024);
        keys(input,{"version","x","prefix","text","prompt","objects","points","objectMask","initFeatures","rootX0","objectX0","timestep","textScale"},"denoiser input");
        require(input.at("version")==1&&input.at("timestep").is_number_integer(),"invalid denoiser input version/timestep");
        Context context(bundle,threads);std::vector<int64_t> tokens;auto condition=tensor_conditions(context,input,tokens);
        auto prediction=context.denoise(vector(input.at("x"),429*60,"x"),input.at("timestep").get<int64_t>(),condition,input.value("textScale",1.0f));
        write_json(output,{{"version",1},{"bundleSha256",context.manifest_sha256()},{"inputSha256",sha256_file(source)},{"prediction",prediction},{"textTokens",tokens}});
        std::cout<<"Dex2HOI native denoiser result: "<<output.string()<<"\n";return 0;
    }
    require(options.contains("--prompt"),"generation requires --prompt");require(output.extension()==".glb","generation output needs .glb");
    fs::path recipe_path=output.string()+".json";new_file(recipe_path);
    SamplingOptions sampling;
    if(options.contains("--seed"))sampling.seed=integer(options.at("--seed"),UINT32_MAX,"seed");
    if(options.contains("--steps"))sampling.steps=integer(options.at("--steps"),500,"steps");
    if(options.contains("--guidance"))sampling.guidance=number(options.at("--guidance"),"guidance");
    (void)sampling_timesteps(sampling.steps);require(sampling.guidance>=0&&sampling.guidance<=10,"guidance must be 0..10");
    auto own=generation_input(source);auto stats=normalizer(bundle);Dex2HoiFormat format;
    auto history=encode_dex2hoi_history(own.rig,own.history,own.objects,format,stats);Conditions condition;
    condition.prefix.resize(429*10);for(size_t f=0;f<10;++f)for(size_t c=0;c<429;++c)condition.prefix[c*10+f]=history.features[f*429+c];
    condition.init_features=dex2hoi_initial_features(history.initial_root);
    condition.root_x0={history.origins.root.x,history.origins.root.y,history.origins.root.z};
    for(size_t s=0;s<2;++s){condition.object_mask[s]=own.objects[s].present;auto p=history.origins.objects[s];condition.obj_x0[3*s]=p.x;condition.obj_x0[3*s+1]=p.y;condition.obj_x0[3*s+2]=p.z;}
    Context context(bundle,threads);std::vector<int64_t> tokens;condition.text=context.text(options.at("--prompt"),&tokens);condition.objects=context.objects(own.points);
    json recipe={{"version",1},{"backend","dex2hoi"},{"bundleSha256",context.manifest_sha256()},{"upstreamRevision",upstream_revision},
        {"prompt",options.at("--prompt")},{"textTokens",tokens},{"input",fs::absolute(source).generic_string()},{"inputSha256",sha256_file(source)},
        {"historySha256",sha256_file(own.history_path)},{"historyManifest",own.manifest},{"frames",70},{"prefixFrames",10},{"fps",30},{"device","cpu"},
        {"onnxRuntime",OrtGetApiBase()->GetVersionString()},{"threads",threads},{"licensingStatus","user-supplied-unverified"}};
    recipe["meshSha256"]=json::array();for(const auto& object:own.objects)recipe["meshSha256"].push_back(object.mesh.empty()?json(nullptr):json(sha256_file(object.mesh)));
    auto start=std::chrono::steady_clock::now();auto prediction=context.sample(condition,sampling,recipe);
    std::vector<float> features(70*429);std::copy(history.features.begin(),history.features.end(),features.begin());
    for(size_t f=0;f<60;++f)for(size_t c=0;c<429;++c)features[(f+10)*429+c]=prediction[c*60+f];
    auto result=decode_dex2hoi_motion(own.rig,features,history.origins,own.objects,10,format,stats,30);
    // Keep every supplied history transform exactly, including unmapped extras;
    // numerical representation round trips do not redefine the user's prefix.
    for(size_t f=0;f<10;++f)std::copy(own.history.frame(f).begin(),own.history.frame(f).end(),result.body.frame(f).begin());
    size_t track=0;for(const auto& object:own.objects)if(object.present&&object.export_track) {
        for(size_t f=0;f<10;++f)result.objects[track].poses[f]=object.poses[f];
        ++track;
    }
    recipe["prefixPreservedExactly"]=true;
    recipe["inferenceSeconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    recipe["decodeReport"]=result.report;result.report["backend"]="dex2hoi";result.report["bundleSha256"]=context.manifest_sha256();
    save_equipment_glb(result,output);write_json(recipe_path,recipe);
    std::cout<<"Dex2HOI generated "<<result.body.frames()<<" frames: "<<output.string()<<"\n";return 0;
}
}
