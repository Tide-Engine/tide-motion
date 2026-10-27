#include "motion.hpp"
#include <kimodo/kimodo.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
using namespace motion;
void help() {
    std::cout<<R"(Tide Motion - native offline animation authoring

  tide-motion models install [all|kimodo|bricks] --accept-licenses [--models DIR]
  tide-motion models check [all|kimodo|bricks] [--models DIR]
  tide-motion generate "PROMPT" -o CLIP.glb [--duration 3] [--seed 0] [--steps 50]
  tide-motion connect FROM.glb TO.glb -o BRIDGE.glb [--tokens 6..16] [--joined]
  tide-motion batch JOBS.json [--models DIR]
  tide-motion inspect CLIP.glb [--clip NAME]
  tide-motion preview CLIP.glb [--tide PATH] [--capture IMAGE.png] [--frames N]

Models are downloaded once and SHA-256 verified. Generation and transitions run
locally without Python, PyTorch, Conda or a server. The initial rig is SOMA-30;
GLB output contains animation and a skeleton, without a skinned character mesh.

--models DIR overrides TIDE_MOTION_MODELS and the OS model cache.
connect preserves target placement; use --offset X Y Z to place the target clip.
--from-clip NAME / --to-clip NAME select clips from multi-animation GLBs.
--joined exports source, generated bridge and target as one animation.
preview uses an installed Tide separately; --tide selects its executable.
batch runs ordered actions and keeps each inference model loaded until completion.
Existing outputs are refused. Use a new output path.

Licenses: Apache-2.0 source; NVIDIA Open Model terms for motion weights;
Meta Llama 3 terms for Kimodo's text encoder. See licenses/ and README.md.
)";
}
double number(const std::string& text,double low,double high,const char* label) {
    size_t used=0;double n=std::stod(text,&used);
    if(used!=text.size()||!std::isfinite(n)||n<low||n>high)throw std::runtime_error(std::string(label)+" is out of range");
    return n;
}
double integer(const std::string& text,double low,double high,const char* label) {
    double value=number(text,low,high,label);
    if(value!=std::floor(value))throw std::runtime_error(std::string(label)+" must be an integer");
    return value;
}
struct RunContext {
    std::unique_ptr<kimodo::model> kimodo;
    std::unique_ptr<BricksModel> bricks;
};
int main_impl(const std::vector<std::string>& args,RunContext* context=nullptr);
int batch(const fs::path& manifest,const fs::path& models,RunContext* context) {
    if(fs::file_size(manifest)>1024*1024)throw std::runtime_error("Batch manifest exceeds 1 MiB");
    std::ifstream stream(manifest);nlohmann::json doc;stream>>doc;
    if(!doc.is_object()||doc.size()!=1||!doc.contains("actions")||!doc["actions"].is_array()||doc["actions"].empty()||doc["actions"].size()>1000)
        throw std::runtime_error("Batch file must contain an actions array with 1..1000 actions");
    const auto base=fs::absolute(manifest).parent_path();
    std::vector<std::vector<std::string>> commands;
    std::set<fs::path> outputs;
    auto path=[&](const nlohmann::json& value) {
        auto p=utf8path(value.get<std::string>());if(p.empty())throw std::runtime_error("Empty batch path");
        return fs::absolute(p.is_absolute()?p:base/p).lexically_normal();
    };
    for(size_t index=0;index<doc["actions"].size();++index)try {
        const auto& action=doc["actions"][index];
        if(!action.is_object())throw std::runtime_error("Action must be an object");
        auto type=action.at("action").get<std::string>();
        std::set<std::string> keys{"action","output"};
        std::vector<std::string> argv{"tide-motion",type};
        if(type=="generate") { argv.push_back(action.at("prompt").get<std::string>());keys.insert({"prompt","duration","seed","steps"}); }
        else if(type=="connect") { argv.push_back(path(action.at("from")).string());argv.push_back(path(action.at("to")).string());keys.insert({"from","to","tokens","joined","offset","fromClip","toClip"}); }
        else throw std::runtime_error("Action must be generate or connect");
        for(const auto& item:action.items())if(!keys.contains(item.key()))throw std::runtime_error("Unknown action field: "+item.key());
        auto output=path(action.at("output"));
        if(output.extension()!=".glb"||fs::exists(output)||fs::exists(output.string()+".part")||fs::exists(output.string()+".json")||!outputs.insert(output).second)throw std::runtime_error("Output must be a new, distinct .glb path: "+output.string());
        argv.insert(argv.end(),{"-o",output.string(),"--models",models.string()});
        for(const char* key:{"duration","seed","steps","tokens"})if(action.contains(key)) {
            if(!action[key].is_number())throw std::runtime_error(std::string(key)+" must be numeric");
            auto text=action[key].dump();
            if(std::string(key)=="duration")number(text,0.2,10,key);
            else if(std::string(key)=="tokens")integer(text,6,16,key);
            else if(std::string(key)=="steps")integer(text,1,1000,key);
            else integer(text,0,4294967295.0,key);
            argv.insert(argv.end(),{"--"+std::string(key),text});
        }
        if(action.contains("joined")&&action["joined"].get<bool>())argv.push_back("--joined");
        if(action.contains("offset")) {
            const auto& offset=action["offset"];
            if(!offset.is_array()||offset.size()!=3)throw std::runtime_error("offset requires three numbers");
            argv.push_back("--offset");
            for(const auto& value:offset) { if(!value.is_number())throw std::runtime_error("offset must be numeric");auto text=value.dump();number(text,-10000,10000,"offset");argv.push_back(text); }
        }
        for(auto pair:{std::pair{"fromClip","--from-clip"},std::pair{"toClip","--to-clip"}})if(action.contains(pair.first))argv.insert(argv.end(),{pair.second,action[pair.first].get<std::string>()});
        commands.push_back(std::move(argv));
    } catch(const std::exception& e) {throw std::runtime_error("Batch action "+std::to_string(index+1)+": "+e.what());}
    // Keep the 8 GB text model resident for repeated prompts rather than reloading
    // every layer per action. Explicit operator environment settings still win.
    if(!std::getenv("KIMODO_TEXT_LAYER_CHUNK")) {
#ifdef _WIN32
        _putenv_s("KIMODO_TEXT_LAYER_CHUNK","32");
#else
        setenv("KIMODO_TEXT_LAYER_CHUNK","32",0);
#endif
    }
    for(size_t i=0;i<commands.size();++i) {
        std::cerr<<"Batch action "<<i+1<<"/"<<commands.size()<<"\n";
        try {int code=main_impl(commands[i],context);if(code)return code;}
        catch(const std::exception& e){throw std::runtime_error("Batch action "+std::to_string(i+1)+" failed: "+e.what()+"; completed outputs have been kept");}
    }
    return 0;
}
int main_impl(const std::vector<std::string>& args,RunContext* context) {
    RunContext owned;if(!context)context=&owned;
    if(args.size()<2||args[1]=="--help"||args[1]=="help") { help();return 0; }
    if(args[1]=="--version") { std::cout<<"tide-motion "<<MOTION_VERSION<<"\n";return 0; }
    std::string command=args[1],clip_name,from_name,to_name,tide="tide";
    fs::path models=model_directory(),output,capture;
    double duration=3;uint64_t seed=0;unsigned steps=50;int frames=0,tokens=0;
    bool accept=false,joined=false;Vec3 offset{};
    std::vector<std::string> pos;
    std::vector<std::string> flags;
    for(size_t i=2;i<args.size();++i) {
        auto value=[&]() { if(++i>=args.size()||args[i].empty())throw std::runtime_error("Missing value for "+args[i-1]);return args[i]; };
        const auto& a=args[i];
        if(a=="--help") {help();return 0;}
        if(a.starts_with('-'))flags.push_back(a);
        if(a=="--models")models=utf8path(value());
        else if(a=="-o"||a=="--output")output=utf8path(value());
        else if(a=="--duration")duration=number(value(),0.2,10,"duration");
        else if(a=="--steps")steps=static_cast<unsigned>(integer(value(),1,1000,"steps"));
        else if(a=="--seed")seed=static_cast<uint64_t>(integer(value(),0,4294967295.0,"seed"));
        else if(a=="--frames")frames=static_cast<int>(integer(value(),1,18000,"frames"));
        else if(a=="--tokens")tokens=static_cast<int>(integer(value(),6,16,"tokens"));
        else if(a=="--offset") {offset.x=static_cast<float>(number(value(),-10000,10000,"offset"));offset.y=static_cast<float>(number(value(),-10000,10000,"offset"));offset.z=static_cast<float>(number(value(),-10000,10000,"offset"));}
        else if(a=="--clip")clip_name=value();
        else if(a=="--from-clip")from_name=value();
        else if(a=="--to-clip")to_name=value();
        else if(a=="--tide")tide=value();
        else if(a=="--capture")capture=utf8path(value());
        else if(a=="--accept-licenses")accept=true;
        else if(a=="--joined")joined=true;
        else if(a.starts_with('-'))throw std::runtime_error("Unknown option: "+a);
        else pos.push_back(a);
    }
    std::set<std::string> allowed;
    if(command=="generate")allowed={"--models","-o","--output","--duration","--seed","--steps"};
    else if(command=="connect")allowed={"--models","-o","--output","--tokens","--joined","--offset","--from-clip","--to-clip"};
    else if(command=="inspect")allowed={"--clip"};
    else if(command=="preview")allowed={"--clip","--tide","--capture","--frames"};
    else if(command=="models")allowed={"--models","--accept-licenses"};
    else if(command=="batch")allowed={"--models"};
    for(const auto& flag:flags)if(!allowed.contains(flag))throw std::runtime_error(command+" does not take "+flag);
    if(command=="batch") {
        if(pos.size()!=1)throw std::runtime_error("batch takes one jobs.json manifest");
        return batch(utf8path(pos[0]),models,context);
    }
    if(command=="models") {
        if(pos.empty()||pos.size()>2||(pos[0]!="install"&&pos[0]!="check"))throw std::runtime_error("Use models install or models check [all|kimodo|bricks]");
        std::string selection=pos.size()==2?pos[1]:"all";
        if(pos[0]=="check")check_models(models,selection);
        else {
            if(!accept)throw std::runtime_error("Read licenses/NVIDIA-KIMODO-MODEL.txt, NVIDIA-OPEN-MODEL.txt and META-LLAMA-3.txt, then pass --accept-licenses to download models");
            setup_models(models,selection);
        }
        return 0;
    }
    if(command=="inspect"||command=="preview") {
        if(pos.size()!=1)throw std::runtime_error(command+" takes one GLB file");
        auto path=utf8path(pos[0]);auto clip=load_glb(path,clip_name);
        if(command=="preview")preview(clip,path,tide,capture,frames);
        else std::cout<<nlohmann::json{{"name",clip.name},{"rig","soma30"},{"joints",clip.joints.size()},{"frames",clip.frames()},{"fps",clip.fps},{"duration",clip.duration()}}.dump(2)<<"\n";
        return 0;
    }
    if(command!="generate"&&command!="connect")throw std::runtime_error("Unknown command: "+command+"; use --help");
    if(output.empty()||output.extension()!=".glb")throw std::runtime_error("Choose a .glb output with -o");
    if(fs::exists(output)||fs::exists(output.string()+".part")||fs::exists(output.string()+".json"))throw std::runtime_error("Output, partial output or recipe already exists: "+output.string());
    Clip clip;
    if(command=="generate") {
        if(pos.size()!=1||pos[0].empty())throw std::runtime_error("generate takes one quoted prompt");
        if(!context->kimodo) {
            check_models(models,"kimodo");
            std::cerr<<"Loading Kimodo SOMA and text encoder (native GGML)...\n";
            auto model=kimodo::model::load((models/"kimodo.gguf").string(),(models/"text/encoder.gguf").string());
            if(!model)throw std::runtime_error(model.error());
            context->kimodo=std::move(*model);
        }
        std::cerr<<"Generating motion...\n";
        auto result=context->kimodo->generate_text(pos[0],static_cast<unsigned>(std::round(duration*30)),steps,seed,2,2);
        if(!result)throw std::runtime_error(result.error());
        if(result->joints!=30||result->local_rotations_xyzw.size()!=static_cast<size_t>(result->frames)*30*4||result->root_positions.size()!=static_cast<size_t>(result->frames)*3)throw std::runtime_error("Kimodo returned an unexpected SOMA output layout");
        clip.joints=soma();clip.name=output.stem().string();
        for(size_t f=0;f<result->frames;++f)for(size_t j=0;j<30;++j) {
            size_t index=(f*30+j)*4;Transform t=clip.joints[j].rest;
            t.q=normalized({result->local_rotations_xyzw[index],result->local_rotations_xyzw[index+1],result->local_rotations_xyzw[index+2],result->local_rotations_xyzw[index+3]});
            if(j==0)t.p={result->root_positions[f*3],result->root_positions[f*3+1],result->root_positions[f*3+2]};
            clip.poses.push_back(t);
        }
    } else {
        if(pos.size()!=2)throw std::runtime_error("connect takes source and target GLB files");
        auto a=load_glb(utf8path(pos[0]),from_name),b=load_glb(utf8path(pos[1]),to_name);
        for(size_t f=0;f<b.frames();++f)b.frame(f)[0].p=b.frame(f)[0].p+offset;
        if(!context->bricks) {
            check_models(models,"bricks");context->bricks=std::make_unique<BricksModel>(models/"motionbricks.onnx");
        }
        clip=context->bricks->generate(a,b,tokens);
        if(joined)clip=join(a,clip,b);
    }
    save_glb(clip,output);
    // Store generation settings alongside the asset; no need to rerun inference at build time.
    nlohmann::json recipe={{"tool","tide-motion"},{"version",MOTION_VERSION},{"command",command},{"inputs",pos},{"rig","soma30"},{"fps",30},{"frames",clip.frames()},{"requestedDuration",duration},{"seed",seed},{"steps",steps},{"tokens",tokens},{"joined",joined},{"offset",{offset.x,offset.y,offset.z}},
        {"fromClip",from_name},{"toClip",to_name},{"kimodoNativeRevision","ed8a931aedb132f768190f6931c365855caba74c"},{"motionbricksRevision","54fc563aeaaf819a753d72cc9fdc6713989cdd43"}};
    std::ofstream metadata(output.string()+".json");metadata<<recipe.dump(2)<<"\n";metadata.close();
    if(!metadata)throw std::runtime_error("Animation saved but could not write its recipe");
    std::cout<<"Saved "<<output.string()<<" ("<<clip.frames()<<" frames, "<<clip.duration()<<" seconds)\n";return 0;
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
    try {
        SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
        std::vector<std::string> args;
        for(int i=0;i<argc;++i) { int n=WideCharToMultiByte(CP_UTF8,0,argv[i],-1,nullptr,0,nullptr,nullptr);std::string s(static_cast<size_t>(n),0);WideCharToMultiByte(CP_UTF8,0,argv[i],-1,s.data(),n,nullptr,nullptr);s.pop_back();args.push_back(s); }
        return main_impl(args);
    } catch(const std::exception& e) { std::cerr<<"tide-motion: "<<e.what()<<"\n";return 1; }
}
#else
int main(int argc,char** argv) {
    try { return main_impl({argv,argv+argc}); }
    catch(const std::exception& e) {std::cerr<<"tide-motion: "<<e.what()<<"\n";return 1;}
}
#endif
