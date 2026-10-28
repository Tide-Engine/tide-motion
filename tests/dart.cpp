#include "dart.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <cmath>
using namespace motion;
namespace {
// Checks real generated assets at half-frame times against an independent
// midpoint formula (quaternion midpoint is normalized q0 + hemisphere(q1)).
void assets(const fs::path& directory) {
    size_t clips=0,samples=0;double worst_root=0,worst_rotation=0;
    for(const auto& file:fs::directory_iterator(directory)) {
        if(file.path().extension()!=".glb")continue;
        auto clip=load_glb(file.path());require_soma(clip);++clips;
        for(size_t f=0;f+1<clip.frames();++f) {
            auto pose=sample(clip,(static_cast<float>(f)+.5f)/clip.fps);
            for(size_t j=0;j<clip.joints.size();++j) {
                auto a=clip.frame(f)[j],b=clip.frame(f+1)[j];
                double expected[3]={(double(a.p.x)+b.p.x)/2,(double(a.p.y)+b.p.y)/2,(double(a.p.z)+b.p.z)/2};
                double actual[3]={pose[j].p.x,pose[j].p.y,pose[j].p.z};
                for(size_t k=0;k<3;++k)worst_root=std::max(worst_root,std::abs(expected[k]-actual[k]));
                double qa[4]={a.q.x,a.q.y,a.q.z,a.q.w},qb[4]={b.q.x,b.q.y,b.q.z,b.q.w};
                double alignment=0;for(size_t k=0;k<4;++k)alignment+=qa[k]*qb[k];
                double q[4],length=0;for(size_t k=0;k<4;++k){q[k]=qa[k]+(alignment<0?-qb[k]:qb[k]);length+=q[k]*q[k];}
                double value[4]={pose[j].q.x,pose[j].q.y,pose[j].q.z,pose[j].q.w},match=0;
                for(size_t k=0;k<4;++k)match+=q[k]/std::sqrt(length)*value[k];
                worst_rotation=std::max(worst_rotation,1-std::min(1.,std::abs(match)));
            }
            ++samples;
        }
    }
    if(!clips||!samples||worst_root>1e-4||worst_rotation>1e-6)throw std::runtime_error("DART asset resampling failed");
    std::cout<<"DART assets "<<clips<<", half-frame samples "<<samples<<", max position error "<<worst_root<<", quaternion dot error "<<worst_rotation<<'\n';
}
}
int main(int argc,char** argv) try {
    if(argc==5&&std::string_view(argv[1])=="--batch-reference"){DartModel model(utf8path(argv[2]),4,argv[4]);std::cout<<"Batched reference max "<<model.verify_batch_reference(utf8path(argv[3]))<<'\n';return 0;}
    if(argc==3&&std::string_view(argv[1])=="--assets"){assets(utf8path(argv[2]));return 0;}
    if(argc==3){DartModel model(utf8path(argv[1]),4);std::cout<<"Native/PyTorch max absolute error "<<model.verify_reference(utf8path(argv[2]))<<'\n';return 0;}
    auto path=fs::temp_directory_path()/("dart-plan-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".json");
    struct Cleanup {fs::path path;~Cleanup(){std::error_code e;fs::remove(path,e);}} cleanup{path};
    auto plan=nlohmann::json{{"version",1},{"segments",nlohmann::json::array({{{"text","walk"},{"chunks",12},{"target",{0,2}}}})}};
    auto load=[&](const nlohmann::json& j){std::ofstream f(path);f<<j.dump();f.close();return load_dart_plan(path);};
    auto p=load(plan);if(p.segments.size()!=1||!p.segments[0].goal||p.segments[0].chunks!=12)throw std::runtime_error("DART plan lost values");
    auto refuses=[&](const nlohmann::json& j){bool refused=false;try{(void)load(j);}catch(const std::exception&){refused=true;}if(!refused)throw std::runtime_error("DART accepted invalid plan");};
    for(const auto& value:{nlohmann::json(-1),nlohmann::json(0),nlohmann::json(1.5),nlohmann::json(513)}){auto bad=plan;bad["segments"][0]["chunks"]=value;refuses(bad);}
    auto bad=plan;bad["segments"][0]["target"]={0,2,3};refuses(bad);
    bad=plan;bad["segments"][0]["text"]="fly";refuses(bad);
    bad=plan;bad["seed"]=-1;refuses(bad);bad=plan;bad["unrecognized"]=true;refuses(bad);
    bad=plan;bad["segments"]=nlohmann::json::array();refuses(bad);
    bad=plan;bad["segments"][0]["target"]={101,0};refuses(bad);
    bad=plan;bad["segments"]=nlohmann::json::array();for(int i=0;i<5;++i)bad["segments"].push_back({{"text","walk"},{"chunks",512}});refuses(bad);
    std::cout<<"DART bounded plan contracts passed\n";return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
