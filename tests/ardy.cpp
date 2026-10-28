#include "ardy.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
int main(int argc,char** argv) try {
    if(argc<2)throw std::runtime_error("ardy-tests MODEL_DIR [REFERENCE_DIR]");
    if(std::string(argv[1])=="--contracts") {
        motion::Clip c;c.joints=motion::soma();
        for(int f=0;f<64;++f)for(const auto& joint:c.joints){auto t=joint.rest;if(&joint==&c.joints[0])t.p={f*.01f,1,f*.02f};c.poses.push_back(t);}
        auto features=motion::encode_ardy_history(c);auto decoded=motion::decode_kimodo(features);
        if(decoded.frames()!=64)throw std::runtime_error("history frame count");
        for(size_t f=0;f<64;++f){auto d=decoded.frame(f)[0].p-c.frame(f)[0].p;if(motion::dot(d,d)>1e-10f)throw std::runtime_error("history pelvis recovery");
            if(std::abs(features[f*369+275]-.3f)>1e-5f||std::abs(features[f*369+277]-.6f)>1e-5f)throw std::runtime_error("history velocity units");}
        c.poses.resize(6*30);bool refused=false;try{(void)motion::encode_ardy_history(c);}catch(const std::exception&){refused=true;}
        if(!refused)throw std::runtime_error("accepted partial history token");
        std::cout<<"PASS ARDY history pelvis, units, frame counts and token bounds\n";return 0;
    }
    if(std::string(argv[1])=="--assets"&&argc==3) {
        auto dir=motion::utf8path(argv[2]);nlohmann::json report;
        motion::Clip cold;
        for(const char* name:{"cold.glb","warm.glb","history-prompts.glb","constrained.glb","long.glb"}){
            auto c=motion::load_glb(dir/name);c.validate();motion::require_soma(c);
            double rootstep=0,slide=0;size_t contacts=0;auto first=c.frame(0)[0].p;
            for(size_t f=0;f<c.frames();++f){auto world=motion::world_pose(c,c.frame(f));for(auto t:world)if(!std::isfinite(t.p.x)||!std::isfinite(t.p.y)||!std::isfinite(t.p.z))throw std::runtime_error("nonfinite FK");
                if(f){auto d=c.frame(f)[0].p-c.frame(f-1)[0].p;rootstep=std::max(rootstep,static_cast<double>(std::sqrt(motion::dot(d,d))));
                    auto prev=motion::world_pose(c,c.frame(f-1));for(size_t foot:{24u,28u})if(world[foot].p.y<.03f&&prev[foot].p.y<.03f){auto v=world[foot].p-prev[foot].p;slide+=std::hypot(v.x,v.z)*100;++contacts;}}
            }
            auto rt=dir/(std::string("roundtrip-")+name);motion::save_glb(c,rt);auto copy=motion::load_glb(rt);
            if(copy.frames()!=c.frames())throw std::runtime_error("GLB roundtrip frame count");
            for(size_t i=0;i<c.poses.size();++i){auto d=c.poses[i].p-copy.poses[i].p;if(motion::dot(d,d)>1e-8f)throw std::runtime_error("GLB roundtrip position");}
            if(std::string(name)=="cold.glb")cold=c;
            if(std::string(name)=="warm.glb")for(size_t i=0;i<c.poses.size();++i){const auto a=cold.poses[i],b=c.poses[i];if(a.p.x!=b.p.x||a.p.y!=b.p.y||a.p.z!=b.p.z||a.q.x!=b.q.x||a.q.y!=b.q.y||a.q.z!=b.q.z||a.q.w!=b.q.w)throw std::runtime_error("same-seed warm output changed");}
            auto last=c.frame(c.frames()-1)[0].p;report[name]={{"frames",c.frames()},{"rootTravelMetres",std::hypot(last.x-first.x,last.z-first.z)},{"maxRootStepMetres",rootstep},{"groundedSlideCmPerFrame",contacts?slide/static_cast<double>(contacts):0},{"groundedSamples",contacts}};
        }
        std::ofstream out(dir/"asset-measurements.json");out<<report.dump(2);std::cout<<report.dump(2)<<'\n';return 0;
    }
    motion::ArdyModel model(motion::utf8path(argv[1]),8);
    if(argc>3&&std::string(argv[2])=="--text"){model.write_text_reference("A person walks forward at a relaxed pace.",motion::utf8path(argv[3]));return 0;}
    if(argc>2)model.verify_reference(motion::utf8path(argv[2]));
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
