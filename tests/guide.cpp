// Sampling guide contracts (src/guide.cpp), and a developer report:
//   guide-tests --report GUIDE.json CLIP.glb   each rule measured on a SOMA clip.
#include "guide.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace motion;
namespace {
void check(bool yes,const char* why) {if(!yes)throw std::runtime_error(why);}
float length(Vec3 v) {return std::sqrt(dot(v,v));}
size_t index_of(const char* name) {auto rig=soma();for(size_t j=0;j<rig.size();++j)if(rig[j].name==name)return j;throw std::runtime_error(name);}
// A SOMA clip of `frames` copies of a pose with the arms lowered differently each frame.
Clip clip_of(size_t frames,float lower) {
    Clip c;c.joints=soma();
    for(size_t f=0;f<frames;++f)for(size_t j=0;j<c.joints.size();++j) {
        Transform t=c.joints[j].rest;
        if(j==0)t.p={0,0.95f,0};
        if(c.joints[j].name=="LeftArm")t.q=normalized(Quat{0,0,std::sin((lower+0.02f*static_cast<float>(f))/2),std::cos((lower+0.02f*static_cast<float>(f))/2)});
        if(c.joints[j].name=="RightArm")t.q=normalized(Quat{0,0,-std::sin(lower/2),std::cos(lower/2)});
        c.poses.push_back(t);
    }
    return c;
}
std::vector<float> features_of(const Clip& c) {
    std::vector<float> out;
    for(size_t f=0;f<c.frames();++f) {auto row=encode_kimodo(c,c.frame(f));out.insert(out.end(),row.begin(),row.end());}
    // Velocities as upstream: fps times the forward difference of global positions.
    const size_t F=c.frames();
    for(size_t f=0;f<F;++f)for(size_t j=0;j<30;++j)for(size_t k=0;k<3;++k) {
        const size_t a=f+1<F?f:f-1;
        const float pa=out[a*kimodo_features+5+3*j+k],pb=out[(a+1)*kimodo_features+5+3*j+k];
        out[f*kimodo_features+275+3*j+k]=30*(pb-pa);
    }
    return out;
}
std::vector<Transform> world_at(const Clip& c,size_t f) {return world_pose(c,c.frame(f));}
template<class F> void refuses(F operation,const char* why) {bool r=false;try{operation();}catch(const std::exception&){r=true;}check(r,why);}
}
int main(int argc,char** argv) try {
    if(argc==4&&std::string(argv[1])=="--report") {
        auto clip=load_glb(argv[3],"",true);
        auto guide=load_guide(nlohmann::json(std::string(argv[2])),fs::current_path());
        resolve_guide(guide,clip.frames());
        std::cout<<guide_report(guide,clip).dump(1)<<"\n";
        return 0;
    }
    if(argc==4&&std::string(argv[1])=="--bench") {
        // One guided pass over a clip's features, timed.
        auto clip=load_glb(argv[3],"",true);
        auto guide=load_guide(nlohmann::json(std::string(argv[2])),fs::current_path());
        resolve_guide(guide,clip.frames());
        auto features=features_of(clip);
        std::cout<<guide_clean(guide,features,clip.frames())<<" ms for "<<clip.frames()<<" frames\n";
        return 0;
    }
    const size_t F=6;
    const auto lh=index_of("LeftHand"),rh=index_of("RightHand"),chest=index_of("Chest"),head=index_of("Head");
    // Grip: the left hand held 10 cm in front of the right hand's frame, from arms at
    // their sides; the root, heading and contacts stay; velocities follow positions.
    {
        auto clip=clip_of(F,1.2f);
        auto features=features_of(clip);const auto before=features;
        nlohmann::json spec={{"version",1},{"objects",{{{"name","stick"},{"bone","RightHand"},{"position",{0,0,0.1}}}}},
            {"grips",{{{"hand","LeftHand"},{"object","stick"},{"position",{0,0,0}},{"orientation",false}}}},{"iterations",30}};
        auto guide=load_guide(spec,".");resolve_guide(guide,F);
        auto far=world_at(clip,0);
        check(length(far[lh].p-(far[rh]*Transform{{0,0,0.1f},{}}).p)>0.2f,"Grip test starts far from its target");
        (void)guide_clean(guide,features,F);
        auto guided=decode_kimodo(features);
        for(size_t f=0;f<F;++f) {
            auto w=world_at(guided,f);
            check(length(w[lh].p-(w[rh]*Transform{{0,0,0.1f},{}}).p)<0.02f,"Guided left hand reaches the grip");
            const float* a=features.data()+f*kimodo_features;const float* b=before.data()+f*kimodo_features;
            for(size_t d=0;d<5;++d)check(a[d]==b[d],"Root and heading stay as predicted");
            for(size_t d=365;d<369;++d)check(a[d]==b[d],"Contacts stay as predicted");
            for(size_t k=0;k<3;++k)check(a[5+k]==b[5+k],"Hips stay as predicted");
        }
        for(size_t f=0;f+1<F;++f)for(size_t j=0;j<30;++j)for(size_t k=0;k<3;++k) {
            const auto at=[&](const std::vector<float>& x,size_t frame,size_t d){return x[frame*kimodo_features+d];};
            const float dv=at(features,f,275+3*j+k)-at(before,f,275+3*j+k);
            const float dp=(at(features,f+1,5+3*j+k)-at(before,f+1,5+3*j+k))-(at(features,f,5+3*j+k)-at(before,f,5+3*j+k));
            check(std::abs(dv-30*dp)<1e-3f,"Velocities move with the positions' forward difference");
        }
        // Strength scales the change; zero leaves the prediction alone.
        auto none=before;spec["strength"]=0;auto off=load_guide(spec,".");resolve_guide(off,F);
        (void)guide_clean(off,none,F);check(none==before,"Strength 0 changes nothing");
        auto report=guide_report(guide,guided);
        check(report["grips"][0]["offMaxMetres"].get<double>()<0.02,"Report measures the grip");
    }
    // A grip spans its frames and eases in: outside it the pose stays as predicted.
    {
        auto clip=clip_of(F,1.2f);auto features=features_of(clip);const auto before=features;
        nlohmann::json spec={{"version",1},{"grips",{{{"hand","LeftHand"},{"bone","RightHand"},{"position",{0,0,0.1}},{"orientation",false},{"start",3},{"end",-1}}}},{"iterations",30}};
        auto guide=load_guide(spec,".");resolve_guide(guide,F);(void)guide_clean(guide,features,F);
        for(size_t f=0;f<3;++f)for(size_t d=0;d<275;++d)check(features[f*kimodo_features+d]==before[f*kimodo_features+d],"Frames before a grip keep their pose");
        check(features[4*kimodo_features+5+3*lh]!=before[4*kimodo_features+5+3*lh],"Frames in the grip move");
    }
    // Capsules: a sphere on the chest put where the left hand is pushes the hand out.
    {
        auto clip=clip_of(F,1.2f);
        auto w=world_at(clip,0);const Vec3 centre=rotate(inverse(w[chest].q),w[lh].p-w[chest].p);
        nlohmann::json capsules={{"version",1},{"parts",{{{"name","ball"},{"from","Chest"},{"centre",{centre.x,centre.y,centre.z}},{"radius",0.1}},
            {{"name","hand"},{"from","LeftHand"},{"to","LeftHandMiddleEnd"},{"radius",0.04}}}},{"pairs",nlohmann::json::array({nlohmann::json::array({"hand","ball"})})}};
        nlohmann::json spec={{"version",1},{"capsules",capsules},{"iterations",30}};
        auto guide=load_guide(spec,".");resolve_guide(guide,F);
        const double inside=guide_report(guide,clip)["capsules"]["worstMetres"].get<double>();
        auto features=features_of(clip);(void)guide_clean(guide,features,F);
        const double after=guide_report(guide,decode_kimodo(features))["capsules"]["worstMetres"].get<double>();
        check(inside>0.1,"Capsule test starts with the hand in the ball");
        check(after<0.01,"Guided hand comes out of the ball");
        (void)head;
    }
    // Hold: both hands where a pose has them on the chest.
    {
        auto pose=clip_of(1,0.4f);
        fs::path dir=fs::temp_directory_path()/"tide-motion-guide-tests";fs::create_directories(dir);
        const auto file=dir/"hold.glb";fs::remove(file);fs::remove(fs::path(file.string()+".part"));save_glb(pose,file);
        auto clip=clip_of(F,1.2f);auto features=features_of(clip);
        nlohmann::json spec={{"version",1},{"holds",{{{"pose",file.string()+"@0"}}}},{"iterations",30},{"weights",{{"holdArm",0}}}};
        auto guide=load_guide(spec,".");resolve_guide(guide,F);(void)guide_clean(guide,features,F);
        auto guided=decode_kimodo(features);auto want=world_at(pose,0);
        for(size_t f=0;f<F;++f) {
            auto w=world_at(guided,f);
            for(size_t h:{lh,rh})check(length((inverse(w[chest])*w[h]).p-(inverse(want[chest])*want[h]).p)<0.03f,"Held hands go where the pose has them");
        }
        check(guide_report(guide,guided)["holds"][0]["offMaxMetres"].get<double>()<0.03,"Report measures the hold");
    }
    // Steps: from/until select a part of the sampling (step counts down to 0).
    {
        nlohmann::json spec={{"version",1},{"grips",{{{"hand","LeftHand"},{"bone","RightHand"},{"position",{0,0,0.1}}}}},{"from",0.5},{"until",1}};
        auto guide=load_guide(spec,".");
        check(!guide_step(guide,49,50)&&guide_step(guide,0,50)&&guide_step(guide,20,50)&&!guide_step(guide,30,50),"from/until pick the later half of the steps");
    }
    // Refusals.
    const nlohmann::json grip={{"hand","LeftHand"},{"bone","RightHand"},{"position",{0,0,0.1}}};
    refuses([&]{(void)load_guide({{"version",2},{"grips",{grip}}},".");},"Unknown version refused");
    refuses([&]{(void)load_guide({{"version",1},{"grips",{grip}},{"typo",1}},".");},"Unknown field refused");
    refuses([&]{(void)load_guide({{"version",1}},".");},"Empty guide refused");
    refuses([&]{auto g=grip;g["hand"]="Wrist";(void)load_guide({{"version",1},{"grips",{g}}},".");},"Unknown joint refused");
    refuses([&]{auto g=grip;g["around"]=90;(void)load_guide({{"version",1},{"grips",{g}}},".");},"Turning grip without an axis refused");
    refuses([&]{auto g=grip;g["bone"]="LeftHand";(void)load_guide({{"version",1},{"grips",{g}}},".");},"A hand holding itself refused");
    refuses([&]{auto g=grip;g["object"]="none";(void)load_guide({{"version",1},{"grips",{g}}},".");},"Grip naming object and bone refused");
    refuses([&]{(void)load_guide({{"version",1},{"grips",{grip}},{"joints",{{"Hips",1}}}},".");},"Turning the root refused");
    refuses([&]{(void)load_guide({{"version",1},{"grips",{grip}},{"from",0.8},{"until",0.2}},".");},"from after until refused");
    refuses([&]{(void)load_guide({{"version",1},{"grips",{grip}},{"strength",2}},".");},"Strength above 1 refused");
    refuses([&]{(void)load_guide({{"version",1},{"capsules",{{"version",1},{"parts",{{{"name","a"},{"from","Head"},{"centre",{0,0,0}}}}},{"pairs",nlohmann::json::array()}}}},".");},"Capsule without radius refused");
    refuses([&]{(void)load_guide({{"version",1},{"capsules",{{"version",1},{"parts",{{{"name","a"},{"from","Head"},{"centre",{0,0,0}},{"radius",0.1}},{{"name","b"},{"from","Chest"},{"to","Neck1"},{"radius",0.1}}}},{"pairs",nlohmann::json::array({nlohmann::json::array({"a","b"}),nlohmann::json::array({"b","a"})})}}}},".");},"Duplicate capsule pair refused");
    refuses([&]{auto g=load_guide({{"version",1},{"grips",{nlohmann::json{{"hand","LeftHand"},{"bone","RightHand"},{"position",{0,0,0.1}},{"start",10}}}}},".");resolve_guide(g,6);},"Grip outside the output refused");
    refuses([&]{(void)load_guide(nlohmann::json("missing-guide.json"),".");},"Missing guide file refused");
    refuses([&]{(void)load_guide({{"version",1},{"capsules",{{"version",1},{"parts",{{{"name","blade"},{"from","RightHand"},{"start",{0,0.1,0}},{"radius",0}}}},{"pairs",nlohmann::json::array()}}}},".");},"Segment capsule without its end refused");
    {
        // A blade (radius 0) along a bone, put through a sphere on the chest, is measured.
        auto clip=clip_of(1,1.2f);auto w=world_at(clip,0);
        const Vec3 at=rotate(inverse(w[rh].q),w[chest].p-w[rh].p);
        nlohmann::json capsules={{"version",1},{"parts",{{{"name","blade"},{"from","RightHand"},{"start",{at.x,at.y-0.2,at.z}},{"end",{at.x,at.y+0.2,at.z}},{"radius",0}},
            {{"name","chest"},{"from","Chest"},{"centre",{0,0,0}},{"radius",0.1}}}},{"pairs",nlohmann::json::array({nlohmann::json::array({"blade","chest"})})}};
        auto guide=load_guide({{"version",1},{"capsules",capsules}},".");resolve_guide(guide,1);
        check(std::abs(guide_report(guide,clip)["capsules"]["worstMetres"].get<double>()-0.095)<0.002,"A blade through a sphere's centre is inside by its radius less slack");
    }
    std::cout<<"Guide contracts passed\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<"guide-tests: "<<e.what()<<"\n";return 1;}
