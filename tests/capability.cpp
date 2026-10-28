// Model-free contracts for SOMA-77 output, dense root paths and the native Kimodo
// MotionCorrection port. `capability-tests --smooth IN.csv OUT.csv` smooths an x,z
// series for the developer reference check (tests/smooth_reference.py).
#include "motion.hpp"
#include <kimodo/kimodo_capi.h>
#include <kimodo/kimodo.hpp>
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace motion;
namespace {
void check(bool yes,const char* why) {if(!yes)throw std::runtime_error(why);}
float angle(Quat a,Quat b) {float d=std::abs(a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);return 2*std::acos(std::min(1.0f,d));}
Quat axis_angle(Vec3 axis,float radians) {axis=normalized(axis);float s=std::sin(radians/2);return {axis.x*s,axis.y*s,axis.z*s,std::cos(radians/2)};}
float length(Vec3 v) {return std::sqrt(dot(v,v));}
// A varied SOMA-30 clip: every joint turning about its own axis over time.
Clip varied(size_t frames) {
    Clip c;c.joints=soma();
    for(size_t f=0;f<frames;++f)for(size_t j=0;j<30;++j) {
        float t=0.37f*static_cast<float>(j)+0.05f*static_cast<float>(f);
        Transform joint=c.joints[j].rest;joint.q=axis_angle({std::sin(t),std::cos(1.3f*t),0.5f+std::sin(0.7f*t)},0.5f*std::sin(2.1f*t));
        if(!j)joint.p={0.02f*static_cast<float>(f),0.9f,0.5f};
        c.poses.push_back(joint);
    }
    return c;
}
void soma77_contracts() {
    auto rig30=soma(),rig77=soma77();
    check(rig77.size()==77,"SOMA-77 has 77 joints");
    Clip r30;r30.joints=rig30;for(const auto& j:rig30)r30.poses.push_back(j.rest);
    Clip r77;r77.joints=rig77;for(const auto& j:rig77)r77.poses.push_back(j.rest);
    auto w30=world_pose(r30,r30.frame(0)),w77=world_pose(r77,r77.frame(0));
    for(size_t i=0;i<30;++i) {
        size_t k=0;while(rig77[k].name!=rig30[i].name)++k;
        check(length(w30[i].p-w77[k].p)<1e-6f,"SOMA-77 neutral joints equal SOMA-30 (upstream joints.p)");
    }
    auto c=varied(40);
    auto full=soma30_to_77(c);
    check(is_soma77(full)&&full.frames()==40,"Expansion is a SOMA-77 clip");
    auto back=soma77_to_30(full);
    for(size_t f=0;f<40;++f)for(size_t j=0;j<30;++j) {
        check(angle(back.frame(f)[j].q,c.frame(f)[j].q)==0,"Reduction returns the 30 rotations exactly");
        if(!j)check(length(back.frame(f)[0].p-c.frame(f)[0].p)==0,"Reduction returns the root exactly");
    }
    // Shared body joints (not the thumb/middle ends, which hang off the fingers) keep world positions.
    for(size_t f:{size_t{0},size_t{39}}) {
        auto a=world_pose(c,c.frame(f)),b=world_pose(full,full.frame(f));
        for(size_t i=0;i<30;++i) {
            if(rig30[i].name.find("End")!=std::string::npos)continue;
            size_t k=0;while(rig77[k].name!=rig30[i].name)++k;
            check(length(a[i].p-b[k].p)<1e-5f&&angle(a[i].q,b[k].q)<2e-3f,"SOMA-77 body poses equal SOMA-30");
        }
    }
    // Fingers hold the relaxed pose; a template keeps its own.
    size_t finger=0;while(rig77[finger].name!="LeftHandIndex2")++finger;
    check(angle(full.frame(0)[finger].q,full.frame(39)[finger].q)==0&&angle(full.frame(0)[finger].q,{})>0.05f,"Constant relaxed fingers");
    auto custom=full;for(size_t f=0;f<40;++f)custom.frame(f)[finger].q=axis_angle({1,0,0},0.01f*static_cast<float>(f));
    auto kept=soma30_to_77(back,&custom);
    check(angle(kept.frame(25)[finger].q,custom.frame(25)[finger].q)<1e-6f,"Expansion keeps template fingers");
    // GLB roundtrip: canonical loads read the subset, raw loads keep all 77.
    auto dir=fs::temp_directory_path()/("tide-motion-soma77-"+std::to_string(std::rand()));fs::create_directories(dir);
    auto path=dir/"full.glb";save_glb(full,path);
    auto canonical=load_glb(path);require_soma(canonical);
    for(size_t f=0;f<40;++f)for(size_t j=0;j<30;++j)check(angle(canonical.frame(f)[j].q,c.frame(f)[j].q)<1e-6f,"Canonical load of SOMA-77 is the SOMA-30 subset");
    auto raw=load_glb(path,"",false);
    check(raw.joints.size()==77&&is_soma77(raw),"Raw load keeps 77 joints");
    // Mirroring the 77 mirrors the body exactly as SOMA-30 and twice restores the fingers.
    auto m30=c;mirror_soma(m30);
    auto m77=mirror_soma77(full);auto m77b=soma77_to_30(m77);
    for(size_t f=0;f<40;++f)for(size_t j=0;j<30;++j)check(angle(m77b.frame(f)[j].q,m30.frame(f)[j].q)<1e-6f,"SOMA-77 mirror agrees with SOMA-30");
    auto twice=mirror_soma77(m77);
    check(angle(twice.frame(10)[finger].q,full.frame(10)[finger].q)<1e-5f,"Mirroring fingers twice restores them");
    bool refused=false;try{auto bad=raw;bad.joints[30].rest.p.x+=0.1f;(void)as_soma77(bad);}catch(const std::exception&){refused=true;}
    check(refused,"SOMA-77 proportions are checked");
    fs::remove_all(dir);
}
void dense_contracts() {
    // Fewer than five samples pass through; a straight constant-speed path stays
    // straight, shifted half a frame as upstream's final averaging does.
    std::vector<std::array<double,2>> line;for(int i=0;i<90;++i)line.push_back({0.04*i,0.01*i});
    auto tiny=smooth_root_path(std::span(line).first(4));
    check(tiny[2][0]==line[2][0],"Short paths are unchanged");
    auto smooth=smooth_root_path(line);
    // Upstream does not converge to the input line: its ADMM pulls the ends in by up to
    // the margin (tests/smooth_reference.py agrees with this to 1e-12 m).
    for(int i=0;i<90;++i) {
        check(std::abs(smooth[i][1]-0.25*smooth[i][0])<1e-6,"A straight path stays on its line");
        check(std::hypot(smooth[i][0]-line[i][0],smooth[i][1]-line[i][1])<0.06+0.03,"A straight path stays within the margin and half a step");
    }
    // A right-angle corner is rounded within the margin and loses its acceleration spike.
    std::vector<std::array<double,2>> corner;
    for(int i=0;i<60;++i)corner.push_back(i<30?std::array<double,2>{0.05*i,0}:std::array<double,2>{1.5,0.05*(i-30)});
    auto round=smooth_root_path(corner);
    double before=0,after=0,offset=0;
    for(int i=1;i+1<60;++i) {
        before=std::max(before,std::hypot(corner[i+1][0]-2*corner[i][0]+corner[i-1][0],corner[i+1][1]-2*corner[i][1]+corner[i-1][1]));
        after=std::max(after,std::hypot(round[i+1][0]-2*round[i][0]+round[i-1][0],round[i+1][1]-2*round[i][1]+round[i-1][1]));
    }
    for(int i=0;i+1<60;++i)offset=std::max(offset,std::hypot(round[i][0]-(corner[i][0]+corner[i+1][0])/2,round[i][1]-(corner[i][1]+corner[i+1][1])/2));
    check(after<0.5*before,"Smoothing removes the corner's acceleration spike");
    check(offset<0.07,"Smoothing stays within the 6 cm margin (of the half-frame-shifted path)");
    // Densify: frame 0 added at the start, waypoints keep yaw, every frame constrained.
    MotionConditions c;
    PathConstraint a;a.frame=30;a.x=1;a.z=0;PathConstraint b;b.frame=60;b.x=1;b.z=1;b.has_heading=true;b.heading=90;
    c.path={b,a};
    densify_path(c,{0.2f,0,0.1f});
    check(c.path.size()==61&&c.path.front().frame==0&&c.path.back().frame==60,"Dense path covers frame 0 to the last waypoint");
    check(c.path.back().has_heading&&c.path.back().heading==90&&!c.path[10].has_heading,"Only waypoints keep yaw");
    for(const auto& p:c.path)check(p.interpolated,"Dense samples are marked");
    MotionConditions two;PathConstraint end;end.frame=20;end.x=2;end.z=0;two.path={end};
    densify_path(two,{});
    check(std::abs(two.path[10].x-1)<1e-6f,"Two waypoints interpolate linearly, unsmoothed");
    // Conditioning: root x/z masked on every dense frame; a pose on the path measures
    // its joints from the path sample (upstream's smooth_root_2d), not its hips.
    Clip rig;rig.joints=soma();
    auto pose=varied(1);PoseConstraint key{"keyframe",45,"k.glb",0,{pose.poses.begin(),pose.poses.end()},0};
    c.poses.push_back(key);
    (void)place_constraints(c,Placement::automatic);
    check(c.poses[0].pose[0].p.x==pose.frame(0)[0].p.x,"Interpolated samples do not move poses");
    auto cond=build_condition(c,90);
    for(size_t f=0;f<=60;++f)check(cond.mask[f*kimodo_features]==1&&cond.mask[f*kimodo_features+2]==1,"Dense frames mask the root on the ground");
    check(cond.mask[61*kimodo_features]==0,"Frames after the last waypoint stay free");
    auto row=encode_kimodo(rig,c.poses[0].pose);
    const float* at=cond.observed.data()+45*kimodo_features;
    const auto& sample=c.path[45];
    check(std::abs(at[0]-(sample.x-cond.origin.x))<1e-5f,"Pose smooth root is the path sample");
    check(std::abs(at[5]-(row[5]+row[0]-sample.x))<1e-5f,"Pose joint positions are relative to the path sample");
}
// A crouched stance on a root sliding forward 1 cm/frame: both ankles skate.
Clip skating(size_t frames) {
    Clip c;c.joints=soma();
    for(size_t f=0;f<frames;++f)for(size_t j=0;j<30;++j) {
        Transform t=c.joints[j].rest;
        if(j==22||j==26)t.q=axis_angle({1,0,0},-0.5f);
        if(j==23||j==27)t.q=axis_angle({1,0,0},0.9f);
        if(j==24||j==28)t.q=axis_angle({1,0,0},-0.4f);
        if(!j)t.p={0,0.85f,0.01f*static_cast<float>(f)};
        c.poses.push_back(t);
    }
    return c;
}
void correction_contracts() {
    // No constraints and no contacts: the clip comes back unchanged.
    auto c=varied(50);auto same=c;
    std::vector<float> none(50*4,0);
    MotionConditions empty;
    (void)correct_kimodo(same,empty,none);
    for(size_t f=0;f<50;++f)for(size_t j=0;j<30;++j)check(angle(same.frame(f)[j].q,c.frame(f)[j].q)<2e-3f,"Unconstrained correction keeps rotations");
    for(size_t f=0;f<50;++f)check(length(same.frame(f)[0].p-c.frame(f)[0].p)<1e-4f,"Unconstrained correction keeps the root");
    // A full-body pose becomes exact, with nearby frames warped smoothly.
    auto target=varied(1);for(auto& t:target.poses)t.q=normalized(axis_angle({0,1,0},0.3f)*t.q);
    target.poses[0].p={0.6f,0.85f,0.7f};
    MotionConditions keyed;keyed.poses.push_back({"keyframe",25,"k.glb",0,{target.poses.begin(),target.poses.end()},0});
    auto fixed=c;(void)correct_kimodo(fixed,keyed,none);
    for(size_t j=0;j<30;++j)check(angle(fixed.frame(25)[j].q,target.poses[j].q)<1e-3f,"Full-body rotations are exact");
    check(length(fixed.frame(25)[0].p-target.poses[0].p)<1e-4f,"Full-body root is exact");
    float worst=0;for(size_t f=1;f<50;++f)worst=std::max(worst,length(fixed.frame(f)[0].p-fixed.frame(f-1)[0].p));
    check(worst<0.08f,"The root warp has no pop");
    // A path point holds the root within upstream's 4 cm margin.
    MotionConditions path;PathConstraint p;p.frame=30;p.x=0.2f;p.z=0.9f;path.path.push_back(p);
    auto pathed=c;(void)correct_kimodo(pathed,path,none);
    check(std::hypot(pathed.frame(30)[0].p.x-0.2f,pathed.frame(30)[0].p.z-0.9f)<0.045f,"Path root within the margin");
    // Contact IK pins skating ankles; translations and bone lengths stay.
    auto skate=skating(40);std::vector<float> contacts(40*4,1);
    auto pinned=skate;auto report=correct_kimodo(pinned,empty,contacts);
    check(report.ankle_frames==80&&report.toe_frames==80,"Every contact frame is solved for both feet");
    for(size_t foot:{size_t{24},size_t{28}}) {
        double before=0,after=0;
        for(size_t f=1;f<40;++f) {
            auto a0=world_pose(skate,skate.frame(f-1))[foot].p,a1=world_pose(skate,skate.frame(f))[foot].p;
            auto b0=world_pose(pinned,pinned.frame(f-1))[foot].p,b1=world_pose(pinned,pinned.frame(f))[foot].p;
            before+=std::hypot(a1.x-a0.x,a1.z-a0.z);after+=std::hypot(b1.x-b0.x,b1.z-b0.z);
        }
        check(after<0.25*before,"Contact IK removes most ankle skating");
    }
    for(size_t f=0;f<40;++f)for(size_t j=1;j<30;++j)check(length(pinned.frame(f)[j].p-skate.frame(f)[j].p)==0,"Local translations are kept");
    // An effector target is reached by two-bone IK.
    // The same frame with only the right arm turned: a reachable target.
    Clip reach;reach.joints=c.joints;reach.poses.assign(c.frame(20).begin(),c.frame(20).end());
    reach.poses[17].q=normalized(reach.poses[17].q*axis_angle({0,0,1},0.5f));reach.poses[18].q=normalized(reach.poses[18].q*axis_angle({0,1,0},0.4f));
    MotionConditions hand;hand.poses.push_back({"effector",20,"h.glb",0,{reach.poses.begin(),reach.poses.end()},8});
    auto handled=c;auto r=correct_kimodo(handled,hand,none);
    auto want=world_pose(reach,reach.frame(0))[19].p,got=world_pose(handled,handled.frame(20))[19].p,was=world_pose(c,c.frame(20))[19].p;
    check(length(was-want)>0.1f,"The effector target is away from the generated hand");
    check(r.effector_frames==1,"One effector frame");
    check(length(got-want)<0.03f,"Right hand reaches its target");
    bool refused=false;try{auto bad=c;(void)correct_kimodo(bad,empty,std::vector<float>(3));}catch(const std::exception&){refused=true;}
    check(refused,"Contacts must be four per frame");
}
void runtime_contracts() {
    // The C API now honours kimodo_runtime_options instead of ignoring them.
    kimodo_runtime_options options{};options.size=sizeof(options);options.backend_dir="plugins";
    char error[256]{};
    check(!kimodo_model_load("missing.gguf",nullptr,nullptr,&options,error,sizeof(error))&&std::strstr(error,"backend_dir"),"backend_dir is refused explicitly");
    options.backend_dir=nullptr;options.device=static_cast<kimodo_device>(9);
    check(!kimodo_model_load("missing.gguf",nullptr,nullptr,&options,error,sizeof(error))&&std::strstr(error,"device"),"Unknown devices are refused");
    options.device=KIMODO_DEVICE_CPU;options.threads=2;
    check(!kimodo_model_load("missing.gguf",nullptr,nullptr,&options,error,sizeof(error))&&!std::strstr(error,"device"),"Valid options reach the loader");
    kimodo::runtime_options cpp;cpp.device=kimodo::runtime_options::device_kind::cpu;cpp.threads=3;
    check(!kimodo::model::load("missing.gguf","",cpp).has_value(),"C++ load with options reports a missing model");
}
}
int main(int argc,char** argv) try {
    if(argc==3&&std::string(argv[1])=="--trace-synthetic") {
        // Every correction stage at once on a skating varied clip, for the upstream parity driver.
        auto c=skating(60);
        for(size_t f=0;f<60;++f)for(size_t j=1;j<22;++j)c.frame(f)[j].q=varied(60).frame(f)[j].q;
        std::vector<float> contacts(60*4,0);
        for(size_t f=0;f<60;++f)for(size_t k=0;k<4;++k)contacts[f*4+k]=(f%30<18)!=(k>=2)?1.f:0.f;
        MotionConditions m;
        Clip reach;reach.joints=c.joints;reach.poses.assign(c.frame(40).begin(),c.frame(40).end());
        reach.poses[17].q=normalized(reach.poses[17].q*axis_angle({0,0,1},0.5f));
        m.poses.push_back({"effector",40,"h.glb",0,{reach.poses.begin(),reach.poses.end()},8});
        auto key=varied(1);key.poses[0].p={0.1f,0.84f,0.2f};
        m.poses.push_back({"keyframe",20,"k.glb",0,{key.poses.begin(),key.poses.end()},0});
        PathConstraint p;p.frame=50;p.x=0.05f;p.z=0.6f;m.path.push_back(p);
        std::ofstream trace(argv[2]);(void)correct_kimodo(c,m,contacts,0.04f,0.5f,&trace);return 0;
    }
    if(argc==6&&std::string(argv[1])=="--replay") {
        // Reruns the Kimodo correction offline: RAW.glb (the same seed with --correction
        // none), the kimodo output's RECIPE.json (contacts and constraints), TRACE, OUT.glb.
        auto raw=load_glb(argv[2]);
        std::ifstream in(argv[3]);nlohmann::json recipe;in>>recipe;
        std::vector<float> contacts(raw.frames()*4,0);const char* names[4]={"leftFoot","leftToe","rightFoot","rightToe"};
        for(size_t k=0;k<4;++k)for(const auto& iv:recipe.at("footContacts").at(names[k]))
            for(size_t f=iv[0].get<size_t>();f<iv[1].get<size_t>();++f)contacts[f*4+k]=1;
        MotionConditions conditions;
        for(const auto& c:recipe.at("constraints")) {
            if(c.at("kind")=="path") {PathConstraint p;p.frame=c["frame"];p.x=c["x"];p.z=c["z"];conditions.path.push_back(p);continue;}
            auto source=load_glb(c.at("file").get<std::string>());long at=c.at("sourceFrame");
            if(at<0)at+=static_cast<long>(source.frames());
            PoseConstraint pose{c["kind"],c["frame"],c["file"],at,{},c.contains("effectors")?parse_effectors(c["effectors"]):0u};
            auto frame=source.frame(static_cast<size_t>(at));pose.pose.assign(frame.begin(),frame.end());pose.anchored=c["anchored"];
            conditions.poses.push_back(pose);
        }
        std::ofstream trace(argv[4]);
        (void)correct_kimodo(raw,conditions,contacts,0.04f,0.5f,&trace);
        save_glb(raw,argv[5]);return 0;
    }
    if(argc==3&&std::string(argv[1])=="--fixture77") {save_glb(soma30_to_77(varied(40)),argv[2]);return 0;}
    if(argc==4&&std::string(argv[1])=="--smooth") {
        std::ifstream in(argv[2]);std::vector<std::array<double,2>> x;std::string line;
        while(std::getline(in,line)) {std::istringstream s(line);double a,b;char comma;if(s>>a>>comma>>b)x.push_back({a,b});}
        auto y=smooth_root_path(x);std::ofstream out(argv[3]);out.precision(17);
        for(const auto& p:y)out<<p[0]<<","<<p[1]<<"\n";
        return 0;
    }
    soma77_contracts();
    dense_contracts();
    correction_contracts();
    runtime_contracts();
    std::cout<<"Capability contracts passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
