#include "motion.hpp"
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdexcept>
using namespace motion;
void check(bool v,const char* m) {if(!v)throw std::runtime_error(m);}
float length(Vec3 p) {return std::sqrt(dot(p,p));}
int main(int argc,char** argv) {try {
    if(argc==3) {
        auto a=load_glb(utf8path(argv[1]),"",false),b=load_glb(utf8path(argv[2]),"",false);
        check(a.frames()==b.frames()&&a.joints.size()==b.joints.size(),"asset shape changed");
        float bone_error=0,root_error=0,position_error=0;
        for(size_t f=0;f<a.frames();++f) {
            auto wa=world_pose(a,a.frame(f)),wb=world_pose(b,b.frame(f));
            root_error=std::max(root_error,length(wa[a.motion_root].p-wb[b.motion_root].p));
            for(size_t j=0;j<a.joints.size();++j) {
                position_error=std::max(position_error,length(a.frame(f)[j].p-b.frame(f)[j].p));
                if(a.joints[j].parent>=0) {auto parent=static_cast<size_t>(a.joints[j].parent);bone_error=std::max(bone_error,std::abs(length(wa[j].p-wa[parent].p)-length(wb[j].p-wb[parent].p)));}
            }
        }
        std::cout<<"max world root position error "<<root_error<<" m; max local translation error "<<position_error<<" m; max bone length error "<<bone_error<<" m\n";
        check(root_error<1e-6f&&position_error<1e-6f&&bone_error<1e-6f,"asset invariant failure");return 0;
    }

    // Independent planar rig: two 0.5 m segments, bent knees, translating root.
    Clip c;c.joints={{"root",-1,{{0,0.85f,0},{}}},{"Lhip",0,{{0.1f,0,0},{}}},{"Lknee",1,{{0,-0.4f,0.3f},{}}},{"Lfoot",2,{{0,-0.4f,-0.3f},{}}},
        {"Rhip",0,{{-0.1f,0,0},{}}},{"Rknee",4,{{0,-0.4f,0.3f},{}}},{"Rfoot",5,{{0,-0.4f,-0.3f},{}}}};
    for(size_t f=0;f<31;++f)for(size_t j=0;j<c.joints.size();++j) {auto t=c.joints[j].rest;if(j==0)t.p.x=static_cast<float>(f)*0.001f;c.poses.push_back(t);}
    RetargetMapping m;m.bones[22]="Lhip";m.bones[23]="Lknee";m.bones[24]="Lfoot";m.bones[26]="Rhip";m.bones[27]="Rknee";m.bones[28]="Rfoot";
    CleanupPlan p;p.contacts={{0,0,30},{1,0,30}};p.fade_frames=0;
    auto original=c;auto r=cleanup_contacts(c,p,&m);
    check(r[0].drift_after<1e-6f&&r[1].drift_after<1e-6f,"planted feet still drift");
    check(r[0].drift_before>0.00099f,"independent drift oracle");
    for(size_t f=0;f<c.frames();++f) {
        auto w=world_pose(c,c.frame(f));
        for(size_t j=0;j<c.joints.size();++j) {
            check(length(c.frame(f)[j].p-original.frame(f)[j].p)==0,"translations changed");
            if(j)check(std::abs(length(w[j].p-w[static_cast<size_t>(c.joints[j].parent)].p)-length(c.joints[j].rest.p))<1e-6f,"bone length changed");
        }
        check(c.frame(f)[0].q.w==1,"root rotation changed");
        check(std::abs(w[3].q.w-1)<1e-6f&&std::abs(w[6].q.w-1)<1e-6f,"foot world orientation changed");
        check(length(w[3].p-Vec3{0.1f,0.05f,0})<1e-6f,"analytical foot target missed");
    }
    c=original;p.fade_frames=3;p.protected_frames={15};cleanup_contacts(c,p,&m);
    for(size_t j=0;j<c.joints.size();++j)check(c.frame(15)[j].q.w==original.frame(15)[j].q.w,"protected pose modified");
    check(c.frame(30)[1].q.w==1,"fade endpoint changed");
    c=original;p.max_correction=0.005f;auto skipped=cleanup_contacts(c,p,&m);check(skipped[0].skipped&&skipped[1].skipped,"unsafe contact not skipped");
    for(size_t f=0;f<c.frames();++f)check(c.frame(f)[1].q.w==1,"skipped contact modified");
    c=original;p.max_correction=0.1f;p.contacts={{0,2,12}};p.protected_frames.clear();cleanup_contacts(c,p,&m);
    check(c.frame(20)[1].q.w==1&&c.frame(20)[2].q.w==1,"outside interval changed (airborne/kick)");
    c=original;p.contacts={{0,0,30}};p.protected_frames.clear();p.fade_frames=3;
    for(size_t f=1;f<c.frames();++f)c.frame(f)[0].p.x=0.02f;
    auto worse=cleanup_contacts(c,p,&m);
    check(worse[0].skipped&&worse[0].skip_reason=="drift-increased","worsening fade was not rolled back");
    check(c.frame(15)[1].q.w==1,"rollback left solver rotations");
    // A centimetre hierarchy exercises ancestor uniform scale and rotated bone axes.
    c=original;c.joints[0].scale=0.01f;
    for(size_t f=0;f<c.frames();++f)for(size_t j=1;j<c.joints.size();++j)c.frame(f)[j].p=c.frame(f)[j].p*100;
    for(size_t j=1;j<c.joints.size();++j)c.joints[j].rest.p=c.joints[j].rest.p*100;
    p.fade_frames=0;auto scaled=cleanup_contacts(c,p,&m);check(scaled[0].drift_after<1e-6f,"scaled contact failed");
    c=original;p.contacts={{0,0,30}};p.protected_frames={0,30};
    // An unreachable anchor is reported as residual while bone lengths remain fixed.
    for(size_t f=0;f<c.frames();++f)c.frame(f)[0].p.y+=0.003f*static_cast<float>(f);
    p.max_correction=0.2f;auto unreachable=cleanup_contacts(c,p,&m);
    check(std::isfinite(unreachable[0].residual_max),"unreachable target produced nonfinite output");
    auto path=fs::temp_directory_path()/("tide-cleanup-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".glb");save_glb(c,path);auto read=load_glb(path,"",false);fs::remove(path);check(read.frames()==c.frames(),"cleanup export");
    p.contacts={{0,0,100}};bool refused=false;try{cleanup_contacts(c,p,&m);}catch(const std::exception&){refused=true;}check(refused,"bad interval accepted");
    std::cout<<"contact contracts passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}}
