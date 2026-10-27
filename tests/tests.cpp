#include "motion.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace motion;
void check(bool yes,const char* why) {if(!yes)throw std::runtime_error(why);}
int main(int argc,char** argv) try {
    Clip c;c.joints=soma();
    for(int f=0;f<8;++f)for(size_t j=0;j<30;++j) { auto t=c.joints[j].rest;if(j==0)t.p={static_cast<float>(f)/30,1,0};c.poses.push_back(t); }
    c.validate();require_soma(c);
    if(argc==3&&std::string(argv[1])=="--fixture") {
        for(size_t f=0;f<c.frames();++f) {c.frame(f)[11].q=yaw(0.2f);c.frame(f)[17].q=yaw(-0.2f);}
        save_glb(c,argv[2]);return 0;
    }
    auto dir=fs::temp_directory_path()/"tide-motion-contract-tests";fs::create_directories(dir);
    auto path=dir/"roundtrip.glb";if(fs::exists(path))fs::remove(path);save_glb(c,path);auto loaded=load_glb(path);
    check(loaded.frames()==c.frames(),"GLB roundtrip frame count");
    for(size_t f=0;f<c.frames();++f)for(size_t j=0;j<30;++j)check(dot(c.frame(f)[j].p-loaded.frame(f)[j].p,c.frame(f)[j].p-loaded.frame(f)[j].p)<1e-12f,"GLB position roundtrip");
    std::array<std::vector<Transform>,8> guides;
    for(size_t f=0;f<8;++f)guides[f]={c.frame(f).begin(),c.frame(f).end()};
    auto in=encode_bricks(c,guides,8);check(in.allowed[2]==1&&in.allowed[0]==0,"Duration mask");
    check(in.has_root[3]==1&&in.has_local_root[3]==0&&in.has_local_root[7]==0,"Conditioning masks");
    check(std::abs(in.root[3*5])<1e-5f&&std::abs(in.root[3*5+2])<1e-5f,"Context ground alignment");
    check(std::abs(in.root[3*5+3])<1e-5f&&std::abs(in.root[3*5+4]-1)<1e-5f,"Context faces +X after alignment");
    auto moved=c;Transform placement{{12,0,-4},yaw(0.7f)};
    for(size_t f=0;f<moved.frames();++f)moved.frame(f)[0]=placement*moved.frame(f)[0];
    std::array<std::vector<Transform>,8> moved_guides;
    for(size_t f=0;f<8;++f)moved_guides[f]={moved.frame(f).begin(),moved.frame(f).end()};
    auto moved_in=encode_bricks(moved,moved_guides,8);
    for(size_t i=0;i<in.root.size();++i)check(std::abs(in.root[i]-moved_in.root[i])<1e-4f,"Root encoding is placement invariant");
    for(size_t i=0;i<in.pose.size();++i)check(std::abs(in.pose[i]-moved_in.pose[i])<1e-4f,"Body encoding is placement invariant");
    std::vector<float> features(64*366);
    for(size_t f=0;f<64;++f)for(size_t j=0;j<30;++j) {features[f*366+92+j*6]=1;features[f*366+92+j*6+4]=1;}
    auto bridge=decode_bricks(c,in,features,8,guides[3],guides[4]);
    check(bridge.frames()==26,"Boundary frame trimming");
    check(bridge.frame(0)[0].p.x==guides[3][0].p.x&&bridge.frame(25)[0].p.x==guides[4][0].p.x,"Exact endpoint positions");
    check(join(c,bridge,c).frames()==c.frames()*2+bridge.frames()-2,"Joined seam frames");
    bool rejected=false;try {auto bad=features;bad[0]=NAN;(void)decode_bricks(c,in,bad,8,guides[3],guides[4]);}catch(...){rejected=true;}
    check(rejected,"Nonfinite model output rejection");
    rejected=false;try{auto bad=c;bad.joints[1].rest.p.y+=0.1f;require_soma(bad);}catch(...){rejected=true;}
    check(rejected,"Different proportions rejected");
    rejected=false;try{auto bad=c;bad.frame(1)[4].p.x+=0.01f;require_soma(bad);}catch(...){rejected=true;}
    check(rejected,"Animated bone lengths rejected");
    for(size_t count:{30u,90u,120u,301u}) {
        auto many=c;many.poses.resize(count*30,c.poses.front());
        for(size_t f=0;f<count;++f)std::copy(c.frame(0).begin(),c.frame(0).end(),many.frame(f).begin());
        fs::remove(path);save_glb(many,path);check(load_glb(path).frames()==count,"Roundtrip fractional frame times");
    }
    fs::remove(path);fs::remove(dir);
    std::cout<<"Motion contracts passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
