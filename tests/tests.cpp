#include "motion.hpp"
#include "motion_decode.hpp"
#include "skeleton.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace motion;
void check(bool yes,const char* why) {if(!yes)throw std::runtime_error(why);}
float angle(Quat a,Quat b) {float d=std::abs(a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);return 2*std::acos(std::min(1.0f,d));}
Quat axis_angle(Vec3 axis,float radians) {axis=normalized(axis);float s=std::sin(radians/2);return {axis.x*s,axis.y*s,axis.z*s,std::cos(radians/2)};}
// A varied SOMA pose: every joint turned about a different axis, root moved and turned.
std::vector<Transform> test_pose(float seed,Vec3 root,float facing) {
    auto rig=soma();std::vector<Transform> pose;
    for(size_t j=0;j<rig.size();++j) {
        float t=seed+static_cast<float>(j);
        Transform joint=rig[j].rest;joint.q=axis_angle({std::sin(t),std::cos(1.3f*t),0.5f+std::sin(0.7f*t)},0.6f*std::sin(2.1f*t));
        if(!j) {joint.p=root;joint.q=normalized(yaw(facing)*joint.q);}
        pose.push_back(joint);
    }
    return pose;
}
// The preview mannequin is a valid skinned GLB on the SOMA-30 skeleton: its rig reads
// back as SOMA, every corner is rigidly bound to a SOMA joint, indices are in range,
// and the closed surface winds outward (positive enclosed volume, glTF front faces).
void mannequin_contracts() {
    auto path=fs::temp_directory_path()/"tide-motion-mannequin-test.glb";
    save_mannequin(path);
    auto rig=load_rig_glb(path);require_soma(rig);
    std::ifstream in(path,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());in.close();
    auto u32=[&](size_t at){uint32_t v;std::memcpy(&v,bytes.data()+at,4);return v;};
    check(u32(0)==0x46546c67&&u32(8)==bytes.size(),"Mannequin GLB header");
    size_t json_length=u32(12);auto doc=nlohmann::json::parse(std::string(bytes.data()+20,json_length));
    const char* bin=bytes.data()+20+json_length+8;
    check(doc["skins"][0]["joints"].size()==30&&doc["meshes"].size()==1,"Mannequin has one mesh skinned to 30 joints");
    auto attribute=[&](const char* name){return doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"][name].get<size_t>()];};
    auto data=[&](const nlohmann::json& accessor){const auto& view=doc["bufferViews"][accessor["bufferView"].get<size_t>()];return bin+view["byteOffset"].get<size_t>();};
    size_t count=attribute("POSITION")["count"];
    const float* p=reinterpret_cast<const float*>(data(attribute("POSITION")));
    const float* n=reinterpret_cast<const float*>(data(attribute("NORMAL")));
    const uint16_t* j=reinterpret_cast<const uint16_t*>(data(attribute("JOINTS_0")));
    const float* w=reinterpret_cast<const float*>(data(attribute("WEIGHTS_0")));
    for(size_t i=0;i<count;++i) {
        check(j[i*4]<30&&w[i*4]==1&&w[i*4+1]==0,"Rigid binding to a SOMA joint");
        check(std::abs(n[i*3]*n[i*3]+n[i*3+1]*n[i*3+1]+n[i*3+2]*n[i*3+2]-1)<1e-4f,"Unit normals");
    }
    const auto& indices=doc["accessors"][doc["meshes"][0]["primitives"][0]["indices"].get<size_t>()];
    const uint32_t* index=reinterpret_cast<const uint32_t*>(data(indices));
    double volume=0;
    for(size_t t=0;t<indices["count"].get<size_t>();t+=3) {
        Vec3 a{p[index[t]*3],p[index[t]*3+1],p[index[t]*3+2]},b{p[index[t+1]*3],p[index[t+1]*3+1],p[index[t+1]*3+2]},c{p[index[t+2]*3],p[index[t+2]*3+1],p[index[t+2]*3+2]};
        check(index[t]<count&&index[t+1]<count&&index[t+2]<count,"Indices in range");
        volume+=dot(a,cross(b,c))/6;
    }
    check(volume>0.02&&volume<0.2,"Closed outward surface with a body-sized volume");
    fs::remove(path);
}
void kimodo_contracts() {
    mannequin_contracts();
    Clip rig;rig.joints=soma();
    // Encoding a pose and decoding it with Kimodo's own decoder (identity statistics,
    // whose eps-adjusted scale is exactly one) returns the same local pose.
    auto pose=test_pose(0.3f,{1.5f,0.93f,-2.25f},0.8f);
    auto row=encode_kimodo(rig,pose);
    check(row.size()==kimodo_features&&kimodo_features==kimodo::detail::soma30_spec.motion_dim(),"Kimodo SOMA feature width");
    std::vector<float> mean5(5,0),std5(5,std::sqrt(1-1e-5f)),mean_body(kimodo_features-5,0),std_body(kimodo_features-5,std::sqrt(1-1e-5f));
    auto decoded=kimodo::detail::decode_motion(row,1,kimodo::detail::soma30_spec,mean5,std5,mean_body,std_body);
    check(decoded.has_value(),"Kimodo decodes encoded features");
    for(size_t j=0;j<30;++j) {
        const float* q=decoded->local_xyzw.data()+j*4;
        check(angle({q[0],q[1],q[2],q[3]},pose[j].q)<1e-3f,"Kimodo decode recovers encoded local rotations");
    }
    for(size_t k=0;k<3;++k)check(std::abs(decoded->root_positions[k]-(k==0?1.5f:k==1?0.93f:-2.25f))<1e-5f,"Kimodo decode recovers the encoded root");
    auto back=decode_kimodo(row);
    for(size_t j=0;j<30;++j)check(angle(back.frame(0)[j].q,pose[j].q)<1e-3f,"Native decode mirrors Kimodo's");
    auto again=encode_kimodo(rig,back.frame(0));
    for(size_t i=0;i<row.size();++i)check(std::abs(again[i]-row[i])<1e-4f,"encode(decode(x)) returns x");
    // Heading follows upstream: atan2 of the right-to-left hip line; 0 faces +Z.
    auto rest=rig.joints;std::vector<Transform> standing;for(const auto& joint:rest)standing.push_back(joint.rest);
    for(float facing:{0.0f,0.5f,1.5707964f,-2.0f}) {standing[0].q=yaw(facing);check(std::abs(std::remainder(kimodo_heading(rig,standing)-facing,6.2831853f))<2e-3f,"Heading is yaw about +Y");}
    // Conditions: a start pose sets the origin and first heading; masks per constraint kind.
    MotionConditions conditions;
    conditions.poses.push_back({"start-pose",0,"a.glb",-1,pose,0});
    conditions.poses.push_back({"end-pose",59,"b.glb",0,test_pose(1.7f,{3,0.9f,1},-1),0});
    conditions.poses.push_back({"effector",30,"c.glb",0,test_pose(2.9f,{2,0.9f,0},0.2f),parse_effectors("right-hand,left-foot")});
    conditions.path.push_back({45,2.5f,0.5f,true,90});
    check(!place_constraints(conditions,Placement::automatic),"A start pose fixes the world");
    for(const auto& p:conditions.poses)check(p.anchored,"Automatic placement anchors poses after a start pose");
    auto c=build_condition(conditions,60);
    check(std::abs(c.origin.x-1.5f)<1e-6f&&std::abs(c.origin.z+2.25f)<1e-6f&&std::abs(c.first_heading-kimodo_heading(rig,pose))<1e-6f,"Start pose sets origin and first heading");
    auto count=[&](size_t frame){size_t n=0;for(size_t d=0;d<kimodo_features;++d)n+=c.mask[frame*kimodo_features+d]!=0;return n;};
    check(count(0)==5+90+24&&count(59)==5+90+24,"Full-body pose masks root, positions and end-effector rotations");
    check(count(30)==5+2*(3+3+6),"Effector pose masks root and its chains");
    check(count(45)==4&&count(10)==0,"Path masks ground position and heading only");
    check(std::abs(c.observed[0])<1e-6f&&std::abs(c.observed[2])<1e-6f,"Start pose sits at the model origin");
    check(std::abs(c.observed[45*kimodo_features]-1)<1e-5f&&std::abs(c.observed[45*kimodo_features+3])<1e-5f&&std::abs(c.observed[45*kimodo_features+4]-1)<1e-5f,"Path is relative to the origin, yaw 90 faces +X");
    // Without a start pose, poses float and the output follows the last one.
    MotionConditions floating;floating.poses.push_back({"end-pose",59,"b.glb",0,test_pose(1.7f,{3,0.9f,1},-1),0});
    check(place_constraints(floating,Placement::automatic)&&!floating.poses[0].anchored,"End pose alone floats");
    auto f=build_condition(floating,60);
    check(f.mask[59*kimodo_features]==0&&f.mask[59*kimodo_features+1]==1&&f.mask[59*kimodo_features+2]==0&&f.mask[59*kimodo_features+3]==1,"Floating pose leaves ground position free");
    check(!place_constraints(floating,Placement::world)&&floating.poses[0].anchored,"World placement anchors");
    bool rejected=false;try{(void)build_condition(conditions,59);}catch(...){rejected=true;}
    check(rejected,"Constraint beyond the output rejected");
    rejected=false;try{auto twice=conditions;twice.poses[1].frame=0;(void)build_condition(twice,60);}catch(...){rejected=true;}
    check(rejected,"Two poses on one frame rejected");
    rejected=false;try{(void)parse_effectors("left-hand,nose");}catch(...){rejected=true;}
    check(rejected,"Unknown effector rejected");
    // Correction makes full-body poses exact and leaves frames beyond its window alone.
    Clip out;out.joints=soma();
    for(size_t frame=0;frame<60;++frame) {auto p=test_pose(0.05f*static_cast<float>(frame),{0.02f*static_cast<float>(frame),0.9f,0},0.1f);out.poses.insert(out.poses.end(),p.begin(),p.end());}
    MotionConditions keys;keys.poses.push_back({"keyframe",20,"k.glb",0,pose,0});keys.poses.push_back({"keyframe",25,"k.glb",0,test_pose(4.0f,{0.4f,0.95f,0.1f},0.3f),0});
    (void)place_constraints(keys,Placement::world);
    auto before=out;
    check(constraint_error(out,keys.poses[0]).rotation_max>1,"Test keyframe starts inexact");
    correct_poses(out,keys);
    for(const auto& key:keys.poses) {auto e=constraint_error(out,key);check(e.root<1e-4f&&e.position_max<1e-4f&&e.rotation_max<0.05f,"Corrected keyframes are exact");}
    for(size_t frame:{0u,9u,36u,59u})for(size_t j=0;j<30;++j) {
        auto a=out.frame(frame)[j],b=before.frame(frame)[j];
        check(a.q.x==b.q.x&&a.q.y==b.q.y&&a.q.z==b.q.z&&a.q.w==b.q.w&&a.p.x==b.p.x&&a.p.y==b.p.y&&a.p.z==b.p.z,"Correction stays within its window");
    }
    check(angle(out.frame(15)[3].q,before.frame(15)[3].q)>1e-3f,"Correction fades in before the keyframe");
}
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
    kimodo_contracts();
    std::cout<<"Motion contracts passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
