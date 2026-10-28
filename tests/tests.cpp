#include "motion.hpp"
#include "author.hpp"
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
// Generation controls without a model: variant names, mirror, in-place, key frames and
// the correction window.
bool same(Quat a,Quat b) {return a.x==b.x&&a.y==b.y&&a.z==b.z&&a.w==b.w;}
void edit_contracts() {
    // Authored support -> lifted leg -> stable landing, no inference fixtures.
    Clip stepping;stepping.joints=soma();
    for(size_t f=0;f<70;++f) {
        std::vector<Transform> pose;for(const auto& joint:stepping.joints)pose.push_back(joint.rest);
        pose[0].p={0,1,0};
        if(f>=10&&f<25) {pose[22].q=axis_angle({1,0,0},0.8f);pose[0].p.z=0.3f*static_cast<float>(f-10)/15;}
        else if(f>=25)pose[0].p.z=0.3f;
        if(f>=40&&f<55) {pose[26].q=axis_angle({1,0,0},0.8f);pose[0].p.z=0.3f+0.3f*static_cast<float>(f-40)/15;}
        else if(f>=55)pose[0].p.z=0.6f;
        stepping.poses.insert(stepping.poses.end(),pose.begin(),pose.end());
    }
    auto falls=footfalls(stepping);
    check(falls.size()==2&&falls[0].foot==0&&falls[1].foot==1,"Two authored travelling footfalls counted in order");
    check(falls[0].lift==10&&falls[0].landing==26&&falls[1].landing==56,"Footfall frames identify complete swings");
    auto placed_steps=stepping;
    for(size_t f=0;f<placed_steps.frames();++f)placed_steps.frame(f)[0]=Transform{{7,2,-3},yaw(0.8f)}*placed_steps.frame(f)[0];
    check(footfalls(placed_steps).size()==2,"Footfalls independent of world translation and heading");
    auto on_spot=stepping;for(size_t f=0;f<on_spot.frames();++f)on_spot.frame(f)[0].p.z=0;
    check(footfalls(on_spot).empty(),"Kicks returning to the same ground position do not count as travelling steps");
    auto unfinished=stepping;unfinished.poses.resize(20*30);
    check(footfalls(unfinished).empty(),"Unfinished swing is excluded");
    auto sliding=stepping;
    for(size_t f=0;f<sliding.frames();++f) {sliding.frame(f)[22].q={};sliding.frame(f)[26].q={};}
    check(footfalls(sliding).empty(),"Ground sliding without a swing does not count as footsteps");
    auto airborne_start=stepping;airborne_start.poses.erase(airborne_start.poses.begin(),airborne_start.poses.begin()+15*30);
    auto later=footfalls(airborne_start);
    check(later.size()==1&&later[0].foot==1,"Incomplete initial swing is excluded");


    check(variant_path(fs::path("out")/"kick.glb",7)==fs::path("out")/"kick.seed7.glb","Variant names carry the seed");
    check(variant_path("a.b.glb",4294967295ull)==fs::path("a.b.seed4294967295.glb"),"Variant names keep earlier dots");
    check(variants_path("kick.glb",".png")==fs::path("kick.variants.png")&&variants_path("kick.glb",".gif")==fs::path("kick.variants.gif"),"Contact sheet names");
    Clip rig;rig.joints=soma();
    // A clip facing 0.6 rad away from +Z, walking and turning, with every joint moving.
    Clip c;c.joints=soma();
    for(size_t f=0;f<40;++f) {auto p=test_pose(0.07f*static_cast<float>(f),{1.0f+0.03f*static_cast<float>(f),0.9f+0.01f*std::sin(0.3f*static_cast<float>(f)),-2.0f+0.02f*static_cast<float>(f)},0.6f+0.01f*static_cast<float>(f));c.poses.insert(c.poses.end(),p.begin(),p.end());}
    auto mirrored=c;mirror_soma(mirrored);
    auto twice=mirrored;mirror_soma(twice);
    for(size_t f=0;f<c.frames();++f)for(size_t j=0;j<30;++j) {
        check(angle(twice.frame(f)[j].q,c.frame(f)[j].q)<5e-3f,"Mirroring twice restores rotations (SOMA is symmetric to 0.2 mm)");
        Vec3 d=twice.frame(f)[j].p-c.frame(f)[j].p;check(dot(d,d)<1e-4f,"Mirroring twice restores the root (within 1 cm: the plane comes from each clip's own heading)");
    }
    check(dot(mirrored.frame(0)[0].p-c.frame(0)[0].p,mirrored.frame(0)[0].p-c.frame(0)[0].p)<1e-10f,"Mirror starts in the same place");
    check(std::abs(std::remainder(kimodo_heading(mirrored,mirrored.frame(0))-kimodo_heading(c,c.frame(0)),6.2831853f))<5e-3f,"Mirror starts facing the same way");
    // A mirrored left foot is the right foot reflected across the first frame's sagittal plane.
    const float h=kimodo_heading(c,c.frame(0));Vec3 origin=c.frame(0)[0].p;origin.y=0;
    for(size_t f:{0u,17u,39u}) {
        auto a=world_pose(c,c.frame(f)),b=world_pose(mirrored,mirrored.frame(f));
        for(auto [from,to]:{std::pair{28,24},std::pair{24,28},std::pair{19,13},std::pair{6,6}}) {
            Vec3 local=rotate(yaw(-h),a[static_cast<size_t>(from)].p-origin);local.x=-local.x;Vec3 want=origin+rotate(yaw(h),local),d=b[static_cast<size_t>(to)].p-want;
            check(std::sqrt(dot(d,d))<0.01f,"Mirror swaps sides across the sagittal plane");
        }
    }
    bool rejected=false;try{Clip other;other.joints=soma();other.joints.pop_back();other.poses.assign(29,{});mirror_soma(other);}catch(...){rejected=true;}
    check(rejected,"Mirror needs a SOMA-30 clip");
    // In place: steady travel goes, frame 0, height, facing and small sway stay.
    Clip walk;walk.joints=soma();
    for(size_t f=0;f<90;++f) {
        auto p=test_pose(0.0f,{0.05f*std::sin(0.4f*static_cast<float>(f)),0.9f+0.02f*std::sin(0.5f*static_cast<float>(f)),0.04f*static_cast<float>(f)},0.2f);
        walk.poses.insert(walk.poses.end(),p.begin(),p.end());
    }
    auto travel=root_travel(walk);
    check(travel.displacement>3.5f&&travel.distance>=travel.displacement&&travel.peak_speed>1.2f,"Root travel measures actual motion");
    auto stationary=walk;
    for(size_t f=0;f<stationary.frames();++f)stationary.frame(f)[0].p=walk.frame(0)[0].p;
    auto zero=root_travel(stationary);
    check(zero.displacement==0&&zero.distance==0&&zero.peak_speed==0,"Stationary root has no travel");
    auto still=walk;remove_root_travel(still);
    for(size_t f=0;f<walk.frames();++f) {
        auto a=walk.frame(f)[0],b=still.frame(f)[0];
        check(a.p.y==b.p.y&&a.q.x==b.q.x&&a.q.y==b.q.y&&a.q.w==b.q.w,"In place keeps height and facing");
        check(std::hypot(b.p.x,b.p.z)<0.1f,"In place removes travel (3.6 m here), keeping 5 cm sway");
    }
    check(still.frame(0)[0].p.x==walk.frame(0)[0].p.x&&still.frame(0)[0].p.z==walk.frame(0)[0].p.z,"In place keeps frame 0");
    float sway=0;for(size_t f=0;f<still.frames();++f)sway=std::max(sway,std::abs(still.frame(f)[0].p.x));
    check(sway>0.03f,"In place keeps sway");
    // Key frames: first, last and the clip's strongest moment.
    Clip kick;kick.joints=soma();
    for(size_t f=0;f<75;++f) {
        auto p=test_pose(0.0f,{0,0.9f,0},0);float lift=std::exp(-0.02f*(static_cast<float>(f)-40)*(static_cast<float>(f)-40));
        p[26].q=axis_angle({1,0,0},-1.4f*lift);kick.poses.insert(kick.poses.end(),p.begin(),p.end());
    }
    auto keys=key_frames(kick,6);
    check(keys.size()==6&&keys.front()==0&&keys.back()==74&&std::is_sorted(keys.begin(),keys.end()),"Key frames span the clip in order");
    check(std::find(keys.begin(),keys.end(),40u)!=keys.end(),"Key frames include the peak");
    check(key_frames(kick,100).size()==75,"Short clips show every frame");
    // Correction window 0 sets only the keyframe.
    Clip out;out.joints=soma();
    for(size_t frame=0;frame<30;++frame) {auto p=test_pose(0.05f*static_cast<float>(frame),{0,0.9f,0},0.1f);out.poses.insert(out.poses.end(),p.begin(),p.end());}
    MotionConditions one;one.poses.push_back({"keyframe",15,"k.glb",0,test_pose(4.0f,{0.1f,0.95f,0},0.3f),0});
    (void)place_constraints(one,Placement::world);
    auto snapped=out;correct_poses(snapped,one,0);
    auto e=constraint_error(snapped,one.poses[0]);check(e.position_max<1e-4f&&e.rotation_max<0.05f,"Window 0 still makes the pose exact");
    for(size_t frame:{14u,16u})for(size_t j=0;j<30;++j)check(same(snapped.frame(frame)[j].q,out.frame(frame)[j].q),"Window 0 leaves neighbours alone");
    auto wide=out;correct_poses(wide,one,3);
    check(angle(wide.frame(13)[3].q,out.frame(13)[3].q)>1e-4f&&same(wide.frame(11)[3].q,out.frame(11)[3].q),"Window 3 fades over three frames");
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
    // A leaf orientation can be wrong while every joint position is exact.
    Clip leaf;leaf.joints=soma();for(const auto& joint:leaf.joints)leaf.poses.push_back(joint.rest);
    PoseConstraint leaf_target{"start-pose",0,"leaf.glb",0,leaf.poses,0};
    leaf.frame(0)[15].q=yaw(1.5707963268f);
    auto leaf_error=constraint_error(leaf,leaf_target);
    check(leaf_error.position_max<1e-6f&&std::abs(leaf_error.rotation_mean-3)<0.01f,"Raw rotation score includes terminal orientations even with exact positions");
    check(leaf_error.rotation_max_joint=="LeftHandMiddleEnd"&&std::abs(leaf_error.rotation_max-90)<0.01f,"Worst raw rotation names its actual joint");
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
    // Colocated paths relocate and turn poses before encoding and correction.
    auto located=conditions;
    located.path.push_back({59,-2,4,true,90});
    (void)place_constraints(located,Placement::free);
    check(located.poses[1].anchored,"Explicit path anchors even in free placement");
    check(located.poses[1].pose[0].p.x==-2&&located.poses[1].pose[0].p.z==4,"Colocated path places pose root");
    check(std::abs(kimodo_heading(rig,located.poses[1].pose)-1.5707964f)<1e-4f,"Colocated yaw turns the pose");
    for(size_t j=1;j<30;++j)check(same(located.poses[1].pose[j].q,conditions.poses[1].pose[j].q),"Placement preserves local body rotations");
    auto lc=build_condition(located,60);
    check(std::abs(lc.observed[59*kimodo_features]-(-2-c.origin.x))<1e-5f,"Merged pose uses path location");
    // --follow ends: an end pose turned to face 60 degrees and moved 1 m turns a target
    // halfway through the clip by half that turn about its original hips and moves it 0.5 m;
    // its height and body-local rotations stay, and the start and end themselves do not move.
    {
        MotionConditions follow;
        auto start=test_pose(0,{0,0.9f,0},0),end=test_pose(0,{0.2f,0.9f,0.1f},0),cut=test_pose(0.4f,{0.6f,0.8f,0.5f},0.3f);
        follow.poses.push_back({"start-pose",0,"idle.glb",-1,start,0});
        follow.poses.push_back({"end-pose",60,"idle.glb",0,end,0});
        follow.poses.push_back({"effector",30,"cut.glb",9,cut,parse_effectors("right-hand")});
        follow.path.push_back({60,1.2f,0.1f,true,60});
        const auto source=follow.poses[1];
        (void)place_constraints(follow,Placement::world);
        auto turns=follow_ends(follow,source);
        const float half=0.5f*std::remainder(1.0471976f-kimodo_heading(rig,end),6.2831853f);
        check(turns.size()==1&&std::abs(turns[0]-half)<1e-4f,"Halfway target takes half the end pose's turn");
        Vec3 d=cut[0].p-Vec3{0.2f,0,0.1f};d.y=0;d=rotate(yaw(half),d);
        const Vec3 want{0.2f+d.x+0.5f,0.8f,0.1f+d.z};
        const Vec3 got=follow.poses[2].pose[0].p;
        check(std::abs(got.x-want.x)<1e-5f&&std::abs(got.y-want.y)<1e-6f&&std::abs(got.z-want.z)<1e-5f,"Target turns about the end pose's hips and moves half its offset");
        check(std::abs(std::remainder(kimodo_heading(rig,follow.poses[2].pose)-kimodo_heading(rig,cut)-half,6.2831853f))<1e-4f,"Target facing turns with it");
        for(size_t j=1;j<30;++j)check(same(follow.poses[2].pose[j].q,cut[j].q),"Following keeps body-local rotations");
        check(follow.poses[0].pose[0].p.x==0&&follow.poses[1].pose[0].p.x==1.2f,"Start and end poses are not moved again");
        auto unturned=follow;unturned.path.clear();unturned.poses[2].pose=cut;unturned.poses[1].pose=end;
        (void)follow_ends(unturned,unturned.poses[1]);
        for(size_t j=0;j<30;++j)check(same(unturned.poses[2].pose[j].q,cut[j].q)&&unturned.poses[2].pose[j].p.x==cut[j].p.x,"An unplaced end pose leaves targets where they are");
        MotionConditions no_end;no_end.poses.push_back(follow.poses[2]);
        bool refused=false;try{(void)follow_ends(no_end,source);}catch(...){refused=true;}
        check(refused,"Following the ends needs an end pose");
    }
    auto duplicate=located;duplicate.path.push_back(located.path.back());
    bool duplicate_refused=false;try{(void)place_constraints(duplicate,Placement::world);}catch(...){duplicate_refused=true;}
    check(duplicate_refused,"Duplicate paths still refused");
    MotionConditions start_only;start_only.poses.push_back(conditions.poses[0]);
    (void)place_constraints(start_only,Placement::world);auto world_start=build_condition(start_only,60);
    (void)place_constraints(start_only,Placement::free);auto free_start=build_condition(start_only,60);
    check(world_start.mask==free_start.mask&&world_start.observed==free_start.observed,"Start-only free/world conditioning is identical");
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
    if(argc>=6&&std::string(argv[1])=="--variant-sheet") {
        std::vector<Clip> clips;std::vector<std::string> labels;
        for(int i=4;i<argc;++i) {clips.push_back(load_glb(argv[i]));labels.push_back(fs::path(argv[i]).stem().string());}
        preview_variants(clips,labels,VariantSheet{argv[2],argv[3],{}});return 0;
    }
    if(argc==7&&std::string(argv[1])=="--verify-placed") {
        auto source=load_glb(argv[2]),output=load_glb(argv[3]);
        std::vector<Transform> end(source.frame(0).begin(),source.frame(0).end());
        if(std::string(argv[6])!="keep") {
            const float facing=std::stof(argv[6])*3.14159265358979323846f/180;
            end[0].q=normalized(yaw(facing-kimodo_heading(source,end))*end[0].q);
        }
        end[0].p.x=std::stof(argv[4]);end[0].p.z=std::stof(argv[5]);
        for(size_t f:{size_t{0},output.frames()-1}) {
            auto want=f?std::span<const Transform>(end):source.frame(source.frames()-1);
            auto got=output.frame(f);
            for(size_t j=0;j<30;++j) {
                const auto d=got[j].p-want[j].p;
                check(dot(d,d)<1e-10f,"Exported placed endpoint position");
                check(angle(got[j].q,want[j].q)<0.002f,"Exported placed endpoint rotation");
            }
        }
        std::cout<<"Exported placed endpoints verified\n";return 0;
    }
    if(argc==3&&std::string(argv[1])=="--feet") {
        // Per frame: left/right ankle and toe heights and horizontal speeds (m, m/s), then hips XYZ.
        auto clip=load_glb(argv[2]);std::array<size_t,4> ids{};
        const char* names[]={"LeftFoot","LeftToeBase","RightFoot","RightToeBase"};
        for(size_t k=0;k<4;++k){auto it=std::find_if(clip.joints.begin(),clip.joints.end(),[&](const Joint& j){return j.name==names[k];});check(it!=clip.joints.end(),"Foot joint");ids[k]=static_cast<size_t>(it-clip.joints.begin());}
        std::vector<Transform> prev;
        for(size_t f=0;f<clip.frames();++f) {
            auto w=world_pose(clip,clip.frame(f));std::cout<<f;
            for(size_t k=0;k<4;++k) {
                auto p=w[ids[k]].p;float v=0;if(!prev.empty()){auto q=prev[ids[k]].p;v=std::sqrt((p.x-q.x)*(p.x-q.x)+(p.z-q.z)*(p.z-q.z))*30;}
                std::cout<<" "<<p.y<<" "<<v;
            }
            std::cout<<" "<<w[0].p.x<<" "<<w[0].p.y<<" "<<w[0].p.z<<"\n";prev=w;
        }
        return 0;
    }
    if(argc==4&&std::string(argv[1])=="--compare") {
        // Largest local translation (m) and rotation (rad) difference between two clips.
        auto a=load_glb(argv[2]),b=load_glb(argv[3]);
        check(a.frames()==b.frames()&&a.joints.size()==b.joints.size(),"Compared clips have the same shape");
        float p=0,q=0;
        for(size_t f=0;f<a.frames();++f)for(size_t j=0;j<a.joints.size();++j) {
            auto d=a.frame(f)[j].p-b.frame(f)[j].p;p=std::max(p,std::sqrt(dot(d,d)));q=std::max(q,angle(a.frame(f)[j].q,b.frame(f)[j].q));
        }
        std::cout<<"frames "<<a.frames()<<" maxPosition "<<p<<" maxRotation "<<q<<"\n";return 0;
    }
    if(argc==3&&std::string(argv[1])=="--end-speed") {
        // Hips ground speed over the last three intervals and the frame count.
        auto clip=load_glb(argv[2]);check(clip.frames()>=4,"End speed needs four frames");
        auto a=clip.frame(clip.frames()-4)[0].p,b=clip.frame(clip.frames()-1)[0].p;
        float dx=b.x-a.x,dz=b.z-a.z;
        std::cout<<"frames "<<clip.frames()<<" endSpeed "<<std::sqrt(dx*dx+dz*dz)*10<<"\n";return 0;
    }
    if(argc==4&&std::string(argv[1])=="--verify-sequence") {
        auto plan=load_pose_sequence(argv[2]);auto output=load_glb(argv[3]);
        size_t frame=0;
        auto verify=[&](std::span<const Transform> expected) {
            auto actual=output.frame(frame);
            for(size_t j=0;j<30;++j) {
                auto d=actual[j].p-expected[j].p;check(dot(d,d)<1e-10f,"Exported pose endpoint position");
                check(angle(actual[j].q,expected[j].q)<0.002f,"Exported pose endpoint rotation");
            }
        };
        verify(plan.start.frame(3));
        // Arrivals come from the effective recipe: the model chooses within token ranges.
        nlohmann::json recipe;{std::ifstream in(std::string(argv[3])+".json");in>>recipe;}
        const auto& targets=recipe.at("authoring").at("targets");check(targets.size()==plan.targets.size(),"Recipe target count");
        size_t guides=0;
        for(size_t i=0;i<plan.targets.size();++i) {
            const auto& t=targets[i];int chosen=t.at("chosenTokens"),early=t.at("early");
            const auto& r=plan.targets[i].request;check(chosen>=r.min_tokens&&chosen<=r.max_tokens,"Chosen tokens within allowed range");
            check(t.at("startFrame").get<size_t>()==frame,"Segment starts where the previous ended");
            frame+=static_cast<size_t>(4*chosen-7-early);check(t.at("endFrame").get<size_t>()==frame,"Recorded arrival frame");
            if(early)++guides;else verify(plan.targets[i].window.frame(0));
        }
        check(output.frames()==frame+1,"Exported exact sequence timing");
        check(recipe.at("predictedFootContacts").at("frames").size()==output.frames(),"Predicted contacts cover every frame");
        std::cout<<"Verified "<<plan.targets.size()-guides<<" exact endpoint poses, "<<guides<<" guide targets and arrival frames: "<<output.frames()<<" frames\n";return 0;
    }
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
    // Independently predictable translating clip: root +X at 1m/s, facing +Z.
    nlohmann::json pose={{"file","roundtrip.glb"},{"frame",3},{"position",{4,2,-3}},{"yaw",90}};
    nlohmann::json target=pose;target["frame"]=1;target["tokens"]=6;
    nlohmann::json doc={{"schema","tide-motion-pose-sequence-1"},{"start",pose},{"targets",{target,target}}};
    auto sequence_file=dir/"poses.json";
    auto write_sequence=[&](){std::ofstream out(sequence_file);out<<doc.dump();};write_sequence();
    auto plan=load_pose_sequence(sequence_file);
    check(std::abs(plan.start.frame(3)[0].p.x-4)<1e-6f&&std::abs(plan.start.frame(3)[0].p.y-2)<1e-6f,"Absolute root placement");
    check(std::abs(plan.start.frame(0)[0].p.z+2.9f)<1e-5f,"Context samples precede anchor and rotate travel");
    check(std::abs(plan.targets[0].window.frame(3)[0].p.z+3.1f)<1e-5f,"Target samples follow anchor and rotate travel");
    check(std::abs(kimodo_heading(plan.start,plan.start.frame(3))-1.5707963f)<1e-4f,"Absolute facing");
    for(size_t j=1;j<30;++j)check(same(plan.start.frame(3)[j].q,c.frame(3)[j].q),"Placement preserves local body rotations");
    check(plan.recipe["targets"][0]["endFrame"]==17&&plan.recipe["targets"][1]["endFrame"]==34,"Exact trimmed timing accumulates intervals");
    check(plan.targets[0].request.target_poses==4&&plan.targets[0].early==0,"Target defaults: four poses, no early transition");
    {
        auto variant=doc;variant["targets"][0]["tokens"]={6,12};variant["targets"][0]["poses"]=1;variant["targets"][0]["early"]=10;
        {std::ofstream out(sequence_file);out<<variant.dump();}
        auto ranged=load_pose_sequence(sequence_file);
        check(ranged.targets[0].request.min_tokens==6&&ranged.targets[0].request.max_tokens==12&&ranged.targets[0].request.target_poses==1&&ranged.targets[0].early==10,"Token range, poses and early parsed");
        check(!ranged.recipe["targets"][0].contains("endFrame")&&!ranged.recipe["targets"][1].contains("endFrame"),"Ranged timing waits for the model");
        auto refused=[&](nlohmann::json d,const char* what) {
            {std::ofstream out(sequence_file);out<<d.dump();}
            bool r=false;try{(void)load_pose_sequence(sequence_file);}catch(const std::exception&){r=true;}check(r,what);
        };
        auto bad=doc;bad["targets"][1]["early"]=4;refused(bad,"Final target early refused");
        bad=doc;bad["targets"][0]["early"]=15;refused(bad,"Early leaving under four frames refused");
        bad=doc;bad["targets"][0]["poses"]=0;refused(bad,"Zero target poses refused");
        bad=doc;bad["targets"][0]["poses"]=2;bad["targets"][0]["hold"]=true;refused(bad,"hold with poses refused");
        bad=doc;bad["targets"][0]["tokens"]={12,8};refused(bad,"Inverted token range refused");
        bad=doc;bad["targets"][0]["tokens"]={6,17};refused(bad,"Token range above 16 refused");
        write_sequence();
    }
    doc["targets"][0]["hold"]=true;doc["start"]["frame"]=-1;write_sequence();plan=load_pose_sequence(sequence_file);
    check(plan.recipe["start"]["frame"]==7,"Negative source frame resolves");
    {std::ofstream out(sequence_file);out<<plan.recipe.dump();}
    auto reloaded=load_pose_sequence(sequence_file);
    check(reloaded.recipe==plan.recipe,"Effective authoring recipe can be reused");
    check(plan.targets[0].window.frame(0)[0].p.z==plan.targets[0].window.frame(3)[0].p.z,"Held pose repeats four guides");
    Clip assembled;assembled.joints=soma();assembled.poses.assign(plan.start.frame(3).begin(),plan.start.frame(3).end());
    Clip fake;fake.joints=soma();for(int f=0;f<18;++f)fake.poses.insert(fake.poses.end(),assembled.frame(0).begin(),assembled.frame(0).end());
    fake.frame(17)[0].p.x=5;append_pose_bridge(assembled,fake);
    check(assembled.frames()==18&&assembled.frame(17)[0].p.x==5,"Append removes only duplicate seam");
    bool seam_refused=false;try{append_pose_bridge(assembled,fake);}catch(const std::exception&){seam_refused=true;}check(seam_refused,"Nonexact seam refused");
    doc["targets"][1]["tokens"]=6.5;write_sequence();bool refused_sequence=false;
    try{(void)load_pose_sequence(sequence_file);}catch(const std::exception&){refused_sequence=true;}check(refused_sequence,"Fractional timing refused");
    doc["targets"][1]["tokens"]=8;doc["targets"][1]["frame"]=99;write_sequence();refused_sequence=false;
    try{(void)load_pose_sequence(sequence_file);}catch(const std::exception&){refused_sequence=true;}check(refused_sequence,"Late invalid source refused");fs::remove(sequence_file);

    for(size_t f=0;f<c.frames();++f)for(size_t j=0;j<30;++j)check(dot(c.frame(f)[j].p-loaded.frame(f)[j].p,c.frame(f)[j].p-loaded.frame(f)[j].p)<1e-12f,"GLB position roundtrip");
    std::array<std::vector<Transform>,8> guides;
    for(size_t f=0;f<8;++f)guides[f]={c.frame(f).begin(),c.frame(f).end()};
    auto in=encode_bricks(c,guides,8);check(in.allowed[2]==1&&in.allowed[0]==0,"Duration mask");
    {
        auto d=encode_bricks(c,guides,0);
        for(int i=0;i<11;++i)check(d.allowed[static_cast<size_t>(i)]==(i+6>=8&&i+6<=14),"Default token range 8..14");
        auto r=parse_bricks_tokens("7..11");check(r.min_tokens==7&&r.max_tokens==11,"Token range parsed");
        auto ranged=encode_bricks(c,guides,r);
        for(int i=0;i<11;++i)check(ranged.allowed[static_cast<size_t>(i)]==(i+6>=7&&i+6<=11),"Token range mask");
        check(parse_bricks_tokens("9").min_tokens==9&&parse_bricks_tokens("9").max_tokens==9,"Fixed token count parsed");
        for(const char* text:{"5","17","12..8","8..","..8","8.5","x"}) {
            bool r2=false;try{(void)parse_bricks_tokens(text);}catch(const std::exception&){r2=true;}check(r2,"Invalid token range refused");
        }
        // Upstream write_motionbricks_conditions_kernel for each target pose count.
        for(int n=1;n<=4;++n) {
            BricksRequest req;req.target_poses=n;auto m=encode_bricks(c,guides,req);
            for(int f=0;f<8;++f) {
                bool given=f<4+n;
                check(m.has_root[static_cast<size_t>(f)]==given&&m.has_pose[static_cast<size_t>(f)]==given,"Target pose mask");
                check(m.has_local_root[static_cast<size_t>(f)]==(given&&f!=3&&f!=3+n),"Root derivative mask");
            }
        }
        bool r3=false;try{BricksRequest req;req.target_poses=5;(void)encode_bricks(c,guides,req);}catch(const std::exception&){r3=true;}check(r3,"Five target poses refused");
    }
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
    {
        // Encoded model-space rotations decode back to the same SOMA-30 pose: a raw
        // prediction repeating the last context frame needs no seam correction.
        auto posed=c;
        for(size_t f=0;f<posed.frames();++f) {posed.frame(f)[22].q=normalized(Quat{0.3f,0.1f,-0.2f,0.9f});posed.frame(f)[11].q=normalized(Quat{-0.1f,0.4f,0.2f,0.85f});posed.frame(f)[1].q=yaw(0.4f);}
        std::array<std::vector<Transform>,8> g;for(size_t f=0;f<8;++f)g[f]={posed.frame(f).begin(),posed.frame(f).end()};
        auto pin=encode_bricks(posed,g,8);std::vector<float> raw(64*366);
        for(size_t f=0;f<64;++f) {
            std::copy_n(pin.root.data()+3*5,3,raw.data()+f*366);
            std::copy_n(pin.pose.data()+3*267,267,raw.data()+f*366+5);
        }
        SeamCorrection seam;(void)decode_bricks(posed,pin,raw,8,g[3],g[3],&seam);
        check(seam.joint_degrees[0]<0.01f&&seam.joint_degrees[1]<0.01f,"Model-space rotations roundtrip");
        auto d=seam.root[0];check(dot(d,d)<1e-10f,"Model-space root roundtrip");
        // The model's frames are BVH bone frames, not world-aligned: Spine1's bone lies along local +X.
        check(std::abs(pin.pose[87+1*6+1]-1)<0.01f,"Rest Spine1 model x axis points up");
    }
    {
        // Contacts are the last four features of each untrimmed frame, clamped.
        auto withContacts=features;
        for(size_t f=0;f<64;++f) {
            float left=f>=5&&f<=9?1.4f:0.2f,right=f>=12&&f<=13?0.9f:-0.3f;
            withContacts[f*366+362]=left;withContacts[f*366+363]=0.7f;withContacts[f*366+364]=right;withContacts[f*366+365]=0.1f;
        }
        auto contacts=decode_bricks_contacts(withContacts,8);
        check(contacts.size()==26,"One contact row per trimmed frame");
        check(contacts[2][0]==1&&contacts[0][0]==0.2f&&contacts[0][2]==0&&contacts[0][1]==0.7f,"Contacts trimmed, ordered and clamped");
        auto planted=planted_contacts(contacts,0.5f,100);
        // Left heel raw 5..9 -> trimmed 2..6; right heel 12..13 is only two frames.
        check(planted.size()==1&&planted[0].foot==0&&planted[0].start==102&&planted[0].end==106,"Planted heel intervals of three frames, offset");
        bool r4=false;try{auto bad=withContacts;bad[10*366+364]=NAN;(void)decode_bricks_contacts(bad,8);}catch(const std::exception&){r4=true;}check(r4,"Nonfinite contacts rejected");
    }
    check(bridge.frame(0)[0].p.x==guides[3][0].p.x&&bridge.frame(25)[0].p.x==guides[4][0].p.x,"Exact endpoint positions");
    check(join(c,bridge,c).frames()==c.frames()*2+bridge.frames()-2,"Joined seam frames");
    const std::vector<size_t> source_pins{0,3,7},target_pins{0,2,7};
    auto pins=connected_protected_frames(8,26,8,source_pins,target_pins,true);
    check(pins==std::vector<size_t>({0,3,7,32,34,39}),"Joined pins omit duplicate seam samples exactly once");
    auto exact_bridge=bridge;
    std::copy(c.frame(7).begin(),c.frame(7).end(),exact_bridge.frame(0).begin());
    std::copy(c.frame(0).begin(),c.frame(0).end(),exact_bridge.frame(25).begin());
    auto assembled_join=join(c,exact_bridge,c);
    for(size_t f:{size_t{32},size_t{34},size_t{39}})
        check(assembled_join.frame(f)[0].p.x==c.frame(f-32)[0].p.x,"Target pin offset agrees with exported join");
    check(connected_protected_frames(8,26,8,source_pins,target_pins,false)==std::vector<size_t>({0,25}),"Bridge-only export protects only retained endpoints");
    check(connected_protected_frames(1,2,1,{}, {},true)==std::vector<size_t>({0,1}),"Minimal seam index mapping");
    bool bad_pins=false;try{(void)connected_protected_frames(8,26,8,std::vector<size_t>{8},{},true);}catch(const std::exception&){bad_pins=true;}check(bad_pins,"Out-of-range inherited pin refused");
    bad_pins=false;try{(void)connected_protected_frames(18000,26,8,{}, {},true);}catch(const std::exception&){bad_pins=true;}check(bad_pins,"Oversized joined frame map refused");
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
    edit_contracts();
    std::cout<<"Motion contracts passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
