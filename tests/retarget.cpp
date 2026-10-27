#include "motion.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <functional>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace motion;
namespace {
void check(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
bool close(Vec3 a,Vec3 b,float tolerance=2e-4f) { auto d=a-b;return dot(d,d)<tolerance*tolerance; }
bool close(Quat a,Quat b) { return std::abs(a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w)>0.99999f; }
Quat zrotation(float angle) { return {0,0,std::sin(angle/2),std::cos(angle/2)}; }
struct Fixture { Clip source,target;RetargetMapping mapping;std::array<size_t,30> index{}; };
Fixture fixture() {
    Fixture result;auto& source=result.source;source.joints=soma();
    std::vector<Transform> rest;for(const auto& j:source.joints)rest.push_back(j.rest);
    auto source_world=world_pose(source,rest);
    float floor=std::min(source_world[25].p.y,source_world[29].p.y);
    for(int f=0;f<8;++f)for(size_t j=0;j<30;++j) {
        auto t=rest[j];float progress=static_cast<float>(f)/7;
        if(j==0) {t.p={progress,-floor,0};t.q=yaw(progress*0.35f);}
        if(j==11)t.q=zrotation(-progress*0.2f);
        if(j==12)t.q=zrotation(progress*1.1f);
        if(j==18)t.q=zrotation(-progress*0.7f);
        source.poses.push_back(t);
    }
    // Independently author a shorter-armed, longer-legged, A-pose rig. Bone axes
    // vary independently of bone directions. A centimetre armature parent and
    // an unmapped twist bone exercise hierarchy and units rather than renaming.
    std::array<Vec3,30> positions{};std::array<Quat,30> orientations{};
    Transform placement{{3,0,-2},yaw(0.65f)};
    for(size_t s=0;s<30;++s) {
        int parent=source.joints[s].parent;
        Vec3 offset=source.joints[s].rest.p;
        float factor=s>=22?1.3f:(s>=10&&s<=21?0.8f:1.1f);
        positions[s]=(parent<0?Vec3{0,-floor*1.3f,0}:positions[static_cast<size_t>(parent)])+offset*factor;
        orientations[s]=normalized(yaw(0.13f*static_cast<float>(s))*zrotation(0.09f*static_cast<float>(s)));
    }
    for(size_t s=12;s<=15;++s)positions[s]=positions[11]+rotate(zrotation(-0.55f),positions[s]-positions[11]);
    for(size_t s=18;s<=21;++s)positions[s]=positions[17]+rotate(zrotation(0.55f),positions[s]-positions[17]);
    for(size_t s=0;s<30;++s) {positions[s]=placement.p+rotate(placement.q,positions[s]);orientations[s]=normalized(placement.q*orientations[s]);}
    auto& target=result.target;
    target.joints.push_back({"Armature",-1,placement,0.01f});
    std::vector<Transform> globals{placement};
    for(size_t s=0;s<30;++s) {
        size_t parent=source.joints[s].parent<0?0:result.index[static_cast<size_t>(source.joints[s].parent)];
        if(s==12) {
            Transform twist{positions[11]*0.5f+positions[12]*0.5f,normalized(orientations[11]*yaw(0.23f))};
            auto local=inverse(globals[parent])*twist;local.p=local.p*100;
            target.joints.push_back({"arm_twist.L",static_cast<int>(parent),local});globals.push_back(twist);parent=target.joints.size()-1;
        }
        auto local=inverse(globals[parent])*Transform{positions[s],orientations[s]};local.p=local.p*100;
        std::string name="Character:"+source.joints[s].name;
        result.mapping.bones[s]=name;result.index[s]=target.joints.size();
        target.joints.push_back({name,static_cast<int>(parent),local});globals.push_back({positions[s],orientations[s]});
    }
    for(const auto& joint:target.joints)target.poses.push_back(joint.rest);
    target.motion_root=result.index[0];
    return result;
}
nlohmann::json mapping_json(const Fixture& f) {
    nlohmann::json doc={{"bones",nlohmann::json::object()},{"rootScale","auto"}};
    for(size_t s=0;s<30;++s)doc["bones"][f.source.joints[s].name]=f.mapping.bones[s];
    return doc;
}
template<class F> void refuses(F operation,const char* why) {
    bool rejected=false;try{operation();}catch(const std::exception&){rejected=true;}check(rejected,why);
}
void rewrite(const fs::path& input,const fs::path& output,const std::function<void(nlohmann::json&,std::vector<char>&)>& edit) {
    std::ifstream stream(input,std::ios::binary);
    std::vector<char> data{std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
    auto word=[&](size_t offset){uint32_t v=0;for(size_t k=0;k<4;++k)v|=static_cast<uint32_t>(static_cast<unsigned char>(data[offset+k]))<<(8*k);return v;};
    size_t json_size=word(12);auto doc=nlohmann::json::parse(data.data()+20,data.data()+20+json_size);
    size_t start=28+json_size,count=doc["buffers"][0]["byteLength"].get<size_t>();
    std::vector<char> bin(data.begin()+static_cast<std::ptrdiff_t>(start),data.begin()+static_cast<std::ptrdiff_t>(start+count));
    edit(doc,bin);doc["buffers"][0]["byteLength"]=bin.size();
    auto text=doc.dump();while(text.size()%4)text+=' ';while(bin.size()%4)bin.push_back(0);
    std::ofstream out(output,std::ios::binary);
    auto write=[&](uint32_t v){for(int k=0;k<4;++k)out.put(static_cast<char>((v>>(8*k))&255));};
    write(0x46546c67);write(2);write(static_cast<uint32_t>(28+text.size()+bin.size()));write(static_cast<uint32_t>(text.size()));write(0x4e4f534a);out<<text;
    write(static_cast<uint32_t>(bin.size()));write(0x004e4942);out.write(bin.data(),static_cast<std::streamsize>(bin.size()));
}
}
// Writes a rig as an ASCII FBX 7.4 the way a Z-up centimetre package would: the
// whole rig turned from Y-up to Z-up and the top node unscaled, so reading it back
// with Tide's conventions (Y-up metres) must undo both. No third-party content.
void write_fbx(const Clip& rig,const fs::path& path) {
    const Quat up{std::sin(0.785398163f),0,0,std::cos(0.785398163f)}; // +90 degrees about X: Y-up to Z-up
    std::ofstream out(path);out.precision(9);
    out<<"; FBX 7.4.0 project file\nFBXHeaderExtension:  {\n\tFBXHeaderVersion: 1003\n\tFBXVersion: 7400\n}\n"
         "GlobalSettings:  {\n\tVersion: 1000\n\tProperties70:  {\n"
         "\t\tP: \"UpAxis\", \"int\", \"Integer\", \"\",2\n\t\tP: \"UpAxisSign\", \"int\", \"Integer\", \"\",1\n"
         "\t\tP: \"FrontAxis\", \"int\", \"Integer\", \"\",1\n\t\tP: \"FrontAxisSign\", \"int\", \"Integer\", \"\",-1\n"
         "\t\tP: \"CoordAxis\", \"int\", \"Integer\", \"\",0\n\t\tP: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1\n"
         "\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1\n\t}\n}\nObjects:  {\n";
    for(size_t j=0;j<rig.joints.size();++j) {
        auto t=rig.joints[j].rest;
        if(rig.joints[j].parent<0) {t.p=rotate(up,t.p)*100;t.q=normalized(up*t.q);} // metres to centimetres
        auto q=t.q;
        float m00=1-2*(q.y*q.y+q.z*q.z),m10=2*(q.x*q.y+q.w*q.z),m20=2*(q.x*q.z-q.w*q.y),m21=2*(q.y*q.z+q.w*q.x),m22=1-2*(q.x*q.x+q.y*q.y);
        const double degrees=57.29577951308232;
        double ex=std::atan2(m21,m22)*degrees,ey=std::asin(std::clamp(-m20,-1.0f,1.0f))*degrees,ez=std::atan2(m10,m00)*degrees;
        out<<"\tModel: "<<1000+j<<", \"Model::"<<rig.joints[j].name<<"\", \"LimbNode\" {\n\t\tVersion: 232\n\t\tProperties70:  {\n"
           <<"\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\","<<t.p.x<<","<<t.p.y<<","<<t.p.z<<"\n"
           <<"\t\t\tP: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\","<<ex<<","<<ey<<","<<ez<<"\n\t\t}\n\t}\n";
    }
    out<<"}\nConnections:  {\n";
    for(size_t j=0;j<rig.joints.size();++j)out<<"\tC: \"OO\","<<1000+j<<","<<(rig.joints[j].parent<0?0:1000+rig.joints[j].parent)<<"\n";
    out<<"}\n";
}
int main(int argc,char** argv) try {
    auto f=fixture();
    auto dir=fs::temp_directory_path()/"tide-motion-retarget-tests";
    if(argc==3&&std::string(argv[1])=="--fixtures")dir=argv[2];
    fs::create_directories(dir);
    for(const char* name:{"source.glb","target.glb","retargeted.glb"}) {fs::remove(dir/name);fs::remove((dir/name).string()+".part");}
    save_glb(f.source,dir/"source.glb");save_glb(f.target,dir/"target.glb");
    {std::ofstream mapping(dir/"bones.json");mapping<<mapping_json(f).dump(2);}
    auto map=load_mapping(dir/"bones.json");
    auto target=load_rig_glb(dir/"target.glb");
    auto result=retarget(f.source,target,map);
    check(std::abs(result.scale-1.3f)<1e-4f,"Root scale follows leg length rather than arm or torso length");
    check(std::abs(result.heading_degrees*3.14159265358979323846f/180-0.65f)<1e-4f,"Character heading comes from geometry, independently of bone axes");
    check(result.clip.frames()==f.source.frames(),"Retarget frame count");
    auto& out=result.clip;
    for(size_t frame=0;frame<out.frames();++frame) {
        auto actual=world_pose(out,out.frame(frame)),original=world_pose(f.source,f.source.frame(frame));
        Quat basis=yaw(result.heading_degrees*3.14159265358979323846f/180);
        for(auto segment:{std::pair{11u,12u},std::pair{12u,13u},std::pair{17u,18u},std::pair{18u,19u},std::pair{22u,23u},std::pair{23u,24u}}) {
            auto a=normalized(actual[f.index[segment.second]].p-actual[f.index[segment.first]].p);
            auto b=rotate(basis,normalized(original[segment.second].p-original[segment.first].p));
            check(dot(a,b)>0.9999f,"Retargeted limb direction matches motion through independent axes and A-pose");
        }
        for(size_t j=0;j<out.joints.size();++j)if(j!=out.motion_root)check(close(out.frame(frame)[j].p,target.joints[j].rest.p),"Retargeting preserves target bone lengths and helper translations");
    }
    auto first=world_pose(out,out.frame(0)),last=world_pose(out,out.frame(out.frames()-1));
    check(std::abs(std::min(first[f.index[25]].p.y,first[f.index[29]].p.y))<2e-4f,"Reference standing toes sit on the inferred ground plane");
    check(close(last[out.motion_root].p-first[out.motion_root].p,rotate(yaw(result.heading_degrees*3.14159265358979323846f/180),{1.3f,0,0})),"Root motion transforms into character heading and metres");
    Clip canonical;canonical.joints=soma();for(const auto& j:canonical.joints)canonical.poses.push_back(j.rest);
    auto roundtrip=retarget(out,canonical,map,true).clip;
    for(size_t frame=0;frame<f.source.frames();++frame)for(size_t j=0;j<30;++j) {
        check(close(roundtrip.frame(frame)[j].p,f.source.frame(frame)[j].p),"Bidirectional root/position roundtrip");
        check(close(roundtrip.frame(frame)[j].q,f.source.frame(frame)[j].q),"Bidirectional rotation roundtrip");
    }
    auto corrected=map;corrected.ground_offset=0.07f;corrected.heading_degrees=20;
    auto shifted=retarget(f.source,target,corrected).clip;
    auto restored=retarget(shifted,canonical,corrected,true).clip;
    check(close(restored.frame(7)[0].p,f.source.frame(7)[0].p),"Root heading and ground corrections invert on roundtrip");
    save_glb(out,dir/"retargeted.glb");auto loaded=load_glb(dir/"retargeted.glb","",false);
    check(loaded.motion_root==out.motion_root,"GLB retains motion root metadata");
    auto loaded_world=world_pose(loaded,loaded.frame(7));
    for(size_t j=0;j<out.joints.size();++j)check(close(loaded_world[j].p,last[j].p),"Custom rig GLB roundtrip preserves uniform scales and poses");
    auto manual=map;manual.reference_rotations[map.bones[6]]=yaw(0.3f);
    auto overridden=retarget(f.source,target,manual).clip;
    check(close(overridden.frame(0)[f.index[6]].q,yaw(0.3f)),"Explicit local reference quaternion overrides calibration");
    auto partial=map;
    for(size_t s:{size_t{1},size_t{2},size_t{4},size_t{5},size_t{7},size_t{8},size_t{9},size_t{10},size_t{14},size_t{15},size_t{16},size_t{20},size_t{21},size_t{25},size_t{29}})partial.bones[s].clear();
    check(retarget(f.source,target,partial).clip.frames()==8,"Missing optional bones and different spine counts are supported");
    refuses([&]{auto bad=map;bad.bones[11].clear();(void)retarget(f.source,target,bad);},"Missing required bone refused");
    refuses([&]{auto bad=map;bad.bones[12]=map.bones[18];bad.bones[18]=map.bones[12];(void)retarget(f.source,target,bad);},"Crossed anatomical hierarchy refused");
    refuses([&]{auto bad=map;bad.bones[13]=map.bones[12];(void)retarget(f.source,target,bad);},"Duplicate mapping refused");
    auto translated=out;translated.frame(1)[f.index[12]].p.x+=1;
    auto fixed=retarget(translated,canonical,map,true);
    check(fixed.discarded_translations&&close(fixed.clip.frame(1)[12].p,canonical.joints[12].rest.p),"Non-root translations are reported and target lengths retained");
    auto changed=dir/"changed.glb";
    rewrite(dir/"target.glb",changed,[&](auto& doc,auto&) {
        auto t=target.joints[0].rest;auto x=rotate(t.q,{0.01f,0,0}),y=rotate(t.q,{0,0.01f,0}),z=rotate(t.q,{0,0,0.01f});
        doc["nodes"][0]={{"name","Armature"},{"children",doc["nodes"][0]["children"]},{"matrix",{x.x,x.y,x.z,0,y.x,y.y,y.z,0,z.x,z.y,z.z,0,t.p.x,t.p.y,t.p.z,1}}};
        doc.erase("animations");
    });
    auto matrix=load_rig_glb(changed);
    check(close(world_pose(matrix,matrix.frame(0))[f.index[13]].p,world_pose(target,target.frame(0))[f.index[13]].p),"Static uniform matrix decomposes without changing rig placement");
    rewrite(dir/"target.glb",changed,[](auto& doc,auto&) {doc["nodes"][0]["scale"]={0.01f,0.02f,0.01f};});
    refuses([&]{(void)load_rig_glb(changed);},"Nonuniform scale rejected");
    auto scale_track=[&](float scale) {
        rewrite(dir/"target.glb",changed,[&](auto& doc,auto& bin) {
            size_t view=doc["bufferViews"].size(),accessor=doc["accessors"].size();
            doc["bufferViews"].push_back({{"buffer",0},{"byteOffset",bin.size()},{"byteLength",12}});
            for(int axis=0;axis<3;++axis) {uint32_t bits;std::memcpy(&bits,&scale,4);for(int byte=0;byte<4;++byte)bin.push_back(static_cast<char>((bits>>(8*byte))&255));}
            doc["accessors"].push_back({{"bufferView",view},{"componentType",5126},{"count",1},{"type","VEC3"}});
            auto& animation=doc["animations"][0];size_t sampler=animation["samplers"].size();
            animation["samplers"].push_back({{"input",animation["samplers"][0]["input"]},{"output",accessor},{"interpolation","LINEAR"}});
            animation["channels"].push_back({{"sampler",sampler},{"target",{{"node",0},{"path","scale"}}}});
        });
    };
    scale_track(0.01f);check(load_glb(changed,"",false).frames()==1,"Constant scale tracks are accepted");
    scale_track(0.02f);refuses([&]{(void)load_glb(changed,"",false);},"Changing scale rejected");
    fs::remove(changed);
    // The same centimetre rig as a Z-up FBX reads back in Y-up metres, and retargets alike.
    auto fbx=dir/"target.fbx";write_fbx(target,fbx);
    check(is_fbx(fbx)&&is_fbx(dir/"TARGET.FBX"),"FBX extension detection");
    auto from_fbx=load_rig_glb(fbx);
    check(from_fbx.joints.size()==target.joints.size()&&std::abs(from_fbx.joints[0].scale-0.01f)<1e-6f,"FBX rig keeps every node, its unit scale folded into the top node");
    auto fbx_world=world_pose(from_fbx,from_fbx.frame(0)),glb_world=world_pose(target,target.frame(0));
    auto named=[](const Clip& c,const std::string& name){for(size_t j=0;j<c.joints.size();++j)if(c.joints[j].name==name)return j;throw std::runtime_error("FBX lost bone "+name);};
    for(size_t j=0;j<target.joints.size();++j)check(close(fbx_world[named(from_fbx,target.joints[j].name)].p,glb_world[j].p,1e-4f),"FBX rig converts Z-up centimetres to Y-up metres");
    auto via_fbx=retarget(f.source,from_fbx,map).clip;
    for(size_t frame=0;frame<out.frames();++frame) {
        auto a=world_pose(via_fbx,via_fbx.frame(frame)),b=world_pose(out,out.frame(frame));
        for(size_t j=0;j<out.joints.size();++j)check(close(a[named(via_fbx,out.joints[j].name)].p,b[j].p,1e-4f),"Retargeting onto the FBX rig matches the GLB rig");
    }
    refuses([&]{(void)load_glb(fbx,"",false);},"FBX without a take has no animation");
    refuses([&]{(void)load_glb(fbx,"",true);},"SOMA clips must be GLB");
    if(argc==3)return 0;
    fs::remove(fbx);
    for(const char* name:{"source.glb","target.glb","retargeted.glb","bones.json"})fs::remove(dir/name);
    fs::remove(dir);
    std::cout<<"Retarget contracts passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
