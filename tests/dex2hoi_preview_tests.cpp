#include "dex2hoi_preview.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace motion;
using nlohmann::json;
namespace {
void check(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
template<class F> void refuses(F operation,const char* reason) {
    bool failed=false;try{operation();}catch(const std::exception&){failed=true;}check(failed,reason);
}
void write(const fs::path& path,const json& data){std::ofstream out(path);out<<data.dump(2);if(!out)throw std::runtime_error("Cannot write preview fixture");}
json point(Vec3 p){return {p.x,p.y,p.z};}
json quaternion(Quat q){return {q.x,q.y,q.z,q.w};}
float error(Transform a,Transform b) {
    auto p=a.p-b.p;auto qa=normalized(a.q),qb=normalized(b.q);
    return std::sqrt(dot(p,p))+std::abs(1-std::abs(qa.x*qb.x+qa.y*qb.y+qa.z*qb.z+qa.w*qb.w));
}
void add_scaled_geometry(const fs::path& source,const fs::path& output) {
    std::ifstream input(source,std::ios::binary);std::vector<unsigned char> bytes(static_cast<size_t>(fs::file_size(source)));
    check(static_cast<bool>(input.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()))),"Cannot read GLB regression fixture");
    auto integer=[&](size_t at){return static_cast<uint32_t>(bytes.at(at))|(static_cast<uint32_t>(bytes.at(at+1))<<8)|(static_cast<uint32_t>(bytes.at(at+2))<<16)|(static_cast<uint32_t>(bytes.at(at+3))<<24);};
    size_t json_size=integer(12),chunk=20+json_size;
    auto doc=json::parse(std::string(reinterpret_cast<const char*>(bytes.data()+20),json_size));
    check(integer(chunk+4)==0x004e4942,"Fixture lacks BIN chunk");
    std::vector<unsigned char> bin(bytes.begin()+static_cast<std::ptrdiff_t>(chunk+8),bytes.end());
    const std::array<float,9> triangle{0,0,0,.1f,0,0,0,.1f,0};size_t offset=bin.size();bin.resize(offset+sizeof(triangle));
    std::memcpy(bin.data()+offset,triangle.data(),sizeof(triangle));
    size_t view=doc["bufferViews"].size();doc["bufferViews"].push_back({{"buffer",0},{"byteOffset",offset},{"byteLength",sizeof(triangle)}});
    size_t accessor=doc["accessors"].size();doc["accessors"].push_back({{"bufferView",view},{"componentType",5126},{"count",3},{"type","VEC3"},{"min",{0,0,0}},{"max",{.1,.1,0}}});
    if(!doc.contains("meshes"))doc["meshes"]=json::array();
    size_t mesh=doc["meshes"].size();doc["meshes"].push_back({{"primitives",json::array({{{"attributes",{{"POSITION",accessor}}}}})}});
    size_t child=doc["nodes"].size();
    for(auto& node:doc["nodes"])if(node.value("name","")=="own_sword") {
        if(!node.contains("children"))node["children"]=json::array();
        node["children"].push_back(child);
    }
    doc["nodes"].push_back({{"name","own_sword_static_nonuniform_geometry"},{"scale",{1,2,1}},{"mesh",mesh}});
    doc["buffers"][0]["byteLength"]=bin.size();
    auto text=doc.dump();while(text.size()%4)text+=' ';
    std::ofstream out(output,std::ios::binary);
    auto u32=[&](uint32_t value){char encoded[4];for(unsigned i=0;i<4;++i)encoded[i]=static_cast<char>((value>>(8*i))&255);out.write(encoded,4);};
    u32(0x46546c67);u32(2);u32(static_cast<uint32_t>(28+text.size()+bin.size()));u32(static_cast<uint32_t>(text.size()));u32(0x4e4f534a);out.write(text.data(),static_cast<std::streamsize>(text.size()));
    u32(static_cast<uint32_t>(bin.size()));u32(0x004e4942);out.write(reinterpret_cast<const char*>(bin.data()),static_cast<std::streamsize>(bin.size()));check(static_cast<bool>(out),"Cannot write GLB regression fixture");
}
Clip body_fixture(size_t frames) {
    Clip body;body.name="own_preview_fixture";
    for(size_t j=0;j<65;++j)body.joints.push_back({"explicit_"+std::to_string(j),j?static_cast<int>(j)-1:-1,{{j?.008f:0,j?.01f:.8f,0},yaw(static_cast<float>(j)*.001f)}});
    body.joints.push_back({"own_helper",10,{{.02f,0,0},{}}});
    for(size_t f=0;f<frames;++f)for(size_t j=0;j<body.joints.size();++j) {
        auto pose=body.joints[j].rest;
        if(!j){pose.p.x=.1f+static_cast<float>(f)*.002f;pose.q=yaw(static_cast<float>(f)*.005f);}
        else if(j<65&&f>=10)pose.q=pose.q*yaw(std::sin(static_cast<float>(f)*.05f)*.01f);
        body.poses.push_back(pose);
    }
    return body;
}
void contracts(const fs::path& root) {
    fs::create_directories(root);auto history=body_fixture(10);save_glb(history,root/"history.glb");
    json config={{"version",1},{"history","history.glb"},{"bones",json::array()},{"objects",json::array()}};
    for(size_t j=0;j<65;++j)config["bones"].push_back(history.joints[j].name);
    EquipmentResult generated;generated.body=body_fixture(70);generated.events=json::array();
    for(size_t s=0;s<2;++s) {
        EquipmentTrack object;object.name=s?"own_shield":"own_sword";
        json spec={{"name",object.name},{"present",true},{"exportTrack",true},{"points",json::array()},{"poses",json::array()}};
        for(size_t k=0;k<128;++k)spec["points"].push_back({static_cast<float>(k%2)*.01f,static_cast<float>(k%3)*.01f,static_cast<float>(k%5)*.01f});
        for(size_t f=0;f<70;++f) {
            Transform t{{static_cast<float>(s)*.4f+static_cast<float>(f)*.003f,1.2f,.3f},yaw(static_cast<float>(f)*.02f)};
            object.poses.push_back(t);object.scales.push_back({1,1,1});
            if(f<10)spec["poses"].push_back({{"position",point(t.p)},{"rotation",quaternion(t.q)}});
        }
        config["objects"].push_back(spec);generated.objects.push_back(object);
    }
    write(root/"input.json",config);save_equipment_glb(generated,root/"generated.glb");
    auto result=dex2hoi::load_preview_result(root/"generated.glb",root/"input.json");
    check(result.body.joints.size()==66&&result.body.frames()==70&&result.objects.size()==2,"body/object shape changed");
    check(result.report.at("defaultCameraBone")=="explicit_5"&&!result.report.at("recipeVerified").get<bool>(),"explicit head binding or sidecar status changed");
    for(size_t f=0;f<70;++f) {
        for(size_t j=0;j<66;++j)check(error(result.body.frame(f)[j],generated.body.frame(f)[j])<2e-6f,"body/root motion not preserved");
        for(size_t s=0;s<2;++s)check(error(result.objects[s].poses[f],generated.objects[s].poses[f])<2e-6f,"independent object world motion changed");
    }
    add_scaled_geometry(root/"generated.glb",root/"nonuniform-geometry.glb");
    refuses([&]{(void)load_glb(root/"nonuniform-geometry.glb","",false);},"all-nodes loader unexpectedly accepted nonuniform geometry");
    auto source_hash=sha256_file(root/"nonuniform-geometry.glb");
    auto filtered=dex2hoi::load_preview_result(root/"nonuniform-geometry.glb",root/"input.json");
    check(source_hash==sha256_file(root/"nonuniform-geometry.glb"),"preview rewrote source geometry");
    check(filtered.body.joints.size()==66&&filtered.objects.size()==2,"static geometry leaked into preview skeleton");
    for(size_t f=0;f<70;++f) {
        for(size_t j=0;j<66;++j)check(error(filtered.body.frame(f)[j],generated.body.frame(f)[j])<2e-6f,"nonuniform descendant changed body tracks");
        for(size_t s=0;s<2;++s)check(error(filtered.objects[s].poses[f],generated.objects[s].poses[f])<2e-6f,"nonuniform descendant changed world object tracks");
    }
    auto changed=config;changed["objects"][0]["poses"][3]["position"][0]=9;write(root/"wrong-object.json",changed);
    refuses([&]{dex2hoi::load_preview_result(root/"generated.glb",root/"wrong-object.json");},"mismatched object prefix accepted");
    changed=config;changed["bones"][4]=changed["bones"][3];write(root/"duplicate-binding.json",changed);
    refuses([&]{dex2hoi::load_preview_result(root/"generated.glb",root/"duplicate-binding.json");},"duplicate explicit binding accepted");
    changed=config;changed["objects"][0]["name"]="missing";write(root/"missing-object.json",changed);
    refuses([&]{dex2hoi::load_preview_result(root/"generated.glb",root/"missing-object.json");},"missing object accepted");
    changed=config;changed["objects"][0]["unexpected"]=1;write(root/"unknown.json",changed);
    refuses([&]{dex2hoi::load_preview_result(root/"generated.glb",root/"unknown.json");},"unknown manifest key accepted");
    auto bad=generated;bad.body.frame(4)[0].p.x+=.1f;save_equipment_glb(bad,root/"wrong-prefix.glb");
    refuses([&]{dex2hoi::load_preview_result(root/"wrong-prefix.glb",root/"input.json");},"mismatched body prefix accepted");
    bad=generated;bad.body.joints[6].rest.q=yaw(.2f);save_equipment_glb(bad,root/"wrong-rest.glb");
    refuses([&]{dex2hoi::load_preview_result(root/"wrong-rest.glb",root/"input.json");},"incompatible rest frames accepted");
    bad=generated;bad.body.joints[6].name="unexpected_bone";save_equipment_glb(bad,root/"missing-bone.glb");
    refuses([&]{dex2hoi::load_preview_result(root/"missing-bone.glb",root/"input.json");},"missing body bone accepted");
    bad=generated;bad.body.joints[65].parent=5;save_equipment_glb(bad,root/"wrong-hierarchy.glb");
    refuses([&]{dex2hoi::load_preview_result(root/"wrong-hierarchy.glb",root/"input.json");},"different body hierarchy accepted");
    bad=generated;bad.body.poses.resize(69*bad.body.joints.size());for(auto& object:bad.objects){object.poses.resize(69);object.scales.resize(69);}save_equipment_glb(bad,root/"wrong-frames.glb");
    refuses([&]{dex2hoi::load_preview_result(root/"wrong-frames.glb",root/"input.json");},"wrong output frame count accepted");
    bad=generated;bad.body.fps=20;save_equipment_glb(bad,root/"wrong-fps.glb");
    refuses([&]{dex2hoi::load_preview_result(root/"wrong-fps.glb",root/"input.json");},"wrong output fps accepted");
    json recipe={{"backend","dex2hoi"},{"inputSha256",sha256_file(root/"input.json")},{"historySha256",sha256_file(root/"history.glb")},{"meshSha256",{nullptr,nullptr}}};write(root/"generated.glb.json",recipe);
    check(dex2hoi::load_preview_result(root/"generated.glb",root/"input.json").report.at("recipeVerified").get<bool>(),"valid sidecar not verified");
    recipe["inputSha256"]=std::string(64,'0');write(root/"generated.glb.json",recipe);
    refuses([&]{dex2hoi::load_preview_result(root/"generated.glb",root/"input.json");},"mismatched recipe pin accepted");

    std::vector<std::string> args{"tide-motion","dex2hoi","preview",(root/"generated.glb").string(),"--input",(root/"input.json").string(),"--character",(root/"character.glb").string(),"--first-person","--camera-mode","root","--camera-bone","explicit_5","--camera-offset","0,0.1,0.2","--camera-rotation","1,2,3","--camera-fov","85","--camera-near","0.02","--frames","12","--capture",(root/"new.png").string()};
    auto request=dex2hoi::parse_preview_request(args);check(request.options.frames==12&&request.options.camera.enabled&&request.options.camera.mode=="root"&&request.camera_bone_explicit,"first-person parsing changed");
    auto invalid=args;invalid.push_back("--steps");invalid.push_back("5");refuses([&]{dex2hoi::parse_preview_request(invalid);},"inference flag accepted in read-only preview");
    invalid=args;invalid.push_back("--capture");invalid.push_back((root/"other.png").string());refuses([&]{dex2hoi::parse_preview_request(invalid);},"duplicate output flag accepted");
    invalid=args;invalid.erase(invalid.begin()+8);refuses([&]{dex2hoi::parse_preview_request(invalid);},"camera options accepted without first-person");
    invalid=args;invalid.back()=(root/"existing.png").string();write(root/"existing.png",json::object());refuses([&]{dex2hoi::parse_preview_request(invalid);},"existing capture accepted");
    invalid=args;invalid.back()="";refuses([&]{dex2hoi::parse_preview_request(invalid);},"empty capture value accepted");
    invalid=args;invalid.push_back("--gif");invalid.push_back((root/"new.png").string());refuses([&]{dex2hoi::parse_preview_request(invalid);},"colliding outputs accepted");
}
}
int main(int argc,char** argv) {
    try {
        auto root=(argc==2?utf8path(argv[1]):fs::temp_directory_path()/"dex2hoi-preview-tests")/std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        contracts(root);std::cout<<"Dex2HOI graph-free preview reconstruction/preflight contracts pass\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
