#include "equipment.hpp"
#include <cgltf.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>

using namespace motion;
using nlohmann::json;
namespace {
void check(bool v,const char* what){if(!v)throw std::runtime_error(what);}
template<class F> void refuses(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"Expected equipment export refusal");}
bool close(float a,float b){return std::abs(a-b)<1e-6f;}
void u32(std::ostream& o,uint32_t v){for(unsigned i=0;i<4;++i)o.put(static_cast<char>((v>>(8*i))&255));}
uint32_t word(const std::vector<char>& b,size_t at){uint32_t v=0;for(unsigned i=0;i<4;++i)v|=static_cast<uint32_t>(static_cast<unsigned char>(b[at+i]))<<(8*i);return v;}
std::vector<char> read(const fs::path& path){std::ifstream in(path,std::ios::binary);return {std::istreambuf_iterator<char>(in),{}};}
void write(const fs::path& path,json d,std::vector<char> bin) {
    d["buffers"]=json::array({{{"byteLength",bin.size()}}});std::string s=d.dump();while(s.size()%4)s+=' ';while(bin.size()%4)bin.push_back(0);
    std::ofstream out(path,std::ios::binary);u32(out,0x46546c67);u32(out,2);u32(out,static_cast<uint32_t>(28+s.size()+bin.size()));
    u32(out,static_cast<uint32_t>(s.size()));u32(out,0x4e4f534a);out<<s;u32(out,static_cast<uint32_t>(bin.size()));u32(out,0x004e4942);out.write(bin.data(),static_cast<std::streamsize>(bin.size()));
}
json document(const std::vector<char>& b){size_t n=word(b,12);return json::parse(b.data()+20,b.data()+20+n);}
std::pair<json,std::vector<char>> fixture() {
    const float v[]={0,0,0,1,0,0,0,1,0, 0,0,1,0,0,1};std::vector<char> bin(sizeof(v));std::memcpy(bin.data(),v,sizeof(v));
    const unsigned char png[]={137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,4,0,0,0,181,28,12,2,0,0,0,11,73,68,65,84,120,218,99,252,255,31,0,2,235,1,245,143,89,151,91,0,0,0,0,73,69,78,68,174,66,96,130};
    size_t image_offset=bin.size();bin.insert(bin.end(),reinterpret_cast<const char*>(png),reinterpret_cast<const char*>(png)+sizeof(png));
    json d={{"asset",{{"version","2.0"},{"copyright","Equipment fixture authored for Tide tests"}}},{"scene",0},{"scenes",json::array({{{"nodes",{0}}}})},
        {"nodes",json::array({{{"name","assembly"},{"children",{1}}},{{"name","surface"},{"mesh",0},{"translation",{1,2,3}},{"scale",{0.5,2,1}}}})},
        {"meshes",json::array({{{"primitives",json::array({{{"attributes",{{"POSITION",0},{"TEXCOORD_0",1}}},{"material",0}}})}}})},
        {"bufferViews",json::array({{{"buffer",0},{"byteOffset",0},{"byteLength",36}},{{"buffer",0},{"byteOffset",36},{"byteLength",24}},{{"buffer",0},{"byteOffset",image_offset},{"byteLength",sizeof(png)}}})},
        {"accessors",json::array({{{"bufferView",0},{"componentType",5126},{"count",3},{"type","VEC3"},{"min",{0,0,0}},{"max",{1,1,0}}},{{"bufferView",1},{"componentType",5126},{"count",3},{"type","VEC2"}}})},
        {"materials",json::parse(R"([{"name":"paint","doubleSided":true,"alphaMode":"MASK","alphaCutoff":0.3,"extensions":{"KHR_materials_unlit":{}},"pbrMetallicRoughness":{"baseColorFactor":[0.2,0.4,0.6,1],"metallicFactor":0.1,"roughnessFactor":0.8,"baseColorTexture":{"index":0,"extensions":{"KHR_texture_transform":{"offset":[0.1,0.2]}}}}}])")},
        {"textures",json::array({{{"source",0},{"sampler",0}}})},{"samplers",json::array({{{"magFilter",9728},{"minFilter",9728}}})},
        {"images",json::array({{{"bufferView",2},{"mimeType","image/png"}}})},{"extensionsUsed",{"KHR_materials_unlit","KHR_texture_transform"}},{"extensionsRequired",{"KHR_materials_unlit"}}};
    return {d,bin};
}
EquipmentResult result(const fs::path& mesh) {
    EquipmentResult r;r.body.name="test_equipment";r.body.joints={{"Root",-1,{},0.01f},{"Hand",0,{{100,100,0},{}},1}};
    for(size_t f=0;f<3;++f)for(const auto& j:r.body.joints)r.body.poses.push_back(j.rest);
    r.body.frame(1)[0].p={0.1f,0,0};
    EquipmentTrack arrow;arrow.name="Arrow";arrow.mesh=mesh;
    for(size_t f=0;f<3;++f){arrow.poses.push_back({{1+static_cast<float>(f),2,3},f==1?Quat{0,0,0,-1}:Quat{}});arrow.scales.push_back({1,1+static_cast<float>(f),1});}
    r.objects.push_back(arrow);arrow.name="Shield";r.objects.push_back(arrow);
    arrow.name="String";arrow.mesh.clear();arrow.builtin_rod=true;r.objects.push_back(arrow);
    arrow.name="Marker";arrow.builtin_rod=false;r.objects.push_back(arrow);
    r.events=json::array({{{"name","release"},{"frame",1},{"object","Arrow"}}});r.report={{"fixture",true}};return r;
}
struct Parsed {
    std::vector<char> bytes;std::unique_ptr<cgltf_data,decltype(&cgltf_free)> data{nullptr,cgltf_free};json d;
    explicit Parsed(const fs::path& path):bytes(read(path)),d(document(bytes)) {
        cgltf_options opts{};cgltf_data* raw=nullptr;check(cgltf_parse(&opts,bytes.data(),bytes.size(),&raw)==cgltf_result_success,"Final equipment GLB parses");data.reset(raw);
        check(cgltf_load_buffers(&opts,raw,nullptr)==cgltf_result_success&&cgltf_validate(raw)==cgltf_result_success,"Final equipment GLB accessors validate");
    }
    cgltf_node* node(const std::string& name) {for(size_t i=0;i<data->nodes_count;++i)if(data->nodes[i].name&&name==data->nodes[i].name)return &data->nodes[i];throw std::runtime_error("Missing exported node "+name);}
    cgltf_accessor* channel(const std::string& name,cgltf_animation_path_type path) {
        auto* n=node(name);for(size_t i=0;i<data->animations[0].channels_count;++i){auto& c=data->animations[0].channels[i];if(c.target_node==n&&c.target_path==path)return c.sampler->output;}throw std::runtime_error("Missing exported channel");
    }
};
}
int main(int argc,char** argv)try {
    check(argc==2,"Need isolated equipment export test directory");fs::path dir=argv[1];fs::create_directories(dir);
    auto [src,bin]=fixture();write(dir/"mesh.glb",src,bin);auto before=read(dir/"mesh.glb");auto r=result(dir/"mesh.glb");
    auto out=dir/"equipped.glb";fs::remove(out);fs::remove(out.string()+".json");fs::remove(out.string()+".part");save_equipment_glb(r,out);
    check(read(dir/"mesh.glb")==before,"Exporter preserves source asset bytes");Parsed final(out);
    check(final.data->buffers_count==1&&!final.data->buffers[0].uri&&final.data->images_count==2&&final.data->textures_count==2,"Repeated meshes/images are self-contained and remapped");
    check(final.node("Arrow")->parent==nullptr&&final.node("Shield")->parent==nullptr&&final.node("String")->parent==nullptr,"Equipment tracks remain independent world roots");
    check(final.node("Hand")->parent==final.node("Root")&&close(final.node("Root")->scale[0],0.01f),"Body hierarchy and centimetre scale preserved");
    check(close(final.node("Arrow")->translation[0],1)&&close(final.node("Arrow/mesh/1/surface")->translation[1],2)&&close(final.node("Arrow/mesh/1/surface")->scale[1],2),"Initial object TRS and nested mesh transforms preserved");
    for(size_t f=0;f<3;++f) {
        float p[3],s[3],q[4];check(cgltf_accessor_read_float(final.channel("Arrow",cgltf_animation_path_type_translation),f,p,3),"Read position track");
        check(cgltf_accessor_read_float(final.channel("Arrow",cgltf_animation_path_type_scale),f,s,3),"Read scale track");
        check(cgltf_accessor_read_float(final.channel("Arrow",cgltf_animation_path_type_rotation),f,q,4),"Read orientation track");
        check(close(p[0],1+static_cast<float>(f))&&close(s[1],1+static_cast<float>(f))&&close(q[3],1),"World trajectories, mesh scales and quaternion continuity preserve samples");
    }
    const auto& extras=final.d["asset"]["extras"]["tideMotion"];
    check(extras["events"]==r.events&&extras["report"]==r.report&&!extras["bodyMeshEmbedded"].get<bool>(),"Events/report and body-mesh scope embedded explicitly");
    check(final.d["nodes"][2]["extras"]["sourceAsset"]["copyright"]==src["asset"]["copyright"],"Upstream asset attribution preserved");
    check(final.d["materials"][0]["doubleSided"]==true&&final.d["materials"][0]["alphaMode"]=="MASK"&&final.d["materials"][1]["pbrMetallicRoughness"]["baseColorTexture"]["index"]==1,"Material properties and duplicate texture offsets retained");
    check(final.d["extensionsRequired"]==src["extensionsRequired"],"Required material extension contract retained");
    for(size_t i=0;i<final.data->images_count;++i) {
        auto* view=final.data->images[i].buffer_view;check(view&&view->size==bin.size()-60&&std::memcmp(cgltf_buffer_view_data(view),bin.data()+60,view->size)==0,"Embedded image bytes preserved");
    }
    auto* string=final.node("String");auto& primitive=string->mesh->primitives[0];const cgltf_accessor *p=nullptr,*n=nullptr;
    for(size_t i=0;i<primitive.attributes_count;++i){if(primitive.attributes[i].type==cgltf_attribute_type_position)p=primitive.attributes[i].data;if(primitive.attributes[i].type==cgltf_attribute_type_normal)n=primitive.attributes[i].data;}
    check(p&&n&&p->count==36,"String has embedded unit rod geometry");
    for(size_t t=0;t<p->count;t+=3) {
        float a[3],c[3],d[3],normal[3];cgltf_accessor_read_float(p,t,a,3);cgltf_accessor_read_float(p,t+1,c,3);cgltf_accessor_read_float(p,t+2,d,3);cgltf_accessor_read_float(n,t,normal,3);
        check(dot(cross(Vec3{c[0],c[1],c[2]}-Vec3{a[0],a[1],a[2]},Vec3{d[0],d[1],d[2]}-Vec3{a[0],a[1],a[2]}),{normal[0],normal[1],normal[2]})>0,"Rod winding agrees with normals");
    }
    auto protected_bytes=read(out);refuses([&]{save_equipment_glb(r,out);});check(read(out)==protected_bytes,"Existing output remains unchanged");
    auto failed=dir/"failed.glb";fs::remove(failed);fs::remove(failed.string()+".part");fs::remove(failed.string()+".json");
    {std::ofstream side(failed.string()+".json");side<<"protected";}refuses([&]{save_equipment_glb(r,failed);});check(!fs::exists(failed),"Existing sidecar protects absent output");fs::remove(failed.string()+".json");
    {std::ofstream partial(failed.string()+".part");partial<<"protected";}refuses([&]{save_equipment_glb(r,failed);});check(read(failed.string()+".part")==std::vector<char>({'p','r','o','t','e','c','t','e','d'}),"Existing partial file remains unchanged");fs::remove(failed.string()+".part");
    auto invalid=r;invalid.objects[0].name="Root";refuses([&]{save_equipment_glb(invalid,failed);});
    invalid=r;invalid.objects[0].poses.pop_back();refuses([&]{save_equipment_glb(invalid,failed);});
    invalid=r;invalid.objects[0].poses[0].p.x=std::numeric_limits<float>::infinity();refuses([&]{save_equipment_glb(invalid,failed);});
    auto bad=src;bad["images"][0]["uri"]="external.png";write(dir/"invalid.glb",bad,bin);invalid=r;invalid.objects[0].mesh=dir/"invalid.glb";refuses([&]{save_equipment_glb(invalid,failed);});
    bad=src;bad["extensionsUsed"].push_back("KHR_draco_mesh_compression");write(dir/"invalid.glb",bad,bin);refuses([&]{save_equipment_glb(invalid,failed);});
    bad=src;bad["skins"]=json::array({{{"joints",{0}}}});write(dir/"invalid.glb",bad,bin);refuses([&]{save_equipment_glb(invalid,failed);});
    check(!fs::exists(failed)&&!fs::exists(failed.string()+".part"),"Failed export leaves no partial/output asset");
    {std::ofstream fbx(dir/"mesh.fbx");fbx<<R"(; FBX 7.4.0 project file
FBXHeaderExtension: { FBXVersion: 7400 }
GlobalSettings: {
 Properties70: {
  P: "UpAxis", "int", "Integer", "",1
  P: "UpAxisSign", "int", "Integer", "",1
  P: "FrontAxis", "int", "Integer", "",2
  P: "FrontAxisSign", "int", "Integer", "",1
  P: "CoordAxis", "int", "Integer", "",0
  P: "CoordAxisSign", "int", "Integer", "",1
  P: "UnitScaleFactor", "double", "Number", "",100
 }
}
Objects: {
 Geometry: 1, "Geometry::Triangle", "Mesh" {
  Vertices: *9 { a: 0,0,0,0.1,0,0,0,0.1,0 }
  PolygonVertexIndex: *3 { a: 0,1,-3 }
 }
 Model: 2, "Model::Triangle", "Mesh" { Version: 232 }
}
Connections: {
 C: "OO",1,2
 C: "OO",2,0
}
)";}
    auto fb=r;fb.objects.resize(1);fb.objects[0].mesh=dir/"mesh.fbx";save_equipment_glb(fb,failed);Parsed fbx(failed);
    check(fbx.d["asset"]["extras"]["tideMotion"]["meshConversions"][0]["format"]=="FBX","FBX conversion notes are visible in asset metadata");
    auto bounds=mesh_bounds(failed);check(bounds.high.x>=1.09f&&bounds.high.y>=2.09f,"FBX mesh positions use metres beneath object root TRS");
    std::cout<<"Equipment GLB body/object trajectories, nested materials/textures, rods, FBX and refusal contracts pass\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
