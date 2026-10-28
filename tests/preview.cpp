#include "motion.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

using namespace motion;
using nlohmann::json;
namespace {
void check(bool v,const char* what){if(!v)throw std::runtime_error(what);}
bool close(float a,float b){return std::abs(a-b)<1e-4f;}
template<class F> void refuses(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"Expected refusal");}
void u32(std::ostream& o,uint32_t v){for(int i=0;i<4;++i)o.put(static_cast<char>((v>>(8*i))&255));}
void unweighted_socket(const fs::path& p) {
    std::ifstream in(p,std::ios::binary);std::vector<char> bytes{std::istreambuf_iterator<char>(in),{}};in.close();
    uint32_t size=0;std::memcpy(&size,bytes.data()+12,4);
    auto d=json::parse(bytes.data()+20,bytes.data()+20+size);d["skins"]=json::array({{{"joints",{0,1}},{"skeleton",0}}});
    std::string j=d.dump();while(j.size()%4)j+=' ';
    auto data=bytes.data()+28+size;size_t count=bytes.size()-28-size;
    std::ofstream out(p,std::ios::binary);u32(out,0x46546c67);u32(out,2);u32(out,static_cast<uint32_t>(28+j.size()+count));
    u32(out,static_cast<uint32_t>(j.size()));u32(out,0x4e4f534a);out<<j;u32(out,static_cast<uint32_t>(count));u32(out,0x004e4942);out.write(data,static_cast<std::streamsize>(count));
}
void mesh(const fs::path& p) {
    // Non-default scene is deliberately huge; the selected scene has nested TRS.
    json d={{"asset",{{"version","2.0"}}},{"scene",0},{"scenes",json::array({{{"nodes",{0}}},{{"nodes",{2}}}})},
        {"nodes",json::array({{{"translation",{2,0,0}},{"children",{1}}},{{"mesh",0},{"scale",{2,2,2}}},{{"mesh",0},{"scale",{100,100,100}}}})},
        {"meshes",json::array({{{"primitives",json::array({{{"attributes",{{"POSITION",0}}}}})}}})},
        {"buffers",json::array({{{"byteLength",36}}})},{"bufferViews",json::array({{{"buffer",0},{"byteLength",36}}})},
        {"accessors",json::array({{{"bufferView",0},{"componentType",5126},{"count",3},{"type","VEC3"},{"min",{0,0,0}},{"max",{1,1,0}}}})}};
    std::string j=d.dump();while(j.size()%4)j+=' ';float v[]={0,0,0,1,0,0,0,1,0};
    std::ofstream out(p,std::ios::binary);u32(out,0x46546c67);u32(out,2);u32(out,static_cast<uint32_t>(28+j.size()+sizeof(v)));
    u32(out,static_cast<uint32_t>(j.size()));u32(out,0x4e4f534a);out<<j;u32(out,sizeof(v));u32(out,0x004e4942);out.write(reinterpret_cast<const char*>(v),sizeof(v));
}
}
int main(int argc,char** argv)try {
    check(argc==2,"Need isolated test directory");fs::path dir=argv[1];fs::create_directories(dir);mesh(dir/"mesh.glb");
    auto b=mesh_bounds(dir/"mesh.glb");check(close(b.low.x,2)&&close(b.high.x,4)&&close(b.high.y,2),"Scene selection and nested prop TRS bounds");
    Clip c;c.joints={{"Root",-1,{},0.01f},{"Hand",0,{{100,100,0},{}},1},{"Socket",1,{{0,50,0},{}},1}};
    for(int f=0;f<3;++f)for(auto j:c.joints)c.poses.push_back(j.rest);
    c.frame(1)[1].p.y=200;
    PreviewOptions o;o.props.push_back({dir/"mesh.glb","Socket",{0,0.5f,0},{90,30,15}});
    auto extent=preview_extent(c,o);
    check(extent.high>7.4f&&extent.reach>5.9f,"Prop framing includes unscaled mesh, offsets, helper socket and odd frame peak");
    auto bare=preview_extent(c,{});check(extent.high>bare.high+4,"Prop enlarges view");
    auto t=preview_offset({}, {90,90,90});auto v=rotate(t.q,{1,0,0});
    check(close(v.x,1)&&close(v.y,0)&&close(v.z,0),"Euler Z-X-Y order agrees with independent axis rotations");
    o.camera.enabled=true;o.camera.bone="Hand";o.camera.mode="root";
    fs::remove(dir/"setup.json");save_preview_setup(o,dir/"setup.json");auto loaded=load_preview_setup(dir/"setup.json");
    check(loaded.camera.mode=="root"&&loaded.camera.enabled&&loaded.props.size()==1&&loaded.props[0].mesh==fs::absolute(dir/"mesh.glb"),"Resolved equipment/camera recipe roundtrip");
    refuses([&]{save_preview_setup(o,dir/"setup.json");});
    {std::ofstream out(dir/"relative.json");out<<R"({"version":1,"props":[{"mesh":"mesh.glb","bone":"Hand"}],"camera":{"fov":90,"near":0.02}})";}
    auto relative=load_preview_setup(dir/"relative.json");check(relative.props[0].mesh==fs::absolute(dir/"mesh.glb")&&relative.camera.enabled,"Setup paths relative to file");
    {std::ofstream out(dir/"bad.json");out<<R"({"version":1,"camera":{"near":0}})";}
    refuses([&]{load_preview_setup(dir/"bad.json");});
    {std::ofstream out(dir/"bad.json");out<<R"({"version":1,"camera":{"ner":0.01}})";}
    refuses([&]{load_preview_setup(dir/"bad.json");});
    auto invalid=o;invalid.camera.fov=180;refuses([&]{validate_preview_setup(invalid);});
    invalid=o;invalid.body=PreviewBody::skeleton;refuses([&]{validate_preview_setup(invalid);});
    fs::remove(dir/"rig.glb");save_glb(c,dir/"rig.glb");o.body=PreviewBody::character;o.character=dir/"rig.glb";
    unweighted_socket(dir/"rig.glb");
    check(load_rig_glb(dir/"rig.glb").joints.size()==2,"Ordinary skin rig excludes unweighted socket");
    auto rig=preview_rig(c,o);auto worlds=world_pose(rig,rig.frame(1));
    check(rig.joints.size()==3,"Preview explicitly includes the unweighted socket");
    check(close(worlds[2].p.y,2.5f),"Socket inherits animated parent and centimetre scale");
    auto missing=o;missing.camera.bone="Missing";refuses([&]{preview_rig(c,missing);});
    save_mannequin(dir/"full.glb");save_mannequin(dir/"arms.glb",true);
    auto full=mesh_bounds(dir/"full.glb"),arms=mesh_bounds(dir/"arms.glb");
    check(arms.low.y>full.low.y+0.5f&&arms.high.y<full.high.y,"Arms-only mannequin omits head/torso/legs");
    check(load_rig_glb(dir/"arms.glb").joints.size()==30,"Arms mannequin preserves complete SOMA hierarchy");
    std::cout<<"Preview bounds, setup, camera validation, Euler and scaled socket contracts pass\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
