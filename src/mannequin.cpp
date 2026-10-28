#include "motion.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numbers>
#include <stdexcept>

// A preview body made here, with no third-party art: a capsule along every SOMA-30
// bone, rigidly skinned to the bone it hangs from, a sphere for the head. Left
// limbs are blue and right limbs orange, as in the stick preview, so a mirrored
// or crossed limb is obvious.
namespace motion {
namespace {
struct Corner { Vec3 p, n; std::array<float,4> color; uint16_t joint; };
struct Body { std::vector<Corner> corners; std::vector<uint32_t> faces; };
std::array<float,4> tint(const std::string& name) {
    if(name.starts_with("Left"))return {0.42f,0.62f,0.92f,1};
    if(name.starts_with("Right"))return {0.95f,0.62f,0.32f,1};
    return {0.84f,0.82f,0.78f,1};
}
// Any two unit vectors perpendicular to axis.
void basis(Vec3 axis,Vec3* u,Vec3* v) {
    Vec3 helper=std::abs(axis.y)<0.9f?Vec3{0,1,0}:Vec3{1,0,0};
    *u=normalized(cross(axis,helper));*v=cross(axis,*u);
}
// A capsule from a to b: rings of a cylinder capped by two hemispheres, outward
// triangles counter-clockwise as glTF expects.
void capsule(Body& body,Vec3 a,Vec3 b,float radius,uint16_t joint,std::array<float,4> color) {
    constexpr int around=14,cap=5;
    Vec3 d=b-a;float length=std::sqrt(dot(d,d));
    Vec3 axis=length>1e-6f?d*(1/length):Vec3{0,1,0},u,v;basis(axis,&u,&v);
    const uint32_t first=static_cast<uint32_t>(body.corners.size());
    int rings=0;
    // Latitude from the a pole (-90 degrees) to the b pole (+90): the hemisphere at a,
    // then the one at b; consecutive equal latitudes at the seam make the cylinder.
    for(int half=0;half<2;++half)for(int r=0;r<=cap;++r) {
        float lat=(half==0?-1.0f+static_cast<float>(r)/cap:static_cast<float>(r)/cap)*std::numbers::pi_v<float>/2;
        Vec3 centre=half==0?a:b;
        for(int k=0;k<=around;++k) {
            float lon=2*std::numbers::pi_v<float>*static_cast<float>(k)/around;
            Vec3 n=(u*std::cos(lon)+v*std::sin(lon))*std::cos(lat)+axis*std::sin(lat);
            body.corners.push_back({centre+n*radius,n,color,joint});
        }
        ++rings;
    }
    for(int r=0;r+1<rings;++r)for(int k=0;k<around;++k) {
        uint32_t i0=first+static_cast<uint32_t>(r*(around+1)+k),i1=i0+1,j0=i0+around+1,j1=j0+1;
        body.faces.insert(body.faces.end(),{i0,i1,j0,i1,j1,j0});
    }
}
void append(std::vector<char>& bin,const void* data,size_t size) {
    const char* bytes=static_cast<const char*>(data);bin.insert(bin.end(),bytes,bytes+size);
    while(bin.size()%4)bin.push_back(0);
}
}
void save_mannequin(const fs::path& path,bool arms_only) {
    auto joints=soma();
    Clip rig;rig.joints=joints;for(const auto& j:joints)rig.poses.push_back(j.rest);
    auto world=world_pose(rig,rig.frame(0));
    auto find=[&](const char* name) {for(size_t j=0;j<joints.size();++j)if(joints[j].name==name)return j;throw std::runtime_error("Missing SOMA joint");};
    auto radius=[](const std::string& name)->float {
        if(name=="Spine1")return 0.11f;
        if(name=="Spine2"||name=="Chest")return 0.12f;
        if(name=="Neck1")return 0.12f;
        if(name=="Neck2")return 0.045f;
        if(name.ends_with("Shoulder"))return 0.05f;
        if(name.ends_with("ForeArm")||name.ends_with("Hand"))return 0.045f;
        if(name.ends_with("Arm"))return 0.05f;
        if(name.ends_with("MiddleEnd"))return 0.035f;
        if(name.ends_with("ThumbEnd"))return 0.018f;
        if(name.ends_with("Leg"))return 0.085f;
        if(name.ends_with("Shin"))return 0.07f;
        if(name.ends_with("Foot"))return 0.055f;
        if(name.ends_with("ToeBase"))return 0.035f;
        return 0;
    };
    Body body;
    for(size_t j=1;j<joints.size();++j) {
        const auto& name=joints[j].name;float r=radius(name);
        if(r<=0)continue;
        if(arms_only&&!(name.ends_with("Arm")||name.ends_with("Hand")||name.ends_with("ThumbEnd")||name.ends_with("MiddleEnd")))continue;
        auto parent=static_cast<size_t>(joints[j].parent);
        capsule(body,world[parent].p,world[j].p,r,static_cast<uint16_t>(parent),tint(joints[parent].name=="Hips"||joints[parent].name=="Chest"?name:joints[parent].name));
    }
    // Keep the complete skin/rig below; first-person changes visible geometry only.
    if(!arms_only) {
    // Toes forward of the ball, the head above the neck.
    for(const char* toe:{"LeftToeBase","RightToeBase"}) {
        size_t t=find(toe);Vec3 foot=world[static_cast<size_t>(joints[t].parent)].p,ball=world[t].p,ahead=normalized(Vec3{ball.x-foot.x,0,ball.z-foot.z});
        capsule(body,ball,ball+ahead*0.06f,0.025f,static_cast<uint16_t>(t),tint(toe));
    }
    size_t head=find("Head");
    capsule(body,world[head].p+Vec3{0,0.06f,0.01f},world[head].p+Vec3{0,0.12f,0.02f},0.1f,static_cast<uint16_t>(head),tint("Head"));
    // Nose, so the facing is plain from any side.
    capsule(body,world[head].p+Vec3{0,0.08f,0.09f},world[head].p+Vec3{0,0.08f,0.13f},0.025f,static_cast<uint16_t>(head),{0.7f,0.3f,0.3f,1});

    }

    std::vector<char> bin;nlohmann::json doc;
    doc["asset"]={{"version","2.0"},{"generator","Tide Motion mannequin"}};
    doc["nodes"]=nlohmann::json::array();doc["bufferViews"]=nlohmann::json::array();doc["accessors"]=nlohmann::json::array();
    auto view=[&](const void* data,size_t size,int target) {
        size_t offset=bin.size();append(bin,data,size);
        nlohmann::json item={{"buffer",0},{"byteOffset",offset},{"byteLength",size}};
        if(target)item["target"]=target;
        doc["bufferViews"].push_back(item);return doc["bufferViews"].size()-1;
    };
    auto accessor=[&](size_t buffer_view,int component,size_t count,const char* type,nlohmann::json extra={}) {
        nlohmann::json item={{"bufferView",buffer_view},{"componentType",component},{"count",count},{"type",type}};
        for(const auto& e:extra.items())item[e.key()]=e.value();
        doc["accessors"].push_back(item);return doc["accessors"].size()-1;
    };
    std::vector<float> positions,normals,colors,weights;std::vector<uint16_t> joint_ids;
    Vec3 low{1e9f,1e9f,1e9f},high{-1e9f,-1e9f,-1e9f};
    for(const auto& c:body.corners) {
        positions.insert(positions.end(),{c.p.x,c.p.y,c.p.z});normals.insert(normals.end(),{c.n.x,c.n.y,c.n.z});
        colors.insert(colors.end(),c.color.begin(),c.color.end());
        joint_ids.insert(joint_ids.end(),{c.joint,0,0,0});weights.insert(weights.end(),{1,0,0,0});
        low={std::min(low.x,c.p.x),std::min(low.y,c.p.y),std::min(low.z,c.p.z)};high={std::max(high.x,c.p.x),std::max(high.y,c.p.y),std::max(high.z,c.p.z)};
    }
    size_t count=body.corners.size();
    auto position=accessor(view(positions.data(),positions.size()*4,34962),5126,count,"VEC3",{{"min",{low.x,low.y,low.z}},{"max",{high.x,high.y,high.z}}});
    auto normal=accessor(view(normals.data(),normals.size()*4,34962),5126,count,"VEC3");
    auto color=accessor(view(colors.data(),colors.size()*4,34962),5126,count,"VEC4");
    auto joint=accessor(view(joint_ids.data(),joint_ids.size()*2,34962),5123,count,"VEC4");
    auto weight=accessor(view(weights.data(),weights.size()*4,34962),5126,count,"VEC4");
    auto index=accessor(view(body.faces.data(),body.faces.size()*4,34963),5125,body.faces.size(),"SCALAR");
    // Bind pose: SOMA's rest has no rotations, so each inverse bind is a translation.
    std::vector<float> inverse_binds;
    for(const auto& w:world)inverse_binds.insert(inverse_binds.end(),{1,0,0,0, 0,1,0,0, 0,0,1,0, -w.p.x,-w.p.y,-w.p.z,1});
    auto binds=accessor(view(inverse_binds.data(),inverse_binds.size()*4,0),5126,world.size(),"MAT4");
    nlohmann::json roots=nlohmann::json::array(),skin_joints=nlohmann::json::array();
    for(size_t j=0;j<joints.size();++j) {
        const auto& p=joints[j].rest.p;
        nlohmann::json node={{"name",joints[j].name},{"translation",{p.x,p.y,p.z}}};
        nlohmann::json children=nlohmann::json::array();
        for(size_t k=j+1;k<joints.size();++k)if(joints[k].parent==static_cast<int>(j))children.push_back(k);
        if(!children.empty())node["children"]=children;
        if(joints[j].parent<0)roots.push_back(j);
        doc["nodes"].push_back(node);skin_joints.push_back(j);
    }
    doc["nodes"].push_back({{"name","Mannequin"},{"mesh",0},{"skin",0}});roots.push_back(joints.size());
    doc["meshes"]=nlohmann::json::array({{{"name","Mannequin"},{"primitives",nlohmann::json::array({{{"attributes",{{"POSITION",position},{"NORMAL",normal},{"COLOR_0",color},{"JOINTS_0",joint},{"WEIGHTS_0",weight}}},{"indices",index},{"mode",4}}})}}});
    doc["skins"]=nlohmann::json::array({{{"joints",skin_joints},{"inverseBindMatrices",binds},{"skeleton",0}}});
    doc["scenes"]=nlohmann::json::array({{{"nodes",roots}}});doc["scene"]=0;
    doc["buffers"]=nlohmann::json::array({{{"byteLength",bin.size()}}});
    auto text=doc.dump();while(text.size()%4)text.push_back(' ');
    std::ofstream out(path,std::ios::binary);
    auto word=[&](uint32_t n){char b[4];for(int i=0;i<4;++i)b[i]=static_cast<char>((n>>(8*i))&255);out.write(b,4);};
    word(0x46546c67);word(2);word(static_cast<uint32_t>(28+text.size()+bin.size()));
    word(static_cast<uint32_t>(text.size()));word(0x4e4f534a);out.write(text.data(),static_cast<std::streamsize>(text.size()));
    word(static_cast<uint32_t>(bin.size()));word(0x004e4942);out.write(bin.data(),static_cast<std::streamsize>(bin.size()));
    out.close();if(!out)throw std::runtime_error("Cannot write the preview mannequin");
}
}
