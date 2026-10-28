#include "equipment.hpp"
#include <cgltf.h>
#include <ufbx.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>

namespace motion {
namespace {
using nlohmann::json;
constexpr size_t limit=512ULL*1024*1024;
void require(bool value,const std::string& message) {if(!value)throw std::runtime_error(message);}
uint32_t word(const std::vector<char>& b,size_t at) {
    require(at<=b.size()&&b.size()-at>=4,"Truncated equipment GLB");uint32_t v=0;
    for(unsigned k=0;k<4;++k)v|=static_cast<uint32_t>(static_cast<unsigned char>(b[at+k]))<<(8*k);
    return v;
}
std::vector<char> read(const fs::path& path) {
    auto size=fs::file_size(path);require(size<=limit,"Equipment mesh exceeds 512 MiB");
    std::vector<char> bytes(static_cast<size_t>(size));std::ifstream in(path,std::ios::binary);
    require(static_cast<bool>(in.read(bytes.data(),static_cast<std::streamsize>(bytes.size()))),"Cannot read equipment mesh: "+path.string());return bytes;
}
void finite(float v) {require(std::isfinite(v),"Nonfinite equipment export value");}
struct Builder {
    json d={{"asset",{{"version","2.0"},{"generator","Tide Motion equipment"}}},{"scene",0},
        {"scenes",json::array({{{"nodes",json::array()}}})},{"nodes",json::array()},
        {"bufferViews",json::array()},{"accessors",json::array()},{"meshes",json::array()},
        {"materials",json::array()},{"images",json::array()},{"textures",json::array()},{"samplers",json::array()}};
    json animation={{"name","equipment"},{"samplers",json::array()},{"channels",json::array()}};
    json conversions=json::array();std::vector<char> bin;std::set<std::string> names;
    void room(size_t bytes) {require(bytes<=limit&&bin.size()<=limit-bytes,"Equipment GLB exceeds 512 MiB");}
    void align() {while(bin.size()%4)bin.push_back(0);}
    size_t floats(const std::vector<float>& values,const char* type,size_t width,bool bounds=false) {
        require(!values.empty()&&values.size()%width==0,"Invalid equipment export accessor");room(values.size()*4+4);align();size_t offset=bin.size();
        std::vector<float> low(width),high(width);for(size_t i=0;i<width;++i)low[i]=high[i]=values[i];
        for(size_t i=0;i<values.size();++i) {
            float v=values[i];finite(v);low[i%width]=std::min(low[i%width],v);high[i%width]=std::max(high[i%width],v);
            uint32_t bits;std::memcpy(&bits,&v,4);for(unsigned k=0;k<4;++k)bin.push_back(static_cast<char>((bits>>(8*k))&255));
        }
        size_t view=d["bufferViews"].size();d["bufferViews"].push_back({{"buffer",0},{"byteOffset",offset},{"byteLength",bin.size()-offset}});
        json a={{"bufferView",view},{"componentType",5126},{"count",values.size()/width},{"type",type}};
        if(bounds){a["min"]=low;a["max"]=high;}size_t index=d["accessors"].size();d["accessors"].push_back(a);return index;
    }
    void channel(size_t node,const char* path,size_t time,const std::vector<float>& values,const char* type,size_t width) {
        size_t output=floats(values,type,width),sampler=animation["samplers"].size();
        animation["samplers"].push_back({{"input",time},{"output",output},{"interpolation","LINEAR"}});
        animation["channels"].push_back({{"sampler",sampler},{"target",{{"node",node},{"path",path}}}});
    }
    void track(size_t node,size_t time,const std::vector<Transform>& poses,const std::vector<Vec3>* scales=nullptr) {
        std::vector<float> p,q,s;Quat previous{};
        for(size_t f=0;f<poses.size();++f) {
            auto t=poses[f];auto rotation=normalized(t.q);
            if(f&&previous.x*rotation.x+previous.y*rotation.y+previous.z*rotation.z+previous.w*rotation.w<0)
                rotation={-rotation.x,-rotation.y,-rotation.z,-rotation.w};
            p.insert(p.end(),{t.p.x,t.p.y,t.p.z});q.insert(q.end(),{rotation.x,rotation.y,rotation.z,rotation.w});previous=rotation;
            if(scales){const auto v=(*scales)[f];require(v.x>0&&v.y>0&&v.z>0,"Equipment mesh scale must be positive");s.insert(s.end(),{v.x,v.y,v.z});}
        }
        channel(node,"translation",time,p,"VEC3",3);channel(node,"rotation",time,q,"VEC4",4);
        if(scales)channel(node,"scale",time,s,"VEC3",3);
    }
    size_t named(json node,const std::string& name) {
        require(!name.empty()&&names.insert(name).second,"Duplicate or empty equipment export node: "+name);node["name"]=name;
        size_t n=d["nodes"].size();d["nodes"].push_back(node);return n;
    }
};
size_t index(const json& j,size_t count,const char* kind) {
    require(j.is_number_integer()&&j.get<double>()>=0&&j.get<double>()<static_cast<double>(count),std::string("Invalid equipment ")+kind+" index");return j.get<size_t>();
}
void extensions(const json& value,std::set<std::string>& used) {
    if(value.is_object())for(const auto& [key,v]:value.items()) {
        if(key=="extensions") {
            require(v.is_object(),"Malformed equipment extensions");
            for(const auto& [name,entry]:v.items()) {
                (void)entry;require(name=="KHR_materials_unlit"||name=="KHR_texture_transform","Unsupported equipment GLB extension: "+name);used.insert(name);
            }
        }
        if(key!="extras")extensions(v,used);
    } else if(value.is_array())for(const auto& v:value)extensions(v,used);
}
void import_glb(Builder& b,const EquipmentTrack& track,size_t root) {
    auto bytes=read(track.mesh);require(bytes.size()>=20&&word(bytes,0)==0x46546c67&&word(bytes,4)==2&&word(bytes,8)==bytes.size(),"Equipment mesh must be GLB 2.0");
    size_t json_size=word(bytes,12);require(word(bytes,16)==0x4e4f534a&&json_size<=bytes.size()-20,"Malformed equipment GLB JSON chunk");
    json src=json::parse(bytes.data()+20,bytes.data()+20+json_size);std::set<std::string> used;extensions(src,used);
    for(const char* key:{"extensionsUsed","extensionsRequired"})if(src.contains(key))for(const auto& x:src[key]) {
        auto name=x.get<std::string>();require(name=="KHR_materials_unlit"||name=="KHR_texture_transform","Unsupported equipment GLB extension: "+name);used.insert(name);
    }
    require(!src.contains("animations")||src["animations"].empty(),"Equipment mesh must be static; author its animation through the equipment plan");
    require(!src.contains("skins")||src["skins"].empty(),"Skinned equipment meshes are unsupported; supply a rigid static mesh");
    require(!src.contains("cameras")||src["cameras"].empty(),"Equipment mesh must not contain cameras");
    require(src.contains("buffers")&&src["buffers"].size()==1&&!src["buffers"][0].contains("uri"),"Equipment GLB must contain one embedded binary buffer");
    size_t at=20+json_size;require(at<=bytes.size()&&bytes.size()-at>=8&&word(bytes,at+4)==0x004e4942,"Equipment GLB has no binary chunk");
    size_t binary_size=word(bytes,at);require(binary_size<=bytes.size()-at-8,"Truncated equipment binary chunk");
    size_t declared=src["buffers"][0].at("byteLength").get<size_t>();require(declared<=binary_size,"Equipment binary buffer is truncated");
    cgltf_options opts{};cgltf_data* raw=nullptr;require(cgltf_parse(&opts,bytes.data(),bytes.size(),&raw)==cgltf_result_success,"Invalid equipment GLB");
    std::unique_ptr<cgltf_data,decltype(&cgltf_free)> validation(raw,cgltf_free);
    require(cgltf_load_buffers(&opts,raw,nullptr)==cgltf_result_success&&cgltf_validate(raw)==cgltf_result_success,"Invalid equipment GLB accessors");
    for(size_t a=0;a<raw->accessors_count;++a)if(raw->accessors[a].component_type==cgltf_component_type_r_32f) {
        const auto* accessor=&raw->accessors[a];require(accessor->count<=10000000,"Equipment accessor exceeds ten million samples");
        if(accessor->is_sparse) {
            size_t components=cgltf_num_components(accessor->type);require(components&&accessor->count<=limit/4/components,"Sparse equipment accessor exceeds the memory limit");
            std::vector<float> values(accessor->count*components);
            require(cgltf_accessor_unpack_floats(accessor,values.data(),values.size())==values.size(),"Cannot unpack sparse equipment accessor");
            for(float value:values)finite(value);
        } else for(size_t i=0;i<accessor->count;++i) {
            float values[16]{};require(cgltf_accessor_read_float(accessor,i,values,16),"Cannot read equipment accessor");
            for(float value:values)finite(value);
        }
    }
    b.room(declared+4);b.align();size_t base=b.bin.size();b.bin.insert(b.bin.end(),bytes.begin()+static_cast<std::ptrdiff_t>(at+8),bytes.begin()+static_cast<std::ptrdiff_t>(at+8+declared));
    auto count=[&](const char* key){return src.contains(key)?src[key].size():size_t{0};};
    const size_t views=b.d["bufferViews"].size(),accessors=b.d["accessors"].size(),materials=b.d["materials"].size(),
        images=b.d["images"].size(),textures=b.d["textures"].size(),samplers=b.d["samplers"].size(),meshes=b.d["meshes"].size(),nodes=b.d["nodes"].size();
    for(auto v:src.value("bufferViews",json::array())) {
        require(index(v.at("buffer"),1,"buffer")==0,"Invalid equipment buffer");v["buffer"]=0;v["byteOffset"]=base+v.value("byteOffset",size_t{0});b.d["bufferViews"].push_back(v);
    }
    for(auto v:src.value("accessors",json::array())) {
        if(v.contains("bufferView"))v["bufferView"]=views+index(v["bufferView"],count("bufferViews"),"bufferView");
        if(v.contains("sparse"))for(const char* key:{"indices","values"})v["sparse"][key]["bufferView"]=views+index(v["sparse"][key].at("bufferView"),count("bufferViews"),"sparse bufferView");
        b.d["accessors"].push_back(v);
    }
    for(const auto& v:src.value("samplers",json::array()))b.d["samplers"].push_back(v);
    for(auto v:src.value("images",json::array())) {
        require(!v.contains("uri")&&v.contains("bufferView")&&v.contains("mimeType"),"Equipment images must be embedded bufferViews; export images into GLB first");
        require(v["mimeType"]=="image/png"||v["mimeType"]=="image/jpeg","Equipment images must be embedded PNG or JPEG");
        v["bufferView"]=views+index(v["bufferView"],count("bufferViews"),"image bufferView");b.d["images"].push_back(v);
    }
    for(auto v:src.value("textures",json::array())) {
        v["source"]=images+index(v.at("source"),count("images"),"image");
        if(v.contains("sampler"))v["sampler"]=samplers+index(v["sampler"],count("samplers"),"sampler");
        b.d["textures"].push_back(v);
    }
    for(auto v:src.value("materials",json::array())) {
        auto info=[&](json& object,const char* key){if(object.contains(key))object[key]["index"]=textures+index(object[key].at("index"),count("textures"),"texture");};
        if(v.contains("pbrMetallicRoughness")){auto& p=v["pbrMetallicRoughness"];info(p,"baseColorTexture");info(p,"metallicRoughnessTexture");}
        for(const char* key:{"normalTexture","occlusionTexture","emissiveTexture"})info(v,key);
        b.d["materials"].push_back(v);
    }
    for(auto v:src.value("meshes",json::array())) {
        require(!v.contains("weights"),"Equipment morph weights are unsupported");
        for(auto& p:v.at("primitives")) {
            require(!p.contains("targets")||p["targets"].empty(),"Equipment morph targets are unsupported");
            for(auto& attr:p.at("attributes").items()) {
                require(!attr.key().starts_with("JOINTS_")&&!attr.key().starts_with("WEIGHTS_"),"Equipment mesh contains skinning attributes");
                attr.value()=accessors+index(attr.value(),count("accessors"),"attribute accessor");
            }
            if(p.contains("indices"))p["indices"]=accessors+index(p["indices"],count("accessors"),"indices accessor");
            if(p.contains("material"))p["material"]=materials+index(p["material"],count("materials"),"material");
        }
        b.d["meshes"].push_back(v);
    }
    require(count("nodes")<=1024,"Too many equipment mesh nodes");std::vector<unsigned> parents(count("nodes")),state(count("nodes"));
    std::function<void(size_t,size_t)> visit=[&](size_t n,size_t depth) {
        require(depth<=128&&state[n]!=1,"Equipment mesh hierarchy contains a cycle or is too deep");if(state[n]==2)return;state[n]=1;
        for(const auto& child:src["nodes"][n].value("children",json::array()))visit(index(child,count("nodes"),"child"),depth+1);
        state[n]=2;
    };
    for(size_t n=0;n<count("nodes");++n)for(const auto& child:src["nodes"][n].value("children",json::array()))
        require(++parents[index(child,count("nodes"),"child")]==1,"Equipment mesh node has multiple parents");
    for(size_t n=0;n<count("nodes");++n)visit(n,0);
    for(size_t n=0;n<count("nodes");++n) {
        auto v=src["nodes"][n];require(!v.contains("skin")&&!v.contains("camera")&&!v.contains("weights"),"Equipment nodes must be rigid, without cameras or morph weights");
        if(v.contains("mesh"))v["mesh"]=meshes+index(v["mesh"],count("meshes"),"mesh");
        if(v.contains("children"))for(auto& child:v["children"])child=nodes+index(child,count("nodes"),"child");
        b.named(v,track.name+"/mesh/"+std::to_string(n)+"/"+v.value("name",std::string("node")));
    }
    json roots=json::array();
    if(count("scenes"))roots=src["scenes"][src.contains("scene")?index(src["scene"],count("scenes"),"scene"):0].value("nodes",json::array());
    else for(size_t n=0;n<count("nodes");++n)if(!parents[n])roots.push_back(n);
    require(!roots.empty(),"Equipment GLB must have scene nodes containing geometry");size_t geometry=0;
    std::function<void(size_t)> geometry_count=[&](size_t n) {
        if(src["nodes"][n].contains("mesh"))++geometry;
        for(const auto& c:src["nodes"][n].value("children",json::array()))geometry_count(c.get<size_t>());
    };
    b.d["nodes"][root]["children"]=json::array();std::set<size_t> unique_roots;
    for(const auto& v:roots) {
        size_t n=index(v,count("nodes"),"scene root");require(parents[n]==0&&unique_roots.insert(n).second,"Equipment scene root is repeated or has a parent");
        b.d["nodes"][root]["children"].push_back(nodes+n);geometry_count(n);
    }
    require(geometry>0,"Equipment scene has no mesh geometry");
    for(const auto& name:used) {
        if(!b.d.contains("extensionsUsed"))b.d["extensionsUsed"]=json::array();
        if(std::find(b.d["extensionsUsed"].begin(),b.d["extensionsUsed"].end(),name)==b.d["extensionsUsed"].end())b.d["extensionsUsed"].push_back(name);
    }
    for(const auto& name:src.value("extensionsRequired",json::array())) {
        if(!b.d.contains("extensionsRequired"))b.d["extensionsRequired"]=json::array();
        if(std::find(b.d["extensionsRequired"].begin(),b.d["extensionsRequired"].end(),name)==b.d["extensionsRequired"].end())b.d["extensionsRequired"].push_back(name);
    }
    b.d["nodes"][root]["extras"]["sourceAsset"]=src.at("asset");
    b.conversions.push_back({{"object",track.name},{"format","GLB"},{"appearance","preserved static hierarchy, attributes, core materials and embedded textures"}});
}
size_t triangle_mesh(Builder& b,const std::vector<float>& p,const std::vector<float>& n,const json& material) {
    auto position=b.floats(p,"VEC3",3,true),normal=b.floats(n,"VEC3",3);size_t mat=b.d["materials"].size();b.d["materials"].push_back(material);
    size_t mesh=b.d["meshes"].size();b.d["meshes"].push_back({{"primitives",json::array({{{"attributes",{{"POSITION",position},{"NORMAL",normal}}},{"material",mat},{"mode",4}}})}});return mesh;
}
void rod(Builder& b,size_t root) {
    std::vector<float> p,n;
    for(unsigned axis=0;axis<3;++axis)for(int sign:{-1,1}) {
        unsigned u=(axis+1)%3,v=(axis+2)%3;std::array<Vec3,4> points{};
        for(size_t k=0;k<4;++k) {
            float values[3]{};values[axis]=static_cast<float>(sign)*0.5f;values[u]=(k==0||k==3)?-0.5f:0.5f;values[v]=k<2?-0.5f:0.5f;points[k]={values[0],values[1],values[2]};
        }
        for(unsigned k:{0u,1u,2u,0u,2u,3u}) {
            auto point=points[sign>0?k:(k==1?3:k==3?1:k)];p.insert(p.end(),{point.x,point.y,point.z});float normal[3]{};normal[axis]=static_cast<float>(sign);n.insert(n.end(),normal,normal+3);
        }
    }
    b.d["nodes"][root]["mesh"]=triangle_mesh(b,p,n,{{"name","Tide authored string"},{"pbrMetallicRoughness",{{"baseColorFactor",{0.8,0.8,0.8,1}},{"metallicFactor",0},{"roughnessFactor",0.7}}}});
}
void import_fbx(Builder& b,const EquipmentTrack& track,size_t root) {
    auto bytes=read(track.mesh);ufbx_load_opts opts{};opts.file_format=UFBX_FILE_FORMAT_FBX;opts.target_axes=ufbx_axes_right_handed_y_up;opts.target_unit_meters=1;
    opts.space_conversion=UFBX_SPACE_CONVERSION_ADJUST_TRANSFORMS;opts.geometry_transform_handling=UFBX_GEOMETRY_TRANSFORM_HANDLING_HELPER_NODES;
    opts.inherit_mode_handling=UFBX_INHERIT_MODE_HANDLING_HELPER_NODES;opts.node_depth_limit=128;opts.ignore_embedded=true;
    opts.temp_allocator.memory_limit=limit;opts.result_allocator.memory_limit=limit;ufbx_error error{};
    std::unique_ptr<ufbx_scene,decltype(&ufbx_free_scene)> scene(ufbx_load_memory(bytes.data(),bytes.size(),&opts,&error),ufbx_free_scene);
    require(static_cast<bool>(scene),"Cannot decode FBX equipment mesh");require(scene->nodes.count<=1024,"Too many FBX equipment nodes");
    require(scene->anim_curves.count==0,"Animated FBX equipment is unsupported; export a static mesh and author tracks in the plan");
    require(scene->skin_deformers.count==0&&scene->blend_deformers.count==0,"Skinned or morphing FBX equipment is unsupported");
    for(const auto* m:scene->materials)require(m->textures.count==0,"Textured FBX equipment must be exported as self-contained GLB to preserve its textures");
    b.d["nodes"][root]["children"]=json::array();size_t corners=0,node_number=0;
    for(const auto* node:scene->nodes)if(node->mesh) {
        const auto* mesh=node->mesh;require(mesh->max_face_triangles<=10000000/3,"FBX equipment face is too large");
        std::vector<uint32_t> indices(mesh->max_face_triangles*3);auto normals=ufbx_matrix_for_normals(&node->geometry_to_world);
        std::map<int,std::pair<std::vector<float>,std::vector<float>>> groups;
        for(size_t f=0;f<mesh->faces.count;++f) {
            auto face=mesh->faces[f];if(face.num_indices<3)continue;
            uint32_t triangles=ufbx_triangulate_face(indices.data(),indices.size(),mesh,face);require(triangles>0,"Cannot triangulate FBX equipment face");
            require(triangles<=10000000/3&&corners<=10000000-static_cast<size_t>(triangles)*3,"FBX equipment exceeds ten million corners");corners+=static_cast<size_t>(triangles)*3;
            int mat=mesh->face_material.count?static_cast<int>(mesh->face_material.data[f]):-1;auto& group=groups[mat];
            for(uint32_t t=0;t<triangles;++t) {
                std::array<ufbx_vec3,3> positions{};for(size_t k=0;k<3;++k)positions[k]=ufbx_transform_position(&node->geometry_to_world,ufbx_get_vertex_vec3(&mesh->vertex_position,indices[t*3+k]));
                Vec3 a{static_cast<float>(positions[0].x),static_cast<float>(positions[0].y),static_cast<float>(positions[0].z)},
                    z{static_cast<float>(positions[1].x),static_cast<float>(positions[1].y),static_cast<float>(positions[1].z)},
                    c{static_cast<float>(positions[2].x),static_cast<float>(positions[2].y),static_cast<float>(positions[2].z)};
                Vec3 geometric=cross(z-a,c-a);if(dot(geometric,geometric)>1e-16f)geometric=normalized(geometric);else geometric={0,1,0};
                for(size_t k=0;k<3;++k) {
                    const auto& p=positions[k];Vec3 normal=geometric;
                    if(mesh->vertex_normal.exists) {
                        auto v=ufbx_transform_direction(&normals,ufbx_get_vertex_vec3(&mesh->vertex_normal,indices[t*3+k]));
                        normal={static_cast<float>(v.x),static_cast<float>(v.y),static_cast<float>(v.z)};if(dot(normal,normal)>1e-16f)normal=normalized(normal);else normal=geometric;
                    }
                    group.first.insert(group.first.end(),{static_cast<float>(p.x),static_cast<float>(p.y),static_cast<float>(p.z)});group.second.insert(group.second.end(),{normal.x,normal.y,normal.z});
                }
            }
        }
        for(const auto& [material,group]:groups) {
            const ufbx_material* m=material>=0&&static_cast<size_t>(material)<node->materials.count?node->materials.data[material]:nullptr;
            ufbx_vec3 color{};color.x=color.y=color.z=0.8;double factor=1,alpha=1,metallic=0,roughness=0.7;
            if(m) {
                if(m->pbr.base_color.has_value){color=m->pbr.base_color.value_vec3;if(m->pbr.base_factor.has_value)factor=m->pbr.base_factor.value_real;}
                else if(m->fbx.diffuse_color.has_value){color=m->fbx.diffuse_color.value_vec3;if(m->fbx.diffuse_factor.has_value)factor=m->fbx.diffuse_factor.value_real;}
                if(m->pbr.opacity.has_value)alpha=m->pbr.opacity.value_real;else if(m->fbx.transparency_factor.has_value)alpha=1-m->fbx.transparency_factor.value_real;
                if(m->pbr.metalness.has_value)metallic=m->pbr.metalness.value_real;
                if(m->pbr.roughness.has_value)roughness=m->pbr.roughness.value_real;
            }
            auto value=[](double v){require(std::isfinite(v),"Nonfinite FBX material value");return std::clamp(v,0.0,1.0);};
            json mat={{"name",m?std::string(m->name.data,m->name.length):"FBX default"},{"pbrMetallicRoughness",{{"baseColorFactor",{value(color.x*factor),value(color.y*factor),value(color.z*factor),value(alpha)}},{"metallicFactor",value(metallic)},{"roughnessFactor",value(roughness)}}}};
            if(alpha<1)mat["alphaMode"]="BLEND";
            size_t id=triangle_mesh(b,group.first,group.second,mat);size_t child=b.named({{"mesh",id}},track.name+"/fbx/"+std::to_string(node_number++));b.d["nodes"][root]["children"].push_back(child);
        }
    }
    require(corners>0,"FBX equipment has no triangle geometry");
    b.conversions.push_back({{"object",track.name},{"format","FBX"},{"appearance","Converted base color/opacity/metalness/roughness; other FBX shader features, UVs and vertex colors are not represented. Use GLB for appearance-preserving export."}});
}
}
void save_equipment_glb(const EquipmentResult& result,const fs::path& output) {
    result.body.validate();require(output.extension()==".glb","Equipment output must be .glb");
    fs::path part=output;part+=".part";
    require(!fs::exists(output)&&!fs::exists(output.string()+".json")&&!fs::exists(part),"Equipment output, sidecar or partial file already exists");
    Builder b;const auto& body=result.body;std::vector<float> times;for(size_t f=0;f<body.frames();++f)times.push_back(static_cast<float>(f)/body.fps);size_t time=b.floats(times,"SCALAR",1,true);
    b.animation["name"]=body.name;
    for(size_t j=0;j<body.joints.size();++j) {
        const auto& bone=body.joints[j];auto q=normalized(bone.rest.q);json n={{"translation",{bone.rest.p.x,bone.rest.p.y,bone.rest.p.z}},{"rotation",{q.x,q.y,q.z,q.w}}};
        if(bone.scale!=1)n["scale"]={bone.scale,bone.scale,bone.scale};
        json children=json::array();
        for(size_t k=j+1;k<body.joints.size();++k)if(body.joints[k].parent==static_cast<int>(j))children.push_back(k);
        if(!children.empty())n["children"]=children;
        size_t node=b.named(n,bone.name);if(bone.parent<0)b.d["scenes"][0]["nodes"].push_back(node);
        std::vector<Transform> poses;poses.reserve(body.frames());for(size_t f=0;f<body.frames();++f)poses.push_back(body.frame(f)[j]);b.track(node,time,poses);
    }
    require(result.objects.size()<=128,"Too many exported equipment tracks");
    for(const auto& object:result.objects) {
        require(object.poses.size()==body.frames()&&object.scales.size()==body.frames(),"Equipment track sample count differs from body");
        require(!object.builtin_rod||object.mesh.empty(),"Built-in equipment rods cannot also specify a source mesh");
        const auto& first=object.poses.front();const auto scale=object.scales.front();auto q=normalized(first.q);
        size_t root=b.named({{"translation",{first.p.x,first.p.y,first.p.z}},{"rotation",{q.x,q.y,q.z,q.w}},{"scale",{scale.x,scale.y,scale.z}},
            {"extras",{{"tideMotion",{{"equipmentObject",true},{"meshScaleOnly",true},{"source",object.mesh.filename().generic_string()}}}}}},object.name);
        b.d["scenes"][0]["nodes"].push_back(root);b.track(root,time,object.poses,&object.scales);
        if(object.builtin_rod)rod(b,root);
        else if(!object.mesh.empty()) {if(is_fbx(object.mesh))import_fbx(b,object,root);else import_glb(b,object,root);}
    }
    b.d["asset"]["extras"]={{"tideMotion",{{"motionRoot",body.joints[body.motion_root].name},{"equipmentVersion",1},{"fps",body.fps},{"frames",body.frames()},
        {"events",result.events},{"report",result.report},{"meshConversions",b.conversions},{"bodyMeshEmbedded",false}}}};
    b.d["animations"]=json::array({b.animation});b.d["buffers"]=json::array({{{"byteLength",b.bin.size()}}});
    for(const char* key:{"meshes","materials","images","textures","samplers"})if(b.d[key].empty())b.d.erase(key);
    std::string text=b.d.dump();while(text.size()%4)text+=' ';b.align();require(text.size()<=limit&&b.bin.size()<=limit-text.size()&&28+text.size()+b.bin.size()<=limit,"Equipment GLB exceeds 512 MiB");
    fs::create_directories(output.has_parent_path()?output.parent_path():fs::path("."));
    // Exclusive creation and a no-replace hard-link commit protect other writers,
    // including a file appearing after preflight. Both paths share a filesystem.
#ifdef _WIN32
    FILE* file=_wfopen(part.c_str(),L"wbx");
#else
    FILE* file=std::fopen(part.c_str(),"wbx");
#endif
    require(file!=nullptr,"Cannot create equipment partial file exclusively");bool owned=true;
    try {
        auto bytes=[&](const char* p,size_t n){require(std::fwrite(p,1,n,file)==n,"Cannot write equipment GLB");};
        auto integer=[&](uint32_t n){char p[4];for(unsigned k=0;k<4;++k)p[k]=static_cast<char>((n>>(8*k))&255);bytes(p,4);};
        integer(0x46546c67);integer(2);integer(static_cast<uint32_t>(28+text.size()+b.bin.size()));
        integer(static_cast<uint32_t>(text.size()));integer(0x4e4f534a);bytes(text.data(),text.size());
        integer(static_cast<uint32_t>(b.bin.size()));integer(0x004e4942);bytes(b.bin.data(),b.bin.size());
        int closed=std::fclose(file);file=nullptr;require(closed==0,"Cannot close equipment GLB");
        fs::create_hard_link(part,output);std::error_code ignored;fs::remove(part,ignored);owned=false;
    } catch(...) {if(file)std::fclose(file);if(owned){std::error_code ignored;fs::remove(part,ignored);}throw;}
}
}
