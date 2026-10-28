#include "motion.hpp"
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>

namespace motion {
using nlohmann::json;
namespace {
void write_u32(std::ostream& out,uint32_t n) { char bytes[4]; for(int i=0;i<4;++i)bytes[i]=static_cast<char>((n>>(8*i))&255); out.write(bytes,4); }
std::array<float,4> read_value(const cgltf_accessor* a,size_t i) {
    std::array<float,4> v{};
    if(!cgltf_accessor_read_float(a,i,v.data(),v.size()))throw std::runtime_error("Cannot read animation accessor");
    for(float f:v)if(!std::isfinite(f))throw std::runtime_error("Nonfinite GLB accessor value");
    return v;
}
}
bool is_fbx(const fs::path& path) {
    auto ext=path.extension().string();for(auto& ch:ext)ch=static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return ext==".fbx";
}
static Clip read_glb(const fs::path& path,const std::string& selected,bool canonical,bool rest_only) {
    if(is_fbx(path)) {
        if(canonical)throw std::runtime_error("SOMA clips must be GLB; convert an FBX clip with retarget --to-soma first");
        return read_fbx(path,selected,rest_only);
    }
    if(path.extension()!=".glb")throw std::runtime_error("Input must be a self-contained .glb or .fbx; export glTF to GLB first");
    if(fs::file_size(path)>512ULL*1024*1024)throw std::runtime_error("GLB exceeds the 512 MiB input limit");
    std::ifstream input(path,std::ios::binary);
    std::vector<char> bytes(static_cast<size_t>(fs::file_size(path)));
    if(!input.read(bytes.data(),static_cast<std::streamsize>(bytes.size())))throw std::runtime_error("Cannot read "+path.string());
    cgltf_options opts{}; cgltf_data* raw=nullptr;
    if(cgltf_parse(&opts,bytes.data(),bytes.size(),&raw)!=cgltf_result_success)throw std::runtime_error("Malformed GLB");
    std::unique_ptr<cgltf_data,decltype(&cgltf_free)> data(raw,cgltf_free);
    for(size_t i=0;i<data->buffers_count;++i)if(data->buffers[i].uri)throw std::runtime_error("GLB must embed all buffers");
    if(cgltf_load_buffers(&opts,data.get(),nullptr)!=cgltf_result_success||cgltf_validate(data.get())!=cgltf_result_success)throw std::runtime_error("Invalid GLB buffers or accessors");
    if(data->nodes_count>1024||data->animations_count>1024)throw std::runtime_error("Too many GLB nodes or animations");
    Clip c; c.name=path.stem().string();
    std::vector<cgltf_node*> nodes;
    if(canonical) {
      // A SOMA-77 presentation clip enters canonical paths as its exact SOMA-30 subset.
      for(size_t i=0;i<data->nodes_count;++i)if(data->nodes[i].name&&std::strcmp(data->nodes[i].name,"LeftHandIndex1")==0) {
          data.reset();return soma77_to_30(read_glb(path,selected,false,rest_only));
      }
      c.joints=soma();
      for(auto& joint:c.joints) {
        cgltf_node* found=nullptr;
        for(size_t i=0;i<data->nodes_count;++i)if(data->nodes[i].name&&joint.name==data->nodes[i].name) {
            if(found)throw std::runtime_error("Duplicate SOMA joint name: "+joint.name);
            found=&data->nodes[i];
        }
        if(!found)throw std::runtime_error("GLB is missing SOMA joint '"+joint.name+"'; retarget to SOMA first");
        if(found->has_matrix)throw std::runtime_error("SOMA nodes must use TRS rather than matrix transforms");
        for(int k=0;k<3;++k)if(!std::isfinite(found->scale[k])||std::abs(found->scale[k]-1)>1e-5f)throw std::runtime_error("SOMA joint scale must be one");
        joint.rest={{found->translation[0],found->translation[1],found->translation[2]},normalized({found->rotation[0],found->rotation[1],found->rotation[2],found->rotation[3]})};
        nodes.push_back(found);
      }
      for(size_t j=0;j<nodes.size();++j) {
        auto parent=c.joints[j].parent;
        if(nodes[j]->parent!=(parent<0?nullptr:nodes[static_cast<size_t>(parent)]))throw std::runtime_error("GLB SOMA joint hierarchy differs from the canonical rig");
      }
    } else {
        std::vector<unsigned char> state(data->nodes_count);
        std::vector<int> indices(data->nodes_count,-1);
        std::function<void(cgltf_node*)> append=[&](cgltf_node* node) {
            size_t index=static_cast<size_t>(node-data->nodes);
            if(state[index]==2)return;
            if(state[index]==1)throw std::runtime_error("Skeleton contains a parent cycle");
            state[index]=1;
            if(node->parent)append(node->parent);
            float scale=node->scale[0];
            float values[3]={node->scale[0],node->scale[1],node->scale[2]};
            Transform rest{{node->translation[0],node->translation[1],node->translation[2]},normalized({node->rotation[0],node->rotation[1],node->rotation[2],node->rotation[3]})};
            if(node->has_matrix) {
                const auto& m=node->matrix;
                for(float value:m)if(!std::isfinite(value))throw std::runtime_error("Nonfinite rig matrix");
                if(std::abs(m[3])+std::abs(m[7])+std::abs(m[11])+std::abs(m[15]-1)>1e-5f)throw std::runtime_error("Rig matrix must be affine");
                Vec3 x{m[0],m[1],m[2]},y{m[4],m[5],m[6]},z{m[8],m[9],m[10]};
                values[0]=std::sqrt(dot(x,x));values[1]=std::sqrt(dot(y,y));values[2]=std::sqrt(dot(z,z));scale=values[0];
                auto nx=normalized(x),ny=normalized(y),nz=normalized(z);
                if(std::abs(dot(nx,ny))+std::abs(dot(nx,nz))+std::abs(dot(ny,nz))>1e-4f||dot(cross(nx,ny),nz)<0.9999f)
                    throw std::runtime_error("Rig matrix has shear or reflection; bake transforms before GLB export");
                rest={{m[12],m[13],m[14]},rotation6(nx,ny)};
            }
            for(float value:values)if(!std::isfinite(value)||value<=0||std::abs(value-scale)>1e-5f*std::max(1e-8f,std::abs(scale)))
                throw std::runtime_error("Rig has nonuniform or negative scale; bake scale before GLB export");
            int parent=node->parent?indices[static_cast<size_t>(node->parent-data->nodes)]:-1;
            indices[index]=static_cast<int>(c.joints.size());
            c.joints.push_back({node->name?node->name:"",parent,rest,scale});
            nodes.push_back(node);
            state[index]=2;
        };
        // A character can have unrelated mesh/camera transforms. Its reference
        // rig consists of skin joints and their ancestors, not render geometry.
        if(rest_only&&data->skins_count) {
            for(size_t s=0;s<data->skins_count;++s)for(size_t j=0;j<data->skins[s].joints_count;++j)append(data->skins[s].joints[j]);
        } else for(size_t i=0;i<data->nodes_count;++i)append(&data->nodes[i]);
        if(data->asset.extras.data) {
            auto extras=json::parse(data->asset.extras.data);
            if(extras.contains("tideMotion")) {
                auto name=extras["tideMotion"].at("motionRoot").get<std::string>();
                for(size_t j=0;j<c.joints.size();++j)if(c.joints[j].name==name)c.motion_root=j;
            }
        }
    }
    if(rest_only) {
        for(const auto& joint:c.joints)c.poses.push_back(joint.rest);
        c.validate();(void)world_pose(c,c.frame(0));return c;
    }
    cgltf_animation* animation=nullptr;
    if(selected.empty()) {
        if(data->animations_count==0)throw std::runtime_error("GLB has no animation; use inspect --skeleton to list rig bones");
        if(data->animations_count!=1)throw std::runtime_error("GLB must have one animation, or select one with --clip/--from-clip/--to-clip");
        animation=&data->animations[0];
    } else {
        for(size_t i=0;i<data->animations_count;++i)if(data->animations[i].name&&selected==data->animations[i].name) {
            if(animation)throw std::runtime_error("Ambiguous animation name");
            animation=&data->animations[i];
        }
        if(!animation)throw std::runtime_error("Animation not found: "+selected);
    }
    float end=0;
    for(size_t i=0;i<animation->samplers_count;++i) {
        auto& s=animation->samplers[i];
        if(!s.input||!s.output||s.input->count==0||s.input->count>1000000||s.input->count!=s.output->count)throw std::runtime_error("Invalid animation sampler size");
        if(s.interpolation!=cgltf_interpolation_type_linear&&s.interpolation!=cgltf_interpolation_type_step)throw std::runtime_error("Bake CUBICSPLINE tracks to LINEAR before importing");
        float previous=-1;
        for(size_t k=0;k<s.input->count;++k) { float time=read_value(s.input,k)[0]; if(time<0||time<=previous)throw std::runtime_error("Animation key times must be nonnegative and strictly increasing"); previous=time; }
        end=std::max(end,previous);
    }
    if(!std::isfinite(end)||end>600)throw std::runtime_error("Animation must be at most 600 seconds");
    double frame_time=static_cast<double>(end)*30;
    double nearest=std::round(frame_time);
    size_t frames=static_cast<size_t>(std::abs(frame_time-nearest)<0.002?nearest:std::ceil(frame_time))+1;
    if(frames>18000)throw std::runtime_error("Animation exceeds 18000 frames");
    c.poses.reserve(frames*c.joints.size());
    for(size_t f=0;f<frames;++f)for(const auto& j:c.joints)c.poses.push_back(j.rest);
    std::vector<std::array<bool,3>> seen(c.joints.size());
    for(size_t i=0;i<animation->channels_count;++i) {
        auto& channel=animation->channels[i];
        auto at=std::find(nodes.begin(),nodes.end(),channel.target_node);
        if(at==nodes.end())throw std::runtime_error("Animation contains a track outside the selected rig");
        size_t joint=static_cast<size_t>(at-nodes.begin());
        int kind=channel.target_path==cgltf_animation_path_type_translation?0:channel.target_path==cgltf_animation_path_type_rotation?1:channel.target_path==cgltf_animation_path_type_scale?2:3;
        if(kind==3||(kind==2&&canonical))throw std::runtime_error("Bake out scale or morph animation before importing");
        if(seen[joint][static_cast<size_t>(kind)])throw std::runtime_error("Duplicate animation channel");
        seen[joint][static_cast<size_t>(kind)]=true;
        auto& s=*channel.sampler;
        if(kind==2) {
            if(s.output->type!=cgltf_type_vec3)throw std::runtime_error("Scale track requires VEC3 values");
            for(size_t k=0;k<s.output->count;++k) {
                auto value=read_value(s.output,k);
                for(size_t axis=0;axis<3;++axis)if(std::abs(value[axis]-c.joints[joint].scale)>1e-5f*c.joints[joint].scale)
                    throw std::runtime_error("Animated scale is unsupported; export rigid bones with static uniform scale");
            }
            continue;
        }
        if(s.input->type!=cgltf_type_scalar||s.output->type!=(kind==0?cgltf_type_vec3:cgltf_type_vec4))throw std::runtime_error("Animation sampler has the wrong accessor type");
        size_t lo=0;
        for(size_t f=0;f<frames;++f) {
            float time=static_cast<float>(f)/30;
            while(lo+1<s.input->count&&read_value(s.input,lo+1)[0]<=time)++lo;
            size_t hi=std::min(lo+1,s.input->count-1);
            float begin=read_value(s.input,lo)[0],finish=read_value(s.input,hi)[0];
            float t=finish>begin&&s.interpolation==cgltf_interpolation_type_linear?std::clamp((time-begin)/(finish-begin),0.0f,1.0f):0;
            auto a=read_value(s.output,lo),b=read_value(s.output,hi);
            auto& pose=c.frame(f)[joint];
            if(kind==0)pose.p=Vec3{a[0],a[1],a[2]}*(1-t)+Vec3{b[0],b[1],b[2]}*t;
            else pose.q=slerp({a[0],a[1],a[2],a[3]},{b[0],b[1],b[2],b[3]},t);
        }
    }
    c.validate();
    if(canonical)require_soma(c);
    return c;
}
Clip load_glb(const fs::path& path,const std::string& selected,bool canonical) { return read_glb(path,selected,canonical,false); }
Clip load_rig_glb(const fs::path& path) { return read_glb(path,"",false,true); }
void save_glb(const Clip& c,const fs::path& path) {
    c.validate();
    json doc={{"asset",{{"version","2.0"},{"generator","Tide Motion"}}},{"scene",0},{"scenes",json::array()},{"nodes",json::array()},{"bufferViews",json::array()},{"accessors",json::array()}};
    doc["asset"]["extras"]={{"tideMotion",{{"motionRoot",c.joints[c.motion_root].name}}}};
    json roots=json::array();
    for(size_t j=0;j<c.joints.size();++j) {
        auto& joint=c.joints[j]; auto p=joint.rest.p; auto q=joint.rest.q;
        json node={{"name",joint.name},{"translation",{p.x,p.y,p.z}},{"rotation",{q.x,q.y,q.z,q.w}}};
        if(joint.scale!=1)node["scale"]={joint.scale,joint.scale,joint.scale};
        json children=json::array();
        for(size_t k=j+1;k<c.joints.size();++k)if(c.joints[k].parent==static_cast<int>(j))children.push_back(k);
        if(!children.empty())node["children"]=children;
        if(joint.parent<0)roots.push_back(j);
        doc["nodes"].push_back(node);
    }
    doc["scenes"].push_back({{"nodes",roots}});
    std::vector<char> bin;
    auto accessor=[&](const std::vector<float>& values,const char* type,int width) {
        size_t offset=bin.size();
        for(float f:values) {
            uint32_t bits; static_assert(sizeof(bits)==sizeof(f)); std::memcpy(&bits,&f,4);
            for(int b=0;b<4;++b)bin.push_back(static_cast<char>((bits>>(8*b))&255));
        }
        size_t view=doc["bufferViews"].size();
        doc["bufferViews"].push_back({{"buffer",0},{"byteOffset",offset},{"byteLength",bin.size()-offset}});
        json item={{"bufferView",view},{"componentType",5126},{"count",values.size()/static_cast<size_t>(width)},{"type",type}};
        if(width==1) { item["min"]={values.front()};item["max"]={values.back()}; }
        size_t index=doc["accessors"].size();doc["accessors"].push_back(item);return index;
    };
    std::vector<float> times;
    for(size_t f=0;f<c.frames();++f)times.push_back(static_cast<float>(f)/c.fps);
    size_t time_accessor=accessor(times,"SCALAR",1);
    json samplers=json::array(),channels=json::array();
    for(size_t j=0;j<c.joints.size();++j) {
        std::vector<float> p,q;
        Quat previous{};
        for(size_t f=0;f<c.frames();++f) {
            auto t=c.frame(f)[j]; t.q=normalized(t.q);
            if(f&&t.q.x*previous.x+t.q.y*previous.y+t.q.z*previous.z+t.q.w*previous.w<0)t.q={-t.q.x,-t.q.y,-t.q.z,-t.q.w};
            previous=t.q;
            p.insert(p.end(),{t.p.x,t.p.y,t.p.z});q.insert(q.end(),{t.q.x,t.q.y,t.q.z,t.q.w});
        }
        for(int kind=0;kind<2;++kind) {
            size_t output=kind==0?accessor(p,"VEC3",3):accessor(q,"VEC4",4);
            channels.push_back({{"sampler",samplers.size()},{"target",{{"node",j},{"path",kind==0?"translation":"rotation"}}}});
            samplers.push_back({{"input",time_accessor},{"output",output},{"interpolation","LINEAR"}});
        }
    }
    doc["animations"]=json::array({{{"name",c.name},{"samplers",samplers},{"channels",channels}}});
    doc["buffers"]=json::array({{{"byteLength",bin.size()}}});
    std::string text=doc.dump();while(text.size()%4)text+=' ';while(bin.size()%4)bin.push_back(0);
    size_t size=12+8+text.size()+8+bin.size();
    if(size>512ULL*1024*1024)throw std::runtime_error("Output exceeds 512 MiB");
    fs::create_directories(path.has_parent_path()?path.parent_path():fs::path("."));
    fs::path temp=path;temp+=".part";
    std::ofstream out(temp,std::ios::binary|std::ios::trunc);
    write_u32(out,0x46546c67);write_u32(out,2);write_u32(out,static_cast<uint32_t>(size));
    write_u32(out,static_cast<uint32_t>(text.size()));write_u32(out,0x4e4f534a);out.write(text.data(),static_cast<std::streamsize>(text.size()));
    write_u32(out,static_cast<uint32_t>(bin.size()));write_u32(out,0x004e4942);out.write(bin.data(),static_cast<std::streamsize>(bin.size()));
    out.close();if(!out)throw std::runtime_error("Cannot write GLB: "+temp.string());
    fs::rename(temp,path);
}
}
