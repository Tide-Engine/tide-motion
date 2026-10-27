#include "motion.hpp"
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
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
Clip load_glb(const fs::path& path,const std::string& selected) {
    if(path.extension()!=".glb")throw std::runtime_error("Input must be a self-contained .glb animation; export FBX/glTF to GLB first");
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
    Clip c; c.joints=soma(); c.name=path.stem().string();
    std::vector<cgltf_node*> nodes;
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
    cgltf_animation* animation=nullptr;
    if(selected.empty()) {
        if(data->animations_count!=1)throw std::runtime_error("GLB must have one animation, or select one with --clip/--from-clip/--to-clip");
        animation=&data->animations[0];
    } else {
        for(size_t i=0;i<data->animations_count;++i)if(data->animations[i].name&&selected==data->animations[i].name) {
            if(animation)throw std::runtime_error("Ambiguous animation name"); animation=&data->animations[i];
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
    c.poses.reserve(frames*c.joints.size());
    for(size_t f=0;f<frames;++f)for(const auto& j:c.joints)c.poses.push_back(j.rest);
    std::array<std::array<bool,3>,30> seen{};
    for(size_t i=0;i<animation->channels_count;++i) {
        auto& channel=animation->channels[i];
        auto at=std::find(nodes.begin(),nodes.end(),channel.target_node);
        if(at==nodes.end())throw std::runtime_error("Animation contains a track outside the canonical SOMA rig");
        size_t joint=static_cast<size_t>(at-nodes.begin());
        int kind=channel.target_path==cgltf_animation_path_type_translation?0:channel.target_path==cgltf_animation_path_type_rotation?1:2;
        if(kind==2)throw std::runtime_error("Bake out scale or morph animation before importing");
        if(seen[joint][static_cast<size_t>(kind)])throw std::runtime_error("Duplicate animation channel");
        seen[joint][static_cast<size_t>(kind)]=true;
        auto& s=*channel.sampler;
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
    c.validate(); require_soma(c); return c;
}
void save_glb(const Clip& c,const fs::path& path) {
    c.validate();
    json doc={{"asset",{{"version","2.0"},{"generator","Tide Motion"}}},{"scene",0},{"scenes",json::array()},{"nodes",json::array()},{"bufferViews",json::array()},{"accessors",json::array()}};
    json roots=json::array();
    for(size_t j=0;j<c.joints.size();++j) {
        auto& joint=c.joints[j]; auto p=joint.rest.p; auto q=joint.rest.q;
        json node={{"name",joint.name},{"translation",{p.x,p.y,p.z}},{"rotation",{q.x,q.y,q.z,q.w}}};
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
