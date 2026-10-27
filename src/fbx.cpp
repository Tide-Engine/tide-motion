#include "motion.hpp"
#include <ufbx.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>

// FBX rigs and clips through ufbx v0.23.1 (MIT), loaded with the options Tide's own
// FBX importer uses (compiler/src/fbx_files.c): right-handed Y-up metres, with the
// conversion folded into the top nodes' transforms, and geometry/inheritance
// helpers as extra nodes. The node names, parents and local transforms read here
// are therefore the ones a Tide game binds a clip to.
namespace motion {
namespace {
struct SceneDeleter { void operator()(ufbx_scene* scene) const { ufbx_free_scene(scene); } };
std::string text(ufbx_string s) { return std::string(s.data,s.length); }
Transform transform(const ufbx_transform& t,float* scale,const std::string& name) {
    const auto& s=t.scale;
    float uniform=static_cast<float>(s.x);
    for(double value:{s.x,s.y,s.z})if(!std::isfinite(value)||value<=0||std::abs(static_cast<float>(value)-uniform)>1e-4f*uniform)
        throw std::runtime_error("FBX node '"+name+"' has nonuniform or negative scale; bake scale before export");
    *scale=uniform;
    Transform out{{static_cast<float>(t.translation.x),static_cast<float>(t.translation.y),static_cast<float>(t.translation.z)},
        normalized({static_cast<float>(t.rotation.x),static_cast<float>(t.rotation.y),static_cast<float>(t.rotation.z),static_cast<float>(t.rotation.w)})};
    for(float v:{out.p.x,out.p.y,out.p.z})if(!std::isfinite(v))throw std::runtime_error("FBX node '"+name+"' has a nonfinite position");
    return out;
}
}
Clip read_fbx(const fs::path& path,const std::string& selected,bool rest_only) {
    if(fs::file_size(path)>512ULL*1024*1024)throw std::runtime_error("FBX exceeds the 512 MiB input limit");
    std::ifstream input(path,std::ios::binary);
    std::vector<char> bytes(static_cast<size_t>(fs::file_size(path)));
    if(!input.read(bytes.data(),static_cast<std::streamsize>(bytes.size())))throw std::runtime_error("Cannot read "+path.string());
    ufbx_load_opts opts{};
    opts.file_format=UFBX_FILE_FORMAT_FBX;
    opts.target_axes=ufbx_axes_right_handed_y_up;
    opts.target_unit_meters=1.0;
    opts.space_conversion=UFBX_SPACE_CONVERSION_ADJUST_TRANSFORMS;
    opts.geometry_transform_handling=UFBX_GEOMETRY_TRANSFORM_HANDLING_HELPER_NODES;
    opts.inherit_mode_handling=UFBX_INHERIT_MODE_HANDLING_HELPER_NODES;
    opts.node_depth_limit=128;
    opts.ignore_embedded=true;
    opts.index_error_handling=UFBX_INDEX_ERROR_HANDLING_CLAMP;
    opts.temp_allocator.memory_limit=512ULL*1024*1024;
    opts.result_allocator.memory_limit=512ULL*1024*1024;
    ufbx_error error{};
    std::unique_ptr<ufbx_scene,SceneDeleter> scene(ufbx_load_memory(bytes.data(),bytes.size(),&opts,&error));
    if(!scene)throw std::runtime_error("Not an FBX that decodes ("+text(error.description)+")");
    if(scene->nodes.count>1024)throw std::runtime_error("Too many FBX nodes");
    Clip c;c.name=path.stem().string();
    // As for GLB: a character's rig is its skin bones and their ancestors; a file
    // without skins (an animation or skeleton export) uses every node. The unnamed
    // scene root, which Tide ignores when binding, is left out.
    std::vector<const ufbx_node*> nodes;
    std::vector<int> index(scene->nodes.count,-1);
    std::vector<unsigned char> state(scene->nodes.count);
    auto append=[&](auto&& self,const ufbx_node* node)->void {
        if(!node||node->is_root)return;
        size_t id=node->typed_id;
        if(state[id]==2)return;
        if(state[id]==1)throw std::runtime_error("FBX skeleton contains a parent cycle");
        state[id]=1;
        self(self,node->parent);
        auto name=text(node->name);float scale=1;
        Transform rest=transform(node->local_transform,&scale,name);
        int parent=node->parent&&!node->parent->is_root?index[node->parent->typed_id]:-1;
        index[id]=static_cast<int>(c.joints.size());
        c.joints.push_back({name,parent,rest,scale});
        nodes.push_back(node);
        state[id]=2;
    };
    bool skinned=false;
    if(rest_only)for(const ufbx_skin_deformer* skin:scene->skin_deformers)for(const ufbx_skin_cluster* cluster:skin->clusters)if(cluster->bone_node) {append(append,cluster->bone_node);skinned=true;}
    if(!skinned)for(const ufbx_node* node:scene->nodes)if(!node->mesh||!rest_only)append(append,node);
    if(c.joints.empty())throw std::runtime_error("FBX has no skeleton nodes");
    if(rest_only) {
        for(const auto& joint:c.joints)c.poses.push_back(joint.rest);
        c.validate();(void)world_pose(c,c.frame(0));return c;
    }
    const ufbx_anim_stack* stack=nullptr;
    if(selected.empty()) {
        if(scene->anim_stacks.count==0)throw std::runtime_error("FBX has no animation; use inspect --skeleton to list rig bones");
        if(scene->anim_stacks.count!=1)throw std::runtime_error("FBX must have one animation take, or select one with --clip/--from-clip/--to-clip");
        stack=scene->anim_stacks.data[0];
    } else {
        for(const ufbx_anim_stack* candidate:scene->anim_stacks)if(text(candidate->name)==selected) {
            if(stack)throw std::runtime_error("Ambiguous animation name");
            stack=candidate;
        }
        if(!stack)throw std::runtime_error("Animation not found: "+selected);
    }
    double length=stack->time_end-stack->time_begin;
    if(!std::isfinite(length)||length<0||length>600)throw std::runtime_error("Animation must be at most 600 seconds");
    double frame_time=length*30,nearest=std::round(frame_time);
    size_t frames=static_cast<size_t>(std::abs(frame_time-nearest)<0.002?nearest:std::ceil(frame_time))+1;
    if(frames>18000)throw std::runtime_error("Animation exceeds 18000 frames");
    c.poses.reserve(frames*c.joints.size());
    for(size_t f=0;f<frames;++f)for(size_t j=0;j<nodes.size();++j) {
        double time=std::min(stack->time_begin+static_cast<double>(f)/30,stack->time_end);
        float scale=1;auto pose=transform(ufbx_evaluate_transform(stack->anim,nodes[j],time),&scale,c.joints[j].name);
        if(std::abs(scale-c.joints[j].scale)>1e-4f*c.joints[j].scale)throw std::runtime_error("Animated scale on FBX node '"+c.joints[j].name+"' is unsupported");
        c.poses.push_back(pose);
    }
    c.validate();
    return c;
}
}
