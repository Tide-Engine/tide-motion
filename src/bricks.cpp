// Native adaptation of NVIDIA MotionBricks' SOMA pose kernels.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "motion.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace motion {
namespace {
// MotionBricks SOMA model space uses the BONES-SEED BVH joint frames (bone along
// local +X); canonical SOMA-30 uses Kimodo's world-aligned standard T-pose. Kimodo's
// from_standard_tpose converts global rotations as G_bvh = G_std * offset, with the
// per-joint offsets of nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954
// kimodo/assets/skeletons/somaskel77/standard_t_pose_global_offsets_rots.p (Apache-2.0),
// here as x,y,z,w for the SOMA-30 joints. Kimodo's somaskel77 bvh_joints reproduce the
// MotionBricks SOMA BVH rest skeleton exactly, so this is that model's convention.
const std::array<Quat,30> bvh_frames{{
    {0.49916838f,0.500830319f,0.49916838f,0.500830289f},{0.498952559f,0.501045314f,0.498952598f,0.501045297f},
    {0.472374587f,0.526177078f,0.472374651f,0.526177036f},{0.499114893f,0.500883608f,0.499115002f,0.500883515f},
    {0.567049458f,0.422439218f,0.567049667f,0.422439043f},{0.570900759f,0.417219654f,0.570901186f,0.417219345f},
    {0.469715252f,0.528552192f,0.469716225f,0.52855174f},{0.469714928f,0.528552557f,0.46971655f,0.528551375f},
    {0.533349433f,0.498629805f,0.49914469f,0.466649062f},{0.499143178f,0.466650337f,0.533350996f,0.498628417f},
    {0.696101723f,0.124267839f,-0.124267847f,0.696101666f},{0.70710684f,3.17987474e-05f,-3.18454965e-05f,0.707106806f},
    {0.70710684f,-3.41346055e-05f,3.40878511e-05f,0.707106806f},{0.764426194f,-2.19231872e-05f,5.26655595e-05f,0.644711365f},
    {1.00000003f,-1.8291871e-08f,4.41242087e-08f,-2.89069215e-09f},{0.684189884f,-0.0385695742f,-0.0509397119f,0.726499716f},
    {-0.696101811f,-0.124267914f,-0.124267911f,0.696101664f},{-0.707106939f,-3.19401913e-05f,-3.19191065e-05f,0.707106792f},
    {-0.707106981f,3.40850479e-05f,3.41061244e-05f,0.707106792f},{-0.644711513f,5.26195441e-05f,2.18605545e-05f,0.764426135f},
    {4.16044856e-09f,-9.23542418e-09f,-2.36186235e-08f,1.00000007f},{-0.726501855f,-0.050941531f,0.0385734824f,0.684187417f},
    {-0.495335f,0.504621965f,-0.495335f,0.504621994f},{-0.478981244f,0.520170195f,-0.478981244f,0.520170224f},
    {0.695349489f,-0.128410077f,0.695349489f,-0.128410055f},{0.699539127f,-0.099537014f,0.70057166f,-0.0996839473f},
    {0.504622192f,0.495334717f,0.504622163f,0.495334717f},{0.520170185f,0.478981223f,0.520170156f,0.478981223f},
    {0.128409981f,0.695349483f,0.128410009f,0.695349451f},{0.0996850378f,0.700571722f,0.0995380792f,0.699538681f}
}};
float heading(const std::vector<Transform>& pose) {
    Vec3 across=pose[26].p-pose[22].p;across.y=0;
    if(dot(across,across)<=1e-12f)return 0;
    Vec3 forward=cross({0,1,0},normalized(across));
    return std::atan2(forward.x,forward.z);
}
}
BricksRequest bricks_tokens(int tokens) {
    if(tokens==0)return {};
    if(tokens<6||tokens>16)throw std::runtime_error("Token count must be 6..16");
    return {tokens,tokens,4};
}
BricksRequest parse_bricks_tokens(const std::string& text) {
    auto whole=[](const std::string& t) {
        if(t.empty()||t.size()>2||!std::all_of(t.begin(),t.end(),[](char c){return c>='0'&&c<='9';}))throw std::runtime_error("--tokens is N or MIN..MAX, 6..16");
        return std::stoi(t);
    };
    auto dots=text.find("..");
    BricksRequest r=dots==std::string::npos?BricksRequest{whole(text),whole(text),4}:BricksRequest{whole(text.substr(0,dots)),whole(text.substr(dots+2)),4};
    if(r.min_tokens<6||r.max_tokens>16||r.min_tokens>r.max_tokens)throw std::runtime_error("--tokens is N or MIN..MAX, 6..16");
    return r;
}
BricksInputs encode_bricks(const Clip& c,const std::array<std::vector<Transform>,8>& guides,int tokens) {
    return encode_bricks(c,guides,bricks_tokens(tokens));
}
BricksInputs encode_bricks(const Clip& c,const std::array<std::vector<Transform>,8>& guides,const BricksRequest& request) {
    require_soma(c);
    if(request.min_tokens<6||request.max_tokens>16||request.min_tokens>request.max_tokens)throw std::runtime_error("Token range must be within 6..16");
    if(request.target_poses<1||request.target_poses>4)throw std::runtime_error("Target poses must be 1..4");
    BricksInputs in;
    std::array<std::vector<Transform>,8> world;
    for(size_t f=0;f<8;++f)world[f]=world_pose(c,guides[f]);
    in.alignment={{world[3][0].p.x,0,world[3][0].p.z},yaw(heading(world[3])-1.5707963267948966f)};
    Transform undo=inverse(in.alignment);
    for(auto& pose:world)for(auto& t:pose)t=undo*t;
    for(size_t f=0;f<8;++f) {
        size_t next=f==3?f:std::min(f+1,size_t(7));
        Vec3 hips=world[f][0].p,velocity=(world[next][0].p-hips)*30;
        float angle=heading(world[f]),delta=heading(world[next])-angle;
        delta=std::atan2(std::sin(delta),std::cos(delta));
        const float root[]={hips.x,hips.y,hips.z,std::cos(angle),std::sin(angle)};
        std::copy_n(root,5,in.root.data()+f*5);
        const float local[]={delta*30,velocity.x,velocity.z,hips.y};
        std::copy_n(local,4,in.local_root.data()+f*4);
        for(size_t j=0;j<30;++j) {
            if(j) { Vec3 p=world[f][j].p-Vec3{hips.x,0,hips.z};size_t base=f*267+(j-1)*3;in.pose[base]=p.x;in.pose[base+1]=p.y;in.pose[base+2]=p.z; }
            Quat model=normalized(world[f][j].q*bvh_frames[j]);
            Vec3 x=rotate(model,{1,0,0}),y=rotate(model,{0,1,0});
            const float rot[]={x.x,x.y,x.z,y.x,y.y,y.z};std::copy_n(rot,6,in.pose.data()+f*267+87+j*6);
        }
        // Upstream write_motionbricks_conditions_kernel: context plus the first
        // target_poses target slots are given; root derivatives are unknown at the
        // last given context and target slot.
        size_t given=4+static_cast<size_t>(request.target_poses);
        in.has_root[f]=in.has_pose[f]=f<given;
        in.has_local_root[f]=f<given&&f!=3&&f!=given-1;
    }
    for(int i=0;i<11;++i)in.allowed[static_cast<size_t>(i)]=i+6>=request.min_tokens&&i+6<=request.max_tokens;
    return in;
}
Clip decode_bricks(const Clip& source,const BricksInputs& in,std::span<const float> features,int tokens,
    std::span<const Transform> start,std::span<const Transform> end,SeamCorrection* correction) {
    if(tokens<6||tokens>16||!in.allowed[static_cast<size_t>(tokens-6)]||features.size()!=64*366||start.size()!=30||end.size()!=30)throw std::runtime_error("MotionBricks returned an invalid tensor shape or duration");
    for(float f:features)if(!std::isfinite(f))throw std::runtime_error("MotionBricks returned nonfinite motion");
    Clip out;out.joints=source.joints;out.name="transition";
    size_t frames=static_cast<size_t>(4*tokens-6);
    for(size_t f=0;f<frames;++f) {
        const float* row=features.data()+(f+3)*366;
        std::array<Quat,30> rotations;
        for(size_t j=0;j<30;++j) { const float* r=row+92+j*6;rotations[j]=normalized(rotation6({r[0],r[1],r[2]},{r[3],r[4],r[5]})*inverse(bvh_frames[j])); }
        out.poses.push_back(in.alignment*Transform{{row[0],row[1],row[2]},rotations[0]});
        for(size_t j=1;j<30;++j)out.poses.push_back({out.joints[j].rest.p,normalized(inverse(rotations[static_cast<size_t>(out.joints[j].parent)])*rotations[j])});
    }
    std::array<Transform,30> a,b;
    a[0]={start[0].p-out.frame(0)[0].p,normalized(inverse(out.frame(0)[0].q)*start[0].q)};
    b[0]={end[0].p-out.frame(frames-1)[0].p,normalized(inverse(out.frame(frames-1)[0].q)*end[0].q)};
    for(size_t j=1;j<30;++j) { a[j]=inverse(out.frame(0)[j])*start[j];b[j]=inverse(out.frame(frames-1)[j])*end[j]; }
    if(correction) {
        correction->root={a[0].p,b[0].p};
        for(size_t j=0;j<30;++j)for(size_t e=0;e<2;++e) {
            const Quat& q=(e?b:a)[j].q;
            float deg=2*std::acos(std::min(1.0f,std::abs(q.w)))*57.29577951308232f;
            if(deg>correction->joint_degrees[e]) {correction->joint_degrees[e]=deg;correction->joint[e]=j;}
        }
    }
    for(size_t f=0;f<frames;++f) {
        float t=static_cast<float>(f)/static_cast<float>(frames-1);
        auto pose=out.frame(f);
        pose[0].p=pose[0].p+a[0].p*(1-t)+b[0].p*t;
        pose[0].q=normalized(pose[0].q*slerp({},a[0].q,1-t)*slerp({},b[0].q,t));
        for(size_t j=1;j<30;++j)pose[j]=pose[j]*blend({},a[j],1-t)*blend({},b[j],t);
    }
    // Preserve the requested seam values exactly, including quaternion signs.
    std::copy(start.begin(),start.end(),out.frame(0).begin());
    std::copy(end.begin(),end.end(),out.frame(frames-1).begin());
    out.validate();return out;
}
FootContacts decode_bricks_contacts(std::span<const float> features,int tokens) {
    if(tokens<6||tokens>16||features.size()!=64*366)throw std::runtime_error("MotionBricks returned an invalid tensor shape or duration");
    FootContacts out(static_cast<size_t>(4*tokens-6));
    for(size_t f=0;f<out.size();++f)for(size_t c=0;c<4;++c) {
        float v=features[(f+3)*366+362+c];
        if(!std::isfinite(v))throw std::runtime_error("MotionBricks returned nonfinite foot contacts");
        out[f][c]=std::clamp(v,0.0f,1.0f);
    }
    return out;
}
std::vector<FootContact> planted_contacts(const FootContacts& contacts,float threshold,size_t offset) {
    std::vector<FootContact> out;
    for(unsigned foot=0;foot<2;++foot) {
        size_t f=0;
        while(f<contacts.size()) {
            if(contacts[f][foot*2]<threshold) {++f;continue;}
            size_t start=f;while(f<contacts.size()&&contacts[f][foot*2]>=threshold)++f;
            if(f-start>=3)out.push_back({foot,offset+start,offset+f-1});
        }
    }
    std::sort(out.begin(),out.end(),[](const FootContact& a,const FootContact& b){return a.start!=b.start?a.start<b.start:a.foot<b.foot;});
    return out;
}
struct BricksModel::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"tide-motion"};
    Ort::Session session{nullptr};
    Impl(const fs::path& model,unsigned threads) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(static_cast<int>(threads?threads:std::max(1u,std::thread::hardware_concurrency()/2)));
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::cerr<<"Loading MotionBricks SOMA (native ONNX CPU inference)...\n";
        session=Ort::Session(env,model.c_str(),opts);
    }
};
BricksModel::BricksModel(const fs::path& model,unsigned threads):impl_(std::make_unique<Impl>(model,threads)) { }
BricksModel::~BricksModel()=default;
Clip BricksModel::generate(const Clip& a,const Clip& b,int tokens) {
    return generate(a,b,bricks_tokens(tokens)).clip;
}
Bridge BricksModel::generate(const Clip& a,const Clip& b,const BricksRequest& request) {
    require_soma(a); require_soma(b);
    if(a.frames()<4||b.frames()<4)throw std::runtime_error("Each clip must contain at least four frames");
    std::array<std::vector<Transform>,8> guides;
    // Unused target slots repeat the last given pose (upstream pad_token_poses_kernel).
    for(size_t f=0;f<4;++f) {
        guides[f]=sample(a,a.duration()-static_cast<float>(3-f)/30);
        guides[f+4]=sample(b,static_cast<float>(std::min<int>(static_cast<int>(f),request.target_poses-1))/30);
    }
    auto in=encode_bricks(a,guides,request);
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    std::array<int64_t,3> root_shape{1,8,5},local_shape{1,8,4},pose_shape{1,8,267};
    std::array<int64_t,2> mask_shape{1,8},allowed_shape{1,11};
    inputs.push_back(Ort::Value::CreateTensor<float>(memory,in.root.data(),in.root.size(),root_shape.data(),3));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory,in.local_root.data(),in.local_root.size(),local_shape.data(),3));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory,in.pose.data(),in.pose.size(),pose_shape.data(),3));
    for(auto* mask:{&in.has_root,&in.has_local_root,&in.has_pose})inputs.push_back(Ort::Value::CreateTensor<int32_t>(memory,mask->data(),mask->size(),mask_shape.data(),2));
    inputs.push_back(Ort::Value::CreateTensor<int32_t>(memory,in.allowed.data(),in.allowed.size(),allowed_shape.data(),2));
    const char* names[]={"global_root_values","local_root_values","local_poses","has_global_root_values","has_local_root_values","has_local_poses","allowed_pred_num_tokens"};
    const char* outputs[]={"pred_global_motions","pred_num_tokens"};
    std::cerr<<"Generating transition...\n";
    auto result=impl_->session.Run(Ort::RunOptions{nullptr},names,inputs.data(),inputs.size(),outputs,2);
    if(result.size()!=2||result[0].GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,64,366}||result[1].GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1})throw std::runtime_error("Unexpected MotionBricks output contract");
    int tokens=result[1].GetTensorData<int32_t>()[0];
    std::span<const float> features{result[0].GetTensorData<float>(),64*366};
    Bridge out;out.tokens=tokens;
    out.clip=decode_bricks(a,in,features,tokens,guides[3],guides[4],&out.correction);
    out.contacts=decode_bricks_contacts(features,tokens);
    return out;
}
Clip connect(const Clip& a,const Clip& b,const fs::path& model,int tokens) {
    BricksModel engine(model);return engine.generate(a,b,tokens);
}
}
