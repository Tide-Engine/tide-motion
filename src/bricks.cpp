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
float heading(const std::vector<Transform>& pose) {
    Vec3 across=pose[26].p-pose[22].p;across.y=0;
    if(dot(across,across)<=1e-12f)return 0;
    Vec3 forward=cross({0,1,0},normalized(across));
    return std::atan2(forward.x,forward.z);
}
}
BricksInputs encode_bricks(const Clip& c,const std::array<std::vector<Transform>,8>& guides,int tokens) {
    require_soma(c);
    if(tokens!=0&&(tokens<6||tokens>16))throw std::runtime_error("Token count must be 6..16");
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
            Vec3 x=rotate(world[f][j].q,{1,0,0}),y=rotate(world[f][j].q,{0,1,0});
            const float rot[]={x.x,x.y,x.z,y.x,y.y,y.z};std::copy_n(rot,6,in.pose.data()+f*267+87+j*6);
        }
        in.has_root[f]=in.has_pose[f]=1;in.has_local_root[f]=(f==3||f==7)?0:1;
    }
    for(int i=0;i<11;++i)in.allowed[static_cast<size_t>(i)]=tokens?i+6==tokens:(i+6>=8&&i+6<=14);
    return in;
}
Clip decode_bricks(const Clip& source,const BricksInputs& in,std::span<const float> features,int tokens,
    std::span<const Transform> start,std::span<const Transform> end) {
    if(tokens<6||tokens>16||!in.allowed[static_cast<size_t>(tokens-6)]||features.size()!=64*366||start.size()!=30||end.size()!=30)throw std::runtime_error("MotionBricks returned an invalid tensor shape or duration");
    for(float f:features)if(!std::isfinite(f))throw std::runtime_error("MotionBricks returned nonfinite motion");
    Clip out;out.joints=source.joints;out.name="transition";
    size_t frames=static_cast<size_t>(4*tokens-6);
    for(size_t f=0;f<frames;++f) {
        const float* row=features.data()+(f+3)*366;
        std::array<Quat,30> rotations;
        for(size_t j=0;j<30;++j) { const float* r=row+92+j*6;rotations[j]=rotation6({r[0],r[1],r[2]},{r[3],r[4],r[5]}); }
        out.poses.push_back(in.alignment*Transform{{row[0],row[1],row[2]},rotations[0]});
        for(size_t j=1;j<30;++j)out.poses.push_back({out.joints[j].rest.p,normalized(inverse(rotations[static_cast<size_t>(out.joints[j].parent)])*rotations[j])});
    }
    std::array<Transform,30> a,b;
    a[0]={start[0].p-out.frame(0)[0].p,normalized(inverse(out.frame(0)[0].q)*start[0].q)};
    b[0]={end[0].p-out.frame(frames-1)[0].p,normalized(inverse(out.frame(frames-1)[0].q)*end[0].q)};
    for(size_t j=1;j<30;++j) { a[j]=inverse(out.frame(0)[j])*start[j];b[j]=inverse(out.frame(frames-1)[j])*end[j]; }
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
    require_soma(a); require_soma(b);
    if(a.frames()<4||b.frames()<4)throw std::runtime_error("Each clip must contain at least four frames");
    std::array<std::vector<Transform>,8> guides;
    for(size_t f=0;f<4;++f) { guides[f]=sample(a,a.duration()-static_cast<float>(3-f)/30);guides[f+4]=sample(b,static_cast<float>(f)/30); }
    auto in=encode_bricks(a,guides,tokens);
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
    return decode_bricks(a,in,{result[0].GetTensorData<float>(),64*366},result[1].GetTensorData<int32_t>()[0],guides[3],guides[4]);
}
Clip connect(const Clip& a,const Clip& b,const fs::path& model,int tokens) {
    BricksModel engine(model);return engine.generate(a,b,tokens);
}
}
