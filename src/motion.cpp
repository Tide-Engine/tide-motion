#include "motion.hpp"
#include "skeleton.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace motion {
Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
Vec3 operator*(Vec3 a, float b) { return {a.x*b,a.y*b,a.z*b}; }
float dot(Vec3 a,Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
Vec3 cross(Vec3 a,Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
Vec3 normalized(Vec3 a) {
    float length=std::sqrt(dot(a,a));
    if(!std::isfinite(length)||length<1e-8f) throw std::runtime_error("Degenerate motion rotation axis");
    return a*(1/length);
}
Quat normalized(Quat q) {
    float n=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
    if(!std::isfinite(n)||n<1e-8f) throw std::runtime_error("Invalid motion quaternion");
    return {q.x/n,q.y/n,q.z/n,q.w/n};
}
Quat inverse(Quat q) { q=normalized(q); return {-q.x,-q.y,-q.z,q.w}; }
Quat operator*(Quat a,Quat b) {
    return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
        a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
        a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
        a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};
}
Vec3 rotate(Quat q,Vec3 v) {
    Vec3 u{q.x,q.y,q.z};
    return v+cross(u,v)*(2*q.w)+cross(u,cross(u,v))*2;
}
Quat slerp(Quat a,Quat b,float t) {
    a=normalized(a); b=normalized(b);
    float d=a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
    if(d<0) { b={-b.x,-b.y,-b.z,-b.w}; d=-d; }
    float x=1-t,y=t;
    if(d<0.9995f) { float angle=std::acos(std::clamp(d,-1.0f,1.0f)); x=std::sin((1-t)*angle)/std::sin(angle); y=std::sin(t*angle)/std::sin(angle); }
    return normalized({a.x*x+b.x*y,a.y*x+b.y*y,a.z*x+b.z*y,a.w*x+b.w*y});
}
Quat yaw(float a) { return {0,std::sin(a/2),0,std::cos(a/2)}; }
Quat rotation6(Vec3 x,Vec3 y) {
    x=normalized(x); Vec3 z=normalized(cross(x,y)); y=cross(z,x);
    const float m[3][3]={{x.x,y.x,z.x},{x.y,y.y,z.y},{x.z,y.z,z.z}};
    Quat q;
    float trace=m[0][0]+m[1][1]+m[2][2];
    if(trace>0) { float s=2*std::sqrt(trace+1); q={(m[2][1]-m[1][2])/s,(m[0][2]-m[2][0])/s,(m[1][0]-m[0][1])/s,s/4}; }
    else {
        int i=0; if(m[1][1]>m[i][i])i=1; if(m[2][2]>m[i][i])i=2;
        int j=(i+1)%3,k=(i+2)%3;
        float s=2*std::sqrt(std::max(0.0f,1+m[i][i]-m[j][j]-m[k][k]));
        if(s<1e-8f) throw std::runtime_error("Degenerate motion rotation matrix");
        float v[3]{}; v[i]=s/4; v[j]=(m[j][i]+m[i][j])/s; v[k]=(m[k][i]+m[i][k])/s;
        q={v[0],v[1],v[2],(m[k][j]-m[j][k])/s};
    }
    return normalized(q);
}
Transform operator*(Transform a,Transform b) { return {a.p+rotate(a.q,b.p),normalized(a.q*b.q)}; }
Transform inverse(Transform t) { Quat q=inverse(t.q); return {rotate(q,t.p*-1),q}; }
Transform blend(Transform a,Transform b,float t) { return {a.p*(1-t)+b.p*t,slerp(a.q,b.q,t)}; }
size_t Clip::frames() const { return joints.empty()?0:poses.size()/joints.size(); }
float Clip::duration() const { return frames()>0?static_cast<float>(frames()-1)/fps:0; }
std::span<Transform> Clip::frame(size_t f) { if(f>=frames())throw std::out_of_range("frame"); return {poses.data()+f*joints.size(),joints.size()}; }
std::span<const Transform> Clip::frame(size_t f) const { if(f>=frames())throw std::out_of_range("frame");return {poses.data()+f*joints.size(),joints.size()}; }
void Clip::validate() const {
    if(joints.empty()||joints.size()>1024||motion_root>=joints.size()||poses.empty()||poses.size()%joints.size()||frames()>18000||!std::isfinite(fps)||fps<=0)
        throw std::runtime_error("Invalid clip size or frame rate");
    for(size_t i=0;i<joints.size();++i) {
        if(joints[i].parent< -1||joints[i].parent>=static_cast<int>(i))throw std::runtime_error("Skeleton must be ordered parent before child");
        auto t=joints[i].rest;
        if(!std::isfinite(t.p.x)||!std::isfinite(t.p.y)||!std::isfinite(t.p.z))throw std::runtime_error("Nonfinite rest position");
        (void)normalized(t.q);
        if(!std::isfinite(joints[i].scale)||joints[i].scale<=0)throw std::runtime_error("Joint scale must be finite and positive");
    }
    for(auto t:poses) { if(!std::isfinite(t.p.x)||!std::isfinite(t.p.y)||!std::isfinite(t.p.z))throw std::runtime_error("Nonfinite animation position"); (void)normalized(t.q); }
}
std::vector<Joint> soma() {
    const auto& s=kimodo::detail::soma30_spec;
    std::vector<Joint> result;
    for(size_t i=0;i<s.joints();++i) { auto o=s.offsets[i]; result.push_back({std::string(s.names[i]),s.parents[i],{{o[0],o[1],o[2]}, {}}}); }
    return result;
}
void require_soma(const Clip& c) {
    c.validate(); auto rig=soma();
    if(c.joints.size()!=rig.size())throw std::runtime_error("MotionBricks currently requires the canonical 30-joint SOMA rig; retarget other rigs before connecting");
    for(size_t i=0;i<rig.size();++i) {
        if(c.joints[i].name!=rig[i].name||c.joints[i].parent!=rig[i].parent)throw std::runtime_error("Clip does not use the canonical SOMA names and hierarchy");
        if(std::abs(c.joints[i].scale-1)>1e-5f)throw std::runtime_error("Canonical SOMA joint scale must be one");
        if(i&&dot(c.joints[i].rest.p-rig[i].rest.p,c.joints[i].rest.p-rig[i].rest.p)>1e-6f)throw std::runtime_error("Clip has different body proportions; retarget to canonical SOMA first");
        for(size_t f=0;f<c.frames();++f)if(i&&dot(c.frame(f)[i].p-c.joints[i].rest.p,c.frame(f)[i].p-c.joints[i].rest.p)>1e-8f)
            throw std::runtime_error("Animated bone lengths are unsupported; bake/retarget to a rigid SOMA skeleton first");
    }
}
std::vector<Transform> world_pose(const Clip& c,std::span<const Transform> pose) {
    if(pose.size()!=c.joints.size())throw std::runtime_error("Pose has the wrong joint count");
    std::vector<Transform> out(pose.begin(),pose.end());
    std::vector<float> scales(out.size());
    for(size_t j=0;j<out.size();++j) {
        scales[j]=c.joints[j].scale;
        if(c.joints[j].parent>=0) {
            auto parent=static_cast<size_t>(c.joints[j].parent);
            out[j].p=out[j].p*scales[parent];
            out[j]=out[parent]*out[j];
            scales[j]*=scales[parent];
        }
        if(!std::isfinite(scales[j])||scales[j]<1e-8f||scales[j]>1e8f)throw std::runtime_error("Accumulated skeleton scale is out of range; bake transforms before export");
    }
    return out;
}
std::vector<Transform> sample(const Clip& c,float time) {
    if(!std::isfinite(time))throw std::runtime_error("Sample time must be finite");
    float f=std::clamp(time*c.fps,0.0f,static_cast<float>(c.frames()-1));
    size_t lo=static_cast<size_t>(f),hi=std::min(lo+1,c.frames()-1);
    std::vector<Transform> result;
    for(size_t j=0;j<c.joints.size();++j)result.push_back(blend(c.frame(lo)[j],c.frame(hi)[j],f-static_cast<float>(lo)));
    return result;
}
Clip join(const Clip& a,const Clip& bridge,const Clip& b) {
    Clip out=a; out.name="connected";
    if(a.joints.size()!=bridge.joints.size()||a.joints.size()!=b.joints.size())throw std::runtime_error("Cannot join different skeletons");
    // The bridge includes both endpoints; avoid repeating seam frames.
    out.poses.insert(out.poses.end(),bridge.poses.begin()+static_cast<std::ptrdiff_t>(a.joints.size()),bridge.poses.end());
    out.poses.insert(out.poses.end(),b.poses.begin()+static_cast<std::ptrdiff_t>(b.joints.size()),b.poses.end());
    out.validate(); return out;
}
}
