/*
 * Native port of NVIDIA Kimodo MotionCorrection (Apache-2.0):
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Source: nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954,
 * MotionCorrection/src/cpp/AnimProcessing/{TrajectoryCorrector,InverseKinematics,Utility}.cpp,
 * BindingsPython.cpp and kimodo/postprocess.py. Modified for Tide Motion: scalar C++ on
 * Tide's math types (no Eigen, SIMD or Python), a banded LDL^T solve in place of
 * Eigen::SparseLU, upstream's reversed (DirectXMath) products rewritten in Hamilton order.
 */
#include "motion.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <ostream>
#include <stdexcept>

namespace motion {
namespace {
using Pose=std::vector<Transform>;
constexpr float pos_weight=0.001f,vel_weight=1.0f,acc_weight=10.0f;
// SOMA-30 indices used by upstream's SOMA working rig.
constexpr int left_hand=13,right_hand=19,left_foot=24,right_foot=28;

Transform global(const std::vector<int>& parents,int index,const Pose& pose) {
    Transform world=pose[static_cast<size_t>(index)];
    for(int p=parents[static_cast<size_t>(index)];p>=0;p=parents[static_cast<size_t>(p)])world=pose[static_cast<size_t>(p)]*world;
    return world;
}
Quat axis_angle(Vec3 axis,float angle) {float s=std::sin(angle/2);return {axis.x*s,axis.y*s,axis.z*s,std::cos(angle/2)};}
float length(Vec3 v) {return std::sqrt(dot(v,v));}
// Quaternion::FromRotationBetweenVectors.
Quat between(Vec3 from,Vec3 to) {
    from=normalized(from);to=normalized(to);
    float d=dot(from,to);
    if(d>=1-1e-6f)return {};
    if(d<=1e-6f-1)return normalized(Quat{-from.z,from.y,from.x,0});
    Vec3 c=cross(from,to);
    float l=std::sqrt(dot(c,c)+d*d);
    return normalized(Quat{c.x,c.y,c.z,d+l});
}
// Quaternion::LookRotation: z axis forward, y axis up (orthogonalized).
Quat look_rotation(Vec3 forward,Vec3 up) {
    Vec3 t=normalized(cross(up,forward));
    return rotation6(t,cross(forward,t));
}
float vector_angle(Vec3 a,Vec3 b) {a=normalized(a);b=normalized(b);return std::atan2(length(cross(a,b)),dot(a,b));}
float cosine_rule(float left,float right,float across) {
    float v=(right*right+left*left-across*across)/(2*left*right);
    return std::acos(std::clamp(v,-1.0f,1.0f));
}
void two_bone_ik(Pose& pose,int c,Vec3 target,const std::vector<int>& parents,Vec3 hint) {
    int b=parents[static_cast<size_t>(c)];if(b<0)return;
    int a=parents[static_cast<size_t>(b)];if(a<0)return;
    Transform above;int p=parents[static_cast<size_t>(a)];if(p>=0)above=global(parents,p,pose);
    Transform aw=above*pose[static_cast<size_t>(a)],bw=aw*pose[static_cast<size_t>(b)],cw=bw*pose[static_cast<size_t>(c)];
    Vec3 pa=aw.p,pb=bw.p,pc=cw.p,t=target;
    const float eps=0.0001f;
    float l_ab=length(pb-pa),l_bc=length(pc-pb),l_at=std::clamp(length(pa-t),eps,(l_ab+l_bc)*0.999f);
    float bac=vector_angle(pa-pb,pa-pc),abc=vector_angle(pb-pa,pb-pc);
    if(l_ab<eps||l_bc<eps||l_at<eps)return;
    float bac_want=cosine_rule(l_ab,l_at,l_bc),abc_want=cosine_rule(l_ab,l_bc,l_at);
    Vec3 axis=cross(pc-pa,(bw.p+rotate(bw.q,hint))-pa);float l=length(axis);
    axis=l==0?Vec3{0,0,1}:axis*(1/l);
    auto& qb=pose[static_cast<size_t>(b)].q;qb=normalized(qb*axis_angle(rotate(inverse(bw.q),axis),abc_want-abc));
    auto& qa=pose[static_cast<size_t>(a)].q;qa=normalized(qa*axis_angle(rotate(inverse(aw.q),axis),bac_want-bac));
    aw=above*pose[static_cast<size_t>(a)];
    // Upstream keeps the chain direction in stage one, then aims a's old a->c ray at t.
    qa=normalized(qa*between(rotate(inverse(aw.q),pc-pa),rotate(inverse(aw.q),target-pa)));
}
void one_bone_ik(Pose& pose,int b,Vec3 target,const std::vector<int>& parents) {
    int a=parents[static_cast<size_t>(b)];if(a<0)return;
    Transform aw=global(parents,a,pose),bw=aw*pose[static_cast<size_t>(b)];
    auto& qa=pose[static_cast<size_t>(a)].q;
    qa=normalized(qa*between(rotate(inverse(aw.q),bw.p-aw.p),rotate(inverse(aw.q),target-aw.p)));
}

// TrajectoryCorrector: minimizes pos|x-x_t|^2 + vel|V(x-x_t)|^2 + acc|A(x-x_t)|^2 with
// pinned frames (margin 0) and ADMM margin balls (margin > 0); margin < 0 is free.
// Upstream's velocity weights never reach V (multVelWeights discards its triplets), so
// none are taken here.
class Corrector {
public:
    Corrector(const std::vector<double>& margins,float pw,float vw,float aw,int iters=100):n_(margins.size()),iters_(iters) {
        d_.assign(n_,pw);e_.assign(n_,0);f_.assign(n_,0);
        for(size_t i=0;i+1<n_;++i) {d_[i]+=vw;d_[i+1]+=vw;e_[i]-=vw;}
        for(size_t i=0;i+2<n_;++i) {
            const size_t c[3]={i,i+1,i+2};const double w[3]={-1,2,-1};
            for(int a=0;a<3;++a)for(int b=a;b<3;++b) {
                double v=aw*w[a]*w[b];size_t gap=c[b]-c[a];
                if(gap==0)d_[c[a]]+=v;else if(gap==1)e_[c[a]]+=v;else f_[c[a]]+=v;
            }
        }
        double diag=0;for(double v:d_)diag=std::max(diag,v);
        rho_=static_cast<double>(0.5f*std::sqrt(static_cast<float>(diag)));
        for(size_t i=0;i<n_;++i) {
            if(margins[i]>0) {margin_.push_back(i);margin_value_.push_back(margins[i]);}
            if(margins[i]==0)pinned_.push_back(i);else free_.push_back(i);
        }
        // Reduced system S M S^T over free frames stays pentadiagonal.
        const size_t m=free_.size();rd_.resize(m);re_.assign(m,0);rf_.assign(m,0);
        std::vector<char> is_margin(n_,0);for(size_t i:margin_)is_margin[i]=1;
        for(size_t k=0;k<m;++k) {
            rd_[k]=d_[free_[k]]+(is_margin[free_[k]]?rho_:0);
            if(k+1<m)re_[k]=band(free_[k],free_[k+1]);
            if(k+2<m)rf_[k]=band(free_[k],free_[k+2]);
        }
    }
    // cols channels of x (N rows each, row-major [frame][col]).
    void interpolate(std::vector<double>& x,const std::vector<double>& observed,const std::vector<double>& reference,size_t cols) const {
        if(pinned_.empty()&&margin_.empty()) {x=reference;return;}
        x=reference;
        if(margin_.empty()) {update_x(x,{},{},reference,observed,cols);return;}
        const size_t m=margin_.size();
        std::vector<double> z(m*cols),zt(m*cols),u(m*cols,0);
        for(size_t i=0;i<m;++i)for(size_t j=0;j<cols;++j) {zt[i*cols+j]=observed[margin_[i]*cols+j];z[i*cols+j]=reference[margin_[i]*cols+j];}
        for(int it=0;it<iters_;++it) {
            update_x(x,z,u,reference,observed,cols);
            for(size_t i=0;i<m;++i) {
                double norm=0;
                for(size_t j=0;j<cols;++j) {double v=x[margin_[i]*cols+j]+u[i*cols+j]-zt[i*cols+j];z[i*cols+j]=v;norm+=v*v;}
                norm=std::sqrt(norm);
                if(norm>margin_value_[i])for(size_t j=0;j<cols;++j)z[i*cols+j]*=margin_value_[i]/norm;
                for(size_t j=0;j<cols;++j)z[i*cols+j]+=zt[i*cols+j];
            }
            for(size_t i=0;i<m;++i)for(size_t j=0;j<cols;++j)u[i*cols+j]+=x[margin_[i]*cols+j]-z[i*cols+j];
        }
    }
private:
    double band(size_t i,size_t j) const {size_t g=j-i;return g==0?d_[i]:g==1?e_[i]:g==2?f_[i]:0;}
    void update_x(std::vector<double>& x,const std::vector<double>& z,const std::vector<double>& u,const std::vector<double>& ref,const std::vector<double>& obs,size_t cols) const {
        std::vector<double> diff=ref;
        for(size_t i:pinned_)for(size_t j=0;j<cols;++j)diff[i*cols+j]-=obs[i*cols+j];
        std::vector<double> r(n_*cols,0);
        for(size_t i=0;i<n_;++i)for(size_t j=0;j<cols;++j) {
            double v=d_[i]*diff[i*cols+j];
            if(i+1<n_)v+=e_[i]*diff[(i+1)*cols+j];
            if(i+2<n_)v+=f_[i]*diff[(i+2)*cols+j];
            if(i>=1)v+=e_[i-1]*diff[(i-1)*cols+j];
            if(i>=2)v+=f_[i-2]*diff[(i-2)*cols+j];
            r[i*cols+j]=v;
        }
        for(size_t k=0;k<margin_.size();++k)for(size_t j=0;j<cols;++j)r[margin_[k]*cols+j]+=rho_*(z[k*cols+j]-u[k*cols+j]);
        for(size_t j=0;j<cols;++j) {
            std::vector<double> rhs(free_.size());
            for(size_t k=0;k<free_.size();++k)rhs[k]=r[free_[k]*cols+j];
            solve(rhs);
            for(size_t k=0;k<free_.size();++k)x[free_[k]*cols+j]=rhs[k];
        }
        for(size_t i:pinned_)for(size_t j=0;j<cols;++j)x[i*cols+j]=obs[i*cols+j];
    }
    void solve(std::vector<double>& r) const {
        const size_t n=rd_.size();if(!n)return;
        std::vector<double> d=rd_,l1(n,0),l2(n,0);
        for(size_t i=0;i<n;++i) {
            if(i>=2)d[i]-=l2[i-2]*l2[i-2]*d[i-2];
            if(i>=1)d[i]-=l1[i-1]*l1[i-1]*d[i-1];
            if(!(d[i]>0))throw std::runtime_error("Motion correction system is not positive definite");
            if(i+1<n) {double v=re_[i];if(i>=1)v-=l2[i-1]*d[i-1]*l1[i-1];l1[i]=v/d[i];}
            if(i+2<n)l2[i]=rf_[i]/d[i];
        }
        for(size_t i=0;i<n;++i) {if(i>=1)r[i]-=l1[i-1]*r[i-1];if(i>=2)r[i]-=l2[i-2]*r[i-2];}
        for(size_t i=0;i<n;++i)r[i]/=d[i];
        for(size_t i=n;i-->0;) {if(i+1<n)r[i]-=l1[i]*r[i+1];if(i+2<n)r[i]-=l2[i]*r[i+2];}
    }
    size_t n_;int iters_;double rho_=0;
    std::vector<double> d_,e_,f_,rd_,re_,rf_,margin_value_;
    std::vector<size_t> margin_,pinned_,free_;
};
std::vector<double> margins_of(const std::vector<float>& mask) {std::vector<double> m(mask.size());for(size_t i=0;i<m.size();++i)m[i]=mask[i]?0.0:-1.0;return m;}
// SmoothChannels: frames entering or leaving a constrained run take their neighbours' mean.
void smooth_channels(std::vector<double>& x,size_t cols,const std::vector<float>& mask) {
    const size_t n=mask.size();
    for(size_t i=0;i<n;++i) {
        size_t prev=i?i-1:0,next=std::min(i+1,n-1);
        if(i>0&&mask[i]>0&&mask[prev]==0)for(size_t j=0;j<cols;++j)x[i*cols+j]=0.5*(x[prev*cols+j]+x[next*cols+j]);
        if(mask[i]>0&&mask[next]==0)for(size_t j=0;j<cols;++j)x[i*cols+j]=0.5*(x[prev*cols+j]+x[next*cols+j]);
    }
}
void correct_bone(std::vector<Pose>& poses,const std::vector<Pose>& targets,const std::vector<float>& mask,const Corrector& corrector,size_t bone,bool smoothing) {
    const size_t n=poses.size();
    std::vector<double> x(n*6),obs(n*6),fixed;
    for(size_t f=0;f<n;++f) {
        Vec3 fw=rotate(poses[f][bone].q,{0,0,1}),up=rotate(poses[f][bone].q,{0,1,0});
        Vec3 tf=rotate(targets[f][bone].q,{0,0,1}),tu=rotate(targets[f][bone].q,{0,1,0});
        const float v[6]={fw.x,fw.y,fw.z,up.x,up.y,up.z},t[6]={tf.x,tf.y,tf.z,tu.x,tu.y,tu.z};
        for(int k=0;k<6;++k) {x[f*6+k]=v[k];obs[f*6+k]=mask[f]*t[k];}
    }
    // Upstream corrects the six channels one at a time with the same system.
    for(size_t k=0;k<6;++k) {
        std::vector<double> xc(n),oc(n);for(size_t f=0;f<n;++f) {xc[f]=x[f*6+k];oc[f]=obs[f*6+k];}
        if(smoothing)smooth_channels(xc,1,mask);
        corrector.interpolate(fixed,oc,xc,1);
        for(size_t f=0;f<n;++f)x[f*6+k]=fixed[f];
    }
    for(size_t f=0;f<n;++f) {
        Vec3 fw=normalized(Vec3{static_cast<float>(x[f*6]),static_cast<float>(x[f*6+1]),static_cast<float>(x[f*6+2])});
        Vec3 up=normalized(Vec3{static_cast<float>(x[f*6+3]),static_cast<float>(x[f*6+4]),static_cast<float>(x[f*6+5])});
        poses[f][bone].q=look_rotation(fw,up);
    }
}
struct Contact { int joint=0; std::vector<float> mask; bool two_bone=true; Vec3 hint; float min_height=0; };
using Intervals=std::vector<std::pair<size_t,size_t>>;
Intervals contact_intervals(const std::vector<float>& contacts,const std::vector<float>& mask,float threshold) {
    Intervals out;long start=-1;
    for(size_t f=0;f<mask.size();++f) {
        bool on=(mask[f]?0.f:contacts[f])>threshold;
        if(on&&start<0)start=static_cast<long>(f);
        else if(!on&&start>=0) {out.emplace_back(static_cast<size_t>(start),f);start=-1;}
    }
    if(start>=0)out.emplace_back(static_cast<size_t>(start),mask.size());
    return out;
}
bool start_pinned(const std::pair<size_t,size_t>& i,const std::vector<float>& mask) {return i.first!=0&&mask[i.first-1];}
bool end_pinned(const std::pair<size_t,size_t>& i,const std::vector<float>& mask) {return i.second!=mask.size()&&mask[i.second];}
void filter_intervals(Intervals& intervals,const std::vector<float>& mask,bool one_bone) {
    std::erase_if(intervals,[&](const auto& i) {
        bool s=start_pinned(i,mask),e=end_pinned(i,mask);
        return one_bone?(s||e):(s&&e);
    });
}
float reach_falloff(const std::vector<int>& parents,const Pose& rest,int joint,Vec3 target,const Pose& pose) {
    float reach=length(rest[static_cast<size_t>(joint)].p);
    joint=parents[static_cast<size_t>(joint)];reach+=length(rest[static_cast<size_t>(joint)].p);
    joint=parents[static_cast<size_t>(joint)];
    float t=std::max(length(target-global(parents,joint,pose).p)/reach-0.99f,0.f)/0.01f;t=t*t;
    return std::exp(-2.f*t*t);
}
void apply_corrections(std::vector<Pose>& poses,const std::vector<Pose>& fixed,const std::map<size_t,std::vector<float>>& masks) {
    for(const auto& [joint,mask]:masks)correct_bone(poses,fixed,mask,Corrector(margins_of(mask),pos_weight*10,vel_weight,acc_weight),joint,false);
}
}

std::vector<std::pair<size_t,size_t>> contact_intervals(std::span<const float> contacts,size_t channel) {
    std::vector<std::pair<size_t,size_t>> out;const size_t n=contacts.size()/4;long start=-1;
    for(size_t f=0;f<=n;++f) {
        bool on=f<n&&contacts[f*4+channel]>0.5f;
        if(on&&start<0)start=static_cast<long>(f);
        if(!on&&start>=0) {out.emplace_back(static_cast<size_t>(start),f);start=-1;}
    }
    return out;
}
KimodoCorrectionReport correct_kimodo(Clip& clip,const MotionConditions& conditions,std::span<const float> raw_contacts,float root_margin,float threshold,std::ostream* trace) {
    const auto started=std::chrono::steady_clock::now();
    require_soma(clip);
    const size_t n=clip.frames(),J=clip.joints.size();
    if(raw_contacts.size()!=n*4)throw std::runtime_error("Kimodo correction needs the model's four foot contacts per frame");
    std::vector<int> parents;for(const auto& j:clip.joints)parents.push_back(j.parent);
    // Working rig (postprocess.create_working_rig_from_skeleton): canonical offsets,
    // identity rotations, the lowest neutral joint 2 cm above the ground (SOMA).
    Pose rest;for(const auto& j:clip.joints)rest.push_back({j.rest.p,{}});
    {
        auto world=world_pose(clip,rest);float lowest=world[0].p.y;for(const auto& w:world)lowest=std::min(lowest,w.p.y);
        rest[0].p.y+=-lowest+0.02f;
    }
    std::vector<Pose> poses(n,rest),targets(n,rest);
    for(size_t f=0;f<n;++f)for(size_t j=0;j<J;++j) {poses[f][j].q=normalized(clip.frame(f)[j].q);if(!j)poses[f][0].p=clip.frame(f)[0].p;}
    for(size_t f=0;f<n;++f)targets[f][0].p={};
    std::vector<float> full(n,0),root(n,0),effector[4];for(auto& m:effector)m.assign(n,0);
    // extract_input_motion_from_constraints: root paths first, then effectors, full bodies last.
    for(const auto& point:conditions.path) {root[point.frame]=1;targets[point.frame][0].p.x=point.x;targets[point.frame][0].p.z=point.z;}
    auto place=[&](const PoseConstraint& pose) {
        auto& target=targets[pose.frame];
        for(size_t j=0;j<J;++j)target[j].q=normalized(pose.pose[j].q);
        Vec3 hips=pose.pose[0].p;
        // A floating pose keeps the ground position the model chose.
        if(!pose.anchored) {hips.x=clip.frame(pose.frame)[0].p.x;hips.z=clip.frame(pose.frame)[0].p.z;}
        // Effector sets use the smooth root (a dense path sample, else the hips) on the ground.
        if(pose.effectors)for(const auto& point:conditions.path)if(point.interpolated&&point.frame==pose.frame) {hips.x=point.x;hips.z=point.z;}
        target[0].p=hips;
    };
    for(const auto& pose:conditions.poses)if(pose.effectors) {
        place(pose);
        // Effector bits: 1 left foot, 2 right foot, 4 left hand, 8 right hand.
        for(int i=0;i<4;++i)if(pose.effectors&(1u<<i))effector[i][pose.frame]=1;
    }
    for(const auto& pose:conditions.poses)if(!pose.effectors) {place(pose);full[pose.frame]=1;}
    // Bindings: pins for hands/feet outside full-body frames; foot and toe contacts.
    std::vector<Contact> pins(4);
    const int pin_joint[4]={left_hand,right_hand,left_foot,right_foot};const int pin_bit[4]={2,3,0,1};
    for(int i=0;i<4;++i) {
        pins[static_cast<size_t>(i)].joint=pin_joint[i];
        pins[static_cast<size_t>(i)].hint=i<2?Vec3{0,0,-0.1f}:Vec3{0,0,0.1f};
        pins[static_cast<size_t>(i)].mask.resize(n);
        for(size_t f=0;f<n;++f)pins[static_cast<size_t>(i)].mask[f]=(1-full[f])*effector[pin_bit[i]][f];
    }
    // Upstream thresholds the decoded contacts (> 0.5) before correction.
    std::vector<float> c(n*4);for(size_t i=0;i<c.size();++i)c[i]=raw_contacts[i]>0.5f?1.f:0.f;
    const auto& lf=effector[0];const auto& rf=effector[1];
    std::vector<Contact> contacts(4);
    contacts[0]={right_foot,{},true,{0,0,0.1f},global(parents,right_foot,rest).p.y};
    contacts[1]={left_foot,{},true,{0,0,0.1f},global(parents,left_foot,rest).p.y};
    const float toe_height=global(parents,right_foot+1,rest).p.y;
    contacts[2]={right_foot+1,{},false,{},toe_height};
    contacts[3]={left_foot+1,{},false,{},toe_height}; // upstream reuses the right toe height
    for(auto& ci:contacts)ci.mask.resize(n);
    for(size_t f=0;f<n;++f) {
        contacts[0].mask[f]=std::min((rf[f]?0:c[4*f+3])+(rf[f]?0:c[4*f+2]),1.f);
        contacts[1].mask[f]=std::min((lf[f]?0:c[4*f+1])+(lf[f]?0:c[4*f+0]),1.f);
        contacts[2].mask[f]=rf[f]?0:c[4*f+3];
        contacts[3].mask[f]=lf[f]?0:c[4*f+1];
    }
    if(trace) {
        // Upstream correct_motion's arguments: working rig, masks, contacts, motion, targets.
        auto& t=*trace;t.precision(9);
        t<<n<<" "<<J<<" "<<root_margin<<" "<<threshold<<'\n';
        for(int p:parents) {t<<p<<" ";}t<<'\n';
        for(const auto& r:rest) {t<<r.p.x<<" "<<r.p.y<<" "<<r.p.z<<" ";}t<<'\n';
        for(const auto* mask:{&full,&effector[2],&effector[3],&effector[0],&effector[1],&root}) {for(float v:*mask) {t<<v<<" ";}t<<'\n';}
        for(float v:c) {t<<v<<" ";}t<<'\n';
        for(const auto* set:{&poses,&targets}) {
            for(const auto& pose:*set) {t<<pose[0].p.x<<" "<<pose[0].p.y<<" "<<pose[0].p.z<<" ";}t<<'\n';
            for(const auto& pose:*set) {for(const auto& j:pose)t<<j.q.x<<" "<<j.q.y<<" "<<j.q.z<<" "<<j.q.w<<" ";}t<<'\n';
        }
    }
    KimodoCorrectionReport report;
    for(size_t f=0;f<n;++f)for(int k=0;k<4;++k)report.contact_frames[k]+=c[4*f+static_cast<size_t>(k)]>0;
    // CorrectHipsY.
    {
        Corrector y(margins_of(full),pos_weight*10,vel_weight,acc_weight*0.1f);
        std::vector<double> x(n),obs(n),out;
        for(size_t f=0;f<n;++f) {x[f]=poses[f][0].p.y;obs[f]=targets[f][0].p.y;}
        y.interpolate(out,obs,x,1);
        for(size_t f=0;f<n;++f)poses[f][0].p.y=static_cast<float>(out[f]);
    }
    // CorrectHipsXZ.
    {
        auto margins=margins_of(full);std::vector<float> combined(n);
        for(size_t f=0;f<n;++f) {
            combined[f]=full[f]>0||root[f]>0;
            if(root[f]>0&&margins[f]!=0)margins[f]=root_margin;
            for(const auto& pin:pins)if(pin.mask[f]&&margins[f]!=0)margins[f]=root_margin;
        }
        Corrector xz(margins,pos_weight,vel_weight,acc_weight);
        std::vector<double> x(n*2),obs(n*2),out;
        for(size_t f=0;f<n;++f) {x[f*2]=poses[f][0].p.x;x[f*2+1]=poses[f][0].p.z;obs[f*2]=targets[f][0].p.x;obs[f*2+1]=targets[f][0].p.z;}
        smooth_channels(x,2,combined);
        xz.interpolate(out,obs,x,2);
        for(size_t f=0;f<n;++f) {poses[f][0].p.x=static_cast<float>(out[f*2]);poses[f][0].p.z=static_cast<float>(out[f*2+1]);}
    }
    // CorrectJointRotations.
    {
        Corrector rotations(margins_of(full),pos_weight*10,vel_weight,acc_weight);
        for(size_t j=0;j<J;++j)correct_bone(poses,targets,full,rotations,j,true);
    }
    // DoEffectorIK.
    {
        std::map<size_t,std::vector<float>> masks;std::vector<Pose> fixed=poses;
        for(const auto& pin:pins) {
            const size_t joint=static_cast<size_t>(pin.joint),parent=static_cast<size_t>(parents[joint]),grand=static_cast<size_t>(parents[parent]);
            for(size_t k:{joint,parent,grand})if(masks[k].empty())masks[k]=full;
            for(size_t f=0;f<n;++f)if(pin.mask[f]) {
                Transform want=global(parents,pin.joint,targets[f]);
                masks[parent][f]=masks[grand][f]=1;
                two_bone_ik(fixed[f],pin.joint,want.p,parents,pin.hint);
                masks[joint][f]=1;
                Transform above=global(parents,static_cast<int>(parent),fixed[f]);
                fixed[f][joint].q=normalized(inverse(above.q)*want.q);
                ++report.effector_frames;
            }
        }
        apply_corrections(poses,fixed,masks);
    }
    // DoContactIK (SOMA has single ankles: no double-ankle phase).
    {
        std::map<size_t,std::vector<float>> masks;std::vector<Pose> fixed=poses;
        auto pin_mask=[&](int joint,int parent,std::vector<float>& mask) {
            const Contact* found=nullptr;
            for(const auto& pin:pins)if(pin.joint==joint)found=&pin;
            if(!found)for(const auto& pin:pins)if(pin.joint==parent)found=&pin;
            if(found)for(size_t f=0;f<n;++f)if(found->mask[f])mask[f]=1;
        };
        for(const auto& ci:contacts) {
            if(!ci.two_bone)continue;
            const size_t joint=static_cast<size_t>(ci.joint),parent=static_cast<size_t>(parents[joint]),grand=static_cast<size_t>(parents[parent]);
            auto mask=full;pin_mask(ci.joint,static_cast<int>(parent),mask);
            for(size_t k:{parent,grand,joint})if(masks[k].empty())masks[k]=mask;
            auto intervals=contact_intervals(ci.mask,mask,threshold);filter_intervals(intervals,mask,false);
            // FindContactPoints: each interval sticks to one representative frame's ankle.
            std::vector<Vec3> points(n);std::vector<char> on(n,0);
            for(const auto& iv:intervals) {
                bool s=start_pinned(iv,mask),e=end_pinned(iv,mask);
                size_t frame=s?iv.first-1:e?iv.second:(iv.first+iv.second)/2;
                Vec3 target=global(parents,ci.joint,poses[frame]).p;
                for(size_t f=iv.first;f<iv.second;++f) {
                    on[f]=1;points[f]=target;
                    if(!s&&!e)points[f].y=std::max(global(parents,ci.joint,poses[f]).p.y,ci.min_height);
                }
            }
            for(size_t f=0;f<n;++f)if(on[f]) {
                masks[parent][f]=masks[grand][f]=masks[joint][f]=1;
                Quat keep=global(parents,ci.joint,fixed[f]).q;
                float w=reach_falloff(parents,rest,ci.joint,points[f],fixed[f]);
                Quat qp=fixed[f][parent].q,qg=fixed[f][grand].q;
                two_bone_ik(fixed[f],ci.joint,points[f],parents,ci.hint);
                fixed[f][parent].q=slerp(qp,fixed[f][parent].q,w);
                fixed[f][grand].q=slerp(qg,fixed[f][grand].q,w);
                Transform above=global(parents,static_cast<int>(parent),fixed[f]);
                fixed[f][joint].q=normalized(inverse(above.q)*keep);
                ++report.ankle_frames;
            }
        }
        apply_corrections(poses,fixed,masks);
        masks.clear();
        for(const auto& ci:contacts) {
            if(ci.two_bone)continue;
            const size_t parent=static_cast<size_t>(parents[static_cast<size_t>(ci.joint)]);
            auto mask=full;pin_mask(ci.joint,static_cast<int>(parent),mask);
            if(masks[parent].empty())masks[parent]=mask;
            auto intervals=contact_intervals(ci.mask,mask,threshold);filter_intervals(intervals,mask,true);
            for(const auto& iv:intervals)for(size_t f=iv.first;f<iv.second;++f) {
                // Stick the toe to the floor, letting it slide sideways.
                Vec3 base=global(parents,static_cast<int>(parent),fixed[f]).p,target=global(parents,ci.joint,fixed[f]).p;
                float len=length(target-base);
                for(int i=0;i<10;++i) {target.y=ci.min_height;target=base+normalized(target-base)*len;}
                one_bone_ik(fixed[f],ci.joint,target,parents);
                masks[parent][f]=1;++report.toe_frames;
            }
        }
        apply_corrections(poses,fixed,masks);
    }
    for(size_t f=0;f<n;++f)for(size_t j=0;j<J;++j) {clip.frame(f)[j].q=poses[f][j].q;if(!j)clip.frame(f)[0].p=poses[f][0].p;}
    if(trace) {
        auto& t=*trace;
        for(const auto& pose:poses) {t<<pose[0].p.x<<" "<<pose[0].p.y<<" "<<pose[0].p.z<<" ";}t<<'\n';
        for(const auto& pose:poses) {for(const auto& j:pose)t<<j.q.x<<" "<<j.q.y<<" "<<j.q.z<<" "<<j.q.w<<" ";}t<<'\n';
    }
    report.milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    return report;
}
}
