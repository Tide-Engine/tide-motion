#include "motion.hpp"
#include "skeleton.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <set>
#include <stdexcept>

// Pose encoding follows NVIDIA Kimodo (Apache-2.0) at e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954:
// kimodo/motion_rep/reps/kimodo_motionrep.py (feature layout, create_conditions),
// kimodo/constraints.py (full-body, end-effector and root-2D sets) and
// kimodo/model/kimodo_model.py (_multiprompt transition constraints).
namespace motion {
namespace {
const auto& spec=kimodo::detail::soma30_spec;
constexpr size_t J=30,positions=5,rotations=5+3*J;
// Chains as upstream SOMASkeleton30: the first joint's rotation and every joint's
// position are constrained. Order matches the effector bits.
constexpr std::array<std::array<size_t,2>,4> chains{{{24,25},{28,29},{13,15},{19,21}}};
constexpr std::array<const char*,4> effector_names{"left-foot","right-foot","left-hand","right-hand"};
float heading_of(const std::vector<Transform>& world) {
    Vec3 d=world[spec.hips[0]].p-world[spec.hips[1]].p;
    return std::atan2(d.z,-d.x);
}
float angle_between(Quat a,Quat b) {
    a=normalized(a);b=normalized(b);
    float d=std::abs(a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
    return 2*std::acos(std::min(1.0f,d))*180/std::numbers::pi_v<float>;
}
}
unsigned parse_effectors(const std::string& text) {
    unsigned bits=0;size_t start=0;
    while(start<=text.size()) {
        size_t end=text.find(',',start);if(end==std::string::npos)end=text.size();
        auto name=text.substr(start,end-start);bool found=false;
        for(size_t i=0;i<4;++i)if(name==effector_names[i]) {bits|=1u<<i;found=true;}
        if(!found)throw std::runtime_error("Unknown effector '"+name+"'; use left-hand, right-hand, left-foot or right-foot, comma-separated");
        start=end+1;
    }
    return bits;
}
std::string effector_list(unsigned bits) {
    std::string out;
    for(size_t i=0;i<4;++i)if(bits&(1u<<i)) {if(!out.empty())out+=',';out+=effector_names[i];}
    return out;
}
float kimodo_heading(const Clip& c,std::span<const Transform> pose) { return heading_of(world_pose(c,pose)); }
std::vector<float> encode_kimodo(const Clip& c,std::span<const Transform> pose) {
    if(c.joints.size()!=J||pose.size()!=J)throw std::runtime_error("Kimodo poses use the 30-joint SOMA rig");
    auto world=world_pose(c,pose);
    std::vector<float> row(kimodo_features);
    // A keyframe's smooth root is its hips (upstream substitutes the real root when
    // no smoothed path is given), so the hips' own horizontal offset is zero.
    const Vec3 root=world[0].p;
    const float angle=heading_of(world);
    row[0]=root.x;row[1]=root.y;row[2]=root.z;row[3]=std::cos(angle);row[4]=std::sin(angle);
    for(size_t j=0;j<J;++j) {
        Vec3 p=world[j].p;
        row[positions+3*j]=p.x-root.x;row[positions+3*j+1]=p.y;row[positions+3*j+2]=p.z-root.z;
        // cont6d: the first two columns of the global rotation matrix.
        Vec3 x=rotate(world[j].q,{1,0,0}),y=rotate(world[j].q,{0,1,0});
        float* r=row.data()+rotations+6*j;r[0]=x.x;r[1]=x.y;r[2]=x.z;r[3]=y.x;r[4]=y.y;r[5]=y.z;
    }
    return row;
}
Clip decode_kimodo(std::span<const float> raw) {
    if(raw.empty()||raw.size()%kimodo_features)throw std::runtime_error("Kimodo features must be rows of 369 values");
    Clip c;c.joints=soma();
    for(size_t f=0;f<raw.size()/kimodo_features;++f) {
        const float* row=raw.data()+f*kimodo_features;
        std::array<Quat,J> global;
        for(size_t j=0;j<J;++j) {const float* r=row+rotations+6*j;global[j]=rotation6({r[0],r[1],r[2]},{r[3],r[4],r[5]});}
        for(size_t j=0;j<J;++j) {
            Transform t=c.joints[j].rest;int parent=c.joints[j].parent;
            t.q=parent<0?global[j]:normalized(inverse(global[static_cast<size_t>(parent)])*global[j]);
            if(!j)t.p={row[0]+row[positions],row[positions+1],row[2]+row[positions+2]};
            c.poses.push_back(t);
        }
    }
    return c;
}
bool place_constraints(MotionConditions& conditions,Placement placement) {
    // A colocated path explicitly places the pose, including its orientation.
    std::set<size_t> path_frames;
    Clip rig;rig.joints=soma();
    for(const auto& point:conditions.path) {
        if(!path_frames.insert(point.frame).second)throw std::runtime_error("Two path points share output frame "+std::to_string(point.frame));
        if(point.interpolated)continue;
        for(auto& pose:conditions.poses)if(pose.frame==point.frame) {
            const float turn=point.has_heading?point.heading*std::numbers::pi_v<float>/180-kimodo_heading(rig,pose.pose):0;
            pose.pose[0].q=normalized(yaw(turn)*pose.pose[0].q);
            pose.pose[0].p.x=point.x;pose.pose[0].p.z=point.z;
        }
    }
    bool start=false;
    for(const auto& pose:conditions.poses)start|=pose.frame==0;
    for(const auto& point:conditions.path)start|=point.frame==0;
    // Without a constrained first frame Kimodo starts at the origin, so the ground
    // positions of later poses would be arbitrary; let them float unless a path or
    // the caller fixes the world.
    bool world=placement==Placement::world||(placement==Placement::automatic&&(start||!conditions.path.empty()));
    for(auto& pose:conditions.poses)pose.anchored=world||pose.frame==0||path_frames.contains(pose.frame);
    bool floating=false;for(const auto& pose:conditions.poses)floating|=!pose.anchored;
    return floating&&!start&&conditions.path.empty();
}
KimodoCondition build_condition(const MotionConditions& conditions,size_t frames) {
    KimodoCondition out;
    out.observed.assign(frames*kimodo_features,0);out.mask.assign(out.observed.size(),0);
    std::set<size_t> used;
    for(const auto& pose:conditions.poses) {
        if(pose.frame>=frames)throw std::runtime_error(pose.kind+" frame "+std::to_string(pose.frame)+" is outside the "+std::to_string(frames)+"-frame output");
        if(!used.insert(pose.frame).second)throw std::runtime_error("Two pose constraints share output frame "+std::to_string(pose.frame));
    }
    std::set<size_t> paths;
    for(const auto& point:conditions.path) {
        if(point.frame>=frames)throw std::runtime_error("Path frame "+std::to_string(point.frame)+" is outside the "+std::to_string(frames)+"-frame output");
        if(!paths.insert(point.frame).second)throw std::runtime_error("Two path points share output frame "+std::to_string(point.frame));
    }
    Clip rig;rig.joints=soma();
    // Kimodo starts generated motion at the origin, facing first_heading. A constraint on
    // frame 0 defines both; the output is moved back to its place afterwards.
    for(const auto& pose:conditions.poses)if(pose.frame==0) {
        auto world=world_pose(rig,pose.pose);out.origin={world[0].p.x,0,world[0].p.z};out.first_heading=heading_of(world);
    }
    for(const auto& point:conditions.path)if(point.frame==0) {
        out.origin={point.x,0,point.z};
        if(point.has_heading)out.first_heading=point.heading*std::numbers::pi_v<float>/180;
    }
    for(const auto& pose:conditions.poses) {
        auto row=encode_kimodo(rig,pose.pose);
        // On a dense path upstream passes the path sample as the pose's smooth root, so
        // its root-relative positions are measured from that sample, not the hips.
        for(const auto& point:conditions.path)if(point.interpolated&&point.frame==pose.frame) {
            const float dx=row[0]-point.x,dz=row[2]-point.z;row[0]=point.x;row[2]=point.z;
            for(size_t j=0;j<J;++j) {row[positions+3*j]+=dx;row[positions+3*j+2]+=dz;}
        }
        row[0]-=out.origin.x;row[2]-=out.origin.z;
        float* observed=out.observed.data()+pose.frame*kimodo_features;float* mask=out.mask.data()+pose.frame*kimodo_features;
        auto keep=[&](size_t first,size_t count) {for(size_t d=first;d<first+count;++d){observed[d]=row[d];mask[d]=1;}};
        // Every set constrains the root height and heading, and an anchored one its ground
        // position. Upstream always pins the ground position; a floating pose leaves it to
        // the model, which keeps the root-relative pose as accurately (notes/implementation.md).
        keep(1,1);keep(3,2);
        if(pose.anchored) {keep(0,1);keep(2,1);}
        if(!pose.effectors) {
            // Full body: all root-relative positions, plus hand and foot rotations as the
            // upstream multi-prompt transition adds with its end-effector set.
            keep(positions,3*J);
            for(const auto& chain:chains)keep(rotations+6*chain[0],6);
        } else for(size_t i=0;i<4;++i)if(pose.effectors&(1u<<i)) {
            for(size_t joint:chains[i])keep(positions+3*joint,3);
            keep(rotations+6*chains[i][0],6);
        }
    }
    for(const auto& point:conditions.path) {
        float* observed=out.observed.data()+point.frame*kimodo_features;float* mask=out.mask.data()+point.frame*kimodo_features;
        observed[0]=point.x-out.origin.x;observed[2]=point.z-out.origin.z;mask[0]=mask[2]=1;
        if(point.has_heading) {
            float angle=point.heading*std::numbers::pi_v<float>/180;
            observed[3]=std::cos(angle);observed[4]=std::sin(angle);mask[3]=mask[4]=1;
        }
    }
    return out;
}
ConstraintError constraint_error(const Clip& result,const PoseConstraint& pose) {
    auto got=world_pose(result,result.frame(pose.frame));
    Clip rig;rig.joints=soma();auto want=world_pose(rig,pose.pose);
    // A floating pose is compared where the model put it on the ground.
    if(!pose.anchored) {Vec3 move{got[0].p.x-want[0].p.x,0,got[0].p.z-want[0].p.z};for(auto& joint:want)joint.p=joint.p+move;}
    ConstraintError e;
    Vec3 d=got[0].p-want[0].p;e.root=std::sqrt(dot(d,d));
    std::vector<size_t> joints;
    if(!pose.effectors)for(size_t j=0;j<J;++j)joints.push_back(j);
    else for(size_t i=0;i<4;++i)if(pose.effectors&(1u<<i))joints.push_back(chains[i][0]);
    for(size_t j:joints) {
        Vec3 p=got[j].p-want[j].p;float distance=std::sqrt(dot(p,p)),angle=angle_between(got[j].q,want[j].q);
        e.position_mean+=distance;e.position_max=std::max(e.position_max,distance);
        e.rotation_mean+=angle;if(angle>e.rotation_max){e.rotation_max=angle;e.rotation_max_joint=result.joints[j].name;}
    }
    e.position_mean/=static_cast<float>(joints.size());e.rotation_mean/=static_cast<float>(joints.size());
    return e;
}
float path_error(const Clip& result,const PathConstraint& point) {
    auto root=result.frame(point.frame)[0].p;
    return std::hypot(root.x-point.x,root.z-point.z);
}
void shift_to_last_pose(Clip& result,const MotionConditions& conditions) {
    const PoseConstraint* last=nullptr;
    for(const auto& pose:conditions.poses)if(!last||pose.frame>last->frame)last=&pose;
    if(!last)return;
    Vec3 at=result.frame(last->frame)[0].p,want=last->pose[0].p,delta{want.x-at.x,0,want.z-at.z};
    for(size_t f=0;f<result.frames();++f)result.frame(f)[0].p=result.frame(f)[0].p+delta;
}
void turn_pose(PoseConstraint& pose,Vec3 pivot,float turn,Vec3 move) {
    // The hips are the rig's root, so turning them turns the whole pose.
    auto& hips=pose.pose[0];const Quat spin=yaw(turn);
    Vec3 d=hips.p-pivot;d.y=0;d=rotate(spin,d);
    hips.p={pivot.x+d.x+move.x,hips.p.y,pivot.z+d.z+move.z};
    hips.q=normalized(spin*hips.q);
}
std::vector<float> follow_ends(MotionConditions& conditions,const PoseConstraint& source) {
    const PoseConstraint* end=nullptr;size_t start=0;
    for(const auto& pose:conditions.poses) {if(pose.kind=="end-pose")end=&pose;if(pose.kind=="start-pose")start=pose.frame;}
    if(!end)throw std::runtime_error("--follow ends needs --end-pose");
    Clip rig;rig.joints=soma();
    // The shorter way round, as a final facing does not say how many turns were made.
    const float turn=std::remainder(kimodo_heading(rig,end->pose)-kimodo_heading(rig,source.pose),2*std::numbers::pi_v<float>);
    const Vec3 pivot{source.pose[0].p.x,0,source.pose[0].p.z},move{end->pose[0].p.x-pivot.x,0,end->pose[0].p.z-pivot.z};
    const float span=static_cast<float>(end->frame-start);
    std::vector<float> turns;
    for(auto& pose:conditions.poses) {
        if(pose.kind!="keyframe"&&pose.kind!="effector")continue;
        const float t=span>0?std::clamp((static_cast<float>(pose.frame)-static_cast<float>(start))/span,0.0f,1.0f):1.0f;
        turn_pose(pose,pivot,t*turn,move*t);turns.push_back(t*turn);
    }
    return turns;
}
void correct_poses(Clip& result,const MotionConditions& conditions,int window) {
    std::vector<size_t> frames;
    for(const auto& pose:conditions.poses)frames.push_back(pose.frame);
    for(const auto& point:conditions.path)frames.push_back(point.frame);
    const long last=static_cast<long>(result.frames())-1;
    for(const auto& pose:conditions.poses) {
        if(pose.effectors)continue;
        // The residual at the keyframe (local rotations, root height and anchored ground
        // position) fades out with a raised cosine over up to `window` frames each side,
        // ending before any neighbouring constraint so every keyframe stays exact.
        const long k=static_cast<long>(pose.frame);long before=std::max(0,window),after=before;
        for(size_t frame:frames) {
            long d=static_cast<long>(frame)-k;
            if(d>0)after=std::min(after,d);
            if(d<0)before=std::min(before,-d);
        }
        auto generated=result.frame(pose.frame);
        std::array<Quat,J> delta;
        for(size_t j=0;j<J;++j)delta[j]=normalized(pose.pose[j].q*inverse(generated[j].q));
        Vec3 root=pose.pose[0].p-generated[0].p;if(!pose.anchored)root.x=root.z=0;
        for(long f=std::max(0L,k-before);f<=std::min(last,k+after);++f) {
            long span=f<k?before:after;
            float x=span?static_cast<float>(std::abs(f-k))/static_cast<float>(span):0,w=0.5f*(1+std::cos(std::numbers::pi_v<float>*x));
            if(w<=0)continue;
            auto frame=result.frame(static_cast<size_t>(f));
            for(size_t j=0;j<J;++j)frame[j].q=normalized(slerp({},delta[j],w)*frame[j].q);
            frame[0].p=frame[0].p+root*w;
        }
    }
}
namespace {
// Solves M x = r in place for a symmetric positive definite pentadiagonal M given by
// its diagonal d and first/second superdiagonals e/f (LDL^T of bandwidth two).
void solve_pentadiagonal(std::vector<double> d,const std::vector<double>& e,const std::vector<double>& f,std::vector<double>& r) {
    const size_t n=d.size();
    std::vector<double> l1(n,0),l2(n,0); // l1[i] = L(i+1,i), l2[i] = L(i+2,i)
    for(size_t i=0;i<n;++i) {
        if(i>=2)d[i]-=l2[i-2]*l2[i-2]*d[i-2];
        if(i>=1)d[i]-=l1[i-1]*l1[i-1]*d[i-1];
        if(!(d[i]>0))throw std::runtime_error("Root path smoothing system is not positive definite");
        if(i+1<n) {double v=e[i];if(i>=1)v-=l2[i-1]*d[i-1]*l1[i-1];l1[i]=v/d[i];}
        if(i+2<n)l2[i]=f[i]/d[i];
    }
    for(size_t i=0;i<n;++i) {if(i>=1)r[i]-=l1[i-1]*r[i-1];if(i>=2)r[i]-=l2[i-2]*r[i-2];}
    for(size_t i=0;i<n;++i)r[i]/=d[i];
    for(size_t i=n;i-->0;) {if(i+1<n)r[i]-=l1[i]*r[i+1];if(i+2<n)r[i]-=l2[i]*r[i+2];}
}
// Upstream TrajectorySmoother with pos_weight 0 and no loop: one margin per sample.
void admm_smooth(const std::vector<std::array<double,2>>& target,std::vector<std::array<double,2>>& x,double margin) {
    const size_t n=target.size();
    // M = A^T A + rho I, A the interior second differences [-1, 2, -1].
    std::vector<double> d(n,0),e(n-1,0),f(n-2,0);
    for(size_t i=1;i+1<n;++i) {
        const size_t c[3]={i-1,i,i+1};const double w[3]={-1,2,-1};
        for(int a=0;a<3;++a)for(int b=0;b<3;++b) {
            size_t r=c[a],q=c[b];if(q<r)continue;
            double v=w[a]*w[b];
            if(q==r)d[r]+=v;else if(q==r+1)e[r]+=v;else f[r]+=v;
        }
    }
    double diag=0;for(double v:d)diag=std::max(diag,std::abs(v));
    const double rho=0.25*std::sqrt(diag);
    for(double& v:d)v+=rho;
    std::vector<std::array<double,2>> z(n,{0,0}),u(n,{0,0});
    std::vector<double> rx(n),rz(n);
    for(int it=0;it<500;++it) {
        for(size_t i=0;i<n;++i) {
            double a=x[i][0]+u[i][0]-target[i][0],b=x[i][1]+u[i][1]-target[i][1],len=std::hypot(a,b);
            if(len>margin) {a*=margin/len;b*=margin/len;}
            z[i]={a+target[i][0],b+target[i][1]};
        }
        for(size_t i=0;i<n;++i)for(int k=0;k<2;++k)u[i][k]+=1.8*(x[i][k]-z[i][k]);
        for(size_t i=0;i<n;++i) {rx[i]=rho*(z[i][0]-u[i][0]);rz[i]=rho*(z[i][1]-u[i][1]);}
        solve_pentadiagonal(d,e,f,rx);solve_pentadiagonal(d,e,f,rz);
        for(size_t i=0;i<n;++i)x[i]={rx[i],rz[i]};
    }
}
}
std::vector<std::array<double,2>> smooth_root_path(std::span<const std::array<double,2>> x,double margin) {
    const size_t n=x.size();
    std::vector<std::array<double,2>> out(x.begin(),x.end());
    if(n<5)return out;
    std::array<double,2> mean{0,0};for(const auto& p:x) {mean[0]+=p[0];mean[1]+=p[1];}
    mean[0]/=static_cast<double>(n);mean[1]/=static_cast<double>(n);
    out.assign(n,mean);
    // Multigrid as upstream: coarse to fine, each level starting from the previous one.
    size_t levels=static_cast<size_t>(std::floor(std::log2(static_cast<double>(n))));
    levels=std::max<size_t>(levels>4?levels-4:0,1);
    size_t step=size_t{1}<<levels;
    while(true) {
        std::vector<size_t> at;for(size_t i=0;i<n;i+=step)at.push_back(i);
        std::vector<std::array<double,2>> target,level;
        for(size_t i:at) {target.push_back(x[i]);level.push_back(out[i]);}
        admm_smooth(target,level,margin);
        for(size_t k=0;k<at.size();++k)out[at[k]]=level[k];
        // Fill the next level by averaging neighbours, extrapolating the last sample when
        // needed. At step 1 upstream's slice is the whole signal (next step 0), so this
        // also averages the finished result with its next frame; kept for parity.
        const size_t next=step/2;
        std::vector<size_t> inter;for(size_t i=next;i<n;i+=step)inter.push_back(i);
        size_t count=inter.size();
        if(count==at.size()) {
            auto a=out[at[at.size()-1]],b=out[at[at.size()-2]];
            out[inter.back()]={a[0]+(a[0]-b[0])/2,a[1]+(a[1]-b[1])/2};
            --count;
        }
        std::vector<std::array<double,2>> mid(count);
        for(size_t k=0;k<count;++k)mid[k]={(out[at[k]][0]+out[at[k+1]][0])/2,(out[at[k]][1]+out[at[k+1]][1])/2};
        for(size_t k=0;k<count;++k)out[inter[k]]=mid[k];
        if(step==1)break;
        step/=2;
    }
    return out;
}
std::vector<std::array<double,2>> smooth_root_of(const Clip& c) {
    std::vector<std::array<double,2>> hips;
    for(size_t f=0;f<c.frames();++f) {auto w=world_pose(c,c.frame(f));hips.push_back({w[c.motion_root].p.x,w[c.motion_root].p.z});}
    return smooth_root_path(hips);
}
void densify_path(MotionConditions& conditions,Vec3 start) {
    auto points=conditions.path;
    std::sort(points.begin(),points.end(),[](const auto& a,const auto& b){return a.frame<b.frame;});
    for(size_t i=1;i<points.size();++i)if(points[i].frame==points[i-1].frame)throw std::runtime_error("Two path points share output frame "+std::to_string(points[i].frame));
    if(points.empty())throw std::runtime_error("A dense path needs --path waypoints");
    // Upstream's dense mode locks a keyframe at frame 0, where the motion starts.
    if(points.front().frame!=0) {PathConstraint first;first.x=start.x;first.z=start.z;points.insert(points.begin(),first);}
    if(points.size()<2)throw std::runtime_error("A dense path needs a waypoint after frame 0");
    const size_t first=points.front().frame,last=points.back().frame;
    std::vector<std::array<double,2>> line;
    size_t k=0;
    for(size_t f=first;f<=last;++f) {
        while(k+2<points.size()&&points[k+1].frame<=f)++k;
        const auto& a=points[k];const auto& b=points[k+1];
        const double t=static_cast<double>(f-a.frame)/static_cast<double>(b.frame-a.frame);
        line.push_back({a.x+(b.x-a.x)*t,a.z+(b.z-a.z)*t});
    }
    const bool smoothed=points.size()>=3;
    if(smoothed)line=smooth_root_path(line);
    std::vector<PathConstraint> dense;
    for(size_t f=first;f<=last;++f) {
        PathConstraint p;p.frame=f;p.x=static_cast<float>(line[f-first][0]);p.z=static_cast<float>(line[f-first][1]);p.interpolated=true;
        for(const auto& w:points)if(w.frame==f) {p.has_heading=w.has_heading;p.heading=w.heading;}
        dense.push_back(p);
    }
    conditions.path=std::move(dense);
}
}

