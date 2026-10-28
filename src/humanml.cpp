// HumanML root/RIC/rotation decoding adapted from MoMask utils/motion_process.py
// and common/skeleton.py at 94a6636c9c463b7a9414c3401a6f1b67e6c51824.
// Copyright (c) 2023 Chuan Guo. MIT; see licenses/MOMASK-SOURCE.txt.
#include "humanml.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace motion::humanml {
namespace {
constexpr int parents[22]={-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19};
constexpr int primary[22]={3,4,5,6,7,8,9,10,11,12,-1,-1,15,16,17,-1,18,19,20,21,-1,-1};
constexpr size_t mapping[22]={0,22,26,1,23,27,2,24,28,3,25,29,5,10,16,6,11,17,12,18,13,19};
void check(std::span<const Row> rows) {
    if(rows.empty()||rows.size()>18000)throw std::runtime_error("Invalid HumanML frame count");
    for(const auto& row:rows)for(float v:row)if(!std::isfinite(v))throw std::runtime_error("Nonfinite HumanML feature");
}
Vec3 vec(const Row& r,size_t i) { return {r[i],r[i+1],r[i+2]}; }
// Each arm chain starts from root in upstream FK, even though its collar is
// attached at spine3. Preserve that convention rather than silently fixing it.
std::vector<std::array<Quat,22>> edge_frames(std::span<const Row> rows,const std::vector<Root>& root) {
    std::vector<std::array<Quat,22>> result(rows.size());
    for(size_t f=0;f<rows.size();++f) {
        auto& q=result[f];q[0]=root[f].world_to_heading;
        for(size_t j=1;j<22;++j) {
            size_t i=67+(j-1)*6;
            Quat local=rotation6(vec(rows[f],i),vec(rows[f],i+3));
            Quat p=(j==13||j==14)?q[0]:q[static_cast<size_t>(parents[j])];
            q[j]=normalized(p*local);
        }
    }
    return result;
}
// Construct a frame with the bone direction as X and a nonparallel roll hint.
Quat frame(Vec3 direction,Vec3 hint) {
    Vec3 x=normalized(direction);
    if(dot(cross(x,hint),cross(x,hint))<1e-10f) {
        hint=std::abs(x.y)<0.9f?Vec3{0,1,0}:Vec3{0,0,1};
    }
    return rotation6(x,hint);
}
float qdistance(Quat a,Quat b) {
    a=normalized(a);b=normalized(b);
    float d=std::abs(a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
    return 1-std::min(1.0f,d);
}
Quat from_to(Vec3 a,Vec3 b) {
    a=normalized(a);b=normalized(b);float d=dot(a,b);
    if(d<-.99999f) { Vec3 axis=normalized(cross(a,std::abs(a.y)<.9f?Vec3{0,1,0}:Vec3{0,0,1}));return {axis.x,axis.y,axis.z,0}; }
    auto c=cross(a,b);return normalized(Quat{c.x,c.y,c.z,1+d});
}
}
void Statistics::validate() const {
    for(size_t i=0;i<features;++i)if(!std::isfinite(mean[i])||!std::isfinite(deviation[i])||deviation[i]<=0)
        throw std::runtime_error("Invalid HumanML normalization statistics");
}
std::vector<Row> normalize(std::span<const Row> input,const Statistics& stats,bool inverse) {
    check(input);stats.validate();std::vector<Row> result(input.begin(),input.end());
    for(auto& row:result)for(size_t i=0;i<features;++i)
        row[i]=inverse?row[i]*stats.deviation[i]+stats.mean[i]:(row[i]-stats.mean[i])/stats.deviation[i];
    check(result);return result;
}
std::vector<Root> roots(std::span<const Row> rows) {
    check(rows);std::vector<Root> result(rows.size());float angle=0;Vec3 position{};
    for(size_t f=0;f<rows.size();++f) {
        if(f)angle+=rows[f-1][0]; // feature is quaternion half-angle per sample
        Quat q{0,std::sin(angle),0,std::cos(angle)};
        if(f)position=position+rotate(inverse(q),{rows[f-1][1],0,rows[f-1][2]});
        position.y=rows[f][3];result[f]={q,position};
        if(!std::isfinite(angle)||!std::isfinite(position.x)||!std::isfinite(position.z))
            throw std::runtime_error("HumanML root integration overflow");
    }
    return result;
}
std::vector<Positions> positions(std::span<const Row> rows) {
    auto root=roots(rows);std::vector<Positions> result(rows.size());
    for(size_t f=0;f<rows.size();++f) {
        result[f][0]=root[f].position;
        for(size_t j=1;j<22;++j) {
            Vec3 p=rotate(inverse(root[f].world_to_heading),vec(rows[f],4+(j-1)*3));
            p.x+=root[f].position.x;p.z+=root[f].position.z;result[f][j]=p;
        }
    }
    return result;
}
std::vector<Positions> rotation_fk(std::span<const Row> rows,const Positions& offsets) {
    auto root=roots(rows);auto edge=edge_frames(rows,root);std::vector<Positions> result(rows.size());
    for(const auto& v:offsets)if(!std::isfinite(v.x)||!std::isfinite(v.y)||!std::isfinite(v.z))
        throw std::runtime_error("Invalid HumanML offsets");
    for(size_t f=0;f<rows.size();++f) {
        result[f][0]=root[f].position;
        for(size_t j=1;j<22;++j)result[f][j]=result[f][static_cast<size_t>(parents[j])]+rotate(edge[f][j],offsets[j]);
    }
    return result;
}
Encoded from_soma(const Clip& clip) {
    require_soma(clip);if(clip.fps!=30)throw std::runtime_error("MoMask source must be canonical 30 Hz SOMA");
    size_t count=(((clip.frames()-1)*2+2)/3+1+3)/4*4;
    if(count<4||count>196)throw std::runtime_error("MoMask source must fit 196 samples at 20 Hz");
    Encoded result;result.origin=clip.frame(0)[0].p;result.origin.y=0;
    result.heading=kimodo_heading(clip,clip.frame(0));Quat alignment=yaw(-result.heading);
    std::vector<Positions> world(count+1);std::vector<float> headings(count+1);
    for(size_t f=0;f<=count;++f) {
        auto pose=sample(clip,static_cast<float>(f)/20);auto w=world_pose(clip,pose);
        for(size_t j=0;j<22;++j)world[f][j]=rotate(alignment,w[mapping[j]].p-result.origin);
        headings[f]=kimodo_heading(clip,pose)-result.heading;
        if(f) {while(headings[f]-headings[f-1]>std::numbers::pi_v<float>)headings[f]-=2*std::numbers::pi_v<float>;
               while(headings[f]-headings[f-1]<-std::numbers::pi_v<float>)headings[f]+=2*std::numbers::pi_v<float>;}
    }
    constexpr Vec3 raw[22]={{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,-1,0},{0,1,0},{0,-1,0},{0,-1,0},{0,1,0},{0,0,1},{0,0,1},{0,1,0},{1,0,0},{-1,0,0},{0,0,1},{0,-1,0},{0,-1,0},{0,-1,0},{0,-1,0},{0,-1,0},{0,-1,0}};
    for(size_t f=0;f<count;++f) {
        Row r{};Quat q=yaw(-headings[f]),next=yaw(-headings[f+1]);
        r[0]=(headings[f]-headings[f+1])/2;Vec3 velocity=rotate(next,world[f+1][0]-world[f][0]);r[1]=velocity.x;r[2]=velocity.z;r[3]=world[f][0].y;
        std::array<Quat,22> edges{};edges[0]=q;
        auto put=[&](size_t i,Vec3 v){r[i]=v.x;r[i+1]=v.y;r[i+2]=v.z;};
        for(size_t j=1;j<22;++j) {
            Vec3 p=world[f][j];p.x-=world[f][0].x;p.z-=world[f][0].z;put(4+(j-1)*3,rotate(q,p));
            edges[j]=from_to(raw[j],world[f][j]-world[f][static_cast<size_t>(parents[j])]);
            Quat parent=(j==13||j==14)?q:edges[static_cast<size_t>(parents[j])];Quat local=normalized(inverse(parent)*edges[j]);
            put(67+(j-1)*6,rotate(local,{1,0,0}));put(70+(j-1)*6,rotate(local,{0,1,0}));
        }
        for(size_t j=0;j<22;++j)put(193+j*3,rotate(q,world[f+1][j]-world[f][j]));
        for(size_t k=0;k<4;++k) {constexpr size_t feet[4]={7,10,8,11};Vec3 d=world[f+1][feet[k]]-world[f][feet[k]];r[259+k]=dot(d,d)<.002f?1.0f:0.0f;}
        result.rows.push_back(r);
    }
    return result;
}
Clip to_soma(std::span<const Row> rows) {
    auto root=roots(rows);auto points=positions(rows);auto edges=edge_frames(rows,root);
    Clip native;native.joints=soma();native.fps=20;native.name="HumanML conversion study";
    std::vector<Transform> rest;for(auto& j:native.joints)rest.push_back(j.rest);
    auto reference=world_pose(native,rest);
    std::array<int,30> semantic;semantic.fill(-1);
    for(size_t j=0;j<22;++j)semantic[mapping[j]]=static_cast<int>(j);
    for(size_t f=0;f<rows.size();++f) {
        std::array<Quat,30> world{};
        for(size_t s=0;s<30;++s) {
            auto local=rest[s];int p=native.joints[s].parent;
            Quat parent=p<0?Quat{}:world[static_cast<size_t>(p)];
            int h=semantic[s];
            if(s==0) { world[s]=inverse(root[f].world_to_heading);local.p=root[f].position; }
            else if(h>=0&&primary[h]>=0) {
                auto child=static_cast<size_t>(primary[h]);
                Vec3 base=reference[mapping[child]].p-reference[s].p;
                Vec3 direction=points[f][child]-points[f][static_cast<size_t>(h)];
                // The source's incoming-edge frame contains roll information;
                // use its Y axis as a hint, with an explicit degenerate fallback.
                Vec3 hint=rotate(edges[f][child],{0,1,0});
                world[s]=normalized(frame(direction,hint)*inverse(frame(base,{0,1,0})));
            } else world[s]=parent; // unpredicted leaves/helpers retain local rest
            local.q=normalized(inverse(parent)*world[s]);native.poses.push_back(local);
        }
    }
    // Native samples cover [0,(N-1)/20]. Keep that interval: no synthetic tail.
    Clip out=native;out.fps=30;out.poses.clear();
    size_t frames=(rows.size()-1)*3/2+1; // exact rational 30/20, no float frame-count drift
    for(size_t f=0;f<frames;++f) {
        auto pose=sample(native,static_cast<float>(f)/30);out.poses.insert(out.poses.end(),pose.begin(),pose.end());
    }
    require_soma(out);return out;
}
Clip splice(const Clip& source,const Clip& generated,size_t begin,size_t end,float ptol,float qtol) {
    require_soma(source);require_soma(generated);
    if(source.fps!=30||generated.fps!=source.fps||source.frames()!=generated.frames()||begin>=end||end>source.frames()||
       !std::isfinite(ptol)||ptol<0||!std::isfinite(qtol)||qtol<0)
        throw std::runtime_error("Invalid HumanML splice contract");
    for(size_t f:{begin,end-1})for(size_t j=0;j<30;++j) {
        Vec3 d=source.frame(f)[j].p-generated.frame(f)[j].p;
        if(dot(d,d)>ptol*ptol||qdistance(source.frame(f)[j].q,generated.frame(f)[j].q)>qtol)
            throw std::runtime_error("HumanML edit endpoints do not match source");
    }
    Clip out=source;
    for(size_t f=begin;f<end;++f)std::copy(generated.frame(f).begin(),generated.frame(f).end(),out.frame(f).begin());
    return out;
}
size_t mask_count(size_t tokens,size_t iteration,size_t iterations) {
    if(tokens==0||tokens>4500||iterations<2||iterations>1000||iteration>=iterations)
        throw std::runtime_error("Invalid MoMask schedule");
    // Match the upstream float32 linspace -> multiply pi -> multiply .5 -> cos
    // schedule. FP64 changes discrete mask counts near half-integer boundaries.
    float t=static_cast<float>(static_cast<double>(iteration)/static_cast<double>(iterations-1));
    double value=std::max(0.0f,std::cos((t*std::numbers::pi_v<float>)*.5f)*static_cast<float>(tokens));
    // PyTorch round is ties-to-even, independent of the process rounding mode.
    double low=std::floor(value),fraction=value-low;
    auto n=static_cast<size_t>(low);
    if(fraction>0.5||(fraction==0.5&&(n%2)))++n;
    return std::clamp(n,size_t{1},tokens);
}
}
