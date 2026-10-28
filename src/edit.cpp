#include "motion.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace motion {
std::vector<Footfall> footfalls(const Clip& c) {
    require_soma(c);
    std::vector<Footfall> events;
    for(unsigned foot=0;foot<2;++foot) {
        const size_t ankle=foot?28:24,toe=ankle+1;
        std::vector<Vec3> positions;std::vector<float> heights;
        for(size_t f=0;f<c.frames();++f) {
            const auto pose=world_pose(c,c.frame(f));
            positions.push_back(pose[ankle].p);heights.push_back(std::min(pose[ankle].p.y,pose[toe].p.y));
        }
        auto sorted=heights;std::sort(sorted.begin(),sorted.end());
        const float floor=sorted[(sorted.size()-1)/10];
        const size_t settle=std::max(size_t{2},static_cast<size_t>(std::ceil(c.fps*0.06f)));
        const size_t swing_min=std::max(size_t{2},static_cast<size_t>(std::ceil(c.fps*0.10f)));
        bool supported=false,air=false;size_t stable=0,high=0,lift=0,landing=0;
        Vec3 anchor{};float peak=0;
        for(size_t f=0;f<c.frames();++f) {
            const float speed=f?std::hypot(positions[f].x-positions[f-1].x,positions[f].z-positions[f-1].z)*c.fps:0;
            const bool low=heights[f]<=floor+0.02f&&speed<=0.8f;
            if(!air) {
                if(low) {if(++stable>=settle){supported=true;anchor=positions[f];}high=0;}
                else {
                    stable=0;
                    if(supported&&heights[f]>=floor+0.04f) {
                        if(!high){lift=f;peak=0;}
                        ++high;peak=std::max(peak,heights[f]-floor);
                        if(high>=swing_min){air=true;stable=0;}
                    } else high=0;
                }
            } else {
                peak=std::max(peak,heights[f]-floor);
                if(low) {
                    if(!stable)landing=f;
                    if(++stable>=settle) {
                        float distance=std::hypot(positions[landing].x-anchor.x,positions[landing].z-anchor.z);
                        if(distance>=0.08f)events.push_back({foot,lift,landing,distance,peak});
                        air=false;high=0;supported=true;anchor=positions[f];
                    }
                } else stable=0;
            }
        }
    }
    std::sort(events.begin(),events.end(),[](const auto& a,const auto& b){return a.landing<b.landing||(a.landing==b.landing&&a.foot<b.foot);});
    return events;
}

RootTravel root_travel(const Clip& c) {
    c.validate();
    RootTravel out;
    const auto start=world_pose(c,c.frame(0))[c.motion_root].p;
    auto previous=start;
    for(size_t f=1;f<c.frames();++f) {
        const auto current=world_pose(c,c.frame(f))[c.motion_root].p;
        float d=std::hypot(current.x-previous.x,current.z-previous.z);
        out.distance+=d;out.peak_speed=std::max(out.peak_speed,d*c.fps);previous=current;
    }
    out.displacement=std::hypot(previous.x-start.x,previous.z-start.z);
    return out;
}

fs::path variant_path(const fs::path& output,uint64_t seed) {
    auto path=output;path.replace_extension(".seed"+std::to_string(seed)+output.extension().string());return path;
}
fs::path variants_path(const fs::path& output,const std::string& extension) {
    auto path=output;path.replace_extension(".variants"+extension);return path;
}
void mirror_soma(Clip& c) {
    require_soma(c);
    // Each joint takes its Left/Right counterpart's rotation; centre joints keep their own.
    std::array<size_t,30> other{};
    for(size_t j=0;j<30;++j) {
        other[j]=j;const auto& name=c.joints[j].name;std::string swapped;
        if(name.starts_with("Left"))swapped="Right"+name.substr(4);
        else if(name.starts_with("Right"))swapped="Left"+name.substr(5);
        for(size_t k=0;k<30&&!swapped.empty();++k)if(c.joints[k].name==swapped)other[j]=k;
    }
    // The plane is the first frame's sagittal plane: through its root, up and its facing.
    // SOMA's rest offsets are symmetric about x (left is +x when facing +z), so a local
    // rotation L becomes M L M with M the x reflection, quaternion (x, -y, -z, w); the root
    // also carries the turn into and out of the plane's frame.
    const float heading=kimodo_heading(c,c.frame(0));
    const Quat to=yaw(-heading),back=yaw(heading);
    Vec3 origin=c.frame(0)[0].p;origin.y=0;
    auto reflect=[](Quat q){return Quat{q.x,-q.y,-q.z,q.w};};
    for(size_t f=0;f<c.frames();++f) {
        auto frame=c.frame(f);std::vector<Transform> in(frame.begin(),frame.end());
        for(size_t j=1;j<30;++j)frame[j].q=normalized(reflect(in[other[j]].q));
        frame[0].q=normalized(back*reflect(to*in[0].q));
        Vec3 local=rotate(to,in[0].p-origin);local.x=-local.x;
        frame[0].p=origin+rotate(back,local);
    }
}
void remove_root_travel(Clip& c) {
    // Travel is the root's ground path averaged over a second; subtracting it (relative to
    // frame 0) keeps sway, height and facing. Ends are padded by point reflection so a
    // steady walk stays centred to the last frame.
    const long n=static_cast<long>(c.frames()),half=15;
    if(n<2)return;
    std::vector<Vec3> ground(static_cast<size_t>(n));
    for(long f=0;f<n;++f) {auto p=c.frame(static_cast<size_t>(f))[c.motion_root].p;ground[static_cast<size_t>(f)]={p.x,0,p.z};}
    auto at=[&](long k) {
        if(k<0) {long m=std::min(-k,n-1);return ground[0]*2-ground[static_cast<size_t>(m)];}
        if(k>=n) {long m=std::max(2*(n-1)-k,0L);return ground[static_cast<size_t>(n-1)]*2-ground[static_cast<size_t>(m)];}
        return ground[static_cast<size_t>(k)];
    };
    std::vector<Vec3> smooth(static_cast<size_t>(n));
    for(long f=0;f<n;++f) {Vec3 sum{};for(long k=f-half;k<=f+half;++k)sum=sum+at(k);smooth[static_cast<size_t>(f)]=sum*(1.0f/static_cast<float>(2*half+1));}
    for(long f=0;f<n;++f) {auto& p=c.frame(static_cast<size_t>(f))[c.motion_root].p;p=p-(smooth[static_cast<size_t>(f)]-smooth[0]);}
}
std::vector<size_t> key_frames(const Clip& c,size_t count) {
    const size_t n=c.frames();
    std::vector<size_t> picked;
    if(n<=count) {for(size_t f=0;f<n;++f)picked.push_back(f);return picked;}
    // How far the body is from its first pose: the largest root-relative joint movement.
    auto relative=[&](size_t f) {
        auto world=world_pose(c,c.frame(f));Vec3 root=world[c.motion_root].p;root.y=0;
        std::vector<Vec3> out;for(const auto& joint:world)out.push_back(joint.p-root);return out;
    };
    auto first=relative(0);std::vector<float> reach(n);
    for(size_t f=0;f<n;++f) {auto now=relative(f);for(size_t j=0;j<now.size();++j) {Vec3 d=now[j]-first[j];reach[f]=std::max(reach[f],std::sqrt(dot(d,d)));}}
    // First, last, then the strongest moments (local peaks), kept apart, then the middles
    // of the widest gaps.
    picked={0,n-1};
    std::vector<size_t> peaks;
    for(size_t f=1;f+1<n;++f)if(reach[f]>=reach[f-1]&&reach[f]>reach[f+1])peaks.push_back(f);
    std::sort(peaks.begin(),peaks.end(),[&](size_t a,size_t b){return reach[a]>reach[b];});
    const size_t gap=std::max<size_t>(2,n/(2*count));
    for(size_t f:peaks) {
        if(picked.size()>=count)break;
        bool apart=true;for(size_t p:picked)apart&=(f>p?f-p:p-f)>=gap;
        if(apart)picked.push_back(f);
    }
    while(picked.size()<count) {
        std::sort(picked.begin(),picked.end());
        size_t widest=0;for(size_t i=1;i<picked.size();++i)if(picked[i]-picked[i-1]>picked[widest+1]-picked[widest])widest=i-1;
        if(picked[widest+1]-picked[widest]<2)break;
        picked.push_back((picked[widest]+picked[widest+1])/2);
    }
    std::sort(picked.begin(),picked.end());
    return picked;
}
}
