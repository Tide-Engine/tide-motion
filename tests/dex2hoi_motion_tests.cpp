#include "dex2hoi_motion.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>

using namespace motion;
namespace {
void check(bool condition,const char* why) {if(!condition)throw std::runtime_error(why);}
template<class F>void refuses(F f) {bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"Expected Dex2HOI representation refusal");}
double distance(Vec3 a,Vec3 b) {const auto d=a-b;return std::sqrt(static_cast<double>(dot(d,d)));}
bool same(Quat a,Quat b) {a=normalized(a);b=normalized(b);return std::abs(a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w)>1-1e-6f;}
Quat axis(Vec3 direction,float angle) {direction=normalized(direction);const float s=std::sin(angle/2);return normalized({direction.x*s,direction.y*s,direction.z*s,std::cos(angle/2)});}
// Independent double-precision homogeneous matrix reference. This explicitly
// evaluates the published rest-world conjugation FK, rather than Clip FK.
using Matrix=std::array<double,16>;
Matrix matrix(Transform t) {
    const double x=t.q.x,y=t.q.y,z=t.q.z,w=t.q.w;
    const double n=x*x+y*y+z*z+w*w,s=2/n;
    return {1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w),t.p.x,
        s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w),t.p.y,
        s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y),t.p.z,0,0,0,1};
}
Matrix multiply(const Matrix& a,const Matrix& b) {
    Matrix c{};for(size_t i=0;i<4;++i)for(size_t j=0;j<4;++j)for(size_t k=0;k<4;++k)c[i*4+j]+=a[i*4+k]*b[k*4+j];return c;
}
Matrix invert(const Matrix& a) {
    Matrix b{};b[15]=1;for(size_t i=0;i<3;++i)for(size_t j=0;j<3;++j)b[i*4+j]=a[j*4+i];
    for(size_t i=0;i<3;++i)for(size_t j=0;j<3;++j)b[i*4+3]-=b[i*4+j]*a[j*4+3];
    return b;
}
Vec3 position(const Matrix& a) {return {static_cast<float>(a[3]),static_cast<float>(a[7]),static_cast<float>(a[11])};}
Dex2HoiRig rig_fixture() {
    Dex2HoiRig rig;
    for(size_t j=0;j<65;++j) {
        int parent=0;
        if(j==0)parent=-1;
        else if(j<=6)parent=static_cast<int>(j)-1;
        else if(j==7||j==31)parent=3;
        else if(j<=10||(j>=32&&j<=34))parent=static_cast<int>(j)-1;
        else if(j>=11&&j<=30)parent=(j-11)%4==0?10:static_cast<int>(j)-1;
        else if(j>=35&&j<=54)parent=(j-35)%4==0?34:static_cast<int>(j)-1;
        else if(j==55||j==60)parent=0;
        else parent=static_cast<int>(j)-1;
        Transform rest;
        if(j==0)rest.p={0.2f,1.1f,-0.15f};
        else if(j>=11&&j<=54)rest.p={0.015f,0.025f,0.005f};
        else rest.p={j%2?0.06f:-0.04f,0.05f,0.015f};
        rest.q=axis({1,static_cast<float>(j%3+1),0.4f},0.025f*static_cast<float>(j%9+1));
        rig.joints.push_back({"owner_bone_"+std::to_string(j),parent,rest});rig.bones[j]=j;
    }
    rig.joints.push_back({"owner_helper",10,{{0.02f,0.01f,0},axis({1,0,0},0.2f)}});
    return rig;
}
Clip history_fixture(const Dex2HoiRig& rig) {
    Clip c;c.joints=rig.joints;c.motion_root=rig.bones[0];
    for(size_t f=0;f<3;++f)for(size_t j=0;j<c.joints.size();++j) {
        Transform delta;
        if(j<65)delta.q=axis({0.4f,1,static_cast<float>(j%5+1)},0.03f*static_cast<float>((j+f)%13));
        if(j==0)delta.p={0.2f+static_cast<float>(f)*0.02f,0.1f,-0.3f};
        c.poses.push_back(c.joints[j].rest*delta);
    }
    return c;
}
std::array<Dex2HoiObject,2> object_fixture() {
    std::array<Dex2HoiObject,2> objects;
    for(size_t s=0;s<2;++s) {
        objects[s].name=s?"owner_shield":"owner_sword";
        objects[s].surface_points={{0,0,0},{0.02f,0.03f,0.04f},{-0.03f,0.05f,-0.02f}};
        for(size_t f=0;f<3;++f)objects[s].poses.push_back({{-0.3f+0.04f*static_cast<float>(f)+0.4f*static_cast<float>(s),1.4f,0.2f},axis({1,2,3},0.3f*static_cast<float>(f+s))});
    }
    return objects;
}
void verify_fk(const Dex2HoiRig& rig,const Clip& clip,const std::vector<float>& raw,const Dex2HoiOrigins& origins) {
    std::vector<Matrix> rest;
    for(size_t j=0;j<rig.joints.size();++j) {
        auto r=matrix(rig.joints[j].rest);if(rig.joints[j].parent>=0)r=multiply(rest[static_cast<size_t>(rig.joints[j].parent)],r);rest.push_back(r);
    }
    for(size_t f=0;f<clip.frames();++f) {
        std::vector<Matrix> deformations;const auto world=world_pose(clip,clip.frame(f));
        for(size_t j=0;j<rig.joints.size();++j) {
            if(j<65) {
                const float* p=raw.data()+f*dex2hoi_features+j*6;
                // Reference row Gram-Schmidt in double, independent of native quaternion reconstruction.
                std::array<double,3> a{p[0],p[1],p[2]},b{p[3],p[4],p[5]};
                auto norm=[](std::array<double,3>& v){double n=0;for(double x:v)n+=x*x;n=std::sqrt(n);for(double& x:v)x/=n;};
                norm(a);double d=0;for(size_t k=0;k<3;++k)d+=a[k]*b[k];for(size_t k=0;k<3;++k)b[k]-=d*a[k];norm(b);
                const std::array<double,3> z{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
                Matrix pose{a[0],a[1],a[2],0,b[0],b[1],b[2],0,z[0],z[1],z[2],0,0,0,0,1};
                if(j==0){pose[3]=raw[f*dex2hoi_features+390]+origins.root.x;pose[7]=raw[f*dex2hoi_features+391]+origins.root.y;pose[11]=raw[f*dex2hoi_features+392]+origins.root.z;}
                auto deformation=multiply(multiply(rest[j],pose),invert(rest[j]));
                if(rig.joints[j].parent>=0)deformation=multiply(deformations[static_cast<size_t>(rig.joints[j].parent)],deformation);
                deformations.push_back(deformation);
            } else deformations.push_back(deformations[static_cast<size_t>(rig.joints[j].parent)]);
            const auto expected=multiply(deformations[j],rest[j]);
            check(distance(position(expected),world[j].p)<3e-6,"Independent rest-conjugation FK matches ordinary Clip positions");
            const auto actual=matrix(world[j]);for(size_t a=0;a<3;++a)for(size_t b=0;b<3;++b)check(std::abs(expected[a*4+b]-actual[a*4+b])<3e-6,"Independent rest-conjugation FK matches ordinary Clip rotations");
        }
    }
}
}
int main()try {
    const auto r=dex2hoi_rotation6(yaw(std::numbers::pi_v<float>/2));
    check(std::abs(r[0])<2e-7f&&std::abs(r[2]-1)<2e-7f&&std::abs(r[4]-1)<2e-7f,"Published 6D values are matrix rows");
    const std::array<float,6> quarter_turn_rows{0,0,1,0,1,0};
    // Allow accumulated float rounding through normalization, basis/quaternion
    // conversion and rotation (including fused multiply-add on ARM).
    constexpr double rotation_tolerance=8*std::numeric_limits<float>::epsilon();
    for(const auto q:{dex2hoi_rotation6(quarter_turn_rows),dex2hoi_rotation6(r)}) {
        check(distance(rotate(q,{1,0,0}),{0,0,-1})<rotation_tolerance,"Row-basis decoding has correct handedness");
        check(distance(rotate(q,{0,1,0}),{0,1,0})<rotation_tolerance,"Row-basis decoding preserves the yaw axis");
        check(distance(rotate(q,{0,0,1}),{1,0,0})<rotation_tolerance,"Row-basis decoding preserves the third basis axis");
    }
    refuses([]{dex2hoi_rotation6(std::array<float,6>{});});
    const auto init=dex2hoi_initial_features({{0.25f,-0.5f,1},yaw(0.3f)});
    check(std::abs(init[3]-std::sqrt(0.5f))<1e-7&&std::abs(init[4]+1)<1e-7&&std::abs(init[8]+1)<1e-7,"Spatial63 frequency/xyz/sin/cos order");
    check(std::abs(init[69])<1e-7&&std::abs(init[70]-0.3f)<1e-7&&std::abs(init[71])<1e-7,"Initial Euler XYZ uses radians");
    // Independent mixture equations stress each learned branch and extreme logits.
    std::mt19937 random(20261028);std::uniform_real_distribution<float> value(-2,2);
    double max_error=0;
    for(size_t trial=0;trial<256;++trial) {
        Transform left{{value(random),value(random),value(random)},axis({1,2,3},value(random))};
        Transform right{{value(random),value(random),value(random)},axis({3,1,2},value(random))};
        Transform object{{value(random),value(random),value(random)},axis({2,3,1},value(random))};
        auto slot=dex2hoi_encode_object18(object,left,right);
        slot[15]=-1000;slot[16]=-1000;slot[17]=-1000;slot[15+trial%3]=1000;
        const auto decoded=dex2hoi_decode_object18(slot,left,right);
        const auto lm=matrix(left),rm=matrix(right);std::array<double,3> expected{};
        for(size_t a=0;a<3;++a) {
            if(trial%3==2)expected[a]=slot[12+a];
            else {const auto& m=trial%3?rm:lm;expected[a]=m[a*4+3];for(size_t b=0;b<3;++b)expected[a]+=m[a*4+b]*slot[(trial%3?9:6)+b];}
        }
        const std::array<float,3> actual{decoded.p.x,decoded.p.y,decoded.p.z};
        for(size_t a=0;a<3;++a)max_error=std::max(max_error,std::abs(expected[a]-actual[a]));
        check(same(decoded.q,object.q),"Object global rotation round trip");
    }
    check(max_error<3e-6,"Native mixture agrees with independent double matrices");
    auto rig=rig_fixture();const auto history=history_fixture(rig);auto objects=object_fixture();
    Dex2HoiNormalizer normalizer;
    for(size_t j=0;j<dex2hoi_features;++j){normalizer.mean[j]=0.001f*static_cast<float>(j%7);normalizer.stddev[j]=0.2f+0.03f*static_cast<float>(j%11);}
    normalizer.stddev[7]=0;normalizer.stddev[12]=1e-8f;
    const auto encoded=encode_dex2hoi_history(rig,history,objects,{},normalizer);
    auto decoded=decode_dex2hoi_motion(rig,encoded.features,encoded.origins,objects,2,{},normalizer);
    for(size_t f=0;f<history.frames();++f) {
        for(size_t j=0;j<history.joints.size();++j) {
            check(distance(history.frame(f)[j].p,decoded.body.frame(f)[j].p)<2e-6,"Local body translations round trip");
            check(same(history.frame(f)[j].q,decoded.body.frame(f)[j].q),"Pose-basis deltas round trip with nonidentity rest rotations");
        }
        for(size_t s=0;s<2;++s)check(distance(objects[s].poses[f].p,decoded.objects[s].poses[f].p)<3e-6,"18D object histories round trip in displaced format");
    }
    auto raw=encoded.features;normalizer.denormalize(raw);verify_fk(rig,decoded.body,raw,encoded.origins);
    // Surface distance, rather than object centre distance, selects the touching hand.
    const std::array<Vec3,1> left_tip{{{0,0,0}}},right_tip{{{3,0,0}}},surface{{{0,0,0}}};
    auto contact=dex2hoi_encode_object18({{2,0,0},{}},{{1,0,0},{}},{{2,0,0},{}},left_tip,right_tip,surface);
    check(contact[15]>contact[16]+10,"Supplied fingertip-surface contact controls GT prefix logits");
    // Prefix uses free translation even if arbitrary branch values disagree.
    auto modified=raw;modified[393+6]=100;modified[393+15]=1000;modified[393+16]=-1000;modified[393+17]=-1000;
    auto prefix=decode_dex2hoi_motion(rig,modified,encoded.origins,objects,1);
    check(distance(prefix.objects[0].poses[0].p,objects[0].poses[0].p)<2e-6,"GT prefix free-translation override ignores learned branch weights");
    auto prediction=decode_dex2hoi_motion(rig,modified,encoded.origins,objects,0);
    check(distance(prediction.objects[0].poses[0].p,objects[0].poses[0].p)>10,"Predicted output decodes actual wrist mixture branches");
    // Every supported displacement/root-relative variant round trips, including legacy object x0.
    for(bool relative:{false,true})for(bool displacement:{false,true})for(bool x0:{false,true}) {
        Dex2HoiFormat format;format.object_relative_to_root=relative;format.object_displacement=displacement;format.x0_as_condition=x0;format.root_displacement=x0;
        const auto e=encode_dex2hoi_history(rig,history,objects,format);
        const auto d=decode_dex2hoi_motion(rig,e.features,e.origins,objects,1,format);
        for(size_t f=0;f<history.frames();++f)for(size_t s=0;s<2;++s)check(distance(d.objects[s].poses[f].p,objects[s].poses[f].p)<3e-6,"Object displacement/root-relative mode round trip");
    }
    auto invalid=rig;invalid.bones[1]=invalid.bones[0];refuses([&]{invalid.validate();});
    invalid=rig;invalid.joints[1].scale=0.01f;refuses([&]{invalid.validate();});
    auto moving=history;moving.frame(1)[1].p.x+=0.01f;refuses([&]{encode_dex2hoi_history(rig,moving,objects);});
    moving=history;moving.frame(1)[65].q=axis({1,0,0},0.4f);refuses([&]{encode_dex2hoi_history(rig,moving,objects);});
    auto malformed=encoded.features;malformed.pop_back();refuses([&]{decode_dex2hoi_motion(rig,malformed,encoded.origins,objects);});
    malformed=encoded.features;malformed[200]=std::numeric_limits<float>::quiet_NaN();refuses([&]{decode_dex2hoi_motion(rig,malformed,encoded.origins,objects);});
    refuses([&]{decode_dex2hoi_motion(rig,encoded.features,encoded.origins,objects,4);});
    auto duplicate=objects;duplicate[1].name=duplicate[0].name;refuses([&]{encode_dex2hoi_history(rig,history,duplicate);});
    duplicate[1].export_track=false;check(decode_dex2hoi_motion(rig,encoded.features,encoded.origins,duplicate).objects.size()==1,"Explicit duplicate conditioning slot can suppress its exported track");
    auto missing=objects;missing[1].present=false;const auto e=encode_dex2hoi_history(rig,history,missing);check(decode_dex2hoi_motion(rig,e.features,e.origins,missing).objects.size()==1,"Absent second object is masked from export");
    std::cout<<"Dex2HOI row rotations, initial72, normalization, explicit rest-basis FK, displacement, surface contact, GT prefix and validation contracts pass; mixture max error "<<max_error<<" m\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
