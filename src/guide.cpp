#include "guide.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numbers>
#include <set>
#include <stdexcept>

// The rules and weights follow tide-prototypes tools/motion/solve.py (a per-frame
// Levenberg-Marquardt pass over the upper body), restated on the SOMA-30 skeleton.
// Feature layout as upstream Kimodo kimodo_motionrep.py: smooth root (3), heading (2),
// root-relative joint positions (90), global cont6d rotations (180), velocities (90,
// fps * forward difference of global positions, the last repeating the one before)
// and foot contacts (4).
namespace motion {
namespace {
constexpr size_t J=30,positions=5,rotations=5+3*J,velocities=rotations+6*J;
constexpr float degrees=180/std::numbers::pi_v<float>;
Vec3 log_of(Quat q) {
    q=normalized(q);if(q.w<0)q={-q.x,-q.y,-q.z,-q.w};
    const float s=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z);
    if(s<1e-8f)return {2*q.x,2*q.y,2*q.z};
    const float angle=2*std::atan2(s,q.w);
    return {q.x/s*angle,q.y/s*angle,q.z/s*angle};
}
Quat exp_of(Vec3 v) {
    const float angle=std::sqrt(dot(v,v));
    if(angle<1e-9f)return {v.x/2,v.y/2,v.z/2,1};
    const float s=std::sin(angle/2)/angle;
    return normalized(Quat{v.x*s,v.y*s,v.z*s,std::cos(angle/2)});
}
Quat about(Vec3 axis,float angle) {const float s=std::sin(angle/2);return {axis.x*s,axis.y*s,axis.z*s,std::cos(angle/2)};}
float length(Vec3 v) {return std::sqrt(dot(v,v));}
// Closest distance between segments p1-q1 and p2-q2.
float segment_distance(Vec3 p1,Vec3 q1,Vec3 p2,Vec3 q2) {
    const Vec3 d1=q1-p1,d2=q2-p2,r=p1-p2;
    const float a=dot(d1,d1),e=dot(d2,d2),f=dot(d2,r);
    float s=0,t=0;
    if(a<=1e-12f&&e<=1e-12f)return length(r);
    if(a<=1e-12f)t=std::clamp(f/e,0.0f,1.0f);
    else {
        const float c=dot(d1,r);
        if(e<=1e-12f)s=std::clamp(-c/a,0.0f,1.0f);
        else {
            const float b=dot(d1,d2),denom=a*e-b*b;
            s=denom>1e-12f?std::clamp((b*f-c*e)/denom,0.0f,1.0f):0;
            t=(b*s+f)/e;
            if(t<0) {t=0;s=std::clamp(-c/a,0.0f,1.0f);}
            else if(t>1) {t=1;s=std::clamp((b-c)/a,0.0f,1.0f);}
        }
    }
    return length((p1+d1*s)-(p2+d2*t));
}
// Raised-cosine weight of a span: 0 outside, 1 inside, easing over `ease` frames.
float weight_in(long start,long end,float ease,size_t frame) {
    const long f=static_cast<long>(frame);
    if(f<start||f>end)return 0;
    if(ease<=0)return 1;
    const float in=static_cast<float>(f-start+1)/(ease+1),out=static_cast<float>(end-f+1)/(ease+1),x=std::min({in,out,1.0f});
    return 0.5f-0.5f*std::cos(std::numbers::pi_v<float>*x);
}
std::array<int,J> parents() {std::array<int,J> p{};auto rig=soma();for(size_t j=0;j<J;++j)p[j]=rig[j].parent;return p;}
const std::array<int,J>& parent_of() {static const auto p=parents();return p;}
void forward(const std::array<Transform,J>& local,std::array<Transform,J>& world) {
    const auto& p=parent_of();
    for(size_t j=0;j<J;++j)world[j]=p[j]<0?local[j]:world[static_cast<size_t>(p[j])]*local[j];
}
size_t joint(const std::string& name,const char* what) {
    static const auto rig=soma();
    for(size_t j=0;j<J;++j)if(rig[j].name==name)return j;
    throw std::runtime_error(std::string("Guide ")+what+" '"+name+"' is not a SOMA-30 joint");
}
// A bone of the rules' skeleton: SOMA, or the character's.
size_t bone(const Guide& g,const std::string& name,const char* what) {
    if(!g.character)return joint(name,what);
    for(size_t j=0;j<g.bones.size();++j)if(g.bones[j]==name)return j;
    throw std::runtime_error(std::string("Guide ")+what+" '"+name+"' is not a bone of the character (list non-skin bones such as a prop bone in character.bones)");
}
// The rules' skeleton posed from SOMA world transforms.
void body_of(const Guide& g,std::span<const Transform> soma_world,std::vector<Transform>& out) {
    if(g.character)proxy_world(*g.character,soma_world,out);
    else out.assign(soma_world.begin(),soma_world.end());
}
// The grip's target for a hand where it is: turned about the axis by the angle that
// best fits, within around.
Transform grip_target(const GuideGrip& g,const Transform& object,const Transform& hand) {
    if(g.around<=0)return object*g.target;
    const Transform here=inverse(object)*hand;
    const Vec3 tp=g.target.p-g.axis*dot(g.target.p,g.axis);
    float phi=0;
    if(length(tp)>0.01f) {
        const Vec3 hp=here.p-g.axis*dot(here.p,g.axis);
        phi=std::atan2(dot(g.axis,cross(tp,hp)),dot(tp,hp));
    } else {
        Quat rel=normalized(here.q*inverse(g.target.q));if(rel.w<0)rel={-rel.x,-rel.y,-rel.z,-rel.w};
        phi=2*std::atan2(rel.x*g.axis.x+rel.y*g.axis.y+rel.z*g.axis.z,rel.w);
    }
    phi=std::remainder(phi,2*std::numbers::pi_v<float>);
    phi=std::clamp(phi,-g.around,g.around);
    return object*Transform{{},about(g.axis,phi)}*g.target;
}
Vec3 capsule_end(const GuideCapsule& c,const std::vector<Transform>& w,bool second) {
    if(c.to<0)return w[c.from].p+rotate(w[c.from].q,second?c.end:c.start);
    return second?w[static_cast<size_t>(c.to)].p:w[c.from].p;
}
float overlap(const Guide& g,const GuidePair& pair,const std::vector<Transform>& w) {
    const auto& a=g.capsules[pair.a];const auto& b=g.capsules[pair.b];
    return a.radius+b.radius-segment_distance(capsule_end(a,w,false),capsule_end(a,w,true),capsule_end(b,w,false),capsule_end(b,w,true))-pair.slack;
}
// One frame's rules as residuals of the turned joints' rotation vectors.
struct Problem {
    const Guide& g; std::array<Transform,J> local; std::vector<Transform> base;
    std::vector<float> grip_k,hold_k; std::vector<float> previous; bool smooth=true;
    std::array<Transform,J> soma_world; std::vector<Transform> world;
    void pose(std::span<const float> x) {
        auto turned=local;
        for(size_t i=0;i<g.joints.size();++i) {
            const Vec3 v{x[3*i],x[3*i+1],x[3*i+2]};
            if(dot(v,v)>0)turned[g.joints[i]].q=normalized(turned[g.joints[i]].q*exp_of(v));
        }
        forward(turned,soma_world);body_of(g,soma_world,world);
    }
    // rules only: everything but the joints' own costs.
    void rules(std::vector<float>& out) {
        const auto& w=g.weights;
        auto place=[&](Vec3 at,Vec3 want,float weight){Vec3 d=(at-want)*(100*weight);out.insert(out.end(),{d.x,d.y,d.z});};
        auto turn=[&](Quat at,Quat want,float weight){Vec3 d=log_of(want*inverse(at))*(degrees*weight);out.insert(out.end(),{d.x,d.y,d.z});};
        for(size_t i=0;i<g.grips.size();++i) {
            const float k=grip_k[i];if(k<=0)continue;
            const auto& grip=g.grips[i];const auto& object=g.objects[grip.object];
            const Transform frame=world[object.bone]*object.offset,hand0=base[grip.hand];
            const Transform want=grip_target(grip,frame,world[grip.hand]);
            place(world[grip.hand].p,hand0.p+(want.p-hand0.p)*k,w.grip_place);
            if(grip.orientation)turn(world[grip.hand].q,slerp(hand0.q,want.q,k),w.grip_turn);
        }
        for(size_t i=0;i<g.holds.size();++i) {
            const float k=hold_k[i];if(k<=0)continue;
            const auto& hold=g.holds[i];
            for(size_t h=0;h<hold.hands.size();++h) {
                const Transform want=world[hold.relative]*hold.placed[h],hand0=base[hold.hands[h]];
                place(world[hold.hands[h]].p,hand0.p+(want.p-hand0.p)*k,w.hold_place);
                turn(world[hold.hands[h]].q,slerp(hand0.q,want.q,k),w.hold_turn);
            }
            for(size_t a=0;a<hold.arms.size();++a) {
                const size_t j=hold.arms[a];const int p=parent_of()[j];
                const Quat now=normalized(inverse(soma_world[static_cast<size_t>(p)].q)*soma_world[j].q);
                turn(now,hold.turns[a],w.hold_arm*k);
            }
        }
        for(const auto& keep:g.keeps) {place(world[keep.bone].p,base[keep.bone].p,w.keep_place);turn(world[keep.bone].q,base[keep.bone].q,w.keep_turn);}
        for(const auto& pair:g.pairs)out.push_back(std::max(0.0f,overlap(g,pair,world)-w.slack)*100*w.inside);
    }
    void residuals(std::span<const float> x,std::vector<float>& out) {
        out.clear();pose(x);
        for(size_t i=0;i<g.joints.size();++i)for(size_t c=0;c<3;++c) {
            out.push_back(x[3*i+c]*degrees*g.stiffness[i]);
            if(smooth)out.push_back((x[3*i+c]-previous[3*i+c])*degrees*g.weights.smooth);
        }
        rules(out);
    }
};
float squared(const std::vector<float>& r) {double s=0;for(float v:r)s+=double(v)*v;return static_cast<float>(s);}
// Levenberg-Marquardt with forward-difference derivatives (0.5 degree steps), as solve.py.
void solve(Problem& problem,std::vector<float>& x,unsigned iterations) {
    const size_t n=x.size();std::vector<float> r,trial_r,step_r;
    problem.residuals(x,r);float cost=squared(r);float damping=1e-2f;
    const float h=0.5f/degrees;
    std::vector<double> jac,a(n*n),g(n),dx(n);
    for(unsigned it=0;it<iterations;++it) {
        const size_t m=r.size();jac.assign(m*n,0);
        for(size_t i=0;i<n;++i) {
            auto stepped=x;stepped[i]+=h;problem.residuals(stepped,step_r);
            for(size_t k=0;k<m;++k)jac[k*n+i]=(step_r[k]-r[k])/h;
        }
        std::fill(a.begin(),a.end(),0.0);std::fill(g.begin(),g.end(),0.0);
        for(size_t k=0;k<m;++k)for(size_t i=0;i<n;++i) {
            const double ji=jac[k*n+i];if(ji==0)continue;
            g[i]+=ji*r[k];for(size_t l=0;l<n;++l)a[i*n+l]+=ji*jac[k*n+l];
        }
        bool improved=false;
        while(damping<1e6f) {
            // (A + damping diag(A)) dx = -g by Cholesky.
            std::vector<double> c=a;
            for(size_t i=0;i<n;++i)c[i*n+i]+=damping*(a[i*n+i]+1e-6);
            bool ok=true;
            for(size_t i=0;i<n&&ok;++i)for(size_t l=0;l<=i;++l) {
                double s=c[i*n+l];for(size_t q=0;q<l;++q)s-=c[i*n+q]*c[l*n+q];
                if(i==l) {if(s<=0){ok=false;break;}c[i*n+i]=std::sqrt(s);} else c[i*n+l]=s/c[l*n+l];
            }
            if(!ok) {damping*=4;continue;}
            for(size_t i=0;i<n;++i) {double s=-g[i];for(size_t q=0;q<i;++q)s-=c[i*n+q]*dx[q];dx[i]=s/c[i*n+i];}
            for(size_t i=n;i-->0;) {double s=dx[i];for(size_t q=i+1;q<n;++q)s-=c[q*n+i]*dx[q];dx[i]=s/c[i*n+i];}
            auto trial=x;for(size_t i=0;i<n;++i)trial[i]+=static_cast<float>(dx[i]);
            problem.residuals(trial,trial_r);const float tc=squared(trial_r);
            if(tc<cost) {x=trial;r=trial_r;cost=tc;damping=std::max(damping/3,1e-6f);improved=true;break;}
            damping*=4;
        }
        if(!improved)break;
        double norm=0;for(double v:dx)norm+=v*v;
        if(std::sqrt(norm)<0.05/degrees)break;
    }
}
Quat parse_quat(const nlohmann::json& v,const char* what) {
    if(!v.is_array()||v.size()!=4)throw std::runtime_error(std::string("Guide ")+what+" is [x, y, z, w]");
    Quat q{v[0].get<float>(),v[1].get<float>(),v[2].get<float>(),v[3].get<float>()};
    const float n=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
    if(!std::isfinite(n)||n<1e-6f)throw std::runtime_error(std::string("Guide ")+what+" is not a rotation");
    return normalized(q);
}
Vec3 parse_vec(const nlohmann::json& v,const char* what,float limit=100) {
    if(!v.is_array()||v.size()!=3)throw std::runtime_error(std::string("Guide ")+what+" is [x, y, z]");
    Vec3 p{v[0].get<float>(),v[1].get<float>(),v[2].get<float>()};
    for(float c:{p.x,p.y,p.z})if(!std::isfinite(c)||std::abs(c)>limit)throw std::runtime_error(std::string("Guide ")+what+" is out of range");
    return p;
}
float parse_number(const nlohmann::json& o,const char* key,float fallback,float low,float high) {
    if(!o.contains(key))return fallback;
    if(!o[key].is_number())throw std::runtime_error(std::string("Guide ")+key+" must be a number");
    const float v=o[key].get<float>();
    if(!std::isfinite(v)||v<low||v>high)throw std::runtime_error(std::string("Guide ")+key+" is out of range");
    return v;
}
long parse_frame(const nlohmann::json& o,const char* key,long fallback) {
    if(!o.contains(key))return fallback;
    if(!o[key].is_number_integer())throw std::runtime_error(std::string("Guide ")+key+" must be an integer frame");
    const long v=o[key].get<long>();
    if(v<-18000||v>17999)throw std::runtime_error(std::string("Guide ")+key+" is out of range");
    return v;
}
void fields(const nlohmann::json& o,std::initializer_list<const char*> allowed,const char* what) {
    if(!o.is_object())throw std::runtime_error(std::string("Guide ")+what+" must be an object");
    for(const auto& item:o.items())if(std::none_of(allowed.begin(),allowed.end(),[&](const char* a){return item.key()==a;}))
        throw std::runtime_error(std::string("Guide ")+what+" has unknown field '"+item.key()+"'");
}
// FILE.glb[@F] relative to base: a SOMA pose (F an integer, negative from the end).
std::vector<Transform> load_pose(const std::string& text,const fs::path& base,fs::path* resolved) {
    std::string file=text;long frame=0;
    auto at=text.rfind('@');
    if(at!=std::string::npos&&at>0) {
        auto suffix=text.substr(at+1);
        if(suffix=="first"||suffix=="last"||(!suffix.empty()&&suffix.find_first_not_of("-0123456789")==std::string::npos)) {
            frame=suffix=="first"?0:suffix=="last"?-1:std::stol(suffix);file=text.substr(0,at);
        }
    }
    fs::path path=utf8path(file);if(path.is_relative())path=base/path;
    auto clip=load_glb(path,"",true);
    const long index=frame<0?static_cast<long>(clip.frames())+frame:frame;
    if(index<0||index>=static_cast<long>(clip.frames()))throw std::runtime_error("Guide pose "+text+" has no such frame");
    *resolved=fs::absolute(path);
    auto pose=clip.frame(static_cast<size_t>(index));
    return {pose.begin(),pose.end()};
}
nlohmann::json read_json(const fs::path& path) {
    if(!fs::exists(path))throw std::runtime_error("Guide file not found: "+path.string());
    if(fs::file_size(path)>1024*1024)throw std::runtime_error("Guide file exceeds 1 MiB: "+path.string());
    std::ifstream in(path);nlohmann::json j;in>>j;return j;
}
}
Guide load_guide(const nlohmann::json& given,const fs::path& given_base) {
    nlohmann::json spec=given;fs::path base=given_base;
    if(spec.is_string()) {fs::path path=utf8path(spec.get<std::string>());if(path.is_relative())path=base/path;spec=read_json(path);base=path.parent_path();}
    fields(spec,{"version","character","joints","objects","grips","holds","capsules","keep","weights","strength","from","until","iterations"},"");
    if(spec.value("version",0)!=1)throw std::runtime_error("Guide version must be 1");
    Guide g;Clip rig;rig.joints=soma();
    nlohmann::json effective=spec;
    if(spec.contains("character")) {
        const auto& c=spec["character"];fields(c,{"rig","map","bones"},"character");
        fs::path file=utf8path(c.at("rig").get<std::string>());if(file.is_relative())file=base/file;
        fs::path map=utf8path(c.at("map").get<std::string>());if(map.is_relative())map=base/map;
        std::vector<std::string> extra;for(const auto& b:c.value("bones",nlohmann::json::array()))extra.push_back(b.get<std::string>());
        g.character=retarget_proxy(load_rig_glb(file,extra),load_mapping(map));
        for(const auto& j:g.character->joints)g.bones.push_back(j.name);
        effective["character"]={{"rig",fs::absolute(file).string()},{"map",fs::absolute(map).string()},{"bones",extra}};
    }
    // A SOMA pose as the rules' skeleton has it.
    auto posed=[&](const std::vector<Transform>& pose){auto w=world_pose(rig,pose);std::vector<Transform> out;body_of(g,w,out);return out;};
    // Turned joints and their stiffness (cost per degree), the upper body as solve.py.
    nlohmann::json joints=spec.value("joints",nlohmann::json{{"Spine2",0.3},{"Chest",0.3},{"Neck1",0.25},{"Head",0.2},{"LeftShoulder",0.15},{"RightShoulder",0.15},
        {"LeftArm",0.08},{"RightArm",0.08},{"LeftForeArm",0.08},{"RightForeArm",0.08},{"LeftHand",0.08},{"RightHand",0.08}});
    if(!joints.is_object()||joints.empty()||joints.size()>29)throw std::runtime_error("Guide joints maps 1..29 SOMA joint names to costs per degree");
    for(const auto& item:joints.items()) {
        const size_t j=joint(item.key(),"joint");if(j==0)throw std::runtime_error("Guide joints cannot turn Hips (the root stays as generated)");
        if(!item.value().is_number()||!(item.value().get<float>()>0)||item.value().get<float>()>100)throw std::runtime_error("Guide joint cost must be in (0, 100]");
        g.joints.push_back(j);g.stiffness.push_back(item.value().get<float>());
    }
    effective["joints"]=joints;
    if(spec.contains("weights")) {
        const auto& w=spec["weights"];
        fields(w,{"smooth","gripPlace","gripTurn","holdPlace","holdTurn","holdArm","keepPlace","keepTurn","inside","slack"},"weights");
        auto& o=g.weights;
        o.smooth=parse_number(w,"smooth",o.smooth,0,100);o.grip_place=parse_number(w,"gripPlace",o.grip_place,0,100);o.grip_turn=parse_number(w,"gripTurn",o.grip_turn,0,100);
        o.hold_place=parse_number(w,"holdPlace",o.hold_place,0,100);o.hold_turn=parse_number(w,"holdTurn",o.hold_turn,0,100);o.hold_arm=parse_number(w,"holdArm",o.hold_arm,0,100);
        o.keep_place=parse_number(w,"keepPlace",o.keep_place,0,100);o.keep_turn=parse_number(w,"keepTurn",o.keep_turn,0,100);
        o.inside=parse_number(w,"inside",o.inside,0,100);o.slack=parse_number(w,"slack",o.slack,0,0.5f);
    }
    effective["weights"]={{"smooth",g.weights.smooth},{"gripPlace",g.weights.grip_place},{"gripTurn",g.weights.grip_turn},{"holdPlace",g.weights.hold_place},
        {"holdTurn",g.weights.hold_turn},{"holdArm",g.weights.hold_arm},{"keepPlace",g.weights.keep_place},{"keepTurn",g.weights.keep_turn},{"inside",g.weights.inside},{"slack",g.weights.slack}};
    g.strength=parse_number(spec,"strength",1,0,1);g.from=parse_number(spec,"from",0,0,1);g.until=parse_number(spec,"until",1,0,1);
    if(g.from>g.until)throw std::runtime_error("Guide from must not be after until");
    g.iterations=static_cast<unsigned>(parse_number(spec,"iterations",6,1,50));
    effective["strength"]=g.strength;effective["from"]=g.from;effective["until"]=g.until;effective["iterations"]=g.iterations;
    if(spec.contains("objects")) {
        if(!spec["objects"].is_array()||spec["objects"].size()>32)throw std::runtime_error("Guide objects is a list of at most 32");
        for(const auto& o:spec["objects"]) {
            fields(o,{"name","bone","position","rotation"},"object");
            GuideObject object{o.at("name").get<std::string>(),bone(g,o.at("bone").get<std::string>(),"object bone"),{}};
            if(object.name.empty())throw std::runtime_error("Guide object needs a name");
            for(const auto& other:g.objects)if(other.name==object.name)throw std::runtime_error("Guide object '"+object.name+"' is named twice");
            if(o.contains("position"))object.offset.p=parse_vec(o["position"],"object position",10);
            if(o.contains("rotation"))object.offset.q=parse_quat(o["rotation"],"object rotation");
            g.objects.push_back(object);
        }
    }
    if(spec.contains("grips")) {
        if(!spec["grips"].is_array()||spec["grips"].size()>64)throw std::runtime_error("Guide grips is a list of at most 64");
        auto& out=effective["grips"];out=nlohmann::json::array();
        for(const auto& o:spec["grips"]) {
            fields(o,{"name","hand","object","bone","position","rotation","pose","axis","around","orientation","start","end","ease"},"grip");
            GuideGrip grip;grip.name=o.value("name",std::string("grip"));grip.hand=bone(g,o.at("hand").get<std::string>(),"grip hand");
            if(o.contains("object")==o.contains("bone"))throw std::runtime_error("Guide grip names one of object and bone");
            if(o.contains("bone")) {
                // A bone itself, as an object with no offset.
                GuideObject object{"bone:"+o["bone"].get<std::string>(),bone(g,o["bone"].get<std::string>(),"grip bone"),{}};
                auto found=std::find_if(g.objects.begin(),g.objects.end(),[&](const auto& x){return x.name==object.name;});
                grip.object=static_cast<size_t>(found-g.objects.begin());if(found==g.objects.end())g.objects.push_back(object);
            } else {
                const auto name=o["object"].get<std::string>();
                auto found=std::find_if(g.objects.begin(),g.objects.end(),[&](const auto& x){return x.name==name;});
                if(found==g.objects.end())throw std::runtime_error("Guide grip names an unknown object '"+name+"'");
                grip.object=static_cast<size_t>(found-g.objects.begin());
            }
            if(g.objects[grip.object].bone==grip.hand)throw std::runtime_error("Guide grip's hand cannot hold an object on itself");
            nlohmann::json item=o;
            if(o.contains("pose")) {
                // The hand's place on the object as a SOMA pose has it.
                if(o.contains("position")||o.contains("rotation"))throw std::runtime_error("Guide grip takes a pose or a position/rotation, not both");
                fs::path file;auto pose=load_pose(o["pose"].get<std::string>(),base,&file);
                auto w=posed(pose);const auto& object=g.objects[grip.object];
                grip.target=inverse(w[object.bone]*object.offset)*w[grip.hand];
                item["poseFile"]=file.string();
            } else {
                grip.target.p=parse_vec(o.at("position"),"grip position",10);
                if(o.contains("rotation"))grip.target.q=parse_quat(o["rotation"],"grip rotation");
            }
            item["position"]={grip.target.p.x,grip.target.p.y,grip.target.p.z};item["rotation"]={grip.target.q.x,grip.target.q.y,grip.target.q.z,grip.target.q.w};
            grip.around=parse_number(o,"around",0,0,180)/degrees;
            if(grip.around>0) {
                if(!o.contains("axis"))throw std::runtime_error("Guide grip with around needs the object's axis to turn about");
                grip.axis=normalized(parse_vec(o["axis"],"grip axis"));
                if(!(length(grip.axis)>0.5f))throw std::runtime_error("Guide grip axis is zero");
            }
            grip.orientation=o.value("orientation",true);
            grip.start=parse_frame(o,"start",0);grip.end=parse_frame(o,"end",-1);grip.ease=parse_number(o,"ease",0,0,600);
            out.push_back(item);g.grips.push_back(grip);
        }
    }
    if(spec.contains("holds")) {
        if(!spec["holds"].is_array()||spec["holds"].size()>16)throw std::runtime_error("Guide holds is a list of at most 16");
        auto& out=effective["holds"];out=nlohmann::json::array();
        for(const auto& o:spec["holds"]) {
            fields(o,{"pose","start","end","ease","hands","relativeTo","arms"},"hold");
            GuideHold hold;fs::path file;auto pose=load_pose(o.at("pose").get<std::string>(),base,&file);
            auto w=posed(pose);
            if(g.character&&(!o.contains("relativeTo")||!o.contains("hands")))throw std::runtime_error("Guide hold on a character names its relativeTo and hands bones");
            hold.relative=bone(g,o.value("relativeTo",std::string("Chest")),"hold relativeTo");
            for(const auto& h:o.value("hands",nlohmann::json{"LeftHand","RightHand"})) {
                const size_t j=bone(g,h.get<std::string>(),"hold hand");
                hold.hands.push_back(j);hold.placed.push_back(inverse(w[hold.relative])*w[j]);
            }
            for(const auto& a:o.value("arms",nlohmann::json{"LeftArm","LeftForeArm","LeftHand","RightArm","RightForeArm","RightHand"})) {
                const size_t j=joint(a.get<std::string>(),"hold arm");
                hold.arms.push_back(j);hold.turns.push_back(normalized(pose[j].q));
            }
            hold.start=parse_frame(o,"start",0);hold.end=parse_frame(o,"end",-1);hold.ease=parse_number(o,"ease",0,0,600);
            nlohmann::json item=o;item["pose"]=file.string();out.push_back(item);
            g.holds.push_back(hold);
        }
    }
    if(spec.contains("keep")) {
        if(!spec["keep"].is_array())throw std::runtime_error("Guide keep is a list of SOMA joint names");
        for(const auto& b:spec["keep"])g.keeps.push_back({bone(g,b.get<std::string>(),"keep")});
    }
    if(spec.contains("capsules")) {
        nlohmann::json c=spec["capsules"];fs::path cbase=base;
        if(c.is_string()) {fs::path path=utf8path(c.get<std::string>());if(path.is_relative())path=base/path;c=read_json(path);cbase=path.parent_path();effective["capsules"]=c;}
        fields(c,{"version","parts","pairs","note"},"capsules");
        if(c.value("version",0)!=1)throw std::runtime_error("Guide capsules version must be 1");
        if(!c.at("parts").is_array()||c["parts"].size()>64)throw std::runtime_error("Guide capsule parts is a list of at most 64");
        for(const auto& p:c["parts"]) {
            fields(p,{"name","from","to","centre","start","end","radius"},"capsule part");
            GuideCapsule part{p.at("name").get<std::string>(),bone(g,p.at("from").get<std::string>(),"capsule joint"),-1,{},{},0};
            for(const auto& other:g.capsules)if(other.name==part.name)throw std::runtime_error("Guide capsule '"+part.name+"' is named twice");
            const int kinds=int(p.contains("to"))+int(p.contains("centre"))+int(p.contains("start")||p.contains("end"));
            if(kinds!=1||p.contains("start")!=p.contains("end"))throw std::runtime_error("Guide capsule '"+part.name+"' takes a to joint, a sphere centre, or start and end points");
            if(p.contains("to"))part.to=static_cast<int>(bone(g,p["to"].get<std::string>(),"capsule joint"));
            else if(p.contains("centre"))part.start=part.end=parse_vec(p["centre"],"capsule centre",2);
            else {part.start=parse_vec(p["start"],"capsule start",5);part.end=parse_vec(p["end"],"capsule end",5);}
            part.radius=parse_number(p,"radius",-1,0,1);
            if(part.radius<0)throw std::runtime_error("Guide capsule '"+part.name+"' needs a radius");
            g.capsules.push_back(part);
        }
        auto find=[&](const std::string& name){
            for(size_t i=0;i<g.capsules.size();++i)if(g.capsules[i].name==name)return i;
            throw std::runtime_error("Guide capsule pair names an unknown part '"+name+"'");
        };
        if(!c.at("pairs").is_array()||c["pairs"].size()>2048)throw std::runtime_error("Guide capsule pairs is a list of at most 2048");
        std::set<std::pair<size_t,size_t>> seen;
        for(const auto& p:c["pairs"]) {
            GuidePair pair;
            nlohmann::json names=p;
            if(p.is_object()) {fields(p,{"pair","slack"},"capsule pair");names=p.at("pair");pair.slack=parse_number(p,"slack",0,0,0.5f);}
            if(!names.is_array()||names.size()!=2)throw std::runtime_error("Guide capsule pair is [part, part] or {\"pair\": [part, part], \"slack\": metres}");
            pair.a=find(names[0].get<std::string>());pair.b=find(names[1].get<std::string>());
            if(pair.a==pair.b)throw std::runtime_error("Guide capsule pair names one part twice");
            if(!seen.insert({std::min(pair.a,pair.b),std::max(pair.a,pair.b)}).second)throw std::runtime_error("Guide capsule pair is listed twice");
            g.pairs.push_back(pair);
        }
        (void)cbase;
    }
    if(g.empty())throw std::runtime_error("Guide has no grips, holds or capsule pairs");
    g.effective=effective;
    return g;
}
void resolve_guide(Guide& g,size_t frames) {
    auto fix=[&](long& start,long& end,const char* what) {
        if(start<0)start+=static_cast<long>(frames);
        if(end<0)end+=static_cast<long>(frames);
        if(start<0||end<start||end>=static_cast<long>(frames))throw std::runtime_error(std::string("Guide ")+what+" span is outside the "+std::to_string(frames)+"-frame output");
    };
    for(auto& grip:g.grips)fix(grip.start,grip.end,"grip");
    for(auto& hold:g.holds)fix(hold.start,hold.end,"hold");
}
bool guide_step(const Guide& g,unsigned step,unsigned steps) {
    const float progress=steps<2?1.0f:static_cast<float>(steps-1-step)/static_cast<float>(steps-1);
    return progress>=g.from-1e-6f&&progress<=g.until+1e-6f;
}
double guide_clean(const Guide& g,std::span<float> features,size_t frames) {
    const auto started=std::chrono::steady_clock::now();
    if(features.size()!=frames*kimodo_features)throw std::runtime_error("Guide features have the wrong size");
    const Clip decoded=decode_kimodo(features);
    std::vector<std::array<Vec3,J>> moved(frames);
    std::vector<std::array<Quat,J>> before(frames),after(frames);
    std::vector<bool> changed(frames,false);
    const size_t n=3*g.joints.size();
    // Each frame's problem; turns from the decoded pose.
    std::vector<Problem> problems;problems.reserve(frames);
    std::vector<std::array<Transform,J>> start(frames);
    std::vector<bool> active(frames,false);std::vector<float> rules;
    for(size_t f=0;f<frames;++f) {
        problems.push_back(Problem{g,{},{},{},{},std::vector<float>(n,0),true,{},{}});
        auto& problem=problems.back();
        auto pose=decoded.frame(f);
        for(size_t j=0;j<J;++j)problem.local[j]=pose[j];
        forward(problem.local,start[f]);body_of(g,start[f],problem.base);
        for(const auto& grip:g.grips)problem.grip_k.push_back(weight_in(grip.start,grip.end,grip.ease,f));
        for(const auto& hold:g.holds)problem.hold_k.push_back(weight_in(hold.start,hold.end,hold.ease,f));
        std::vector<float> zero(n,0);problem.pose(zero);rules.clear();problem.rules(rules);
        active[f]=!g.keeps.empty()||squared(rules)>0;
    }
    // Forward from the frame before, then back from the frame after starting at the
    // forward answer (solve.py's forward pass alone let a correction fade for half a
    // second after a grip). Frames the forward pass left alone stay as predicted.
    std::vector<std::vector<float>> x(frames,std::vector<float>(n,0));
    auto zero=[](const std::vector<float>& v){return std::all_of(v.begin(),v.end(),[](float c){return c==0;});};
    for(size_t f=0;f<frames;++f) {
        auto& problem=problems[f];
        problem.smooth=f>0;if(f>0)problem.previous=x[f-1];
        if(!active[f]&&(f==0||zero(x[f-1])))continue;
        x[f]=f>0?x[f-1]:x[f];solve(problem,x[f],g.iterations);
    }
    for(size_t f=frames;f-->0;) {
        auto& problem=problems[f];
        problem.smooth=f+1<frames;if(f+1<frames)problem.previous=x[f+1];
        if(!active[f]&&zero(x[f]))continue;
        solve(problem,x[f],g.iterations);
    }
    for(size_t f=0;f<frames;++f) {
        if(zero(x[f]))continue;
        auto& problem=problems[f];problem.pose(x[f]);
        for(size_t j=0;j<J;++j) {before[f][j]=start[f][j].q;after[f][j]=problem.soma_world[j].q;moved[f][j]=problem.soma_world[j].p-start[f][j].p;}
        changed[f]=true;
    }
    // Positions, rotations and velocities move by the solved pose's change; the root,
    // heading and contacts stay as the model predicted them.
    auto rot6=[](Quat q,float* out){Vec3 a=rotate(q,{1,0,0}),b=rotate(q,{0,1,0});out[0]=a.x;out[1]=a.y;out[2]=a.z;out[3]=b.x;out[4]=b.y;out[5]=b.z;};
    const float s=g.strength,fps=30;
    auto dp=[&](size_t f,size_t j){return changed[f]?moved[f][j]:Vec3{};};
    for(size_t f=0;f<frames;++f) {
        float* row=features.data()+f*kimodo_features;
        for(size_t j=0;j<J;++j) {
            if(changed[f]) {
                const Vec3 d=moved[f][j];
                row[positions+3*j]+=s*d.x;row[positions+3*j+1]+=s*d.y;row[positions+3*j+2]+=s*d.z;
                float a[6],b[6];rot6(after[f][j],a);rot6(before[f][j],b);
                for(size_t c=0;c<6;++c)row[rotations+6*j+c]+=s*(a[c]-b[c]);
            }
            const size_t from=f+1<frames?f:(frames>1?f-1:f);
            const Vec3 v=(frames>1)?(dp(from+1,j)-dp(from,j))*fps:Vec3{};
            row[velocities+3*j]+=s*v.x;row[velocities+3*j+1]+=s*v.y;row[velocities+3*j+2]+=s*v.z;
        }
    }
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}
nlohmann::json guide_report(const Guide& g,const Clip& clip) {
    require_soma(clip);
    nlohmann::json out=nlohmann::json::object();
    std::vector<std::vector<Transform>> world(clip.frames());
    for(size_t f=0;f<clip.frames();++f) {std::array<Transform,J> local,soma_world;auto p=clip.frame(f);for(size_t j=0;j<J;++j)local[j]=p[j];forward(local,soma_world);body_of(g,soma_world,world[f]);}
    auto round=[](double v,double unit){return std::round(v/unit)*unit;};
    if(!g.grips.empty()) {
        out["grips"]=nlohmann::json::array();
        for(const auto& grip:g.grips) {
            double sum=0,worst=0,turn=0;size_t count=0;
            for(long f=grip.start;f<=grip.end&&f<static_cast<long>(clip.frames());++f) {
                const auto& w=world[static_cast<size_t>(f)];const auto& object=g.objects[grip.object];
                const Transform want=grip_target(grip,w[object.bone]*object.offset,w[grip.hand]);
                const double d=length(w[grip.hand].p-want.p);sum+=d;worst=std::max(worst,d);++count;
                turn=std::max(turn,static_cast<double>(length(log_of(want.q*inverse(w[grip.hand].q))))*degrees);
            }
            out["grips"].push_back({{"name",grip.name},{"start",grip.start},{"end",grip.end},{"offMeanMetres",round(count?sum/static_cast<double>(count):0,1e-4)},
                {"offMaxMetres",round(worst,1e-4)},{"turnMaxDegrees",round(turn,0.1)}});
        }
    }
    if(!g.holds.empty()) {
        out["holds"]=nlohmann::json::array();
        for(const auto& hold:g.holds) {
            double sum=0,worst=0;size_t count=0;
            for(long f=hold.start;f<=hold.end&&f<static_cast<long>(clip.frames());++f)for(size_t h=0;h<hold.hands.size();++h) {
                const auto& w=world[static_cast<size_t>(f)];
                const double d=length(w[hold.hands[h]].p-(w[hold.relative]*hold.placed[h]).p);sum+=d;worst=std::max(worst,d);++count;
            }
            out["holds"].push_back({{"start",hold.start},{"end",hold.end},{"offMeanMetres",round(count?sum/static_cast<double>(count):0,1e-4)},{"offMaxMetres",round(worst,1e-4)}});
        }
    }
    if(!g.pairs.empty()) {
        double worst=0;std::string where;size_t at=0,over=0;
        for(size_t f=0;f<clip.frames();++f) {
            double frame_worst=0;
            for(const auto& pair:g.pairs) {
                const double depth=overlap(g,pair,world[f])-g.weights.slack;
                frame_worst=std::max(frame_worst,depth);
                if(depth>worst) {worst=depth;where=g.capsules[pair.a].name+" in "+g.capsules[pair.b].name;at=f;}
            }
            if(frame_worst>0.02)++over;
        }
        out["capsules"]={{"worstMetres",round(worst,1e-4)},{"worstPair",where},{"worstFrame",at},{"framesOver2cm",over}};
    }
    return out;
}
}
