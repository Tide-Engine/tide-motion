#include "equipment.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace motion;
using nlohmann::json;
namespace {
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
float distance(Vec3 a,Vec3 b){auto d=a-b;return std::sqrt(dot(d,d));}
bool same(Transform a,Transform b){return distance(a.p,b.p)<1e-7f&&std::abs(a.q.x*b.q.x+a.q.y*b.q.y+a.q.z*b.q.z+a.q.w*b.q.w)>0.999999f;}
bool exact(Transform a,Transform b) {
    return a.p.x==b.p.x&&a.p.y==b.p.y&&a.p.z==b.p.z&&a.q.x==b.q.x&&a.q.y==b.q.y&&a.q.z==b.q.z&&a.q.w==b.q.w;
}
template<class F>void refuses(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"Expected equipment refusal");}
Clip body(float scale=1) {
    Clip c;c.joints={{"Root",-1,{},scale},{"LeftArm",0,{{-0.3f/scale,1/scale,0},{}}},{"LeftForeArm",1,{{0,-0.4f/scale,0},{}}},{"LeftHand",2,{{0,-0.4f/scale,0},{}}},
        {"RightArm",0,{{0.3f/scale,1/scale,0},{}}},{"RightForeArm",4,{{0,-0.4f/scale,0},{}}},{"RightHand",5,{{0,-0.4f/scale,0},{}}}};
    for(size_t f=0;f<5;++f)for(size_t j=0;j<c.joints.size();++j){auto t=c.joints[j].rest;if(j==0)t.p.x=static_cast<float>(f)*0.01f;c.poses.push_back(t);}
    return c;
}
EquipmentPlan plan() {
    EquipmentPlan p;EquipmentObject o;o.name="sword";EquipmentSegment s;s.start=0;s.end=4;s.keys={{0,{{0,0.25f,0.1f},{}},{1,1,1}},{4,{{0,0.25f,0.14f},{}},{1,1,1}}};o.segments.push_back(s);p.objects.push_back(o);
    for(int side=0;side<2;++side) {EquipmentGrip g;g.name=side?"right":"left";g.object="sword";g.bone=side?"RightHand":"LeftHand";
        g.chain=side?std::vector<std::string>{"RightArm","RightForeArm","RightHand"}:std::vector<std::string>{"LeftArm","LeftForeArm","LeftHand"};
        g.start=1;g.end=3;g.target.p={side?0.3f:-0.3f,0,0};g.position_tolerance=0.0001f;g.angle_tolerance=0.05f;g.max_rotation=150;g.max_step_increase=180;p.grips.push_back(g);}
    return p;
}
Clip held_body(float scale) {
    auto source=body(scale);Clip c=source;c.poses.clear();
    c.joints.push_back({"LeftFinger",3,{{0.025f/scale,0.01f/scale,0},{}}});
    c.joints.push_back({"RightFinger",6,{{-0.025f/scale,0.01f/scale,0},{}}});
    for(size_t f=0;f<source.frames();++f) {
        for(size_t j=0;j<source.joints.size();++j) {
            auto t=source.frame(f)[j];
            if(j==0)t.q=yaw(0.15f*static_cast<float>(f));
            if(j==1||j==4)t.q=preview_offset({}, {12,8*static_cast<float>(f),3}).q;
            if(j==2||j==5)t.q=preview_offset({}, {-9,13,5*static_cast<float>(f)}).q;
            if(j==3||j==6)t.q=preview_offset({}, {4*static_cast<float>(f),17,-11}).q;
            c.poses.push_back(t);
        }
        for(size_t j=7;j<9;++j) {auto t=c.joints[j].rest;t.q=yaw(0.1f*static_cast<float>(f));c.poses.push_back(t);}
    }
    c.validate();return c;
}
EquipmentPlan held_plan() {
    EquipmentPlan p;EquipmentObject o;o.name="held-sword";EquipmentSegment seg;seg.space="grip";seg.anchor="right-palm";seg.end=4;
    seg.keys={{0,{},{2,0.5f,3}},{4,{},{2,0.5f,3}}};o.segments.push_back(seg);p.objects.push_back(o);
    EquipmentGrip g;g.name="right-palm";g.object=o.name;g.bone="RightHand";g.end=4;
    g.hand_offset=preview_offset({0.02f,0.01f,-0.03f},{-12,9,28});
    g.target=preview_offset({0.06f,-0.03f,0.04f},{17,32,-13});
    g.position_tolerance=0.00001f;g.angle_tolerance=0.01f;p.grips.push_back(g);return p;
}
void grip_anchor_contracts() {
    const auto p=held_plan();std::vector<EquipmentResult> outputs;
    for(float scale:{1.0f,0.01f}) {
        const auto c=held_body(scale);auto result=author_equipment(c,p);
        check(result.report["allGripsSatisfied"].get<bool>(),"Named grip aligns nonidentity palm and object markers");
        check(result.report["objectTrajectoryBasis"]=="source-body-before-grip-correction","Grip anchor uses original body basis");
        for(size_t f=0;f<c.frames();++f) {
            const auto world=world_pose(c,c.frame(f));
            const auto palm=world[6]*p.grips[0].hand_offset;
            const auto target=result.objects[0].poses[f]*p.grips[0].target;
            check(same(palm,target),"Independent FK marker equality at every grip sample");
            check(result.objects[0].scales[f].x==2&&result.objects[0].scales[f].y==0.5f&&result.objects[0].scales[f].z==3,"Grip anchor preserves mesh-only scale keys");
            for(size_t j=0;j<c.joints.size();++j)check(exact(c.frame(f)[j],result.body.frame(f)[j]),"Grip attachment preserves exact body/root/finger transforms");
        }
        outputs.push_back(std::move(result));
    }
    for(size_t f=0;f<5;++f)check(same(outputs[0].objects[0].poses[f],outputs[1].objects[0].poses[f]),"Grip offsets stay in metres on centimetre source rigs");
    auto c=held_body(1);auto bad=p;bad.objects[0].segments[0].anchor="missing";refuses([&]{author_equipment(c,bad);});
    bad=p;bad.grips[0].bone="missing";refuses([&]{author_equipment(c,bad);});
    bad=p;bad.grips.push_back(bad.grips[0]);refuses([&]{author_equipment(c,bad);});
    bad=p;bad.grips[0].name="";refuses([&]{author_equipment(c,bad);});
    bad=p;bad.grips[0].start=1;refuses([&]{author_equipment(c,bad);});
    bad=p;bad.grips[0].end=3;refuses([&]{author_equipment(c,bad);});
    bad=p;auto other=bad.objects[0];other.name="other";other.segments[0].space="world";other.segments[0].anchor="";bad.objects.push_back(other);bad.grips[0].object="other";refuses([&]{author_equipment(c,bad);});
    auto offset=p;for(auto& key:offset.objects[0].segments[0].keys)key.local.p.x=0.02f;
    auto moved=author_equipment(c,offset);check(!moved.report["allGripsSatisfied"].get<bool>(),"Authored grip-space offsets remain visible to ordinary grip validation");
    auto release=p;auto& held=release.objects[0].segments[0];held.end=2;held.keys.back().frame=2;release.grips[0].end=2;
    EquipmentSegment world;world.start=3;world.end=4;world.keys={{3,{{1,1,1},{}},{1,1,1}},{4,{{1,1,2},{}},{1,1,1}}};release.objects[0].segments.push_back(world);
    release.events.push_back({{"name","release"},{"frame",3},{"object","held-sword"}});auto detached=author_equipment(c,release);
    check(detached.report["allGripsSatisfied"].get<bool>()&&same(detached.objects[0].poses[4],{{1,1,2},{}}),"Grip-to-world release keeps explicit flight independent");
}
}
int main(int argc,char** argv)try {
    check(argc==2,"Need isolated equipment fixture directory");fs::path dir=argv[1];fs::create_directories(dir);
    auto c=body();auto p=plan();auto diagnosed=author_equipment(c,p);
    grip_anchor_contracts();
    check(!diagnosed.report["allGripsSatisfied"].get<bool>(),"Diagnosis catches wrist drift on every contact frame");
    auto result=author_equipment(c,p,true);check(result.report["allGripsSatisfied"].get<bool>(),"Both hands meet moving object targets after IK");
    for(size_t f=0;f<c.frames();++f)for(size_t j=0;j<c.joints.size();++j) {
        check(distance(c.frame(f)[j].p,result.body.frame(f)[j].p)<1e-8f,"All body translations preserved");
        if(f==0||f==4||j==0)check(same(c.frame(f)[j],result.body.frame(f)[j]),"Root and outside-contact poses unchanged");
    }
    for(size_t f=1;f<=3;++f) {
        auto w=world_pose(result.body,result.body.frame(f));
        check(distance(w[3].p,result.objects[0].poses[f].p+p.grips[0].target.p)<0.0001f,"Independent FK verifies left target");
        check(distance(w[6].p,result.objects[0].poses[f].p+p.grips[1].target.p)<0.0001f,"Independent FK verifies right target");
    }
    auto scaled=author_equipment(body(0.01f),p,true);check(scaled.report["allGripsSatisfied"].get<bool>(),"Centimetre rig correction works in metres");
    auto palm=p;palm.grips[0].orientation=false;palm.grips[0].hand_offset.p={0.02f,0.01f,0.03f};
    auto palm_result=author_equipment(c,palm,true);check(palm_result.report["allGripsSatisfied"].get<bool>(),"Position-only palm offset uses preserved hand orientation");
    auto guarded=p;guarded.protected_frames={2};auto no=author_equipment(c,guarded,true);
    check(no.report["grips"][0]["status"]=="rejected"&&!no.report["allGripsSatisfied"].get<bool>(),"Protected frame rejects incomplete correction");
    for(size_t f=0;f<c.frames();++f)for(size_t j=0;j<c.joints.size();++j)check(same(c.frame(f)[j],no.body.frame(f)[j]),"Rejected intervals roll back fully");
    guarded=p;guarded.grips[0].max_correction=0.01f;no=author_equipment(c,guarded,true);check(no.report["grips"][0]["reason"]=="correction-bound","Distance safeguard");
    guarded=p;guarded.grips[0].max_rotation=1;no=author_equipment(c,guarded,true);check(no.report["grips"][0]["reason"]=="rotation-bound","Rotation safeguard");
    guarded=p;guarded.grips[0].max_step_increase=1;no=author_equipment(c,guarded,true);check(no.report["grips"][0]["reason"]=="step-increase-bound","Interval boundary step safeguard");
    guarded=p;guarded.objects[0].segments[0].keys[0].local.p.z=0.9f;guarded.objects[0].segments[0].keys[1].local.p.z=0.9f;guarded.grips[0].max_correction=1;
    no=author_equipment(c,guarded,true);check(no.report["grips"][0]["reason"]=="unreachable","Unreachable grip reported");
    guarded=p;guarded.objects[0].segments[0].end=100;refuses([&]{author_equipment(c,guarded);});
    guarded=p;guarded.grips.push_back(guarded.grips.front());refuses([&]{author_equipment(c,guarded,true);});
    guarded=p;guarded.grips[0].chain={"Root","LeftForeArm","LeftHand"};refuses([&]{author_equipment(c,guarded,true);});
    // A bow holds the arrow before a release, then its authored world track is independent.
    EquipmentPlan bow;EquipmentObject b;b.name="bow";EquipmentSegment held;held.end=4;held.space="bone";held.anchor="LeftHand";held.keys={{0,{},{1,1,1}},{4,{},{1,1,1}}};b.segments.push_back(held);bow.objects.push_back(b);
    EquipmentObject arrow;arrow.name="arrow";EquipmentSegment nocked;nocked.end=2;nocked.space="object";nocked.anchor="bow";nocked.keys={{0,{{0,0,0.1f},{}},{1,1,1}},{2,{{0,0,0.3f},{}},{1,1,1}}};arrow.segments.push_back(nocked);
    EquipmentSegment flight;flight.start=3;flight.end=4;flight.keys={{3,{{-0.28f,0.2f,0.3f},{}},{1,1,1}},{4,{{-0.28f,0.2f,2},{}},{1,1,1}}};arrow.segments.push_back(flight);bow.objects.push_back(arrow);
    EquipmentString string;string.name="string";string.object="bow";string.top={0,0.6f,0};string.bottom={0,-0.6f,0};string.nock={{0,{{0,0,0.1f},{}},{1,1,1}},{2,{{0,0,0.3f},{}},{1,1,1}},{3,{},{1,1,1}},{4,{},{1,1,1}}};bow.strings.push_back(string);
    bow.events.push_back({{"name","release"},{"frame",3},{"object","arrow"}});auto drawn=author_equipment(c,bow);
    check(drawn.objects.size()==4&&drawn.objects[2].builtin_rod,"Two animated string pieces baked");
    check(distance(drawn.objects[1].poses[4].p,{-0.28f,0.2f,2})<1e-6f,"Arrow world flight is independent after release");
    check(std::abs(drawn.events[0]["timeSeconds"].get<double>()-0.1)<1e-7,"Release event frame/time preserved");
    auto rod=drawn.objects[2].poses[2];float half=drawn.objects[2].scales[2].y*0.5f;
    check(distance(rod.p+rotate(rod.q,{0,-half,0}),drawn.objects[0].poses[2].p+Vec3{0,0.6f,0})<1e-6f,"String first endpoint independently verified");
    check(distance(rod.p+rotate(rod.q,{0,half,0}),drawn.objects[0].poses[2].p+Vec3{0,0,0.3f})<1e-6f,"String nock endpoint independently verified");
    guarded=bow;guarded.objects[0].segments[0].space="object";guarded.objects[0].segments[0].anchor="arrow";refuses([&]{author_equipment(c,guarded);});
    guarded=bow;guarded.events[0]["frame"]=2;refuses([&]{author_equipment(c,guarded);});
    json recipe={{"version",1},{"objects",json::array({{{"name","sword"},{"segments",json::array({{{"start",0},{"end",4},{"keys",json::array({{{"frame",0},{"position",{0,0.25,0.1}}},{{"frame",4},{"position",{0,0.25,0.14}}}})}}})}}})}};
    {std::ofstream out(dir/"plan.json");out<<recipe.dump(2);}auto parsed=load_equipment_plan(dir/"plan.json");check(author_equipment(c,parsed).objects.size()==1,"Strict versioned equipment parser");
    auto grip_recipe=recipe;grip_recipe["objects"][0]["segments"][0]["space"]="grip";grip_recipe["objects"][0]["segments"][0]["anchor"]="right-palm";
    grip_recipe["objects"][0]["segments"][0]["keys"]=json::array({{{"frame",0}},{{"frame",4}}});
    grip_recipe["grips"]=json::array();
    grip_recipe["grips"].push_back({{"name","right-palm"},{"object","sword"},{"bone","RightHand"},{"start",0},{"end",4},
        {"position",{0.01,-0.02,0.03}},{"rotation",{7,14,-9}},
        {"handOffset",{{"position",{0.02,0.03,-0.01}},{"rotation",{-4,8,13}}}}});
    {std::ofstream out(dir/"grip-plan.json");out<<grip_recipe.dump(2);}auto parsed_grip=load_equipment_plan(dir/"grip-plan.json");check(author_equipment(c,parsed_grip).report["allGripsSatisfied"].get<bool>(),"JSON grip anchor resolves declared marker transforms natively");
    recipe["typo"]=1;{std::ofstream out(dir/"bad.json");out<<recipe.dump();}refuses([&]{load_equipment_plan(dir/"bad.json");});
    fs::remove(dir/"body.glb");save_glb(c,dir/"body.glb");
    std::cout<<"Equipment grip anchors, IK, independent FK, guarded rollback, scale, bow/string/release and parser contracts pass\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
