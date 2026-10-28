#include "review.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace motion;
using nlohmann::json;
namespace {
void check(bool b,const char* why){if(!b)throw std::runtime_error(why);}
template<class F>void refuses(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"Expected refusal");}
Clip fixture(){Clip c;c.joints=soma();for(int f=0;f<8;++f)for(const auto& j:c.joints)c.poses.push_back(j.rest);return c;}
size_t joint(const Clip& c,const std::string& name){for(size_t j=0;j<c.joints.size();++j)if(c.joints[j].name==name)return j;throw std::runtime_error("Missing fixture joint");}
}
int main(int argc,char** argv)try {
    Clip c=fixture();const auto original=c.poses;auto hand=joint(c,"RightHand");
    // Same physical rotation with opposite quaternion sign must not create a jump.
    c.frame(1)[hand].q={0,0,0,-1};
    c.frame(3)[hand].q=yaw(1.57079632679f);c.frame(4)[hand].q=yaw(1.57079632679f);
    auto r=review_motion(c);size_t steps=0;
    for(const auto& o:r["observations"])if(o["rule"]=="local-rotation-step"){
        ++steps;check(o["subjects"][0]=="RightHand","Localized joint");check(o["startFrame"]==3||o["startFrame"]==5,"Quaternion sign invariant and intervals split across gaps");check(o["startFrame"]==o["endFrame"],"Inclusive single-sample interval");check(std::abs(o["peakValue"].get<float>()-90)<.001f,"Measured 90 degree step");check(o["unit"]=="degrees/sample","Explicit sample-based unit");
    }
    check(steps==2,"Two separated steps, no extra sign jump");
    bool measured=false;for(const auto& m:r["measurements"])if(m["rule"]=="local-rotation-step"&&m["subjects"][0]=="RightHand"){
        measured=true;check(m["samples"]==7&&m["exceedanceSamples"]==2,"Rotation statistics use adjacent pairs only");check(std::abs(m["meanValue"].get<double>()-180.0/7)<.001,"Continuous measurement independent of hint interval count");}
    check(measured,"Below-threshold samples remain available to agents");
    auto unavailable=review_motion(c,{{"groundY",-1000}});for(const auto& m:unavailable["measurements"])if(m["rule"]=="low-foot-horizontal-motion")check(m["samples"]==0&&m["meanValue"].is_null()&&m["maxValue"].is_null()&&m["peakFrame"].is_null(),"No eligible samples is unavailable, not zero movement");
    check(c.frame(0)[hand].q.w==original[hand].q.w&&c.frame(3)[hand].q.y>.7f,"Analysis preserves input");
    check(r["coverage"]["anatomicalReachability"].get<std::string>().starts_with("not-evaluated"),"Unknown is explicit");
    json settings={{"props",json::array({{{"name","probe"},{"bone","Hips"},{"start",{0,0,0}},{"end",{0,0,0}},{"radius",.02}}})}};
    auto prop=review_motion(fixture(),settings);bool overlap=false;for(const auto& o:prop["observations"])if(o["rule"]=="prop-torso-proxy-overlap"){overlap=true;check(o["startFrame"]==0&&o["endFrame"]==7,"Contiguous intervals merged");check(std::abs(o["peakValue"].get<float>()-.15f)<1e-5f,"Degenerate segment distance and capsule radii");}
    check(overlap,"Explicit prop proxy evaluated");
    settings["props"][0]["start"]={10,10,10};settings["props"][0]["end"]={11,10,10};auto far=review_motion(fixture(),settings);for(const auto& o:far["observations"])check(o["rule"]!="prop-torso-proxy-overlap","Separated capsule stays clear");
    auto custom=fixture();custom.joints[hand].name="OtherHand";auto partial=review_motion(custom);check(partial["coverage"]["bodyAndGroundProxies"].get<std::string>().starts_with("not-evaluated"),"Custom rig never assumed anatomical");
    refuses([&]{review_motion(c,{{"groundY","wrong"}});});refuses([&]{review_motion(c,{{"unknown",1}});});refuses([&]{review_motion(c,{{"props",json::array({{{"name","p"},{"bone","missing"},{"start",{0,0,0}},{"end",{0,1,0}},{"radius",.02}}})}});});
    json bundle={{"title","</script><script>BAD</script>"},{"cases",json::array()}};auto html=review_html(bundle);check(html.find("<script>BAD")==std::string::npos,"Embedded JSON cannot escape its script element");check(html.find("\\u003c/script>")!=std::string::npos,"Script data escaped");
    if(argc==2){fs::create_directories(utf8path(argv[1]));save_glb(c,utf8path(argv[1])/"fixture.glb");}
    std::cout<<"Advisory review contracts pass\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
