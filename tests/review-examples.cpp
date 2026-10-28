#include "motion.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace motion;
// Diagnostic fixtures only. This never generates animation in place of a model.
int main(int argc,char** argv)try {
    if(argc!=3)throw std::runtime_error("review-examples CANONICAL_SOURCE.glb NEW_DIRECTORY");
    auto base=load_glb(utf8path(argv[1]));require_soma(base);auto dir=utf8path(argv[2]);
    if(fs::exists(dir))throw std::runtime_error("Choose a new diagnostic fixture directory");fs::create_directories(dir);
    auto lowered=base;for(size_t f=0;f<lowered.frames();++f)lowered.frame(f)[0].p.y-=.15f;save_glb(lowered,dir/"lowered.glb");
    auto step=base;size_t hand=0;for(size_t j=0;j<step.joints.size();++j)if(step.joints[j].name=="RightHand")hand=j;
    size_t middle=step.frames()/2;for(size_t f=middle;f<std::min(middle+3,step.frames());++f)step.frame(f)[hand].q=normalized(step.frame(f)[hand].q*yaw(1.2217304764f));save_glb(step,dir/"rotation-step.glb");
    auto pose=base;pose.poses.assign(base.frame(middle).begin(),base.frame(middle).end());save_glb(pose,dir/"guide-pose.glb");
    const auto world=world_pose(pose,pose.frame(0));size_t hips=0,chest=0;for(size_t j=0;j<pose.joints.size();++j){if(pose.joints[j].name=="Hips")hips=j;if(pose.joints[j].name=="Chest")chest=j;}
    Vec3 end=rotate(inverse(world[hand].q),(world[hips].p+world[chest].p)*.5f-world[hand].p);
    nlohmann::json props={{"props",nlohmann::json::array({{{"name","misaligned-test-blade"},{"bone","RightHand"},{"start",{0,0,0}},{"end",{end.x,end.y,end.z}},{"radius",.02}}})}};
    std::ofstream settings(dir/"prop-settings.json");settings<<props.dump(2);if(!settings)throw std::runtime_error("Could not write diagnostic prop settings");
    std::cout<<"Created explicit diagnostic mutations (-.15 m root, +70 degree wrist pulse) and one extracted guide pose. Not model outputs.\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
