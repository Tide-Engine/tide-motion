#include "dex2hoi.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace motion;
using namespace motion::dex2hoi;
using nlohmann::json;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F> void refuses(F operation,const char* message) {
    bool failed=false;try{operation();}catch(const std::exception&){failed=true;}check(failed,message);
}
void sampler() {
    auto full=sampling_timesteps(500);check(full.size()==500,"full schedule size");
    for(size_t i=0;i<full.size();++i)check(full[i]==static_cast<int64_t>(i),"full schedule changed");
    check(sampling_timesteps(5)==std::vector<int64_t>({0,125,250,374,499}),"respaced schedule reference");
    refuses([]{(void)sampling_timesteps(1);},"one-step schedule accepted");
    refuses([]{(void)sampling_timesteps(501);},"too many timesteps accepted");
    auto alpha=cosine_alphas();
    // Independent NumPy float64 evaluation of the pinned cosine schedule.
    const std::array<std::pair<size_t,double>,6> reference{{{0,.9999125759273678},{1,.9998057312026716},
        {100,.8968160434297451},{249,.4938435904406379},{498,9.715044036169857e-6},{499,9.715044036169866e-9}}};
    for(auto [index,value]:reference)check(std::abs(alpha[index]-value)<2e-15,"cosine alpha reference mismatch");
    for(size_t i=1;i<alpha.size();++i)check(alpha[i]<alpha[i-1]&&alpha[i]>0,"nondecreasing/invalid cosine schedule");
    const std::array<float,4> x{2,-1,.5f,-3},x0{1,2,-.5f,.25f};
    auto next=ddim_step(x,x0,.25,.64);
    // sqrt(.64)*x0 + sqrt(.36)*(2*x-x0)/sqrt(3).
    const std::array<double,4> expected{1.8392304845413264,.21435935394489846,.11961524227066311,-1.965063509461097};
    for(size_t i=0;i<next.size();++i)check(std::abs(next[i]-expected[i])<5e-7,"DDIM reference mismatch");
    auto last=ddim_step(x,x0,alpha[0],1);check(std::equal(last.begin(),last.end(),x0.begin()),"final DDIM step did not return x0");
    refuses([&]{(void)ddim_step(x,std::span<const float>{x0.data(),3},.25,.64);},"mismatched DDIM tensors accepted");
    refuses([&]{(void)ddim_step(x,x0,.5,.25);},"reversed DDIM alphas accepted");
    auto invalid=x;invalid[0]=std::numeric_limits<float>::quiet_NaN();
    refuses([&]{(void)ddim_step(invalid,x0,.25,.64);},"nonfinite DDIM state accepted");
}
template<size_t N> std::array<float,N> arr(const json& j) {
    auto v=j.get<std::vector<float>>();check(v.size()==N,"fixture array size");std::array<float,N> result{};std::copy(v.begin(),v.end(),result.begin());return result;
}
void parity(const fs::path& bundle,const fs::path& directory) {
    Context context(bundle,4);
    for(size_t case_index=0;case_index<3;++case_index) {
        std::ifstream in(directory/("denoise-"+std::to_string(case_index)+".json"));json input;in>>input;
        auto data=[&](const char* name)->const json&{return input.at(name).at("data");};
        Conditions condition;
        condition.prefix=data("prefix").get<std::vector<float>>();condition.text=data("text").get<std::vector<float>>();
        condition.objects=data("objects").get<std::vector<float>>();condition.object_mask={data("object_mask")[0].get<bool>(),data("object_mask")[1].get<bool>()};
        condition.init_features=arr<72>(data("init_features"));if(input.contains("root_x0"))condition.root_x0=arr<3>(data("root_x0"));condition.obj_x0=arr<6>(data("obj_x0"));
        auto result=context.denoise(data("x").get<std::vector<float>>(),data("timestep")[0].get<int64_t>(),condition,data("text_scale")[0].get<float>());
        auto expected=data("expected").get<std::vector<float>>();check(result.size()==expected.size(),"learned parity result size");
        double maximum=0,mean=0;
        for(size_t i=0;i<result.size();++i){double error=std::abs(static_cast<double>(result[i])-expected[i]);maximum=std::max(maximum,error);mean+=error;}
        mean/=static_cast<double>(result.size());
        std::cout<<"learned case "<<case_index<<" maximum="<<maximum<<" mean="<<mean<<"\n";
        check(maximum<1e-4&&mean<5e-6,"real trained denoiser parity failed");
    }
}
}
int main(int argc,char** argv) {
    try {
        sampler();if(argc==3)parity(utf8path(argv[1]),utf8path(argv[2]));else check(argc==1,"usage: dex2hoi-runtime-tests [BUNDLE FIXTURE-DIRECTORY]");
        std::cout<<"Dex2HOI runtime contracts passed\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
