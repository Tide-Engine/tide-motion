#include "upgrade.hpp"
#include <iostream>
#include <stdexcept>
using namespace motion;
void check(bool v,const char* why){if(!v)throw std::runtime_error(why);}
template<class F> void refuses(F call){bool bad=false;try{call();}catch(const std::exception&){bad=true;}check(bad,"Invalid upgrade contract accepted");}
int main() {try {
    check(upgrade_version("v1.20.3")==std::array<unsigned,3>{1,20,3},"version components");
    check(upgrade_version("0.10.0")>upgrade_version("0.9.99"),"numeric version order");
    for(const char* text:{"1.2","1.2.3.4","-1.2.3","01.2.3","1.2.3-dev","1.2.3/../x","1.2.3;echo","1.2.3\n"})refuses([&]{(void)upgrade_version(text);});
    for(const auto& platform:{"windows-x64","linux-x64","linux-arm64","macos-arm64"}) {
        std::string name=std::string("tide-motion-1.2.3-")+platform+(std::string(platform).starts_with("windows")?".zip":".tar.gz");
        std::string base="https://github.com/Tide-Engine/tide-motion/releases/download/v1.2.3/";
        nlohmann::json doc={{"draft",false},{"prerelease",false},{"tag_name","v1.2.3"},{"assets",nlohmann::json::array({{{"name",name},{"size",1024},{"browser_download_url",base+name}},{{"name",name+".sha256"},{"size",120},{"browser_download_url",base+name+".sha256"}}})}};
        auto release=upgrade_release(doc,platform);check(release.version=="1.2.3"&&release.archive==name&&release.bytes==1024,"platform release selection");
        auto bad=doc;bad["draft"]=true;refuses([&]{(void)upgrade_release(bad,platform);});
        bad=doc;bad["prerelease"]=true;refuses([&]{(void)upgrade_release(bad,platform);});
        bad=doc;bad["assets"][0]["browser_download_url"]="https://example.com/other.zip";refuses([&]{(void)upgrade_release(bad,platform);});
        bad=doc;bad["assets"][0]["size"]=536870913;refuses([&]{(void)upgrade_release(bad,platform);});
        bad=doc;bad["assets"][1]["size"]=-1;refuses([&]{(void)upgrade_release(bad,platform);});
        bad=doc;bad["assets"].erase(1);refuses([&]{(void)upgrade_release(bad,platform);});
        bad=doc;bad["assets"].push_back(doc["assets"][0]);refuses([&]{(void)upgrade_release(bad,platform);});
    }
    std::cout<<"Upgrade release contracts passed\n";return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
