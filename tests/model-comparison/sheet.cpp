#include "motion.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("soma-sheet MANIFEST.json CHECKOUT_BUILT_TIDE OUTPUT.png");
        std::ifstream input(argv[1]);nlohmann::json doc;input>>doc;
        std::vector<motion::Clip> clips;std::vector<std::string> labels;
        for(const auto& item:doc) {clips.push_back(motion::load_glb(item.at("file").get<std::string>()));labels.push_back(item.at("label").get<std::string>());}
        motion::VariantSheet sheet;sheet.tide=argv[2];if(motion::fs::path(argv[3]).extension()==".gif")sheet.gif=argv[3];else sheet.png=argv[3];
        motion::preview_variants(clips,labels,sheet);return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

