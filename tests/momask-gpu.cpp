// Opt-in real-checkpoint GPU/reference test; fixtures remain outside Git.
#include "momask_gpu.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
using namespace motion;
using nlohmann::json;
std::vector<unsigned char> read(const fs::path& path,size_t size) {
    if(size>256*1024*1024||fs::file_size(path)!=size)throw std::runtime_error("Invalid fixture size");
    std::vector<unsigned char> bytes(size);std::ifstream in(path,std::ios::binary);
    if(!in.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(size)))throw std::runtime_error("Cannot read fixture");
    return bytes;
}
int main(int argc,char** argv) try {
    if(argc!=2)throw std::runtime_error("Usage: momask-gpu-tests PRIVATE_EXPORT_DIRECTORY");
    auto root=utf8path(argv[1]);json report;std::ifstream(root/"report.json")>>report;
    MomaskGpu gpu(root,4);size_t cases=0;
    for(const auto& [name,component]:report.at("components").items()) {
        if(name=="text")continue; // CLIP remains the already-validated CPU path.
        for(const auto& test:component.at("cases")) {
            std::vector<MomaskTensor> inputs;
            for(const auto& fixture:test.at("fixtures")) {
                if(fixture.at("name")=="expected")continue;
                if(fixture.at("name")!="input"+std::to_string(inputs.size()))throw std::runtime_error("Fixture input order mismatch");
                auto shape=fixture.at("shape").get<std::vector<int64_t>>();size_t count=1;
                for(auto d:shape){if(d<1||count>10000000/static_cast<size_t>(d))throw std::runtime_error("Invalid fixture shape");count*=static_cast<size_t>(d);}
                std::string dtype=fixture.at("dtype");int type=dtype=="float32"?1:dtype=="int64"?7:dtype=="bool"?9:0;
                if(!type)throw std::runtime_error("Invalid fixture type");
                auto file=fs::path(fixture.at("file").get<std::string>());if(file.filename()!=file)throw std::runtime_error("Invalid fixture name");
                inputs.push_back({type,shape,read(root/file,count*(type==1?4:type==7?8:1))});
            }
            auto start=std::chrono::steady_clock::now();auto actual=gpu.run(name,std::move(inputs));
            const auto& expected=test.at("fixtures").back();
            if(actual.shape!=expected.at("shape").get<std::vector<int64_t>>())throw std::runtime_error("GPU output shape mismatch");
            auto bytes=read(root/expected.at("file").get<std::string>(),actual.bytes.size());double error=0;
            if(expected.at("dtype")=="int64") {
                if(actual.type!=7||bytes!=actual.bytes)throw std::runtime_error("GPU RVQ indices differ from PyTorch reference");
            } else {
                if(actual.type!=1)throw std::runtime_error("GPU output type mismatch");
                for(size_t offset=0;offset<bytes.size();offset+=4) {
                    float a,b;std::memcpy(&a,actual.bytes.data()+offset,4);std::memcpy(&b,bytes.data()+offset,4);
                    double delta=std::abs(static_cast<double>(a)-b);error=std::max(error,delta);
                    // Fixed GPU bound; CPU's tighter existing bound remains unchanged.
                    if(!std::isfinite(a)||delta>0.002+0.002*std::abs(b))throw std::runtime_error("GPU tensor parity failure: "+name+" delta "+std::to_string(delta));
                }
            }
            std::cout<<name<<" tokens "<<test.at("tokens")<<" max_abs "<<error<<" seconds "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n";++cases;
        }
    }
    if(cases!=12)throw std::runtime_error("Expected all twelve motion-network cases");
    std::cout<<"MoMask GPU reference cases passed: "<<cases<<"; sampled GPU local peak bytes: "<<gpu.peak_memory_bytes()<<"\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
