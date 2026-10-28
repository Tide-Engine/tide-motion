// Opt-in model-free native ONNX check of tests/momask-export.py artifacts.
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using nlohmann::json;
namespace fs=std::filesystem;
std::vector<char> read(const fs::path& path,size_t expected) {
    if(expected>256*1024*1024||fs::file_size(path)!=expected)throw std::runtime_error("Invalid fixture size");
    std::vector<char> bytes(expected);std::ifstream stream(path,std::ios::binary);
    if(!stream.read(bytes.data(),static_cast<std::streamsize>(expected)))throw std::runtime_error("Cannot read fixture");
    return bytes;
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::runtime_error("Usage: momask-ort-tests ignored-export-directory");
        std::u8string argument;for(unsigned char c:std::string(argv[1]))argument.push_back(static_cast<char8_t>(c));
        fs::path root(argument);json report;std::ifstream(root/"report.json")>>report;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"momask-source-proof");Ort::SessionOptions options;
        options.SetIntraOpNumThreads(4);options.SetInterOpNumThreads(1);
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        std::cout<<"Native ONNX Runtime "<<OrtGetApiBase()->GetVersionString()<<'\n';
        for(auto& [name,component]:report.at("components").items()) {
            auto start=std::chrono::steady_clock::now();
            Ort::Session session(env,(root/(name+".onnx")).c_str(),options);
            std::cout<<name<<" session load ms "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
            for(auto& test:component.at("cases")) {
                std::vector<std::vector<char>> storage;
                std::vector<Ort::Value> inputs;std::vector<std::string> strings;
                std::vector<const char*> input_names;
                auto fixtures=test.at("fixtures");storage.reserve(fixtures.size());strings.reserve(fixtures.size());
                for(const auto& fixture:fixtures) {
                    if(fixture.at("name")=="expected")continue;
                    auto shape=fixture.at("shape").get<std::vector<int64_t>>();size_t count=1;
                    for(auto d:shape) {if(d<1||d>100000||count>10000000/static_cast<size_t>(d))throw std::runtime_error("Invalid fixture shape");count*=static_cast<size_t>(d);}
                    std::string type=fixture.at("dtype");
                    auto dtype=type=="float32"?ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:type=="int64"?ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
                    if(type!="float32"&&type!="int64"&&type!="bool")throw std::runtime_error("Invalid fixture dtype");
                    size_t bytes=count*(type=="float32"?4:type=="int64"?8:1);
                    std::string file=fixture.at("file");if(fs::path(file).filename()!=fs::path(file))throw std::runtime_error("Invalid fixture path");
                    storage.push_back(read(root/file,bytes));auto& data=storage.back();
                    inputs.push_back(Ort::Value::CreateTensor(memory,data.data(),data.size(),shape.data(),shape.size(),dtype));
                    strings.push_back(fixture.at("name"));
                }
                for(auto& s:strings)input_names.push_back(s.c_str());
                const char* output_name="output";start=std::chrono::steady_clock::now();
                auto outputs=session.Run(Ort::RunOptions{nullptr},input_names.data(),inputs.data(),inputs.size(),&output_name,1);
                double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                const auto& fixture=fixtures.back();auto shape=fixture.at("shape").get<std::vector<int64_t>>();
                auto info=outputs[0].GetTensorTypeAndShapeInfo();if(info.GetShape()!=shape)throw std::runtime_error("Output shape mismatch");
                size_t count=info.GetElementCount();double max_error=0;
                if(fixture.at("dtype")=="int64") {
                    auto bytes=read(root/fixture.at("file").get<std::string>(),count*8);
                    const auto* actual=outputs[0].GetTensorData<int64_t>();
                    for(size_t i=0;i<count;++i) {int64_t expected;std::memcpy(&expected,bytes.data()+i*8,8);if(expected!=actual[i])throw std::runtime_error("RVQ index mismatch");}
                } else {
                    auto bytes=read(root/fixture.at("file").get<std::string>(),count*4);
                    const auto* actual=outputs[0].GetTensorData<float>();
                    for(size_t i=0;i<count;++i) {
                        float expected;std::memcpy(&expected,bytes.data()+i*4,4);
                        double delta=std::abs(static_cast<double>(expected)-actual[i]);max_error=std::max(max_error,delta);
                        if(!std::isfinite(actual[i])||delta>3e-5+3e-4*std::abs(expected))throw std::runtime_error("Native ORT parity failure");
                    }
                }
                std::cout<<"  tokens "<<test.at("tokens")<<" max_abs "<<max_error<<" run ms "<<elapsed<<'\n';
            }
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
