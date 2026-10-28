// An independently authored, in-memory ONNX operator fixture, not a DART export.
// Minimal protobuf writer avoids adding a Python/protobuf dependency or model file.
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using Bytes=std::string;
Bytes var(uint64_t n) {
    Bytes b;
    while(n>=128) { b+=static_cast<char>((n&127)|128); n>>=7; }
    b+=static_cast<char>(n); return b;
}
Bytes integer(int field,uint64_t n) { return var(static_cast<uint64_t>(field)*8)+var(n); }
Bytes message(int field,const Bytes& b) { return var(static_cast<uint64_t>(field)*8+2)+var(b.size())+b; }
Bytes value(const Bytes& name) {
    const auto shape=message(1,integer(1,1))+message(1,integer(1,4));
    return message(1,name)+message(2,message(1,integer(1,1)+message(2,shape)));
}
Bytes tensor(const Bytes& name, const std::initializer_list<int64_t> dims, const std::initializer_list<float> data) {
    Bytes b;
    for(auto d:dims) b+=integer(1,static_cast<uint64_t>(d));
    return b+integer(2,1)+message(8,name)+message(9,Bytes(reinterpret_cast<const char*>(data.begin()),data.size()*sizeof(float)));
}
Bytes node(const Bytes& op,const std::initializer_list<Bytes> inputs,const Bytes& output) {
    Bytes b;
    for(const auto& input:inputs) b+=message(1,input);
    return message(1,b+message(2,output)+message(4,op));
}
int main() try {
    if(std::string(OrtGetApiBase()->GetVersionString())!="1.31.1") throw std::runtime_error("requires pinned ORT 1.31.1");
    Bytes graph=node("MatMul",{"x","w"},"pre")
        +node("LayerNormalization",{"pre","scale","bias"},"norm")
        +node("Gelu",{"norm"},"y")+message(2,"tide-source-only-operator-probe")
        +message(5,tensor("w",{4,4},{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}))
        +message(5,tensor("scale",{4},{1,1,1,1}))
        +message(5,tensor("bias",{4},{0,0,0,0}))
        +message(11,value("x"))+message(12,value("y"));
    const Bytes model=integer(1,9)+message(7,graph)+message(8,integer(2,20));
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"dart-source-probe");
    Ort::SessionOptions options; options.SetIntraOpNumThreads(1);
    Ort::Session session(env,model.data(),model.size(),options);
    std::array<float,4> x{1,2,3,4};
    const std::array<int64_t,2> shape{1,4};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto input=Ort::Value::CreateTensor<float>(memory,x.data(),x.size(),shape.data(),shape.size());
    const char* in="x"; const char* out="y";
    auto output=session.Run(Ort::RunOptions{nullptr},&in,&input,1,&out,1);
    const auto* y=output[0].GetTensorData<float>();
    double worst=0;
    for(size_t i=0;i<x.size();++i) {
        const double z=(x[i]-2.5)/std::sqrt(1.25+1e-5);
        const double expected=.5*z*(1+std::erf(z/std::sqrt(2.0)));
        worst=std::fmax(worst,std::abs(y[i]-expected));
        if(!std::isfinite(y[i]) || std::abs(y[i]-expected)>2e-6) throw std::runtime_error("operator mismatch");
    }
    // The shipped inference distribution has no training/backward implementation.
    const bool training=Ort::GetApi().GetTrainingApi(1)!=nullptr;
    std::cout << "ORT 1.31.1 CPU MatMul/LayerNormalization/Gelu max error " << worst
              << "; training API available: " << training << '\n';
    std::cout << "Synthetic operator fixture only; DART export and VJPs unvalidated.\n";
    return training ? 1 : 0;
} catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
