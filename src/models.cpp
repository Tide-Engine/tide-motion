#include "motion.hpp"
#include <picosha2.h>
#include <array>
#include <bit>
#include <cstring>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#endif

namespace motion {
namespace {
struct ModelFile { const char* group; const char* file; uint64_t bytes; const char* hash; const char* url; };
constexpr ModelFile v12_model={"kimodo-v1.2","kimodo-v1.2.gguf",1133166784ULL,"9c24c5ee9b6b3683947ab2796727ffb2fbd69b2ebc7683214809013687d87a19",nullptr};
constexpr ModelFile files[]={
    v12_model,
    {"kimodo","kimodo.gguf",1133166784ULL,"3bf1229f4c1eff1d28f5196a854113da2df9a11a5e21c60694630903bf948ee4",
     "https://huggingface.co/LocalAI-io/Kimodo-SOMA-RP-v1.1-GGML/resolve/65f8b8ab34ee7524161cc87b20ae4cb4f5d3a12d/models/kimodo-soma-rp-v1.1-f32.gguf"},
    {"kimodo-text","text/encoder.gguf",8142058752ULL,"b26d0e74b115b33a7f2df5aca04426548365e99217cd590986f8abc4172e4c5c",
     "https://huggingface.co/LocalAI-io/Llama-3-Kimodo-GGML/resolve/3e8d958803beaddb6011ac534f2be972e2710c7d/Llama-3-Kimodo-Q8_0.gguf"},
    {"kimodo-text","text/tokenizer.gguf",7295054ULL,"81614aca62a98846c02b72cc2e5378e5bdce8b4d507b88f96eb1faa90dae607e",
     "https://huggingface.co/LocalAI-io/Llama-3-Kimodo-GGML/resolve/3e8d958803beaddb6011ac534f2be972e2710c7d/tokenizer.gguf"},
    {"bricks","motionbricks.onnx",757864273ULL,"856419dc72aeb14e591eebf4dd0ea05734a2506ddb00bcd39dd6dac2ad60f4fb",
     "https://media.githubusercontent.com/media/NVlabs/motionbricks/54fc563aeaaf819a753d72cc9fdc6713989cdd43/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx"}
};
constexpr ModelFile v12_sources[]={
    {"kimodo-v1.2",".soma-v1.2/LICENSE",10229ULL,"11ccdcb47db87ac30b63765b5e5154403026c4fbb52c19919d99d912ec454e19","https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/resolve/0e3c89024d896064b6851201373f7834784a386b/LICENSE"},
    {"kimodo-v1.2",".soma-v1.2/config.yaml",675ULL,"82f1e9f44d0df517d8ebde03d2e9396fd0bf372360714757901c1d97a2626599","https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/resolve/0e3c89024d896064b6851201373f7834784a386b/config.yaml"},
    {"kimodo-v1.2",".soma-v1.2/model.safetensors",1133177724ULL,"fdcef624140a12c0a4840192ce0b75cafb4e73228a893da3864b4a990d3e919e","https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/resolve/0e3c89024d896064b6851201373f7834784a386b/model.safetensors"},
    {"kimodo-v1.2",".soma-v1.2/stats/motion/body/mean.npy",3040ULL,"d42f35f8bb7ef52fdec66ae29c5cf590bd038f4f1aabd2d560d1d0f78663ed6f","https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/resolve/0e3c89024d896064b6851201373f7834784a386b/stats/motion/body/mean.npy"},
    {"kimodo-v1.2",".soma-v1.2/stats/motion/body/std.npy",3040ULL,"965bbad497726ef4e6c46b519827e2cbcb29f974214b25718d20a725b82279ed","https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/resolve/0e3c89024d896064b6851201373f7834784a386b/stats/motion/body/std.npy"},
    {"kimodo-v1.2",".soma-v1.2/stats/motion/global_root/mean.npy",168ULL,"6c110980f9a0386e57a8c186f8eac7f55dc05bade92f44c4b391450ada598da1","https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/resolve/0e3c89024d896064b6851201373f7834784a386b/stats/motion/global_root/mean.npy"},
    {"kimodo-v1.2",".soma-v1.2/stats/motion/global_root/std.npy",168ULL,"fb15aeed7143f5f0754611f0acf5fa5f55d1f6492a49c23e490fc62f7bf11fb0","https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/resolve/0e3c89024d896064b6851201373f7834784a386b/stats/motion/global_root/std.npy"},
    {"kimodo-v1.2",".soma-v1.2/stats/motion/local_root/mean.npy",160ULL,"4c36e8e33d6d51e6c8206ac796a52b7b3a4da7f0dc49ebdbb4f130178e3e9d31","https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/resolve/0e3c89024d896064b6851201373f7834784a386b/stats/motion/local_root/mean.npy"},
    {"kimodo-v1.2",".soma-v1.2/stats/motion/local_root/std.npy",160ULL,"5f57b41f51ced7832cffa81702f6e04d8694e16475c184baec6c8be0e98c8dba","https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/resolve/0e3c89024d896064b6851201373f7834784a386b/stats/motion/local_root/std.npy"},
};
void prepare_v12(const fs::path& root);
void selection_valid(const std::string& selection) {
    if(selection!="all"&&selection!="kimodo"&&selection!="bricks"&&selection!="kimodo-v1.2"&&selection!="text")throw std::runtime_error("Model selection must be all, kimodo, kimodo-v1.2, text or bricks");
}
bool selected(const ModelFile& f,const std::string& s) { return (s=="all"&&std::string(f.group)!="kimodo-v1.2")||s==f.group||((s=="kimodo"||s=="kimodo-v1.2"||s=="text")&&std::string(f.group)=="kimodo-text"); }
void verify(const fs::path& path,const ModelFile& f) {
    if(!fs::exists(path)||fs::file_size(path)!=f.bytes)throw std::runtime_error("Missing or incomplete model '"+path.string()+"'; run tide-motion models install "+std::string(std::string(f.group)=="kimodo-text"?"kimodo":f.group));
    if(sha256_file(path)!=f.hash)throw std::runtime_error("Model checksum mismatch: "+path.string()+"; remove this file and reinstall models");
}
}
std::string sha256_file(const fs::path& path) {
    std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("Cannot open "+path.string());
    std::vector<char> buffer(1024*1024);
#ifdef _WIN32
    // Use the OS implementation for multi-GB models; it can use the CPU's SHA
    // instructions and adds no redistributable dependency.
    struct Handles {
        std::vector<unsigned char> state;
        BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
        ~Handles(){if(hash)BCryptDestroyHash(hash);if(algorithm)BCryptCloseAlgorithmProvider(algorithm,0);}
    } handles;
    auto checked=[](NTSTATUS status){if(status<0)throw std::runtime_error("Windows SHA-256 failed");};
    checked(BCryptOpenAlgorithmProvider(&handles.algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0));
    DWORD length=0,received=0;
    checked(BCryptGetProperty(handles.algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&length),sizeof(length),&received,0));
    handles.state.resize(length);
    checked(BCryptCreateHash(handles.algorithm,&handles.hash,handles.state.data(),length,nullptr,0,0));
    while(input) { input.read(buffer.data(),static_cast<std::streamsize>(buffer.size()));auto count=input.gcount();checked(BCryptHashData(handles.hash,reinterpret_cast<PUCHAR>(buffer.data()),static_cast<ULONG>(count),0)); }
    if(!input.eof())throw std::runtime_error("Cannot read "+path.string());
    std::array<unsigned char,32> digest{};checked(BCryptFinishHash(handles.hash,digest.data(),static_cast<ULONG>(digest.size()),0));
    const char hex[]="0123456789abcdef";std::string result;
    for(unsigned char c:digest) {result+=hex[c>>4];result+=hex[c&15];}
    return result;
#else
    picosha2::hash256_one_by_one hash;
    while(input) { input.read(buffer.data(),static_cast<std::streamsize>(buffer.size()));auto count=input.gcount();hash.process(buffer.data(),buffer.data()+count); }
    if(!input.eof())throw std::runtime_error("Cannot read "+path.string());
    hash.finish();return picosha2::get_hash_hex_string(hash);
#endif
}
void check_models(const fs::path& root,const std::string& selection) {
    selection_valid(selection);
    for(const auto& f:files)if(selected(f,selection)) { std::cerr<<"Verifying "<<f.file<<"...\n";verify(root/f.file,f); }
}
namespace {
void install_file(const fs::path& root,const ModelFile& f) {
        auto path=root/f.file;fs::create_directories(path.parent_path());
        if(fs::exists(path)) { verify(path,f);std::cerr<<"Verified "<<path.string()<<"\n";return; }
        if(!f.url) {prepare_v12(root);verify(path,f);return;}
        auto temp=path;temp+=".part";
        // A full interrupted download can be verified and committed without another request.
        if(!fs::exists(temp)||fs::file_size(temp)!=f.bytes) {
            std::cerr<<"Downloading "<<f.file<<" ("<<f.bytes/1000000<<" MB)...\n";
            std::vector<std::string> args{"curl","--fail","--location","--proto","=https","--proto-redir","=https","--retry","3","--connect-timeout","20","--speed-time","60","--speed-limit","1024","--max-time","7200","--continue-at","-","--output",temp.string(),f.url};
            int code=1;
            for(int attempt=0;attempt<3;++attempt) {
                if(attempt)std::cerr<<"Connection interrupted; resuming "<<f.file<<"...\n";
                code=run_process(args);if(code==0)break;
            }
            if(code)throw std::runtime_error("Model download failed; run the same install command to resume");
        }
        verify(temp,f);fs::rename(temp,path);
}
// The input files are size/hash verified before parsing. This is the same F32
// GGUF boundary as the development reference converter, without a Python runtime.
void prepare_v12(const fs::path& root) {
    for(const auto& source:v12_sources)install_file(root,source);
    if constexpr(std::endian::native!=std::endian::little)throw std::runtime_error("v1.2 conversion requires a little-endian host");
    const auto source=root/".soma-v1.2";
    struct Tensor {std::string name;std::vector<uint64_t> shape;uint64_t start=0,size=0,offset=0;fs::path file;bool stats=false;};
    std::ifstream input(source/"model.safetensors",std::ios::binary);
    uint64_t header_size=0;input.read(reinterpret_cast<char*>(&header_size),8);
    if(!input||header_size>128*1024*1024)throw std::runtime_error("Invalid safetensors header");
    std::string json(static_cast<size_t>(header_size),'\0');input.read(json.data(),static_cast<std::streamsize>(header_size));
    auto tensors_json=nlohmann::json::parse(json);
    std::vector<Tensor> tensors;
    for(const auto& item:tensors_json.items()) {
        if(item.key()=="__metadata__")continue;
        const auto& desc=item.value();Tensor t;t.name=item.key();
        const std::string prefix="denoiser.backbone.";if(t.name.starts_with(prefix))t.name.erase(0,prefix.size());
        t.shape=desc.at("shape").get<std::vector<uint64_t>>();
        t.start=8+header_size+desc.at("data_offsets").at(0).get<uint64_t>();
        t.size=desc.at("data_offsets").at(1).get<uint64_t>()-desc.at("data_offsets").at(0).get<uint64_t>();
        t.file=source/"model.safetensors";tensors.push_back(t);
    }
    std::sort(tensors.begin(),tensors.end(),[](const auto& a,const auto& b){return a.name<b.name;});
    for(const auto& part:{"global_root","local_root","body"})for(const auto& stat:{"mean","std"}) {
        Tensor t;t.name=std::string("stats.")+part+"."+stat;t.file=source/"stats/motion"/part/(std::string(stat)+".npy");t.stats=true;
        std::ifstream npy(t.file,std::ios::binary);npy.seekg(8);uint16_t length=0;npy.read(reinterpret_cast<char*>(&length),2);t.start=10+length;
        t.shape={std::string(part)=="global_root"?5ULL:std::string(part)=="local_root"?4ULL:364ULL};t.size=4*t.shape[0];tensors.push_back(t);
    }
    std::string header;
    auto integer=[&](uint64_t value,unsigned bytes){for(unsigned i=0;i<bytes;++i)header.push_back(static_cast<char>((value>>(8*i))&255));};
    auto string=[&](const std::string& value){integer(value.size(),8);header+=value;};
    integer(0x46554747,4);integer(3,4);integer(tensors.size(),8);integer(23,8);
    auto text=[&](const std::string& key,const std::string& value){string(key);integer(8,4);string(value);};
    auto number=[&](const std::string& key,uint64_t value){string(key);integer(10,4);integer(value,8);};
    text("general.architecture","kimodo-motion");text("general.name","Kimodo-SOMA-RP-v1.2");
    string("general.alignment");integer(4,4);integer(32,4);
    number("kimodo.format_version",1);text("kimodo.skeleton","soma30");
    text("kimodo.model_identity","nvidia/Kimodo-SOMA-RP-v1.2@0e3c89024d896064b6851201373f7834784a386b");
    text("kimodo.license","nvidia-open-model-license");number("kimodo.commercial_use",1);
    text("kimodo.source_revision","0e3c89024d896064b6851201373f7834784a386b");
    text("kimodo.source_sha256","fdcef624140a12c0a4840192ce0b75cafb4e73228a893da3864b4a990d3e919e");
    number("kimodo.text_embedding_width",4096);number("kimodo.motion_dim",369);number("kimodo.global_root_dim",5);
    number("kimodo.local_root_dim",4);number("kimodo.body_dim",364);number("kimodo.hidden_size",1024);
    number("kimodo.layers",16);number("kimodo.heads",8);number("kimodo.feed_forward_size",2048);
    number("kimodo.num_text_tokens",50);number("kimodo.base_diffusion_steps",1000);number("kimodo.fps",30);
    string("kimodo.normalization_epsilon");integer(6,4);integer(std::bit_cast<uint32_t>(1e-5f),4);
    uint64_t cursor=0;
    for(auto& t:tensors) {
        cursor=(cursor+31)&~uint64_t{31};t.offset=cursor;cursor+=t.size;
        string(t.name);integer(t.shape.size(),4);for(auto dim=t.shape.rbegin();dim!=t.shape.rend();++dim)integer(*dim,8);
        integer(0,4);integer(t.offset,8);
    }
    header.append((32-header.size()%32)%32,'\0');
    auto temporary=root/"kimodo-v1.2.gguf.part";
    std::ofstream output(temporary,std::ios::binary|std::ios::trunc);output.write(header.data(),static_cast<std::streamsize>(header.size()));
    uint64_t written=0;std::vector<char> block(1024*1024);
    for(const auto& t:tensors) {
        for(;written<t.offset;++written)output.put(0);
        std::ifstream tensor(t.file,std::ios::binary);tensor.seekg(static_cast<std::streamoff>(t.start));
        if(t.stats)for(uint64_t i=0;i<t.size/4;++i) {double d=0;tensor.read(reinterpret_cast<char*>(&d),8);float f=static_cast<float>(d);output.write(reinterpret_cast<const char*>(&f),4);}
        else for(uint64_t left=t.size;left;) {auto n=static_cast<std::streamsize>(std::min<uint64_t>(left,block.size()));tensor.read(block.data(),n);output.write(block.data(),n);left-=static_cast<uint64_t>(n);}
        if(!tensor||!output)throw std::runtime_error("v1.2 conversion I/O failed");
        written=t.offset+t.size;
    }
    for(;written%32;++written)output.put(0);
    output.close();
    if(!output)throw std::runtime_error("v1.2 conversion write failed");
    verify(temporary,v12_model);fs::rename(temporary,root/"kimodo-v1.2.gguf");
}

}
void setup_models(const fs::path& root,const std::string& selection) {
    selection_valid(selection);
    for(const auto& f:files)if(selected(f,selection))install_file(root,f);
    std::cout<<"Models ready in "<<fs::absolute(root).string()<<"\n";
}
}
