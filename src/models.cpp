#include "motion.hpp"
#include <picosha2.h>
#include <array>
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
constexpr ModelFile files[]={
    {"kimodo","kimodo.gguf",1133166784ULL,"3bf1229f4c1eff1d28f5196a854113da2df9a11a5e21c60694630903bf948ee4",
     "https://huggingface.co/LocalAI-io/Kimodo-SOMA-RP-v1.1-GGML/resolve/65f8b8ab34ee7524161cc87b20ae4cb4f5d3a12d/models/kimodo-soma-rp-v1.1-f32.gguf"},
    {"kimodo","text/encoder.gguf",8142058752ULL,"b26d0e74b115b33a7f2df5aca04426548365e99217cd590986f8abc4172e4c5c",
     "https://huggingface.co/LocalAI-io/Llama-3-Kimodo-GGML/resolve/3e8d958803beaddb6011ac534f2be972e2710c7d/Llama-3-Kimodo-Q8_0.gguf"},
    {"kimodo","text/tokenizer.gguf",7295054ULL,"81614aca62a98846c02b72cc2e5378e5bdce8b4d507b88f96eb1faa90dae607e",
     "https://huggingface.co/LocalAI-io/Llama-3-Kimodo-GGML/resolve/3e8d958803beaddb6011ac534f2be972e2710c7d/tokenizer.gguf"},
    {"bricks","motionbricks.onnx",757864273ULL,"856419dc72aeb14e591eebf4dd0ea05734a2506ddb00bcd39dd6dac2ad60f4fb",
     "https://media.githubusercontent.com/media/NVlabs/motionbricks/54fc563aeaaf819a753d72cc9fdc6713989cdd43/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx"}
};
void selection_valid(const std::string& selection) {
    if(selection!="all"&&selection!="kimodo"&&selection!="bricks")throw std::runtime_error("Model selection must be all, kimodo or bricks");
}
bool selected(const ModelFile& f,const std::string& s) { return s=="all"||s==f.group; }
void verify(const fs::path& path,const ModelFile& f) {
    if(!fs::exists(path)||fs::file_size(path)!=f.bytes)throw std::runtime_error("Missing or incomplete model '"+path.string()+"'; run tide-motion models install "+f.group);
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
void setup_models(const fs::path& root,const std::string& selection) {
    selection_valid(selection);
    for(const auto& f:files)if(selected(f,selection)) {
        auto path=root/f.file;fs::create_directories(path.parent_path());
        if(fs::exists(path)) { verify(path,f);std::cerr<<"Verified "<<path.string()<<"\n";continue; }
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
    std::cout<<"Models ready in "<<fs::absolute(root).string()<<"\n";
}
}
