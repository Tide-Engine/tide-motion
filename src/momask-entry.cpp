#include "momask.hpp"
#include "motion.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
namespace {
int launch(std::vector<std::string> args) {
#ifdef _WIN32
    std::array<wchar_t,32768> path{};DWORD length=GetModuleFileNameW(nullptr,path.data(),static_cast<DWORD>(path.size()));
    if(!length||length>=path.size())throw std::runtime_error("Cannot locate tide-momask executable");
    auto bytes=std::filesystem::path(std::wstring(path.data(),length)).u8string();args[0]=std::string(bytes.begin(),bytes.end());
#endif
    auto executable=motion::utf8path(args.at(0));
    args.insert(args.begin()+1,"momask");
    if(args.size()>2&&args[2]!="--help"&&args[2]!="help"&&std::find(args.begin(),args.end(),"--models")==args.end()) {
        args.push_back("--models");args.push_back((std::filesystem::absolute(executable).parent_path().parent_path()/"models").string());
    }
    return momask_main(args);
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
    try {
        SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
        std::vector<std::string> args;
        for(int i=0;i<argc;++i){int size=WideCharToMultiByte(CP_UTF8,0,argv[i],-1,nullptr,0,nullptr,nullptr);std::string value(static_cast<size_t>(size),0);WideCharToMultiByte(CP_UTF8,0,argv[i],-1,value.data(),size,nullptr,nullptr);value.pop_back();args.push_back(value);}
        return launch(std::move(args));
    }catch(const std::exception& e){std::cerr<<"tide-momask: "<<e.what()<<'\n';return 1;}
}
#else
int main(int argc,char** argv) {
    try{return launch({argv,argv+argc});}catch(const std::exception& e){std::cerr<<"tide-momask: "<<e.what()<<'\n';return 1;}
}
#endif
