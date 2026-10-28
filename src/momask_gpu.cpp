#include "momask_gpu.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#ifdef MOTION_MOMASK_DIRECTML
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <dml-ort/onnxruntime_c_api.h>
#include <dml-ort/dml_provider_factory.h>
namespace motion {
namespace {
fs::path runtime_directory() {
    std::wstring exe(32768,L'\0');DWORD count=GetModuleFileNameW(nullptr,exe.data(),static_cast<DWORD>(exe.size()));
    if(!count||count>=exe.size())throw MomaskGpuFailure("MoMask GPU: cannot locate executable");
    exe.resize(count);return fs::path(exe).parent_path()/"ardy-directml";
}
struct Module {
    HMODULE handle=nullptr;
    explicit Module(const fs::path& path) {
        handle=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        if(!handle)throw MomaskGpuFailure("MoMask GPU: cannot load "+path.string()+" (Windows error "+std::to_string(GetLastError())+")");
    }
    ~Module(){if(handle)FreeLibrary(handle);}
};
size_t element_size(int type) {
    if(type==1)return 4;
    if(type==7)return 8;
    if(type==9)return 1;
    throw MomaskGpuFailure("MoMask GPU: unsupported tensor type");
}
}
struct MomaskGpu::Impl {
    Module directml{runtime_directory()/"DirectML.dll"},ort{runtime_directory()/"onnxruntime.dll"};
    const OrtApi* api=nullptr;OrtEnv* env=nullptr;OrtMemoryInfo* memory=nullptr;
    std::map<std::string,OrtSession*> sessions;
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;uint64_t peak=0;
    void check(OrtStatus* status) const {
        if(status){std::string message=api->GetErrorMessage(status);api->ReleaseStatus(status);throw MomaskGpuFailure("MoMask GPU: "+message);}
    }
    Impl(const fs::path& dir,unsigned threads) {
        using GetBase=const OrtApiBase* (ORT_API_CALL*)();
        const FARPROC proc=GetProcAddress(ort.handle,"OrtGetApiBase");GetBase getbase=nullptr;
        static_assert(sizeof(getbase)==sizeof(proc));std::memcpy(&getbase,&proc,sizeof(proc));
        if(!getbase)throw MomaskGpuFailure("MoMask GPU: missing ORT API");
        api=getbase()->GetApi(ORT_API_VERSION);
        if(!api)throw MomaskGpuFailure("MoMask GPU: incompatible ORT API version");
        try {
            check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"tide-momask-directml",&env));
            check(api->DisableTelemetryEvents(env));
            check(api->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&memory));
            OrtSessionOptions* options=nullptr;check(api->CreateSessionOptions(&options));
            try {
                check(api->DisableMemPattern(options));check(api->SetSessionExecutionMode(options,ORT_SEQUENTIAL));
                check(api->SetIntraOpNumThreads(options,static_cast<int>(threads)));
                check(api->SetSessionGraphOptimizationLevel(options,ORT_ENABLE_ALL));
                const OrtDmlApi* dml=nullptr;
                check(api->GetExecutionProviderApi("DML",ORT_API_VERSION,reinterpret_cast<const void**>(&dml)));
                OrtDmlDeviceOptions device{HighPerformance,Gpu};check(dml->SessionOptionsAppendExecutionProvider_DML2(options,&device));
                Microsoft::WRL::ComPtr<IDMLDevice> ml;Microsoft::WRL::ComPtr<ID3D12Device> d3d;Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
                check(dml->GetDMLDevice(options,&ml));
                if(SUCCEEDED(ml->GetParentDevice(IID_PPV_ARGS(&d3d)))&&SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
                    factory->EnumAdapterByLuid(d3d->GetAdapterLuid(),IID_PPV_ARGS(&adapter));
                    DXGI_ADAPTER_DESC desc{};
                    if(adapter&&SUCCEEDED(adapter->GetDesc(&desc))) {
                        int count=WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,nullptr,0,nullptr,nullptr);
                        std::string name(static_cast<size_t>(count),0);WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,name.data(),count,nullptr,nullptr);name.pop_back();
                        std::cerr<<"MoMask DirectML adapter: "<<name<<"\n";
                    }
                }
                for(const char* name:{"masked_step","residual_step","rvq_encoder","rvq_decoder"}) {
                    if(const char* prefix=std::getenv("TIDE_MOMASK_PROFILE"))check(api->EnableProfiling(options,utf8path(std::string(prefix)+"-"+name).c_str()));
                    OrtSession* session=nullptr;check(api->CreateSession(env,(dir/(std::string(name)+".onnx")).c_str(),options,&session));sessions[name]=session;
                }
            } catch(...) {api->ReleaseSessionOptions(options);throw;}
            api->ReleaseSessionOptions(options);sample_memory();
        } catch(...) {release();throw;}
        std::cerr<<"MoMask motion networks: DirectML GPU (ORT 1.24.4); CLIP text: CPU ORT 1.31.1\n";
    }
    void release() {
        for(auto [name,session]:sessions){(void)name;api->ReleaseSession(session);}sessions.clear();
        if(memory){api->ReleaseMemoryInfo(memory);memory=nullptr;}
        if(env){api->ReleaseEnv(env);env=nullptr;}
    }
    ~Impl(){release();}
    void sample_memory(){DXGI_QUERY_VIDEO_MEMORY_INFO info{};if(adapter&&SUCCEEDED(adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&info)))peak=std::max(peak,info.CurrentUsage);}
    MomaskTensor run(const std::string& name,std::vector<MomaskTensor> tensors) {
        static const char* names[]={"input0","input1","input2","input3"};const char* output_name="output";
        if(tensors.empty()||tensors.size()>4)throw MomaskGpuFailure("MoMask GPU: invalid feed count");
        std::vector<OrtValue*> inputs;OrtValue* output=nullptr;
        try {
            for(auto& tensor:tensors) {
                size_t count=1;for(auto d:tensor.shape){if(d<1||count>10000000/static_cast<size_t>(d))throw MomaskGpuFailure("MoMask GPU: invalid tensor shape");count*=static_cast<size_t>(d);}
                if(tensor.bytes.size()!=count*element_size(tensor.type))throw MomaskGpuFailure("MoMask GPU: invalid tensor bytes");
                OrtValue* value=nullptr;check(api->CreateTensorWithDataAsOrtValue(memory,tensor.bytes.data(),tensor.bytes.size(),tensor.shape.data(),tensor.shape.size(),static_cast<ONNXTensorElementDataType>(tensor.type),&value));inputs.push_back(value);
            }
            check(api->Run(sessions.at(name),nullptr,names,const_cast<const OrtValue* const*>(inputs.data()),inputs.size(),&output_name,1,&output));
            OrtTensorTypeAndShapeInfo* info=nullptr;check(api->GetTensorTypeAndShape(output,&info));
            MomaskTensor result;size_t dims=0,count=0;ONNXTensorElementDataType type;
            try {
                check(api->GetDimensionsCount(info,&dims));result.shape.resize(dims);check(api->GetDimensions(info,result.shape.data(),dims));
                check(api->GetTensorShapeElementCount(info,&count));check(api->GetTensorElementType(info,&type));result.type=static_cast<int>(type);
            } catch(...) {api->ReleaseTensorTypeAndShapeInfo(info);throw;}
            api->ReleaseTensorTypeAndShapeInfo(info);
            if(count>10000000)throw MomaskGpuFailure("MoMask GPU: oversized output");
            result.bytes.resize(count*element_size(result.type));void* ptr=nullptr;check(api->GetTensorMutableData(output,&ptr));std::memcpy(result.bytes.data(),ptr,result.bytes.size());
            sample_memory();api->ReleaseValue(output);output=nullptr;for(auto* value:inputs)api->ReleaseValue(value);
            return result;
        } catch(...) {if(output)api->ReleaseValue(output);for(auto* value:inputs)api->ReleaseValue(value);throw;}
    }
};
MomaskGpu::MomaskGpu(const fs::path& dir,unsigned threads):impl(std::make_unique<Impl>(dir,threads)){}
MomaskGpu::~MomaskGpu()=default;
MomaskTensor MomaskGpu::run(const std::string& name,std::vector<MomaskTensor> inputs){return impl->run(name,std::move(inputs));}
uint64_t MomaskGpu::peak_memory_bytes() const{return impl->peak;}
}
#else
namespace motion {
struct MomaskGpu::Impl {};
MomaskGpu::MomaskGpu(const fs::path&,unsigned){throw MomaskGpuFailure("MoMask GPU: this build has no DirectML support");}
MomaskGpu::~MomaskGpu()=default;
MomaskTensor MomaskGpu::run(const std::string&,std::vector<MomaskTensor>){throw MomaskGpuFailure("MoMask GPU: unavailable");}
uint64_t MomaskGpu::peak_memory_bytes() const{return 0;}
}
#endif
