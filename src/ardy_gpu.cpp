// Optional native Windows DirectML runtime. Microsoft headers remain unmodified.
#include "ardy_gpu.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#ifdef MOTION_ARDY_DIRECTML
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
// This translation unit exclusively uses the pinned DirectML ORT C API.
#include <dml-ort/onnxruntime_c_api.h>
#include <dml-ort/dml_provider_factory.h>
#endif
namespace motion {
#ifdef MOTION_ARDY_DIRECTML
namespace {
struct Module {
    HMODULE handle=nullptr;
    explicit Module(const fs::path& path) {
        handle=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        if(!handle)throw std::runtime_error("ARDY GPU: cannot load "+path.string()+" (Windows error "+std::to_string(GetLastError())+")");
    }
    ~Module(){if(handle)FreeLibrary(handle);}
};
fs::path runtime_directory() {
    std::wstring exe(32768,L'\0');DWORD count=GetModuleFileNameW(nullptr,exe.data(),static_cast<DWORD>(exe.size()));
    if(!count||count>=exe.size())throw std::runtime_error("ARDY GPU: cannot locate executable");
    exe.resize(count);return fs::path(exe).parent_path()/"ardy-directml";
}
}
struct ArdyGpu::Impl {
    // Modules outlive every API-owned object, including partially constructed sessions.
    Module directml{runtime_directory()/"DirectML.dll"};
    Module ort{runtime_directory()/"onnxruntime.dll"};
    const OrtApi* api=nullptr;OrtEnv* env=nullptr;OrtMemoryInfo* memory=nullptr;
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;uint64_t peak_memory=0;
    std::map<std::string,OrtSession*> sessions;
    void check(OrtStatus* status) const {
        if(status){std::string error=api->GetErrorMessage(status);api->ReleaseStatus(status);throw std::runtime_error("ARDY GPU: "+error);}
    }
    Impl(const fs::path& dir,unsigned threads) {
        using GetBase=const OrtApiBase* (ORT_API_CALL*)();
        // Copy the FARPROC bits; newer clang rejects the direct cast under -Wcast-function-type-mismatch.
        const FARPROC proc=GetProcAddress(ort.handle,"OrtGetApiBase");GetBase getbase=nullptr;
        static_assert(sizeof(getbase)==sizeof(proc));std::memcpy(&getbase,&proc,sizeof(proc));
        if(!getbase)throw std::runtime_error("ARDY GPU: missing ORT API");
        api=getbase()->GetApi(ORT_API_VERSION);
        if(!api)throw std::runtime_error("ARDY GPU: incompatible DirectML ORT API version");
        // The redistributable is built with telemetry support; opt out explicitly.
        try {
            check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"tide-ardy-directml",&env));
            check(api->DisableTelemetryEvents(env));
            check(api->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&memory));
            OrtSessionOptions* options=nullptr;check(api->CreateSessionOptions(&options));
            try {
                check(api->DisableMemPattern(options));check(api->SetSessionExecutionMode(options,ORT_SEQUENTIAL));
                check(api->SetIntraOpNumThreads(options,static_cast<int>(threads?threads:8)));
                check(api->SetSessionGraphOptimizationLevel(options,ORT_ENABLE_ALL));
                const OrtDmlApi* dml=nullptr;
                check(api->GetExecutionProviderApi("DML",ORT_API_VERSION,reinterpret_cast<const void**>(&dml)));
                OrtDmlDeviceOptions device{HighPerformance,Gpu};
                check(dml->SessionOptionsAppendExecutionProvider_DML2(options,&device));
                Microsoft::WRL::ComPtr<IDMLDevice> ml;
                Microsoft::WRL::ComPtr<ID3D12Device> d3d;
                Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
                check(dml->GetDMLDevice(options,&ml));
                if(SUCCEEDED(ml->GetParentDevice(IID_PPV_ARGS(&d3d)))&&SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))){
                    factory->EnumAdapterByLuid(d3d->GetAdapterLuid(),IID_PPV_ARGS(&adapter));
                    DXGI_ADAPTER_DESC desc{};if(adapter&&SUCCEEDED(adapter->GetDesc(&desc))){
                        const int n=WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,nullptr,0,nullptr,nullptr);
                        std::string name(static_cast<size_t>(n),0);WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,name.data(),n,nullptr,nullptr);name.pop_back();
                        std::cerr<<"ARDY DirectML adapter: "<<name<<"\n";
                    }
                }
                // A developer may collect real provider-placement evidence, not merely
                // infer GPU use from successfully creating a session.
                for(const char* name:{"denoiser","encoder","decoder"}){
                    if(const char* prefix=std::getenv("TIDE_ARDY_PROFILE"))check(api->EnableProfiling(options,utf8path(std::string(prefix)+"-"+name).c_str()));
                    OrtSession* session=nullptr;check(api->CreateSession(env,(dir/(std::string(name)+".onnx")).c_str(),options,&session));sessions[name]=session;
                }
                sample_memory();
            }catch(...){api->ReleaseSessionOptions(options);throw;}
            api->ReleaseSessionOptions(options);
        }catch(...){release();throw;}
        std::cerr<<"ARDY motion networks: DirectML GPU (ORT 1.24.4, high-performance adapter); text encoder: CPU\n";
    }
    void release() {
        for(auto [name,session]:sessions){(void)name;api->ReleaseSession(session);}sessions.clear();
        if(memory){api->ReleaseMemoryInfo(memory);memory=nullptr;}
        if(env){api->ReleaseEnv(env);env=nullptr;}
    }
    ~Impl(){release();}
    void sample_memory(){DXGI_QUERY_VIDEO_MEMORY_INFO info{};if(adapter&&SUCCEEDED(adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&info)))peak_memory=std::max(peak_memory,info.CurrentUsage);}
    std::vector<float> run(const std::string& network,ArdyFeed feed) {
        std::vector<const char*> names;std::vector<OrtValue*> inputs;OrtValue* output=nullptr;
        try {
            for(auto& [name,t]:feed){OrtValue* value=nullptr;
                if(t.i.empty())check(api->CreateTensorWithDataAsOrtValue(memory,t.f.data(),t.f.size()*sizeof(float),t.shape.data(),t.shape.size(),ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&value));
                else check(api->CreateTensorWithDataAsOrtValue(memory,t.i.data(),t.i.size()*sizeof(int64_t),t.shape.data(),t.shape.size(),ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,&value));
                names.push_back(name.c_str());inputs.push_back(value);
            }
            const char* out="output";
            check(api->Run(sessions.at(network),nullptr,names.data(),const_cast<const OrtValue* const*>(inputs.data()),inputs.size(),&out,1,&output));
            OrtTensorTypeAndShapeInfo* shape=nullptr;check(api->GetTensorTypeAndShape(output,&shape));size_t count=0;
            auto status=api->GetTensorShapeElementCount(shape,&count);api->ReleaseTensorTypeAndShapeInfo(shape);check(status);
            float* ptr=nullptr;check(api->GetTensorMutableData(output,reinterpret_cast<void**>(&ptr)));
            std::vector<float> result(ptr,ptr+count);
            sample_memory();
            for(float v:result)if(!std::isfinite(v))throw std::runtime_error("ARDY GPU: nonfinite output");
            api->ReleaseValue(output);output=nullptr;for(auto* value:inputs)api->ReleaseValue(value);
            return result;
        }catch(...){if(output)api->ReleaseValue(output);for(auto* value:inputs)api->ReleaseValue(value);throw;}
    }
};
bool ardy_gpu_compiled(){return true;}
ArdyGpu::ArdyGpu(const fs::path& dir,unsigned threads):impl_(std::make_unique<Impl>(dir,threads)){}
ArdyGpu::~ArdyGpu()=default;
std::vector<float> ArdyGpu::run(const std::string& network,ArdyFeed feed){return impl_->run(network,std::move(feed));}
uint64_t ArdyGpu::peak_memory_bytes() const{return impl_->peak_memory;}
#else
struct ArdyGpu::Impl{};
bool ardy_gpu_compiled(){return false;}
ArdyGpu::ArdyGpu(const fs::path&,unsigned){throw std::runtime_error("ARDY GPU: this build has no DirectML support");}
ArdyGpu::~ArdyGpu()=default;
std::vector<float> ArdyGpu::run(const std::string&,ArdyFeed){throw std::runtime_error("ARDY GPU: unavailable");}
uint64_t ArdyGpu::peak_memory_bytes() const{return 0;}
#endif
}
