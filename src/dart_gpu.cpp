// Native DirectML execution follows the isolated C API boundary used by ARDY.
#include "dart_gpu.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#ifdef MOTION_ARDY_DIRECTML
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <dml-ort/onnxruntime_c_api.h>
#include <dml-ort/dml_provider_factory.h>
#endif
namespace motion {
#ifdef MOTION_ARDY_DIRECTML
namespace {
fs::path runtime() {
    std::wstring path(32768,L'\0');auto n=GetModuleFileNameW(nullptr,path.data(),static_cast<DWORD>(path.size()));
    if(!n||n>=path.size())throw DartGpuFailure("DART GPU: cannot locate executable");
    path.resize(n);return fs::path(path).parent_path()/"ardy-directml";
}
struct Module {
    HMODULE value=nullptr;
    explicit Module(const fs::path& p){value=LoadLibraryExW(p.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);if(!value)throw DartGpuFailure("DART GPU: cannot load "+p.string());}
    ~Module(){if(value)FreeLibrary(value);}
};
}
struct DartGpu::Impl {
    Module dml{runtime()/"DirectML.dll"},ort{runtime()/"onnxruntime.dll"};
    const OrtApi* api=nullptr;OrtEnv* env=nullptr;OrtMemoryInfo* memory=nullptr;OrtSessionOptions* options=nullptr;
    fs::path directory;std::map<std::string,OrtSession*> sessions;
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;Microsoft::WRL::ComPtr<IDMLDevice> ml;Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;uint64_t peak=0;
    void check(OrtStatus* s) const {if(s){std::string e=api->GetErrorMessage(s);api->ReleaseStatus(s);throw DartGpuFailure("DART GPU: "+e);}}
    Impl(const fs::path& dir,unsigned threads):directory(dir) {
        using GetBase=const OrtApiBase* (ORT_API_CALL*)();const FARPROC proc=GetProcAddress(ort.value,"OrtGetApiBase");GetBase base=nullptr;
        static_assert(sizeof(base)==sizeof(proc));std::memcpy(&base,&proc,sizeof(proc));
        if(!base||(api=base()->GetApi(ORT_API_VERSION))==nullptr)throw DartGpuFailure("DART GPU: incompatible ORT API");
        try {
            check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"tide-dart-directml",&env));check(api->DisableTelemetryEvents(env));
            check(api->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&memory));check(api->CreateSessionOptions(&options));
            check(api->DisableMemPattern(options));check(api->SetSessionExecutionMode(options,ORT_SEQUENTIAL));
            check(api->SetIntraOpNumThreads(options,static_cast<int>(threads?threads:4)));check(api->SetSessionGraphOptimizationLevel(options,ORT_ENABLE_ALL));
            const OrtDmlApi* provider=nullptr;check(api->GetExecutionProviderApi("DML",ORT_API_VERSION,reinterpret_cast<const void**>(&provider)));
            OrtDmlDeviceOptions device{HighPerformance,Gpu};check(provider->SessionOptionsAppendExecutionProvider_DML2(options,&device));
            Microsoft::WRL::ComPtr<ID3D12Device> d3d;Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
            check(provider->GetDMLDevice(options,&ml));
            if(FAILED(ml->GetParentDevice(IID_PPV_ARGS(&d3d))))throw DartGpuFailure("DART GPU: cannot obtain D3D12 device");
            D3D12_COMMAND_QUEUE_DESC desc{};desc.Type=D3D12_COMMAND_LIST_TYPE_COMPUTE;
            if(FAILED(d3d->CreateCommandQueue(&desc,IID_PPV_ARGS(&queue))))throw DartGpuFailure("DART GPU: cannot create command queue");
            if(SUCCEEDED(ml->GetParentDevice(IID_PPV_ARGS(&d3d)))&&SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
                factory->EnumAdapterByLuid(d3d->GetAdapterLuid(),IID_PPV_ARGS(&adapter));DXGI_ADAPTER_DESC desc{};
                if(adapter&&SUCCEEDED(adapter->GetDesc(&desc))){int n=WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,nullptr,0,nullptr,nullptr);std::string name(static_cast<size_t>(n),0);WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,name.data(),n,nullptr,nullptr);name.pop_back();std::cerr<<"DART DirectML adapter: "<<name<<'\n';}
            }
        }catch(...){release();throw;}
    }
    void release(){for(auto& [key,s]:sessions){(void)key;api->ReleaseSession(s);}sessions.clear();if(options){api->ReleaseSessionOptions(options);options=nullptr;}if(memory){api->ReleaseMemoryInfo(memory);memory=nullptr;}if(env){api->ReleaseEnv(env);env=nullptr;}}
    ~Impl(){release();}
    void sample_memory(){DXGI_QUERY_VIDEO_MEMORY_INFO info{};if(adapter&&SUCCEEDED(adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&info)))peak=std::max(peak,info.CurrentUsage);}
    std::vector<float> run(const std::string& network,DartFeed feed,const char* output_name) {
        auto batch=feed.begin()->second.shape.at(0);auto key=network+"-"+std::to_string(batch);
        if(!sessions.contains(key)) {
            OrtSessionOptions* opts=nullptr;check(api->CreateSessionOptions(&opts));
            try {
                check(api->DisableMemPattern(opts));check(api->SetSessionExecutionMode(opts,ORT_SEQUENTIAL));check(api->SetSessionGraphOptimizationLevel(opts,ORT_ENABLE_ALL));
                const OrtDmlApi* provider=nullptr;check(api->GetExecutionProviderApi("DML",ORT_API_VERSION,reinterpret_cast<const void**>(&provider)));
                check(provider->SessionOptionsAppendExecutionProvider_DML1(opts,ml.Get(),queue.Get()));
                if(network!="text")check(api->AddFreeDimensionOverrideByName(opts,"batch",batch));
                if(const char* profile=std::getenv("TIDE_DART_PROFILE"))check(api->EnableProfiling(opts,utf8path(std::string(profile)+"-"+key).c_str()));
                OrtSession* s=nullptr;check(api->CreateSession(env,(directory/(network+(network=="text"?".onnx":"-batch.onnx"))).c_str(),opts,&s));sessions[key]=s;
            }catch(...){api->ReleaseSessionOptions(opts);throw;}
            api->ReleaseSessionOptions(opts);
            std::cerr<<"DART DirectML session: "<<key<<'\n';
        }
        std::vector<const char*> names;std::vector<OrtValue*> values;OrtValue* output=nullptr;
        try {
            for(auto& [name,t]:feed) {
                OrtValue* value=nullptr;
                if(t.i.empty())check(api->CreateTensorWithDataAsOrtValue(memory,t.f.data(),t.f.size()*4,t.shape.data(),t.shape.size(),ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&value));
                else check(api->CreateTensorWithDataAsOrtValue(memory,t.i.data(),t.i.size()*8,t.shape.data(),t.shape.size(),ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,&value));
                names.push_back(name.c_str());values.push_back(value);
            }
            check(api->Run(sessions.at(key),nullptr,names.data(),const_cast<const OrtValue* const*>(values.data()),values.size(),&output_name,1,&output));
            OrtTensorTypeAndShapeInfo* shape=nullptr;check(api->GetTensorTypeAndShape(output,&shape));size_t n=0;auto status=api->GetTensorShapeElementCount(shape,&n);api->ReleaseTensorTypeAndShapeInfo(shape);check(status);
            float* data=nullptr;check(api->GetTensorMutableData(output,reinterpret_cast<void**>(&data)));std::vector<float> result(data,data+n);
            for(float v:result)if(!std::isfinite(v))throw DartGpuFailure("DART GPU: nonfinite output");
            sample_memory();api->ReleaseValue(output);output=nullptr;for(auto* v:values)api->ReleaseValue(v);return result;
        }catch(...){if(output)api->ReleaseValue(output);for(auto* v:values)api->ReleaseValue(v);throw;}
    }
};
bool dart_gpu_compiled(){return true;}
DartGpu::DartGpu(const fs::path& p,unsigned t):impl_(std::make_unique<Impl>(p,t)){}
DartGpu::~DartGpu()=default;
std::vector<float> DartGpu::run(const std::string& n,DartFeed f,const char* o){return impl_->run(n,std::move(f),o);}
uint64_t DartGpu::peak_memory_bytes() const{return impl_->peak;}
#else
struct DartGpu::Impl{};
bool dart_gpu_compiled(){return false;}
DartGpu::DartGpu(const fs::path&,unsigned){throw DartGpuFailure("DART GPU: this build has no DirectML support");}
DartGpu::~DartGpu()=default;
std::vector<float> DartGpu::run(const std::string&,DartFeed,const char*){throw DartGpuFailure("DART GPU: unavailable");}
uint64_t DartGpu::peak_memory_bytes() const{return 0;}
#endif
}
