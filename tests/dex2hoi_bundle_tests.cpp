// Model-free filesystem/manifest contracts. Tiny stand-in files are not ONNX
// networks and this test makes no inference/parity or license-grant assertion.
#include "dex2hoi_bundle.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace motion;
using nlohmann::json;
namespace {
void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
template<class F> void refuses(F f,const char* message) {
    bool refused=false;
    try {f();}catch(const std::exception&){refused=true;}
    check(refused,message);
}
void write(const fs::path& path,const std::string& bytes) {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    out<<bytes;
    if(!out)throw std::runtime_error("Cannot write bundle fixture");
}
void manifest(const fs::path& root,const json& value) {write(root/"bundle.json",value.dump(2));}
json fixture(const fs::path& root) {
    fs::create_directories(root);
    json files=json::object();
    for(auto name:{"denoiser.onnx","object.onnx","clip.onnx","tokenizer.json","normalization.json","LICENSE.txt"}) {
        write(root/name,std::string("stand-in contract fixture ")+name);
        files[name]={{"sha256",sha256_file(root/name)},{"size",fs::file_size(root/name)}};
    }
    write(root/"unlisted.txt","This unlisted file must not be imported.");
    json pin={{"sha256",std::string(64,'0')},{"size",1}};
    json value={
        {"schema",1},{"backend","dex2hoi"},{"upstreamRevision",dex2hoi::upstream_revision},
        {"representation",{{"bones",65},{"numObjects",2},{"objectSlotSize",18},{"features",429},{"contextFrames",10},{"predictionFrames",60},{"fps",30}}},
        {"architecture",{{"objectEncoder","bps"},{"basisPoints",512},{"latentDim",512},{"decoderLayers",8},{"crossAttentionLayers",8},{"hiddenDim",1024},{"fusion","transformer"},{"fusionHeads",3}}},
        {"diffusion",{{"steps",500},{"schedule","cosine"},{"prediction","x0"},{"eta",0}}},
        {"origin",{{"checkpoint",pin},{"mean",pin},{"std",pin},{"eps",pin},{"state","model_avg"}}},
        {"licensingStatus","user-supplied-unverified"},{"files",files}
    };
    manifest(root,value);
    return value;
}
}

int main(int argc,char** argv)try {
    check(argc==2,"Need isolated bundle-test directory");
    auto nonce=std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path root=fs::path(argv[1])/std::to_string(nonce);
    auto source=root/"source";
    auto good=fixture(source);
    auto checked=dex2hoi::inspect_bundle(source);
    check(checked.manifest==good&&checked.manifest_sha256==sha256_file(source/"bundle.json"),"Exact user manifest is returned");
    auto imported=dex2hoi::import_bundle(source,root/"cache"/"dex2hoi");
    check(imported.manifest_sha256==checked.manifest_sha256,"Import retains manifest digest");
    check(fs::is_regular_file(imported.root/"LICENSE.txt"),"Listed license is retained");
    check(!fs::exists(imported.root/"unlisted.txt"),"Unlisted files are not imported");
    check(!fs::exists(root/"cache"/"dex2hoi.part"),"Successful staging is published");
    dex2hoi::inspect_bundle(imported.root);
    refuses([&]{dex2hoi::import_bundle(source,imported.root);},"Existing destination must refuse");
    refuses([&]{dex2hoi::import_bundle(source,source);},"Source cannot overwrite itself");
    fs::create_directory(root/"reserved.part");
    refuses([&]{dex2hoi::import_bundle(source,root/"reserved");},"Existing staging must refuse");
    check(!fs::exists(root/"reserved"),"Refused staging does not publish a destination");

    // Same-length corruption exercises hashing instead of only file-size checks.
    auto damaged=imported.root/"denoiser.onnx";
    auto count=fs::file_size(damaged);
    write(damaged,std::string(static_cast<size_t>(count),'x'));
    refuses([&]{dex2hoi::inspect_bundle(imported.root);},"Same-length tamper must refuse");
    auto try_bad=[&](json value,const char* message) {
        manifest(source,value);
        refuses([&]{dex2hoi::inspect_bundle(source);},message);
        manifest(source,good);
    };
    auto bad=good;bad["unexpected"]=true;try_bad(bad,"Unknown top-level fields must refuse");
    bad=good;bad["architecture"]["objectEncoder"]="pointwise";try_bad(bad,"Wrong trained encoder must refuse");
    bad=good;bad["representation"]["features"]=411;try_bad(bad,"Wrong feature count must refuse");
    bad=good;bad["diffusion"]["steps"]=1000;try_bad(bad,"Wrong trained schedule must refuse");
    bad=good;bad["upstreamRevision"]=std::string(40,'0');try_bad(bad,"Unpinned upstream must refuse");
    bad=good;bad["licensingStatus"]="commercially-cleared";try_bad(bad,"Integrity check cannot imply rights clearance");
    bad=good;bad["origin"]["mean"]["sha256"]="not-a-hash";try_bad(bad,"Malformed provenance hash must refuse");
    bad=good;bad["files"]["clip.onnx"]["size"]=-1;try_bad(bad,"Negative lengths must refuse");
    bad=good;bad["files"]["clip.onnx"]["size"]=1.5;try_bad(bad,"Fractional lengths must refuse");
    bad=good;bad["files"]["clip.onnx"]["sha256"]=std::string(64,'A');try_bad(bad,"Noncanonical hashes must refuse");
    bad=good;bad["files"].erase("clip.onnx");try_bad(bad,"Missing required graph must refuse");
    auto pin=good["files"]["clip.onnx"];
    for(auto name:{"../outside.onnx","sub/clip.onnx","sub\\clip.onnx","C:clip.onnx","clip.onnx:stream","NUL.onnx","COM1.onnx","clip.onnx.","bundle.json"}) {
        bad=good;bad["files"][name]=pin;try_bad(bad,"Unsafe flat filename must refuse");
    }
    bad=good;bad["files"]["CLIP.ONNX"]=pin;try_bad(bad,"Windows case alias must refuse on every platform");
    auto duplicate=good.dump();duplicate.insert(1,"\"schema\":1,");write(source/"bundle.json",duplicate);
    refuses([&]{dex2hoi::inspect_bundle(source);},"Duplicate JSON keys must refuse");
    manifest(source,good);
    fs::remove(source/"clip.onnx");
    refuses([&]{dex2hoi::inspect_bundle(source);},"Missing listed file must refuse");

    // A platform without symbolic-link privileges still runs all other checks.
    fixture(source);
    auto link_source=root/"linked-source";
    std::error_code ec;
    fs::create_directory_symlink(fs::absolute(source),link_source,ec);
    if(!ec)refuses([&]{dex2hoi::inspect_bundle(link_source);},"Symlink bundle directory must refuse");
    fs::remove(source/"clip.onnx");
    ec.clear();fs::create_symlink(fs::absolute(imported.root/"clip.onnx"),source/"clip.onnx",ec);
    if(!ec)refuses([&]{dex2hoi::inspect_bundle(source);},"Symlink graph with matching bytes must refuse");
    std::cout<<"Dex2HOI private bundle pins, import, no-overwrite, tamper, paths and schema contracts pass\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
