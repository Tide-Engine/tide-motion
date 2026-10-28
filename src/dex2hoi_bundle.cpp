#include "dex2hoi_bundle.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <cerrno>
#include <cstdio>
#include <cstring>
#endif

namespace motion::dex2hoi {
namespace {
using nlohmann::json;
constexpr uint64_t max_manifest_bytes=1024*1024;
constexpr uint64_t max_file_bytes=16ULL*1024*1024*1024;
constexpr uint64_t max_bundle_bytes=32ULL*1024*1024*1024;

[[noreturn]] void fail(const std::string& reason) {
    throw std::runtime_error("Dex2HOI bundle: "+reason);
}
bool path_present(const fs::path& path) {
    // exists() follows links and misses dangling ones, which must also refuse.
    std::error_code ec;
    auto status=fs::symlink_status(path,ec);
    if(ec&&ec!=std::errc::no_such_file_or_directory)fail("cannot inspect "+path.string());
    return status.type()!=fs::file_type::not_found&&status.type()!=fs::file_type::none;
}
void require_plain_path(const fs::path& path,bool directory) {
    std::error_code ec;
    auto status=fs::symlink_status(path,ec);
    if(ec||fs::is_symlink(status)||(directory?!fs::is_directory(status):!fs::is_regular_file(status)))
        fail(std::string(directory?"missing/plain directory required: ":"missing/plain file required: ")+path.string());
#ifdef _WIN32
    auto attributes=GetFileAttributesW(path.c_str());
    if(attributes==INVALID_FILE_ATTRIBUTES||(attributes&FILE_ATTRIBUTE_REPARSE_POINT))
        fail("reparse points are not accepted: "+path.string());
#endif
}
void publish_directory(const fs::path& stage,const fs::path& destination) {
    // std::filesystem::rename can replace an existing empty directory on POSIX.
    // Use the supported hosts' exclusive rename so a racing importer cannot
    // overwrite a destination after the earlier existence check.
#ifdef _WIN32
    if(!MoveFileW(stage.c_str(),destination.c_str()))fail("exclusive directory publication failed (Windows error "+std::to_string(GetLastError())+")");
#elif defined(__linux__) && defined(SYS_renameat2)
    constexpr unsigned rename_noreplace=1;
    if(syscall(SYS_renameat2,AT_FDCWD,stage.c_str(),AT_FDCWD,destination.c_str(),rename_noreplace)!=0)
        fail(std::string("exclusive directory publication failed: ")+std::strerror(errno));
#elif defined(__APPLE__)
    if(renamex_np(stage.c_str(),destination.c_str(),RENAME_EXCL)!=0)
        fail(std::string("exclusive directory publication failed: ")+std::strerror(errno));
#else
    (void)stage;(void)destination;
    fail("this host has no supported exclusive directory publication primitive");
#endif
}
std::string lower_ascii(std::string text) {
    for(char& c:text)if(c>='A'&&c<='Z')c=static_cast<char>(c-'A'+'a');
    return text;
}
void require_filename(const std::string& name) {
    if(name.empty()||name.size()>128||name.front()=='.'||name.back()=='.')fail("invalid flat filename: "+name);
    for(unsigned char c:name)
        if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.'))
            fail("invalid flat filename: "+name);
    auto stem=lower_ascii(name.substr(0,name.find('.')));
    if(stem=="con"||stem=="prn"||stem=="aux"||stem=="nul"||
       (stem.size()==4&&(stem.starts_with("com")||stem.starts_with("lpt"))&&stem[3]>='0'&&stem[3]<='9'))
        fail("reserved filename: "+name);
    if(lower_ascii(name)=="bundle.json")fail("bundle.json cannot list itself");
}
void require_fields(const json& value,std::initializer_list<const char*> allowed,const char* label) {
    if(!value.is_object())fail(std::string(label)+" must be an object");
    std::set<std::string> keys;
    for(auto key:allowed)keys.insert(key);
    for(const auto& [key,v]:value.items()) {
        (void)v;
        if(!keys.contains(key))fail(std::string("unknown ")+label+" field: "+key);
    }
    for(auto key:allowed)if(!value.contains(key))fail(std::string("missing ")+label+" field: "+key);
}
json read_manifest(const fs::path& path) {
    require_plain_path(path,false);
    auto size=fs::file_size(path);
    if(!size||size>max_manifest_bytes)fail("bundle.json must contain 1..1048576 bytes");
    std::ifstream in(path,std::ios::binary);
    std::string data(static_cast<size_t>(size),'\0');
    if(!in.read(data.data(),static_cast<std::streamsize>(data.size())))fail("cannot read bundle.json");
    std::vector<std::set<std::string>> object_keys;
    auto reject_duplicates=[&](int,json::parse_event_t event,json& parsed) {
        if(event==json::parse_event_t::object_start)object_keys.emplace_back();
        else if(event==json::parse_event_t::key) {
            if(object_keys.empty()||!object_keys.back().insert(parsed.get<std::string>()).second)
                fail("duplicate JSON key");
        } else if(event==json::parse_event_t::object_end)object_keys.pop_back();
        return true;
    };
    try {return json::parse(data,reject_duplicates);}
    catch(const json::exception& e){fail(std::string("invalid bundle.json: ")+e.what());}
}
void require_sha256(const json& value) {
    if(!value.is_string())fail("sha256 must be a lowercase hexadecimal string");
    auto hash=value.get<std::string>();
    if(hash.size()!=64||!std::all_of(hash.begin(),hash.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}))
        fail("sha256 must contain 64 lowercase hexadecimal digits");
}
uint64_t validate_pin(const json& pin) {
    require_fields(pin,{"sha256","size"},"file pin");
    require_sha256(pin.at("sha256"));
    if(!pin.at("size").is_number_unsigned()&&!pin.at("size").is_number_integer())fail("file size must be an integer");
    if(pin.at("size").is_number_integer()&&pin.at("size").get<int64_t>()<=0)fail("file size must be positive");
    auto size=pin.at("size").get<uint64_t>();
    if(!size||size>max_file_bytes)fail("file size exceeds native import limits");
    return size;
}

// Kept separate from filesystem verification so incompatible architecture cannot
// be mistaken for a compatible model merely because its files hash correctly.
void validate_manifest(const json& manifest) {
    require_fields(manifest,{"schema","backend","upstreamRevision","representation","architecture","diffusion","origin","licensingStatus","files"},"manifest");
    if(manifest.at("schema")!=1||manifest.at("backend")!="dex2hoi"||manifest.at("upstreamRevision")!=upstream_revision)
        fail("unsupported schema, backend or upstream revision");
    if(manifest.at("licensingStatus")!="user-supplied-unverified")
        fail("licensingStatus must be user-supplied-unverified; verification does not establish artifact rights");
    const auto& representation=manifest.at("representation");
    require_fields(representation,{"bones","numObjects","objectSlotSize","features","contextFrames","predictionFrames","fps"},"representation");
    if(representation.at("bones")!=65||representation.at("numObjects")!=2||representation.at("objectSlotSize")!=18||
       representation.at("features")!=429||representation.at("contextFrames")!=10||representation.at("predictionFrames")!=60||representation.at("fps")!=30)
        fail("unsupported motion representation");
    const auto& architecture=manifest.at("architecture");
    require_fields(architecture,{"objectEncoder","basisPoints","latentDim","decoderLayers","crossAttentionLayers","hiddenDim","fusion","fusionHeads"},"architecture");
    if(architecture.at("objectEncoder")!="bps"||architecture.at("basisPoints")!=512||architecture.at("latentDim")!=512||
       architecture.at("decoderLayers")!=8||architecture.at("crossAttentionLayers")!=8||architecture.at("hiddenDim")!=1024||
       architecture.at("fusion")!="transformer"||architecture.at("fusionHeads")!=3)
        fail("unsupported trained architecture");
    const auto& diffusion=manifest.at("diffusion");
    require_fields(diffusion,{"steps","schedule","prediction","eta"},"diffusion");
    if(diffusion.at("steps")!=500||diffusion.at("schedule")!="cosine"||diffusion.at("prediction")!="x0"||diffusion.at("eta")!=0)
        fail("unsupported diffusion schedule");
    const auto& origin=manifest.at("origin");
    require_fields(origin,{"checkpoint","mean","std","eps","state"},"origin");
    for(auto name:{"checkpoint","mean","std","eps"})(void)validate_pin(origin.at(name));
    if(origin.at("state")!="model_avg")fail("only the trained EMA model_avg state is supported");
    const auto& files=manifest.at("files");
    if(!files.is_object()||files.empty()||files.size()>64)fail("files must list 1..64 explicitly pinned files");
    std::set<std::string> filenames;
    uint64_t total=0;
    for(const auto& [name,pin]:files.items()) {
        require_filename(name);
        if(!filenames.insert(lower_ascii(name)).second)fail("case-insensitive filename collision");
        auto size=validate_pin(pin);
        if(size>max_bundle_bytes-total)fail("bundle size exceeds native import limits");
        total+=size;
    }
    for(auto name:{"denoiser.onnx","object.onnx","clip.onnx","tokenizer.json","normalization.json"})
        if(!files.contains(name))fail(std::string("missing required prepared asset: ")+name);
}
}

Bundle inspect_bundle(const fs::path& root) {
    require_plain_path(root,true);
    auto manifest_path=root/"bundle.json";
    require_plain_path(manifest_path,false);
    if(fs::file_size(manifest_path)>max_manifest_bytes)fail("bundle.json exceeds 1048576 bytes");
    auto digest=sha256_file(manifest_path);
    auto manifest=read_manifest(manifest_path);
    validate_manifest(manifest);
    for(const auto& [name,pin]:manifest.at("files").items()) {
        auto file=root/utf8path(name);
        require_plain_path(file,false);
        if(fs::file_size(file)!=pin.at("size").get<uint64_t>()||sha256_file(file)!=pin.at("sha256").get<std::string>())
            fail("checksum/size mismatch: "+name);
    }
    if(sha256_file(manifest_path)!=digest)fail("manifest changed during verification");
    return {fs::absolute(root).lexically_normal(),std::move(manifest),std::move(digest)};
}

Bundle import_bundle(const fs::path& source,const fs::path& destination) {
    if(destination.empty()||destination.filename().empty()||destination.filename()=="."||destination.filename()=="..")
        fail("destination must name a new bundle directory");
    if(path_present(destination))fail("import destination already exists: "+destination.string());
    auto original=inspect_bundle(source);
    auto stage=destination;stage+=".part";
    if(path_present(stage))fail("import staging path already exists: "+stage.string());
    if(!stage.parent_path().empty())fs::create_directories(stage.parent_path());
    if(!fs::create_directory(stage))fail("cannot reserve staging directory: "+stage.string());
    try {
        for(const auto& [name,pin]:original.manifest.at("files").items()) {
            (void)pin;
            require_plain_path(original.root/utf8path(name),false);
            fs::copy_file(original.root/utf8path(name),stage/utf8path(name),fs::copy_options::none);
        }
        fs::copy_file(original.root/"bundle.json",stage/"bundle.json",fs::copy_options::none);
        auto staged=inspect_bundle(stage);
        if(staged.manifest_sha256!=original.manifest_sha256)fail("source manifest changed during import");
        if(path_present(destination))fail("import destination appeared during staging");
        publish_directory(stage,destination);
        staged.root=fs::absolute(destination).lexically_normal();
        return staged;
    } catch(const std::exception& e) {
        fail(std::string("import failed; staging is retained at ")+stage.string()+": "+e.what());
    }
}
}
