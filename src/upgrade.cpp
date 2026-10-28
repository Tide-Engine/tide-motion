#include "upgrade.hpp"
#include <algorithm>
#include <charconv>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

namespace motion {
namespace {
constexpr uint64_t archive_limit=512*1024*1024;
const std::string releases="https://github.com/Tide-Engine/tide-motion/releases/download/";
std::string path_text(const fs::path& path) {
    auto text=path.u8string();return {reinterpret_cast<const char*>(text.data()),text.size()};
}
std::string read(const fs::path& path,uint64_t limit) {
    if(fs::file_size(path)>limit)throw std::runtime_error("Upgrade file exceeds size limit: "+path_text(path));
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot read upgrade file");
    return {std::istreambuf_iterator<char>(in),{}};
}
void write(const fs::path& path,const std::string& value) {
    std::ofstream out(path,std::ios::binary);out<<value;out.close();if(!out)throw std::runtime_error("Cannot write upgrade file");
}
std::string extension(const std::string& platform) {return platform.starts_with("windows-")?".zip":".tar.gz";}
std::string archive_name(const std::string& version,const std::string& platform) {return "tide-motion-"+version+"-"+platform+extension(platform);}
void download(const std::string& url,const fs::path& path,uint64_t limit) {
    std::cout<<"Downloading "<<path.filename().string()<<"...\n"<<std::flush;
    int code=run_process({"curl","-q","--fail","--location","--proto","=https","--proto-redir","=https","--retry","2","--connect-timeout","15","--max-time","300","--max-filesize",std::to_string(limit),"--user-agent","tide-motion-upgrade","--header","Accept: application/vnd.github+json","--output",path_text(path),url});
    if(code)throw std::runtime_error("Upgrade download failed; nothing installed (check the release version, network or GitHub rate limit)");
    if(fs::file_size(path)>limit)throw std::runtime_error("Upgrade download exceeds size limit");
}
fs::path executable() {
#ifdef _WIN32
    std::wstring buffer(32768,0);DWORD n=GetModuleFileNameW(nullptr,buffer.data(),static_cast<DWORD>(buffer.size()));
    if(!n||n==buffer.size())throw std::runtime_error("Cannot locate running executable");buffer.resize(n);return fs::canonical(buffer);
#elif defined(__APPLE__)
    uint32_t size=0;_NSGetExecutablePath(nullptr,&size);std::string buffer(size,0);
    if(_NSGetExecutablePath(buffer.data(),&size))throw std::runtime_error("Cannot locate running executable");
    return fs::canonical(buffer.c_str());
#else
    return fs::canonical("/proc/self/exe");
#endif
}
fs::path scratch(const fs::path& root,const std::string& prefix) {
    for(int i=0;i<16;++i) {
        auto name=prefix+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+std::to_string(std::random_device{}());
        auto path=root/name;if(fs::create_directory(path)) {
#ifndef _WIN32
            fs::permissions(path,fs::perms::owner_all,fs::perm_options::replace);
#endif
            return path;
        }
    }
    throw std::runtime_error("Cannot create private upgrade directory");
}
bool child(const fs::path& root,const fs::path& path) {return path.parent_path()==root&&!fs::is_symlink(path);}
void remove_scratch(const fs::path& root,const fs::path& path) {
    if(!child(root,path))throw std::runtime_error("Refusing upgrade cleanup outside its root");
    std::error_code ignored;fs::remove_all(path,ignored);
}
void remove_committed_backup(const fs::path& root,const fs::path& path) {
    remove_scratch(root,path);
    // A mapped Windows runtime can prevent deletion after the marker was removed.
    // Restore the committed marker so a later invocation can finish cleanup.
    if(fs::exists(path)) {try{write(path/"marker","tide-motion-upgrade-v1-committed");}catch(...) {}}
}
void cleanup_backups(const fs::path& root) {
    std::vector<fs::path> backups;
    for(const auto& e:fs::directory_iterator(root))if(e.path().filename().string().starts_with(".upgrade-backup-")&&child(root,e.path())&&e.is_directory()) {
        try {if(read(e.path()/"marker",64)=="tide-motion-upgrade-v1-committed")backups.push_back(e.path());}catch(...) {}
    }
    for(const auto& backup:backups)remove_committed_backup(root,backup);
}
struct Work { fs::path root,path; ~Work(){try{remove_scratch(root,path);}catch(...) {}} };
struct Lock {
#ifdef _WIN32
    HANDLE handle=INVALID_HANDLE_VALUE;
    explicit Lock(const fs::path& root) {
        auto path=root/".upgrade.lock";if(fs::is_symlink(path))throw std::runtime_error("Unsafe upgrade lock");
        handle=CreateFileW(path.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(handle==INVALID_HANDLE_VALUE)throw std::runtime_error("Another upgrade is running, or the installation is not writable");
    }
    ~Lock(){CloseHandle(handle);}
#else
    int fd=-1;
    explicit Lock(const fs::path& root) {
        auto path=root/".upgrade.lock";if(fs::is_symlink(path))throw std::runtime_error("Unsafe upgrade lock");
        fd=open(path.c_str(),O_CREAT|O_RDWR|O_NOFOLLOW,0600);
        if(fd<0)throw std::runtime_error("Installation is not writable");
        if(flock(fd,LOCK_EX|LOCK_NB)) {close(fd);fd=-1;throw std::runtime_error("Another upgrade is running");}
    }
    ~Lock(){if(fd>=0)close(fd);}
#endif
};
#ifndef _WIN32
void safe_member(std::string name) {
    if(name.empty()||name.front()=='/'||name.find('\\')!=std::string::npos||name.find(':')!=std::string::npos)throw std::runtime_error("Unsafe archive entry");
    for(const auto& part:fs::path(name))if(part=="..")throw std::runtime_error("Unsafe archive entry");
}
#endif
void extract(const fs::path& archive,const fs::path& work) {
    fs::create_directory(work/"payload");
#ifdef _WIN32
    // Fixed trusted extraction code, never a downloaded install script. Arguments
    // are passed separately; no path is interpolated into PowerShell source.
    auto script=work/"extract.ps1";
    write(script,R"PS(param([string]$Archive,[string]$Destination)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip=[IO.Compression.ZipFile]::OpenRead($Archive)
try {
  if($zip.Entries.Count -gt 10000){throw 'Too many archive entries'}
  [long]$total=0
  $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
  foreach($entry in $zip.Entries){
    $name=$entry.FullName
    if($name.Contains('\') -or $name.Contains(':') -or $name.StartsWith('/') -or (($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000){throw 'Unsafe archive entry'}
    foreach($part in $name.Split('/')){
      if($part -eq '..' -or $part -match '[. ]$' -or $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])([.]|$)'){throw 'Unsafe archive entry'}
    }
    $path=[IO.Path]::GetFullPath((Join-Path $Destination $name))
    if(!$path.StartsWith($Destination+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or !$seen.Add($path)){throw 'Unsafe archive entry'}
    $total+=$entry.Length
    if($total -gt 536870912){throw 'Unpacked archive exceeds size limit'}
  }
} finally {$zip.Dispose()}
Expand-Archive -LiteralPath $Archive -DestinationPath $Destination
)PS");
    if(run_process({"powershell.exe","-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File",path_text(script),"-Archive",path_text(archive),"-Destination",path_text(work/"payload")}))throw std::runtime_error("Archive extraction failed; nothing installed");
#else
    auto members=work/"members",types=work/"types";
    if(run_process({"tar","-tzf",path_text(archive)},members)||run_process({"tar","-tvzf",path_text(archive)},types))throw std::runtime_error("Cannot list upgrade archive");
    std::istringstream names(read(members,1024*1024));std::string line;size_t count=0;
    while(std::getline(names,line)) {safe_member(line);if(++count>10000)throw std::runtime_error("Too many archive entries");}
    std::istringstream entries(read(types,2*1024*1024));
    while(std::getline(entries,line)) {
        if(line.empty())continue;
        if(line[0]=='-'||line[0]=='d')continue;
        auto arrow=line.rfind(" -> ");std::string target=arrow==std::string::npos?"":line.substr(arrow+4);
        if(line[0]!='l'||!target.starts_with("libonnxruntime.")||target.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.")!=std::string::npos)
            throw std::runtime_error("Unsafe archive entry type or link");
    }
    if(run_process({"tar","-xzf",path_text(archive),"-C",path_text(work/"payload")}))throw std::runtime_error("Archive extraction failed; nothing installed");
#endif
}
std::string checksum(const std::string& text,const std::string& archive) {
    std::istringstream in(text);std::string hash,name,extra;in>>hash;
    if(hash.size()!=64||hash.find_first_not_of("0123456789abcdefABCDEF")!=std::string::npos)throw std::runtime_error("Malformed archive checksum");
    if(in>>name) {
        if(name.starts_with('*'))name.erase(0,1);
        if(name!=archive||(in>>extra))throw std::runtime_error("Checksum does not name the selected archive");
    }
    std::transform(hash.begin(),hash.end(),hash.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});return hash;
}
void install(const fs::path& archive,const fs::path& root,const fs::path& work,const std::string& version) {
    extract(archive,work);auto payload=work/"payload";
#ifdef _WIN32
    const fs::path binary="bin/tide-motion.exe";
#else
    const fs::path binary="bin/tide-motion";
#endif
    if(!fs::is_regular_file(payload/binary)) {
        std::vector<fs::path> entries;for(const auto& e:fs::directory_iterator(payload))entries.push_back(e.path());
        if(entries.size()!=1||!fs::is_directory(entries[0])||fs::is_symlink(entries[0]))throw std::runtime_error("Incomplete Tide Motion package");
        payload=entries[0];
    }
    const std::vector<fs::path> items{"licenses","examples","notes","README.md","PROMPTING.md","LICENSE","install.ps1","install.sh","bin"};
    for(const auto& item:items) {
        if(item=="PROMPTING.md"&&!fs::exists(payload/item))continue; // Added in 0.5.1.
        if(!fs::exists(payload/item)||fs::is_symlink(payload/item))throw std::runtime_error("Incomplete Tide Motion package: "+item.string());
    }
    if(!fs::is_regular_file(payload/"licenses/META-LLAMA-3.txt"))throw std::runtime_error("Missing package license");
    auto log=work/"probe.txt";
    if(run_process({path_text(payload/binary),"--version"},log))throw std::runtime_error("Packaged executable cannot run; nothing installed");
    std::string observed=read(log,4096);while(!observed.empty()&&(observed.back()=='\n'||observed.back()=='\r'))observed.pop_back();
    if(observed!="tide-motion "+version)throw std::runtime_error("Packaged executable version does not match release; nothing installed");
    // Swap only distribution-owned entries. Models, settings and arbitrary user
    // files are never moved. Old files remain available for rollback until all swaps pass.
    auto backup=scratch(root,".upgrade-backup-");write(backup/"marker","tide-motion-upgrade-v1");
    std::vector<fs::path> saved,placed;
    try {
        for(const auto& item:items) {
            auto target=root/item;
            if(fs::is_symlink(target))throw std::runtime_error("Refusing to replace a symlink in the installation: "+item.string());
            if(fs::exists(target)) {fs::rename(target,backup/item);saved.push_back(item);}
            if(fs::exists(payload/item)) {fs::rename(payload/item,target);placed.push_back(item);}
        }
        write(backup/"marker","tide-motion-upgrade-v1-committed");
    } catch(...) {
        auto failure=std::current_exception();
        try {
            for(auto i=placed.rbegin();i!=placed.rend();++i)fs::rename(root/ *i,payload/ *i);
            for(auto i=saved.rbegin();i!=saved.rend();++i)fs::rename(backup/ *i,root/ *i);
        } catch(...) {throw std::runtime_error("Upgrade failed and rollback needs attention; original files retained in "+path_text(backup));}
        remove_scratch(root,backup);std::rethrow_exception(failure);
    }
    // Only committed backups can be collected automatically on a later run.
    remove_committed_backup(root,backup); // Windows may retain the mapped old runtime.
    if(fs::exists(backup))std::cout<<"Previous running runtime retained in "<<path_text(backup)<<"; a later upgrade will finish cleanup after it exits.\n";
}
}
std::array<unsigned,3> upgrade_version(std::string_view text) {
    if(text.starts_with('v'))text.remove_prefix(1);
    std::array<unsigned,3> out{};
    for(size_t i=0;i<3;++i) {
        auto dot=text.find('.');auto part=text.substr(0,dot);
        if(part.empty()||part.size()>6||(part.size()>1&&part[0]=='0'))throw std::runtime_error("Version must be MAJOR.MINOR.PATCH");
        auto result=std::from_chars(part.data(),part.data()+part.size(),out[i]);
        if(result.ec!=std::errc{}||result.ptr!=part.data()+part.size()||(i<2&&dot==std::string_view::npos)||(i==2&&dot!=std::string_view::npos))throw std::runtime_error("Version must be MAJOR.MINOR.PATCH");
        if(i<2)text.remove_prefix(dot+1);
    }
    return out;
}
std::string upgrade_platform() {
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
    return "windows-x64";
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    return "macos-arm64";
#elif defined(__linux__) && (defined(__aarch64__) || defined(__arm64__))
    return "linux-arm64";
#elif defined(__linux__) && defined(__x86_64__)
    return "linux-x64";
#else
    throw std::runtime_error("No published upgrade package for this platform");
#endif
}
UpgradeRelease upgrade_release(const nlohmann::json& doc,const std::string& platform) {
    if(!doc.is_object()||doc.at("draft").get<bool>()||doc.at("prerelease").get<bool>())throw std::runtime_error("Upgrade requires a published stable release");
    auto tag=doc.at("tag_name").get<std::string>();(void)upgrade_version(tag);
    if(!tag.starts_with('v'))throw std::runtime_error("Release tag must start with v");
    UpgradeRelease out;out.version=tag.substr(1);out.archive=archive_name(out.version,platform);
    const std::string base=releases+tag+"/";std::set<std::string> seen;
    const auto& assets=doc.at("assets");if(!assets.is_array())throw std::runtime_error("Invalid release assets");
    for(const auto& a:assets) {
        auto name=a.at("name").get<std::string>();if(name!=out.archive&&name!=out.archive+".sha256")continue;
        if(!seen.insert(name).second||a.at("browser_download_url")!=base+name||!a.at("size").is_number_integer())throw std::runtime_error("Invalid upgrade release asset");
        auto bytes=a.at("size").get<int64_t>();
        if(bytes<=0||static_cast<uint64_t>(bytes)>(name==out.archive?archive_limit:4096))throw std::runtime_error("Upgrade asset exceeds size limit");
        if(name==out.archive) {out.url=base+name;out.bytes=static_cast<uint64_t>(bytes);}else out.checksum_url=base+name;
    }
    if(out.url.empty()||out.checksum_url.empty())throw std::runtime_error("Release has no archive/checksum for "+platform);
    return out;
}
int upgrade(const std::vector<std::string>& args,const std::string& current) {
    bool check=false,force=false;std::string wanted;fs::path archive,root;
    for(size_t i=2;i<args.size();++i) {
        const auto& flag=args[i];
        auto value=[&](){if(++i>=args.size()||args[i].empty())throw std::runtime_error("Missing upgrade option value");return args[i];};
        if(flag=="--check")check=true;
        else if(flag=="--force")force=true;
        else if(flag=="--version") {wanted=value();(void)upgrade_version(wanted);if(wanted.starts_with('v'))wanted.erase(0,1);}
        else if(flag=="--archive")archive=fs::absolute(utf8path(value()));
        else if(flag=="--root")root=fs::absolute(utf8path(value()));
        else throw std::runtime_error("Unknown upgrade option: "+flag);
    }
    if(!wanted.empty()&&!archive.empty())throw std::runtime_error("Use --version or --archive, not both");
    const auto platform=upgrade_platform();
    Work work{fs::canonical(fs::temp_directory_path()),{}};work.path=scratch(work.root,"tide-motion-upgrade-");
    UpgradeRelease selected;
    if(!archive.empty()) {
        auto name=archive.filename().string(),tail="-"+platform+extension(platform);
        if(!name.starts_with("tide-motion-")||!name.ends_with(tail))throw std::runtime_error("Archive name must identify its version and this platform");
        selected.version=name.substr(12,name.size()-12-tail.size());(void)upgrade_version(selected.version);selected.archive=name;
        if(fs::file_size(archive)>archive_limit)throw std::runtime_error("Upgrade archive exceeds size limit");
        auto sums=archive;sums+=".sha256";
        auto hash=checksum(read(sums,4096),name);
        auto local_copy=work.path/name;fs::copy_file(archive,local_copy);archive=local_copy;
        if(fs::file_size(archive)>archive_limit||sha256_file(archive)!=hash)throw std::runtime_error("Archive checksum mismatch; nothing installed");
    } else {
        auto metadata=work.path/"release.json";
        download("https://api.github.com/repos/Tide-Engine/tide-motion/releases/"+(wanted.empty()?"latest":"tags/v"+wanted),metadata,1024*1024);
        selected=upgrade_release(nlohmann::json::parse(read(metadata,1024*1024)),platform);
        if(!wanted.empty()&&selected.version!=wanted)throw std::runtime_error("Release version differs from requested version");
    }
    std::cout<<"Installed executable: "<<current<<"; selected release: "<<selected.version<<" ("<<platform<<")\n";
    if(check)return 0;
    if(!force&&wanted.empty()&&upgrade_version(selected.version)<=upgrade_version(current)) {
        std::cout<<"Already at or newer than the selected release; use --force to reinstall it.\n";return 0;
    }
    if(root.empty()) {
        auto bin=executable().parent_path();root=bin.parent_path();
        if(bin.filename()!="bin"||!fs::is_regular_file(root/"licenses/META-LLAMA-3.txt"))throw std::runtime_error("Not an installed Tide Motion layout; use --root INSTALL_DIR explicitly");
    }
    fs::create_directories(root);root=fs::canonical(root);
    if(root==root.root_path())throw std::runtime_error("Refusing to install into a filesystem root");
    Lock lock(root);cleanup_backups(root);
    if(archive.empty()) {
        archive=work.path/selected.archive;auto sums=archive;sums+=".sha256";
        download(selected.checksum_url,sums,4096);auto hash=checksum(read(sums,4096),selected.archive);
        download(selected.url,archive,archive_limit);
        if(fs::file_size(archive)!=selected.bytes||sha256_file(archive)!=hash)throw std::runtime_error("Archive checksum/size mismatch; nothing installed");
    }
    // Extraction is on the destination volume, so every rename is atomic.
    Work stage{root,scratch(root,".upgrade-work-")};
    install(archive,root,stage.path,selected.version);
    std::cout<<"Installed Tide Motion "<<selected.version<<" in "<<path_text(root)<<". Models and settings preserved.\n";return 0;
}
}
