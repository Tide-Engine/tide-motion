#include "motion.hpp"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <set>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace motion {
fs::path utf8path(std::string_view text) {
    std::u8string value;value.reserve(text.size());
    for(unsigned char c:text)value.push_back(static_cast<char8_t>(c));
    return fs::path(value);
}
#ifdef _WIN32
static std::wstring wide(const std::string& text) {
    if(text.empty())return {};
    int count=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,text.data(),static_cast<int>(text.size()),nullptr,0);
    if(!count)throw std::runtime_error("Invalid UTF-8 process argument");
    std::wstring out(static_cast<size_t>(count),0);
    MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,text.data(),static_cast<int>(text.size()),out.data(),count);return out;
}
static std::wstring quote(const std::string& text) {
    std::wstring out=L"\"";size_t slashes=0;
    for(wchar_t c:wide(text)) {
        if(c==L'\\') { ++slashes;continue; }
        out.append(slashes*(c==L'"'?2:1),L'\\');slashes=0;
        if(c==L'"')out+=L'\\';out+=c;
    }
    out.append(slashes*2,L'\\');out+=L'"';return out;
}
#endif
int run_process(const std::vector<std::string>& args,const fs::path& output) {
    if(args.empty())throw std::runtime_error("Missing executable");
#ifdef _WIN32
    std::wstring command;
    for(const auto& arg:args) { if(!command.empty())command+=L' ';command+=quote(arg); }
    STARTUPINFOW si{};si.cb=sizeof(si);PROCESS_INFORMATION pi{};
    si.dwFlags=STARTF_USESTDHANDLES;
    si.hStdInput=GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput=GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError=GetStdHandle(STD_ERROR_HANDLE);
    HANDLE file=INVALID_HANDLE_VALUE;
    if(!output.empty()) {
        SECURITY_ATTRIBUTES inherit{sizeof(inherit),nullptr,TRUE};
        file=CreateFileW(output.wstring().c_str(),GENERIC_WRITE,FILE_SHARE_READ,&inherit,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(file==INVALID_HANDLE_VALUE)throw std::runtime_error("Cannot create "+output.string());
        si.hStdOutput=si.hStdError=file;
    }
    DWORD flags=0;const char* hidden=std::getenv("TIDE_HIDDEN");if(hidden&&*hidden&&std::string(hidden)!="0")flags=CREATE_NO_WINDOW;
    BOOL started=CreateProcessW(nullptr,command.data(),nullptr,nullptr,TRUE,flags,nullptr,nullptr,&si,&pi);
    DWORD error=GetLastError();if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);
    if(!started)throw std::runtime_error("Cannot start '"+args.front()+"' (Windows error "+std::to_string(error)+"); install it or supply its path");
    CloseHandle(pi.hThread);WaitForSingleObject(pi.hProcess,INFINITE);
    DWORD code=1;GetExitCodeProcess(pi.hProcess,&code);CloseHandle(pi.hProcess);return static_cast<int>(code);
#else
    std::vector<char*> argv;for(const auto& a:args)argv.push_back(const_cast<char*>(a.c_str()));argv.push_back(nullptr);
    pid_t pid=fork();if(pid<0)throw std::runtime_error("Cannot start child process");
    if(pid==0) {
        if(!output.empty()) {int fd=open(output.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0644);if(fd<0)_exit(127);dup2(fd,1);dup2(fd,2);close(fd);}
        execvp(argv[0],argv.data());_exit(127);
    }
    int status;while(waitpid(pid,&status,0)<0)if(errno!=EINTR)throw std::runtime_error("Cannot wait for child");
    return WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
#endif
}
fs::path model_directory() {
    if(const char* p=std::getenv("TIDE_MOTION_MODELS"))return utf8path(p);
#ifdef _WIN32
    if(const char* p=std::getenv("LOCALAPPDATA"))return utf8path(p)/"TideMotion/models";
#else
    if(const char* p=std::getenv("XDG_CACHE_HOME"))return utf8path(p)/"tide-motion/models";
    if(const char* p=std::getenv("HOME"))return utf8path(p)/".cache/tide-motion/models";
#endif
    throw std::runtime_error("Cannot determine model cache; use --models <directory>");
}
void preview(const Clip& c,const fs::path& source,const std::string& tide,const fs::path& capture,int frames,const fs::path& gif) {
    c.validate();
    std::set<std::string> names;
    for(const auto& joint:c.joints)if(joint.name.empty()||!names.insert(joint.name).second)
        throw std::runtime_error("Skeleton preview needs unique, nonempty bone and ancestor names");
    const char* hidden=std::getenv("TIDE_HIDDEN");
    if(hidden&&*hidden&&std::string(hidden)!="0"&&capture.empty()&&gif.empty()&&frames==0)throw std::runtime_error("Hidden preview requires --capture, --gif or --frames");
    if(!gif.empty()&&gif.extension()!=".gif")throw std::runtime_error("--gif needs a .gif output path");
    // A default GIF shows the clip once, holding its last frame rather than wrapping.
    const bool once=!gif.empty()&&frames==0;
    auto root=fs::temp_directory_path()/"tide-motion-preview";
    fs::create_directories(root);
    // A per-clip directory keeps compiler caches reusable without conflating assets.
    size_t key=std::hash<std::string>{}(fs::absolute(source).generic_string());
    root/=std::to_string(key);fs::create_directories(root);
    fs::path exported=root/"motion.glb";
    if(fs::exists(exported))fs::remove(exported);
    save_glb(c,exported);
    std::ofstream scene(root/"preview.tide");
    scene<<R"(asset SkinnedMesh<Vertex3> rig = "motion.glb";
local singleton PreviewClock { float elapsed; }
local scene Main { }
view Preview(mut PreviewClock clock)
{
    var seconds = clock.elapsed;
    clock.elapsed += Frame.dt;
    rig.animations.TryGetAt(0, out var animation);
)";
    // Sampling before advancing makes the first rendered frame the clip's first frame.
    scene<<"    var pose = animation.Sample(seconds, Playback."<<(once?"Clamp":"Loop")<<");\n";
    scene<<"    rig.skeleton.bones.TryGetByName("<<nlohmann::json(c.joints[c.motion_root].name).dump()<<", out var centerBone);\n";
    scene<<"    var center = Math.Transform(pose.Model(centerBone), float3(0));\n";
    auto first=world_pose(c,c.frame(0));
    float low=first.front().p.y,high=low;
    for(const auto& joint:first) {low=std::min(low,joint.p.y);high=std::max(high,joint.p.y);}
    float drawing_scale=350/std::max(0.1f,high-low);
    // Project into two 2D panels. This uses Tide's ordinary drawing path and works
    // without depending on a particular game camera or skinned character asset.
    for(size_t j=0;j<c.joints.size();++j) {
        if(c.joints[j].parent<0)continue;
        // Tide adds a synthetic scene root when importing GLB. Bind by the
        // authored joint names instead of reusing GLB node indices.
        scene<<"    rig.skeleton.bones.TryGetByName("<<nlohmann::json(c.joints[j].name).dump()<<", out var b"<<j<<");\n";
        scene<<"    rig.skeleton.bones.TryGetByName("<<nlohmann::json(c.joints[static_cast<size_t>(c.joints[j].parent)].name).dump()<<", out var p"<<j<<");\n";
        scene<<"    var a"<<j<<" = Math.Transform(pose.Model(b"<<j<<"), float3(0)) - float3(center.x, 0, center.z);\n";
        scene<<"    var z"<<j<<" = Math.Transform(pose.Model(p"<<j<<"), float3(0)) - float3(center.x, 0, center.z);\n";
        scene<<"    Draw.Line(float2(-240 + a"<<j<<".x * "<<drawing_scale<<", -200 + (a"<<j<<".y - "<<low<<") * "<<drawing_scale<<"), float2(-240 + z"<<j<<".x * "<<drawing_scale<<", -200 + (z"<<j<<".y - "<<low<<") * "<<drawing_scale<<"), 4, Color(0.3, 0.8, 1));\n";
        scene<<"    Draw.Line(float2(240 + a"<<j<<".z * "<<drawing_scale<<", -200 + (a"<<j<<".y - "<<low<<") * "<<drawing_scale<<"), float2(240 + z"<<j<<".z * "<<drawing_scale<<", -200 + (z"<<j<<".y - "<<low<<") * "<<drawing_scale<<"), 4, Color(1, 0.7, 0.3));\n";
    }
    scene<<"    Draw.Line(float2(-450, -200), float2(-30, -200), 1, Color(0.3, 0.3, 0.3));\n";
    scene<<"    Draw.Line(float2(30, -200), float2(450, -200), 1, Color(0.3, 0.3, 0.3));\n}\n";
    scene.close();if(!scene)throw std::runtime_error("Cannot write Tide preview scene");
    std::string executable=tide;
    if(tide.find('/')!=std::string::npos||tide.find('\\')!=std::string::npos)executable=fs::absolute(utf8path(tide)).string();
    std::vector<std::string> args{executable,"run",root.string(),"--size","960","540"};
    if(!gif.empty()) {
        // tide run --record arrived in 0.5.0-nightly.45; older builds lack its help topic.
        auto probe=root/"record-help.txt";
        int probed=run_process({executable,"help","run","--record"},probe);
        std::ifstream text(probe);std::string usage((std::istreambuf_iterator<char>(text)),std::istreambuf_iterator<char>());
        if(probed||usage.find("--record")==std::string::npos)throw std::runtime_error("preview --gif needs Tide 0.5.0-nightly.45 or later (tide run --record); update Tide or use --capture");
        // Headless runs use fixed 1/60 s frames and the GIF keeps every second one
        // (30 a second), so 2(F-1)+1 frames show each of a clip's F frames once.
        if(once)frames=static_cast<int>(2*(c.frames()-1)+1);
        args.push_back("--record");args.push_back(fs::absolute(gif).string());
    }
    if(!capture.empty()) { args.push_back("--capture");args.push_back(fs::absolute(capture).string()); }
    if(frames) { args.push_back("--frames");args.push_back(std::to_string(frames)); }
    int code=run_process(args);if(code)throw std::runtime_error("Tide preview failed (exit "+std::to_string(code)+")");
}
}
