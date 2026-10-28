#include "motion.hpp"
#include "equipment.hpp"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <locale>
#include <numbers>
#include <sstream>
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
// Runs a preview project. A GIF needs tide run --record; with clip_frames it records a
// clip of that many frames exactly once (unless frames is given).
static void run_tide(const std::string& tide,const fs::path& root,int width,int height,const fs::path& gif,const fs::path& capture,int frames,size_t clip_frames) {
    std::string executable=tide;
    if(tide.find('/')!=std::string::npos||tide.find('\\')!=std::string::npos)executable=fs::absolute(utf8path(tide)).string();
    std::vector<std::string> args{executable,"run",root.string(),"--size",std::to_string(width),std::to_string(height)};
    if(!gif.empty()) {
        // tide run --record arrived in 0.5.0-nightly.45; older builds lack its help topic.
        auto probe=root/"record-help.txt";
        int probed=run_process({executable,"help","run","--record"},probe);
        std::ifstream text(probe);std::string usage((std::istreambuf_iterator<char>(text)),std::istreambuf_iterator<char>());
        if(probed||usage.find("--record")==std::string::npos)throw std::runtime_error("preview --gif needs Tide 0.5.0-nightly.45 or later (tide run --record); update Tide or use --capture");
        // Headless runs use fixed 1/60 s frames and the GIF keeps every second one
        // (30 a second), so 2(F-1)+1 frames show each of a clip's F frames once.
        if(!frames&&clip_frames)frames=static_cast<int>(2*(clip_frames-1)+1);
        args.push_back("--record");args.push_back(fs::absolute(gif).string());
    }
    if(!capture.empty()) { args.push_back("--capture");args.push_back(fs::absolute(capture).string()); }
    if(frames) { args.push_back("--frames");args.push_back(std::to_string(frames)); }
    int code=run_process(args);if(code)throw std::runtime_error("Tide preview failed (exit "+std::to_string(code)+")");
}
// The vertical extent of a clip and its widest horizontal reach from the motion root,
// over every second frame and the last. ground is the median of each frame's lowest
// joint, so a hand pushing through the floor for a moment does not move it.
struct Extent { float low=0, high=0, reach=0, ground=0; };
static Extent clip_extent(const Clip& c) {
    Extent e;bool first=true;
    std::vector<size_t> frames;
    for(size_t f=0;f<c.frames();f+=2)frames.push_back(f);
    if(c.frames()%2==0)frames.push_back(c.frames()-1);
    std::vector<float> lows;
    for(size_t f:frames) {
        auto pose=world_pose(c,c.frame(f));const auto centre=pose[c.motion_root].p;
        float lowest=pose.front().p.y;
        for(const auto& joint:pose) {
            if(first) {e.low=e.high=joint.p.y;first=false;}
            lowest=std::min(lowest,joint.p.y);
            e.low=std::min(e.low,joint.p.y);e.high=std::max(e.high,joint.p.y);
            e.reach=std::max(e.reach,std::hypot(joint.p.x-centre.x,joint.p.z-centre.z));
        }
        lows.push_back(lowest);
    }
    std::nth_element(lows.begin(),lows.begin()+static_cast<std::ptrdiff_t>(lows.size()/2),lows.end());
    e.ground=lows[lows.size()/2];
    return e;
}
// A lit skinned body: the procedural mannequin or the user's character, drawn twice
// through one camera, facing it and in profile, each centred on the motion root with
// a floor of tiles that slides under it as the root travels.
static void body_scene(const Clip& c,const fs::path& root,const PreviewOptions& options,bool once) {
    std::string model="character.glb",texture;
    if(options.body==PreviewBody::mannequin) {
        require_soma(c);
        save_mannequin(root/model,options.camera.enabled);
    } else {
        auto ext=options.character.extension().string();for(auto& ch:ext)ch=static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if(ext!=".glb"&&ext!=".fbx")throw std::runtime_error("--character must be a .glb or .fbx model");
        model="character"+ext;
        fs::copy_file(options.character,root/model,fs::copy_options::overwrite_existing);
        if(!options.texture.empty()) {
            auto tex=options.texture.extension().string();for(auto& ch:tex)ch=static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if(tex!=".png"&&tex!=".jpg"&&tex!=".jpeg")throw std::runtime_error("--texture must be a .png or .jpg image");
            texture="skin"+tex;
            fs::copy_file(options.texture,root/texture,fs::copy_options::overwrite_existing);
        }
    }
    std::vector<std::string> props;
    for(size_t i=0;i<options.props.size();++i) {
        const auto& attachment=options.props[i];
        auto ext=attachment.mesh.extension().string();for(auto& ch:ext)ch=static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if(ext!=".glb"&&ext!=".fbx")throw std::runtime_error("--prop must be a .glb or .fbx mesh");
        props.push_back("prop"+std::to_string(i)+ext);
        fs::copy_file(attachment.mesh,root/props.back(),fs::copy_options::overwrite_existing);
    }
    std::vector<std::string> equipment_meshes;
    if(options.equipment) {
        const auto& eq=*options.equipment;Clip tracks;tracks.name="equipment";tracks.fps=c.fps;
        for(size_t i=0;i<eq.objects.size();++i) {
            tracks.joints.push_back({"Object"+std::to_string(i),-1,{}});
            tracks.joints.push_back({"Scale"+std::to_string(i),-1,{}});
            const auto& object=eq.objects[i];std::string mesh;
            if(object.builtin_rod) {
                mesh="equipment-rod.glb";
                if(!fs::exists(root/mesh)) {
                    EquipmentResult unit;unit.body.joints.push_back({"PreviewRodRoot",-1,{}});unit.body.poses.push_back({});
                    EquipmentTrack rod;rod.name="PreviewRod";rod.builtin_rod=true;rod.poses.push_back({});rod.scales.push_back({1,1,1});unit.objects.push_back(rod);
                    save_equipment_glb(unit,root/mesh);
                }
            } else if(!object.mesh.empty()) {
                mesh="equipment-mesh-"+std::to_string(i)+object.mesh.extension().string();
                fs::copy_file(object.mesh,root/mesh,fs::copy_options::overwrite_existing);
            }
            equipment_meshes.push_back(mesh);
        }
        for(size_t f=0;f<c.frames();++f)for(const auto& object:eq.objects) {
            tracks.poses.push_back(object.poses.at(f));tracks.poses.push_back({object.scales.at(f),{}});
        }
        save_glb(tracks,root/"equipment-tracks.glb");
    }
    // Framing from the whole clip (a get-up starts lying down): where its feet are, its
    // height above them and how far it reaches out from its root.
    auto extent=preview_extent(c,options);
    float low=extent.ground;
    // Generated and retargeted clips stand on y=0 with joints (toes, ankles) above their
    // soles; keep that ground unless the clip is clearly elsewhere (mostly in the air).
    // A retargeted body lying down can sink hands into it (no contacts are solved).
    if(low>-0.3f&&low<0.15f)low=0;
    // The camera aims at the middle of the body's height and stands back far enough for
    // its farthest reach as well: a panel spans 0.63 view heights beside its centre.
    const float bottom=std::min(low,extent.low);
    const float height=std::max(0.5f,extent.high-bottom),view=std::max(height,(extent.reach+0.15f)/0.63f);
    auto num=[](float v){std::ostringstream o;o.imbue(std::locale::classic());o<<std::setprecision(6)<<std::fixed<<v;return o.str();};
    std::ofstream scene(root/"preview.tide");
    scene<<R"(struct Corner
{
    float3 position;
    float3 normal;
    float2 uv;
    Color color = Color.white;
}

asset SkinnedMesh<Corner> body = ")"<<model<<R"(";
asset Animation motion = "motion.glb";
)";
    if(!texture.empty())scene<<"asset Texture skin = \""<<texture<<"\";\n";
    for(size_t i=0;i<props.size();++i)scene<<"asset Mesh<Corner> prop"<<i<<" = \""<<props[i]<<"\";\n";
    if(options.equipment) {
        scene<<"asset SkinnedMesh<Vertex3> equipmentTracks = \"equipment-tracks.glb\";\n";
        for(size_t i=0;i<equipment_meshes.size();++i)if(!equipment_meshes[i].empty())
            scene<<"asset Mesh<Corner> equipmentMesh"<<i<<" = \""<<equipment_meshes[i]<<"\";\n";
    }
    scene<<"const float HEIGHT = "<<num(height)<<";\nconst float VIEW = "<<num(view)<<";\nconst float FLOOR = "<<num(low)<<";\n";
    scene<<R"(const float TILE = HEIGHT * 0.22;
const float SPREAD = VIEW * 0.66;

local singleton PreviewClock { float elapsed; }
local scene Main { }

// A sun from above the camera's left and a sky from above.
)";
    // The mannequin is painted by its vertex colours; a character by its texture or
    // in plain clay (FBX vertex colours are often unrelated authoring data).
    if(options.body==PreviewBody::mannequin)scene<<"frag Lit(Corner corner, out Color color, Instance instance)\n{\n    var paint = corner.color;\n";
    else if(texture.empty())scene<<"frag Lit(Corner corner, out Color color, Instance instance)\n{\n    var paint = Color(0.82, 0.8, 0.76);\n";
    else scene<<"frag Lit(Corner corner, out Color color, Texture skin, Instance instance)\n{\n    var paint = skin.Sample(corner.uv);\n";
    scene<<R"(    var normal = Math.Normalize(instance.Normal(corner.normal));
    var sun = Math.Normalize(float3(-0.45, 0.8, -0.55));
    var light = 0.3 + 0.2 * (0.5 + 0.5 * normal.y) + 0.6 * Math.Max(0, Math.Dot(normal, sun));
    color = Color(paint.r * light, paint.g * light, paint.b * light, 1);
}
)";
    // A prop is drawn in plain light steel on its bone's model matrix without the rig's
    // scale (Synty bones carry 0.01), then offset and turned within that bone.
    if(!props.empty()||options.camera.enabled||options.equipment) {
        scene<<R"(
frag Steel(Corner corner, out Color color, Instance instance)
{
    var normal = Math.Normalize(instance.Normal(corner.normal));
    var sun = Math.Normalize(float3(-0.45, 0.8, -0.55));
    var light = 0.3 + 0.2 * (0.5 + 0.5 * normal.y) + 0.6 * Math.Max(0, Math.Dot(normal, sun));
    color = Color(0.78 * light, 0.8 * light, 0.85 * light, 1);
}

float4x4 Unscaled(float4x4 m)
{
    return float4x4(float4(Math.Normalize(m.c0.xyz), 0), float4(Math.Normalize(m.c1.xyz), 0), float4(Math.Normalize(m.c2.xyz), 0), m.c3);
}
)";
        for(size_t i=0;i<options.props.size();++i) {
            const auto& o=options.props[i].offset;const auto& r=options.props[i].rotation;
            scene<<"\nconst float4x4 HELD"<<i<<" = float4x4.TRS(float3("<<num(o.x)<<", "<<num(o.y)<<", "<<num(o.z)<<"), quaternion.Euler("
             <<num(r.x)<<"deg, "<<num(r.y)<<"deg, "<<num(r.z)<<"deg), float3(1));\n";
        }
    }
    scene<<"\n// One copy of the body at a place on screen, turned about its own centre.\n";
    std::string grip_parameters=options.equipment?"Pose equipmentPose, ":"",grip_arguments=options.equipment?"equipmentPose, ":"";
    for(size_t i=0;i<props.size();++i) {
        grip_parameters+="Bone grip"+std::to_string(i)+", ";
        grip_arguments+="grip"+std::to_string(i)+", ";
    }
    scene<<"void Panel(Pose pose, "<<grip_parameters<<"float3 centre, float3 at, float turn)\n";
    scene<<R"({
    var spin = quaternion.Euler(0, turn, 0);
    var place = Math.Mul(float4x4.TRS(at, spin, float3(1)), float4x4.Translate(float3(-centre.x, 0, -centre.z)));
    for (int i = -2; i <= 2; i++)
    {
        for (int k = -2; k <= 2; k++)
        {
            var tx = Math.Floor(centre.x / TILE) + i;
            var tz = Math.Floor(centre.z / TILE) + k;
            var odd = (tx + tz) - 2 * Math.Floor((tx + tz) / 2);
            var local = float3((tx + 0.5) * TILE - centre.x, FLOOR - 0.002, (tz + 0.5) * TILE - centre.z);
            Draw.Plane(at + Math.Rotate(spin, local), float2(TILE, TILE), spin, odd < 0.5 ? Color(0.36, 0.37, 0.4) : Color(0.28, 0.29, 0.32));
        }
    }
)";
    scene<<(texture.empty()?"    Draw.SkinnedMesh(body, place, pose, Lit);\n":"    Draw.SkinnedMesh(body, place, pose, skin, Lit);\n");
    for(size_t i=0;i<props.size();++i)scene<<"    Draw.Mesh(prop"<<i<<", Math.Mul(Math.Mul(place, Unscaled(pose.Model(grip"<<i<<"))), HELD"<<i<<"), Steel);\n";
    if(options.equipment)for(size_t i=0;i<equipment_meshes.size();++i)if(!equipment_meshes[i].empty()) {
        scene<<"    equipmentTracks.skeleton.bones.TryGetByName(\"Object"<<i<<"\", out var equipmentBone"<<i<<");\n";
        scene<<"    equipmentTracks.skeleton.bones.TryGetByName(\"Scale"<<i<<"\", out var scaleBone"<<i<<");\n";
        scene<<"    var meshScale"<<i<<" = Math.Transform(equipmentPose.Model(scaleBone"<<i<<"), float3(0));\n";
        scene<<"    Draw.Mesh(equipmentMesh"<<i<<", Math.Mul(place, Math.Mul(equipmentPose.Model(equipmentBone"<<i<<"), float4x4.Scale(meshScale"<<i<<"))), Steel);\n";
    }
    scene<<"}\n";
    scene<<R"(
view Preview(mut PreviewClock clock)
{
    // Sample before advancing so the first rendered frame is the clip's first frame.
    var seconds = clock.elapsed;
    clock.elapsed += Frame.dt;
    mut var pose = body.defaultPose;
)";
    scene<<"    pose.Sample(motion, seconds, Playback."<<(once?"Clamp":"Loop")<<");\n";
    if(options.equipment) {
        scene<<"    equipmentTracks.animations.TryGetAt(0, out var equipmentAnimation);\n";
        scene<<"    var equipmentPose = equipmentAnimation.Sample(seconds, Playback."<<(once?"Clamp":"Loop")<<");\n";
    }
    scene<<"    body.skeleton.TryGetByName("<<nlohmann::json(c.joints[c.motion_root].name).dump()<<", out var centreBone);\n";
    for(size_t i=0;i<props.size();++i)scene<<"    body.skeleton.TryGetByName("<<nlohmann::json(options.props[i].bone).dump()<<", out var grip"<<i<<");\n";
    scene<<R"(    var centre = Math.Transform(pose.Model(centreBone), float3(0));
    Draw.Clear(Color(0.17, 0.18, 0.21));
)";
    if(options.camera.enabled) {
        const auto& camera=options.camera;
        auto local=preview_offset(camera.offset,camera.rotation);
        std::string anchor="eyeBone";
        if(camera.mode=="root") {
            auto rig=preview_rig(c,options);auto pose=world_pose(rig,rig.frame(0));
            size_t eye=0;bool found=false;
            for(size_t j=0;j<rig.joints.size();++j)if(rig.joints[j].name==camera.bone){eye=j;found=true;}
            if(!found)throw std::runtime_error("Camera bone not found: "+camera.bone);
            local=inverse(pose[rig.motion_root])*pose[eye]*local;anchor="centreBone";
        } else scene<<"    body.skeleton.TryGetByName("<<nlohmann::json(camera.bone).dump()<<", out var eyeBone);\n";
        scene<<"    var eyeLocal = float4x4.TRS(float3("<<num(local.p.x)<<", "<<num(local.p.y)<<", "<<num(local.p.z)
             <<"), quaternion("<<num(local.q.x)<<", "<<num(local.q.y)<<", "<<num(local.q.z)<<", "<<num(local.q.w)<<"), float3(1));\n";
        scene<<"    var eye = Math.Mul(Unscaled(pose.Model("<<anchor<<")), eyeLocal);\n";
        scene<<"    Draw.SetCamera(eye, float4x4.PerspectiveFov("<<num(camera.fov)<<"deg, Screen.width / Screen.height, "<<num(camera.near_plane)<<", 1000));\n";
        // Keep the original world space so the camera and props share the body transform.
        scene<<"    Panel(pose, "<<grip_arguments<<"centre, float3(centre.x, 0, centre.z), 0);\n";
    } else {
        scene<<"    Draw.SetCamera(float3(0, "<<num(bottom+height*0.72f)<<", -VIEW * 2.7), quaternion.Euler(6deg, 0, 0), 30);\n";
        // Turned so the first frame faces the camera on the left and shows its right side,
        // facing screen right, on the right (the camera looks along +z).
        float facing=options.facing*180/std::numbers::pi_v<float>;

        scene<<"    Panel(pose, "<<grip_arguments<<"centre, float3(-SPREAD, 0, 0), "<<num(180-facing)<<"deg);\n";
        scene<<"    Panel(pose, "<<grip_arguments<<"centre, float3(SPREAD, 0, 0), "<<num(90-facing)<<"deg);\n";
    }
    scene<<"}\n";
    scene.close();if(!scene)throw std::runtime_error("Cannot write Tide preview scene");
}
void preview(const Clip& c,const fs::path& source,const PreviewOptions& options) {
    const auto& tide=options.tide;const auto& capture=options.capture;const auto& gif=options.gif;int frames=options.frames;
    c.validate();validate_preview_setup(options);
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
    // Files of an earlier preview of the same clip with another body.
    for(const auto& entry:fs::directory_iterator(root)) {
        auto stem=entry.path().stem().string();
        const bool prop_asset=stem=="prop"||(stem.starts_with("prop")&&stem.size()>4&&
            std::all_of(stem.begin()+4,stem.end(),[](unsigned char ch){return std::isdigit(ch);}));
        if(entry.is_regular_file()&&(stem=="character"||stem=="skin"||prop_asset||stem.starts_with("equipment-")))fs::remove(entry.path());
    }
    if(options.body!=PreviewBody::skeleton)body_scene(c,root,options,once);
    else {
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
    // Scaled to the whole clip's height, as the bodies are.
    const auto extent=clip_extent(c);const float low=extent.low;
    float drawing_scale=350/std::max(0.1f,extent.high-low);
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
    }
    run_tide(tide,root,960,540,gif,capture,frames,once?c.frames():0);
}
// Several SOMA clips on the mannequin through one orthographic camera, a labelled cell
// each. The sheet shows key poses (key_frames) along two rows, in right profile and
// facing the camera; the GIF plays every clip at once, facing and in profile.
static void variants_scene(const std::vector<Clip>& clips,const std::vector<std::string>& labels,const fs::path& root,bool animated,float* aspect,size_t* across) {
    save_mannequin(root/"character.glb");
    auto first=world_pose(clips[0],clips[0].frame(0));
    float low=first.front().p.y,high=low;
    for(const auto& joint:first) {low=std::min(low,joint.p.y);high=std::max(high,joint.p.y);}
    if(low>-0.05f&&low<0.15f)low=0;
    // Cells fit the tallest and widest poses of any clip (a clip may crouch, jump or
    // kick out from its first frame), at least a standing body's height.
    float top_y=high,reach=0;
    for(const auto& c:clips)for(size_t f=0;f<c.frames();f+=2) {
        auto pose=world_pose(c,c.frame(f));const auto centre=pose[c.motion_root].p;
        for(const auto& joint:pose) {top_y=std::max(top_y,joint.p.y);reach=std::max(reach,std::hypot(joint.p.x-centre.x,joint.p.z-centre.z));}
    }
    // The body's flesh reaches about 0.15 m past its joints.
    constexpr float flesh=0.15f;
    const float height=std::max(0.5f,top_y-low+flesh);
    const size_t n=clips.size(),poses=animated?1:6;
    size_t columns=animated?std::min(n,static_cast<size_t>(std::ceil(std::sqrt(2.0*static_cast<double>(n)))))
        :n<=3?1:static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(n)/2)));
    columns=std::max<size_t>(1,columns);
    const size_t rows=(n+columns-1)/columns;
    // Metres: a pose every 0.8 heights, or wider for poses that reach out; a cell is a
    // label strip and one row (GIF: the two views side by side) or two rows (sheet:
    // profile above, front below).
    const float step=std::max(0.8f*height,1.5f*reach+flesh),pad=0.35f*height,label=0.32f*height,row=1.3f*height;
    const float cell_w=static_cast<float>(animated?2:poses)*step+pad,cell_h=label+row*static_cast<float>(animated?1:2)+(animated?0.25f:0.0f)*height;
    const float view_w=static_cast<float>(columns)*cell_w,view_h=static_cast<float>(rows)*cell_h;
    *aspect=view_w/view_h;*across=columns;
    auto num=[](float v){std::ostringstream o;o.imbue(std::locale::classic());o<<std::setprecision(5)<<std::fixed<<v;return o.str();};
    std::ofstream scene(root/"preview.tide");
    scene<<R"(struct Corner
{
    float3 position;
    float3 normal;
    float2 uv;
    Color color = Color.white;
}

asset SkinnedMesh<Corner> body = "character.glb";
)";
    for(size_t i=0;i<n;++i)scene<<"asset Animation v"<<i<<" = \"v"<<i<<".glb\";\n";
    scene<<"const float HEIGHT = "<<num(height)<<";\nconst float FLOOR = "<<num(low)<<";\n";
    scene<<"const float VIEW_W = "<<num(view_w)<<";\nconst float VIEW_H = "<<num(view_h)<<";\n";
    scene<<R"(
local singleton PreviewClock { float elapsed; }
local scene Main { }

frag Lit(Corner corner, out Color color, Instance instance)
{
    var paint = corner.color;
    var normal = Math.Normalize(instance.Normal(corner.normal));
    var sun = Math.Normalize(float3(-0.45, 0.8, -0.55));
    var light = 0.3 + 0.2 * (0.5 + 0.5 * normal.y) + 0.6 * Math.Max(0, Math.Dot(normal, sun));
    color = Color(paint.r * light, paint.g * light, paint.b * light, 1);
}

// One copy of the body standing at a place, turned about its own centre.
void Body(Pose pose, float3 at, float turn)
{
)";
    scene<<"    body.skeleton.TryGetByName("<<nlohmann::json(clips[0].joints[clips[0].motion_root].name).dump()<<", out var centreBone);\n";
    scene<<R"(    var centre = Math.Transform(pose.Model(centreBone), float3(0));
    var spin = quaternion.Euler(0, turn, 0);
    var place = Math.Mul(float4x4.TRS(at - float3(0, FLOOR, 0), spin, float3(1)), float4x4.Translate(float3(-centre.x, 0, -centre.z)));
    Draw.SkinnedMesh(body, place, pose, Lit);
}

// A cell's backdrop and a floor line under each row.
void Cell(float left, float top, float width, float height, float floorA, float floorB)
{
    Draw.Plane(float3(left + width * 0.5, top - height * 0.5, 4), float2(width - HEIGHT * 0.06, height - HEIGHT * 0.06), quaternion.Euler(-90deg, 0, 0), Color(0.21, 0.22, 0.25));
    Draw.Cube(float3(left + width * 0.5, floorA - 0.01, 0), float3(width - HEIGHT * 0.3, 0.02, 0.6), Color(0.45, 0.46, 0.5));
    Draw.Cube(float3(left + width * 0.5, floorB - 0.01, 0), float3(width - HEIGHT * 0.3, 0.02, 0.6), Color(0.45, 0.46, 0.5));
}

float2 OnScreen(float x, float y) { return float2(x * Screen.width / VIEW_W, -y * Screen.height / VIEW_H); }

)";
    // Only the GIF plays; the sheet is one still frame.
    if(animated)scene<<"view Sheet(mut PreviewClock clock)\n{\n    var seconds = clock.elapsed;\n    clock.elapsed += Frame.dt;\n";
    else scene<<"view Sheet()\n{\n";
    scene<<R"(    Draw.Clear(Color(0.13, 0.14, 0.16));
    Draw.SetCamera(float4x4.TRS(float3(VIEW_W * 0.5, -VIEW_H * 0.5, -50), quaternion.Euler(0, 0, 0), float3(1)), float4x4.Ortho(VIEW_W, VIEW_H, 0.1, 200));
)";
    struct Text { std::string text; float x,y,size; bool centred; };
    std::vector<Text> texts;
    for(size_t i=0;i<n;++i) {
        const auto& c=clips[i];
        const float left=static_cast<float>(i%columns)*cell_w,top=-static_cast<float>(i/columns)*cell_h;
        const float floor_a=top-label-1.1f*height,floor_b=animated?floor_a:floor_a-row;
        const float facing=kimodo_heading(c,c.frame(0))*180/std::numbers::pi_v<float>;
        const std::string front=num(180-facing),side=num(90-facing);
        scene<<"    Cell("<<num(left)<<", "<<num(top)<<", "<<num(cell_w)<<", "<<num(cell_h)<<", "<<num(floor_a)<<", "<<num(floor_b)<<");\n";
        const auto metrics=root_travel(c);
        std::ostringstream travel;travel.imbue(std::locale::classic());
        travel<<labels[i]<<std::fixed<<std::setprecision(2)<<" | displacement "<<metrics.displacement
              <<" m | path "<<metrics.distance<<" m | peak "<<metrics.peak_speed<<" m/s";
        // A GIF cell has only two poses' width. Its full metrics line can spill
        // over the next candidate; keep that detail on the wide still sheet.
        texts.push_back({animated?labels[i]:travel.str(),left+pad*0.5f,top-0.06f*height,0.13f*height,false});
        if(!animated)texts.push_back({std::to_string(footfalls(c).size())+" travelling footfalls",left+pad*0.5f,top-0.21f*height,0.10f*height,false});
        if(animated) {
            scene<<"    {\n        mut var pose = body.defaultPose;\n        pose.Sample(v"<<i<<", seconds, Playback.Clamp);\n";
            scene<<"        Body(pose, float3("<<num(left+pad*0.5f+0.5f*step)<<", "<<num(floor_a)<<", 0), "<<front<<"deg);\n";
            scene<<"        Body(pose, float3("<<num(left+pad*0.5f+1.5f*step)<<", "<<num(floor_a)<<", 0), "<<side<<"deg);\n    }\n";
            continue;
        }
        auto frames=key_frames(c,poses);
        for(size_t k=0;k<frames.size();++k) {
            const float x=left+pad*0.5f+(static_cast<float>(k)+0.5f)*step,seconds=static_cast<float>(frames[k])/c.fps;
            scene<<"    {\n        mut var pose = body.defaultPose;\n        pose.Sample(v"<<i<<", "<<num(seconds)<<", Playback.Clamp);\n";
            scene<<"        Body(pose, float3("<<num(x)<<", "<<num(floor_a)<<", 0), "<<side<<"deg);\n";
            scene<<"        Body(pose, float3("<<num(x)<<", "<<num(floor_b)<<", 0), "<<front<<"deg);\n    }\n";
            std::ostringstream time;time.imbue(std::locale::classic());time<<std::setprecision(1)<<std::fixed<<seconds<<" s";
            texts.push_back({time.str(),x,floor_a-0.04f*height,0.11f*height,true});
        }
    }
    scene<<"    Draw.SetScreen();\n";
    for(const auto& t:texts)scene<<"    Draw.Text("<<nlohmann::json(t.text).dump()<<", OnScreen("<<num(t.x)<<", "<<num(t.y)<<"), "<<num(t.size)<<" * Screen.height / VIEW_H, "
        <<(t.centred?"Color(0.7, 0.72, 0.76), Anchor.UpperCenter":"Color.white")<<");\n";
    scene<<"}\n";
    scene.close();if(!scene)throw std::runtime_error("Cannot write Tide variants scene");
}
void preview_variants(const std::vector<Clip>& clips,const std::vector<std::string>& labels,const VariantSheet& options) {
    if(clips.empty()||clips.size()!=labels.size())throw std::runtime_error("Variant sheet needs one label per clip");
    for(const auto& c:clips)require_soma(c);
    auto root=fs::temp_directory_path()/"tide-motion-preview";
    size_t key=std::hash<std::string>{}(fs::absolute(options.png.empty()?options.gif:options.png).generic_string());
    root/="variants-"+std::to_string(key);fs::create_directories(root);
    for(const auto& entry:fs::directory_iterator(root))if(entry.is_regular_file()&&entry.path().extension()==".glb")fs::remove(entry.path());
    size_t longest=0;
    for(size_t i=0;i<clips.size();++i) {save_glb(clips[i],root/("v"+std::to_string(i)+".glb"));longest=std::max(longest,clips[i].frames());}
    auto even=[](double v){return std::max(64,static_cast<int>(std::lround(v/2))*2);};
    if(!options.png.empty()) {
        // About 960 pixels a column, at most 4096 either way.
        float aspect=1;size_t columns=1;variants_scene(clips,labels,root,false,&aspect,&columns);
        double width=std::min(4096.0,static_cast<double>(columns)*960),height=width/aspect;
        if(height>4096) {height=4096;width=height*aspect;}
        run_tide(options.tide,root,even(width),even(height),{},options.png,3,0);
    }
    if(!options.gif.empty()) {
        float aspect=1;size_t columns=1;variants_scene(clips,labels,root,true,&aspect,&columns);
        double width=std::min(1920.0,std::max(640.0,static_cast<double>(columns)*360)),height=width/aspect;
        if(height>1080) {height=1080;width=height*aspect;}
        run_tide(options.tide,root,even(width),even(height),options.gif,{},0,longest);
    }
}
}
