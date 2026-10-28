#include "motion.hpp"
#include "author.hpp"
#include <kimodo/kimodo.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#ifdef MOTION_VULKAN
#include <ggml-vulkan.h>
#endif

namespace {
using namespace motion;
#ifdef MOTION_VULKAN
// Empty when Kimodo can use Vulkan; otherwise why not. Asks GGML, as Kimodo does,
// whose initialization throws for an unusable driver. On Windows the loader is
// delay-loaded, so it must be found before any Vulkan function is called.
std::string vulkan_unavailable() {
#ifdef _WIN32
    if(!LoadLibraryW(L"vulkan-1.dll"))return "no Vulkan loader";
#endif
    try { if(ggml_backend_vk_get_device_count()>0)return {}; }
    catch(const std::exception& e) {return e.what();}
    return "no supported Vulkan device";
}
#endif
// Sets an environment variable for Kimodo, which reads them when it loads; empty removes it.
void set_env(const char* name,const std::string& value) {
#ifdef _WIN32
    _putenv_s(name,value.c_str());
#else
    if(value.empty())unsetenv(name);else setenv(name,value.c_str(),1);
#endif
}
std::optional<std::string> get_env(const char* name) { const char* v=std::getenv(name);if(!v)return std::nullopt;return std::string(v); }
void use_cpu(const std::string& reason) {
    std::cerr<<reason<<"; generating on the CPU.\n";
    set_env("KIMODO_BACKEND","cpu");
}
// Whether Kimodo will try the GPU. It takes the first Vulkan device unless
// KIMODO_BACKEND=cpu, and fails rather than falling back when Vulkan is present
// but unusable, so check first; a GPU that still fails is retried on the CPU.
bool kimodo_on_gpu() {
#ifdef MOTION_VULKAN
    const char* chosen=std::getenv("KIMODO_BACKEND");
    if(chosen&&std::string_view(chosen)=="cpu") {std::cerr<<"Generating on the CPU (KIMODO_BACKEND=cpu).\n";return false;}
    std::string reason=vulkan_unavailable();
    if(reason.empty())return true;
    use_cpu("Vulkan unavailable ("+reason+")");
#endif
    return false;
}
// Usage of one command; the global text lists every command and option.
const char* command_help(std::string_view command) {
    if(command=="generate")return R"(tide-motion generate "PROMPT" -o CLIP.glb [options]

Generates a SOMA-30 clip from text with Kimodo (30 fps).

      --model v1.1|v1.2      Smooth-root SOMA-RP checkpoint (default v1.1)
  -o, --output FILE.glb       New output (a FILE.glb.json recipe is written beside it)
  --duration SECONDS          Length of the prompt before it, 0.2..10 (default 3)
  --then "PROMPT"             Continue with another prompt (up to 16 in total); a
                              --duration after it sets that prompt's length
  --transition FRAMES         Frames blended between prompts, 1..60 (default 5)
  --start-pose FILE.glb[@F]   Begin in frame F of a SOMA clip (default: its last
                              frame, so the output continues from it)
  --end-pose FILE.glb[@F]     End in frame F of a SOMA clip (default: its first
                              frame, so the output leads into it)
  --keyframe FRAME FILE.glb[@F]
                              Reach frame F (default first) at output FRAME
  --effector PARTS FRAME FILE.glb[@F]
                              Reach only the hands/feet of frame F (default first)
                              at output FRAME; PARTS is a comma list of left-hand,
                              right-hand, left-foot and right-foot
  --path X,Z[,YAW]@FRAME      Put the root over ground point X,Z (metres) at output
                              FRAME, optionally facing YAW degrees (0 faces +Z,
                              90 faces +X); repeat for waypoints
  --placement auto|world|free
                              Ground placement of poses after frame 0 (see below)
  --no-correction             Keep the model's poses as generated instead of making
                              full-body poses exact
  --correction-frames N       Frames each side the correction fades over, 0..60
                              (default 10; 0 sets only the constrained frame)
  --in-place                  Remove horizontal root travel, keeping height, facing
                              and sway (not with --path)
  --seed N                    Random seed, 0..4294967295 (default 0)
  --variants N                Make N clips, 1..64, from seeds --seed, --seed + 1...
                              (default 1): CLIP.seedS.glb each, and a contact sheet
                              CLIP.variants.png when Tide is installed
  --variants-gif              Also write CLIP.variants.gif playing them all at once
  --tide PATH                 Tide executable for the sheet (default: tide on PATH)
  --guidance W                How strongly the prompt steers the motion, 0..10
                              (default 2; try 3-4 when the move is missing, as
                              higher exaggerates the body and slides the feet)
  --constraint-guidance W     How strongly poses and paths steer it, 0..10
                              (default 2, the closest poses in measurements)
  --steps N                   Diffusion steps, 1..1000 (default 50)
  --device auto|gpu|cpu       Where Kimodo runs (default auto: the GPU, or the CPU
                              when it is unavailable or fails; gpu never falls back)
  --threads N                 CPU threads, 1..1024 (default: all)
  --models DIR                Model directory (default: TIDE_MOTION_MODELS or the
                              OS cache)

F is a frame of the pose file; negative counts from its end (-1 is the last),
and "first" or "last" also work. Output FRAME is 0-based at 30 fps; negative
counts from the end. Pose files must be SOMA-30 clips (convert other rigs with
retarget --to-soma).

Placement: a start pose (or a path point at frame 0) fixes where the output
begins, in that clip's world. world also pins later poses to their ground
positions in their files; free keeps their body, height and facing but lets the
motion choose where on the ground they happen. auto (default) is world when the
first frame or a path fixes the world, otherwise free, and then moves the whole
output so its last pose lands where it is in its file (so an end pose leads
straight into that clip). Without a start pose the output begins facing +Z.

Kimodo reaches poses closely but not exactly; the model's error at each
constraint is printed and stored in the recipe. Full-body poses are then made
exact, fading the correction out over up to --correction-frames frames either
side (effector targets and path points are left as generated). --in-place is
applied last: frame 0 stays put and later poses move with the removed travel.

Variants: -o kick.glb --seed 4 --variants 3 writes kick.seed4.glb,
kick.seed5.glb and kick.seed6.glb, each with its recipe; generate with --seed S
and no --variants to remake one exactly. kick.variants.png shows each in a
labelled cell: key moments in right profile (top) and facing (bottom).

PROMPTING.md (beside README.md) covers what to write and how to nudge a result.
)";
    if(command=="connect")return R"(tide-motion connect FROM.glb TO.glb -o BRIDGE.glb [options]

Generates a MotionBricks transition from the end of one SOMA clip to the start
of another.

  -o, --output FILE.glb       New output
  --tokens N                  Transition length in tokens, 6..16 (default: model)
  --joined                    Export source, bridge and target as one animation
  --offset X Y Z              Move the target clip before connecting (metres)
  --from-clip NAME            Animation to use from a multi-animation FROM.glb
  --to-clip NAME              Animation to use from a multi-animation TO.glb
  --threads N                 CPU threads, 1..1024 (default: half)
  --models DIR                Model directory
)";
    if(command=="pose-sequence")return R"(tide-motion pose-sequence POSES.json -o CLIP.glb [--models DIR] [--threads N]

Native MotionBricks offline authoring through 1..128 explicit SOMA pose targets.
Poses use schema tide-motion-pose-sequence-1, start and targets. Each pose has
file, optional clip/frame/hold, required position [root X,Y,Z metres] and yaw
(degrees, 0 faces +Z, 90 +X). Each target requires tokens 6..16; it adds
4*tokens-7 intervals at 30 Hz. Start defaults to the last source frame, targets
to the first. Negative frame indices count from the end. Windows sample the last
four context / first four target samples, clamped at clip ends; hold:true repeats
the selected pose. Later contexts use the last four generated samples.
Paths are relative to POSES.json. All source clips must exist before the run
(including in batch). Output is ordinary GLB plus an effective JSON recipe.
No scene understanding, contacts, IK, arbitrary timing or velocity seam guarantee.
See examples/pose-sequence.json and README.md.
)";
    if(command=="cleanup")return R"(tide-motion cleanup CLIP.glb|.fbx --contacts PLAN.json -o CLEAN.glb [--map BONES.json] [--clip NAME]

Optional native foot contact IK; no model is loaded. PLAN explicitly declares
inclusive left-foot/right-foot intervals (at least three frames), protectedFrames,
fadeFrames (0..60, default 3), maxCorrection (metres, (0,0.5], default 0.1).
Each foot anchors to its first interval sample. Intervals exceeding maxCorrection
are skipped in full and reported. Intervals with increased mean drift are restored.
No contact detection or ground projection.
SOMA-30 or explicitly mapped direct thigh/calf/foot chains only.
Translations, root motion and world foot orientation are kept. Authored constraint
frames in an adjacent Tide recipe are protected automatically; use protectedFrames
for clips without that recipe. Do not label kicks/airborne frames as contacts.
Recipes report horizontal mean drift per frame before/after, max anchor residual
(including fades/protected frames), max joint correction and skipped intervals.
)";
    if(command=="mirror")return R"(tide-motion mirror CLIP.glb -o MIRRORED.glb [--clip NAME]

Swaps left and right in a SOMA-30 clip and reflects it across the character's
sagittal plane at its first frame, so a right kick becomes a left kick. The
clip starts in the same place, facing the same way. No model is loaded.

  -o, --output FILE.glb       New output (a FILE.glb.json recipe is written beside it)
  --clip NAME                 Animation to use from a multi-animation CLIP.glb
)";
    if(command=="retarget")return R"(tide-motion retarget SOURCE.glb --target CHARACTER.glb|.fbx --map BONES.json -o CLIP.glb
tide-motion retarget SOURCE.glb|.fbx --to-soma --map BONES.json -o SOMA.glb

Converts animation between SOMA and a mapped humanoid without loading a model.
Characters and source clips may be GLB or FBX (read as Tide imports FBX: Y-up
metres); examples/synty-polygon.json maps Synty POLYGON characters.

  --target CHARACTER.glb|.fbx Character skeleton to receive SOMA motion
  --to-soma                   Convert the source character's animation to SOMA
  --map BONES.json            SOMA-to-character bone names (see examples/)
  --clip NAME                 Animation to use from a multi-animation SOURCE.glb
  -o, --output FILE.glb       New output
)";
    if(command=="batch")return R"(tide-motion batch JOBS.json [--models DIR]

Runs ordered generate, connect, pose-sequence, retarget, cleanup and mirror actions, loading each model
once.
The whole manifest and every output path are checked before anything runs.
Paths are relative to the manifest. README.md lists every action field.
)";
    if(command=="inspect")return R"(tide-motion inspect FILE.glb|FILE.fbx [--clip NAME | --skeleton]

Prints a clip's rig, joints, frames and duration as JSON.

  --clip NAME                 Animation to inspect from a multi-animation GLB
  --skeleton                  List bone names, parents, rest transforms and scales
)";
    if(command=="preview")return R"(tide-motion preview CLIP.glb|CLIP.fbx [options]

Plays a clip in a temporary Tide project, lit, facing the camera on the left and
in profile on the right, following its root over a tiled floor.

  --mannequin                 On a built-in SOMA-30 mannequin (the default for
                              SOMA clips; left limbs blue, right limbs orange)
  --character MODEL.glb|.fbx  On your skinned character instead
  --map BONES.json            Retarget a SOMA clip onto --character first (as
                              retarget --target does); leave it out for a clip
                              already on the character's rig
  --texture IMAGE.png|.jpg    The character's texture (default: plain clay)
  --skeleton                  Draw bones as lines (the default for other rigs
                              when no --character is given)
  --tide PATH                 Tide executable (default: tide on PATH)
  --capture IMAGE.png         Save the last frame as a PNG
  --gif OUT.gif               Record the clip once from its first frame as a GIF
                              (needs Tide 0.5.0-nightly.45 or later)
  --frames N                  Stop after N frames at 60 fps (with --gif the
                              default covers the clip exactly once)
  --clip NAME                 Animation to preview from a multi-animation file

With TIDE_HIDDEN set, give --capture, --gif or --frames.
)";
    if(command=="models")return R"(tide-motion models install [all|kimodo|kimodo-v1.2|bricks] --accept-licenses [--models DIR]
tide-motion models check [all|kimodo|kimodo-v1.2|bricks] [--models DIR]

Downloads (resumably) and verifies the pinned model files by SHA-256.
kimodo-v1.2 downloads pinned official safetensors/statistics and converts them
to verified GGUF natively, then adds the shared text files. No Python is needed.
all retains v1.1 and MotionBricks.

  --accept-licenses           Accept the model licences in licenses/ (install)
  --models DIR                Model directory (default: TIDE_MOTION_MODELS or the
                              OS cache)
)";
    return nullptr;
}
void help(std::string_view command={}) {
    if(const char* text=command_help(command)) {std::cout<<text;return;}
    std::cout<<R"(Tide Motion - native offline animation authoring

  tide-motion models install [all|kimodo|kimodo-v1.2|bricks] --accept-licenses [--models DIR]
  tide-motion models check [all|kimodo|kimodo-v1.2|bricks] [--models DIR]
  tide-motion generate "PROMPT" -o CLIP.glb [--duration 3] [--seed 0] [--steps 50]
      [--variants N [--variants-gif]] [--guidance 2] [--constraint-guidance 2]
      [--then "PROMPT" [--duration S]]... [--transition 5]
      [--start-pose FILE.glb[@F]] [--end-pose FILE.glb[@F]]
      [--keyframe FRAME FILE.glb[@F]]... [--effector PARTS FRAME FILE.glb[@F]]...
      [--path X,Z[,YAW]@FRAME]... [--correction-frames 10] [--in-place]
      [--device auto|gpu|cpu] [--threads N]
  tide-motion connect FROM.glb TO.glb -o BRIDGE.glb [--tokens 6..16] [--joined]
      [--offset X Y Z] [--from-clip NAME] [--to-clip NAME] [--threads N]
  tide-motion pose-sequence POSES.json -o CLIP.glb [--threads N]
  tide-motion cleanup CLIP.glb --contacts PLAN.json -o CLEAN.glb [--map BONES.json]
  tide-motion mirror CLIP.glb -o MIRRORED.glb [--clip NAME]
  tide-motion retarget SOURCE.glb --target CHARACTER.glb|.fbx --map BONES.json -o CLIP.glb
  tide-motion retarget SOURCE.glb|.fbx --to-soma --map BONES.json -o SOMA.glb [--clip NAME]
  tide-motion batch JOBS.json [--models DIR]
  tide-motion inspect FILE.glb|FILE.fbx [--clip NAME | --skeleton]
  tide-motion preview CLIP.glb [--mannequin | --skeleton | --character MODEL.glb|.fbx
      [--map BONES.json] [--texture IMAGE.png]] [--tide PATH] [--capture IMAGE.png]
      [--gif OUT.gif] [--frames N] [--clip NAME]
  tide-motion help [COMMAND]       one command's options (also COMMAND --help)
  tide-motion --version

Models are downloaded once and SHA-256 verified. Generation and transitions run
locally without Python, PyTorch, Conda or a server. Models use SOMA-30; retarget
converts mapped humanoids to/from SOMA without loading a model. GLB output
contains animation and a skeleton, without a skinned character mesh.

--models DIR overrides TIDE_MOTION_MODELS and the OS model cache.
generate --then chains prompts; --start-pose, --end-pose, --keyframe, --effector
and --path hold the motion to SOMA poses and ground points; --variants N makes N
seeds and a contact sheet to pick from. mirror turns a right kick into a left one.
connect preserves target placement; use --offset X Y Z to place the target clip.
--from-clip NAME / --to-clip NAME select clips from multi-animation GLBs.
--joined exports source, generated bridge and target as one animation.
preview uses an installed Tide separately (--tide selects it) to play a clip on a
built-in mannequin, your GLB/FBX character or as lines; --gif records it once.
retarget, inspect and preview read FBX characters and clips as Tide imports them.
batch runs ordered actions and keeps each inference model loaded until completion.
Existing outputs are refused. Use a new output path.

Licenses: Apache-2.0 source; NVIDIA Open Model terms for motion weights;
Meta Llama 3 terms for Kimodo's text encoder. See licenses/ and README.md.
)";
}
double number(const std::string& text,double low,double high,const char* label) {
    size_t used=0;double n=std::stod(text,&used);
    if(used!=text.size()||!std::isfinite(n)||n<low||n>high)throw std::runtime_error(std::string(label)+" is out of range");
    return n;
}
double integer(const std::string& text,double low,double high,const char* label) {
    double value=number(text,low,high,label);
    if(value!=std::floor(value))throw std::runtime_error(std::string(label)+" must be an integer");
    return value;
}
// Preserve authored frames through model-free edits and mapped retargeting.
nlohmann::json source_metadata(const fs::path& source) {
    auto path=fs::path(source.string()+".json");
    if(!fs::exists(path))return nlohmann::json::object();
    if(fs::file_size(path)>1024*1024)throw std::runtime_error("Source recipe exceeds 1 MiB");
    std::ifstream input(path);nlohmann::json recipe;input>>recipe;
    if(!recipe.is_object())throw std::runtime_error("Source recipe must be an object");
    return recipe;
}
std::vector<size_t> protected_recipe_frames(const nlohmann::json& recipe,size_t frames) {
    std::vector<size_t> out;
    auto add=[&](const nlohmann::json& value) {
        if(!value.is_number_integer()||value.get<double>()<0||value.get<double>()>=static_cast<double>(frames))throw std::runtime_error("Source constraint frame outside clip");
        out.push_back(value.get<size_t>());
    };
    if(recipe.contains("constraints")) {
        if(!recipe["constraints"].is_array())throw std::runtime_error("Invalid source constraints");
        for(const auto& constraint:recipe["constraints"])add(constraint.at("frame"));
    }
    if(recipe.contains("protectedFrames")) {
        if(!recipe["protectedFrames"].is_array())throw std::runtime_error("Invalid source protectedFrames");
        for(const auto& frame:recipe["protectedFrames"])add(frame);
    }
    if(recipe.value("command",std::string{})=="pose-sequence") {
        add(0);
        for(const auto& target:recipe.at("authoring").at("targets"))add(target.at("endFrame"));
    }
    std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return out;
}
struct RunContext {
    std::unique_ptr<kimodo::model> kimodo;
    // strict: an explicit --device, which never falls back. gpu_failed: auto fell back
    // in this run, so later auto actions stay on the CPU.
    bool kimodo_gpu=false,kimodo_strict=false,gpu_failed=false,env_saved=false;
    std::string kimodo_version="v1.1";std::set<std::string> checked_versions;
    std::string kimodo_device;unsigned kimodo_threads=0;
    std::optional<std::string> backend_env,threads_env;
    std::unique_ptr<BricksModel> bricks;unsigned bricks_threads=0;
};
struct Guidance { float text=2,constraint=2; };
void load_kimodo(const fs::path& models,RunContext* context) {
    std::cerr<<"Loading Kimodo SOMA and text encoder (native GGML)...\n";
    std::string failure;
    try {
        auto model=kimodo::model::load((models/(context->kimodo_version=="v1.2"?"kimodo-v1.2.gguf":"kimodo.gguf")).string(),(models/"text/encoder.gguf").string());
        if(model) {context->kimodo=std::move(*model);return;}
        failure=model.error();
    } catch(const std::exception& e) {failure=e.what();}
    if(!context->kimodo_gpu||context->kimodo_strict)throw std::runtime_error(failure);
    context->kimodo_gpu=false;context->gpu_failed=true;use_cpu("The GPU could not load Kimodo ("+failure+")");
    load_kimodo(models,context);
}
// --device and --threads apply when Kimodo loads; a batch action asking for others reloads it.
void ensure_kimodo(const fs::path& models,RunContext* context,const std::string& device,unsigned threads,const std::string& version) {
    if(context->kimodo&&context->kimodo_device==device&&context->kimodo_threads==threads&&context->kimodo_version==version)return;
    if(!context->env_saved) {context->backend_env=get_env("KIMODO_BACKEND");context->threads_env=get_env("KIMODO_THREADS");context->env_saved=true;}
    if(context->kimodo) {std::cerr<<"Reloading Kimodo for another model, --device or --threads.\n";context->kimodo.reset();}
    set_env("KIMODO_THREADS",threads?std::to_string(threads):context->threads_env.value_or(""));
    context->kimodo_strict=device!="auto";
    if(device=="cpu") {set_env("KIMODO_BACKEND","cpu");context->kimodo_gpu=false;std::cerr<<"Generating on the CPU (--device cpu).\n";}
    else if(device=="gpu") {
#ifdef MOTION_VULKAN
        std::string reason=vulkan_unavailable();
        if(!reason.empty())throw std::runtime_error("--device gpu: Vulkan is unavailable ("+reason+")");
        set_env("KIMODO_BACKEND","vulkan");context->kimodo_gpu=true;
#else
        throw std::runtime_error("--device gpu: this build has no GPU support; use --device auto or cpu");
#endif
    } else if(context->gpu_failed) {set_env("KIMODO_BACKEND","cpu");context->kimodo_gpu=false;std::cerr<<"Generating on the CPU (the GPU failed earlier in this run).\n";}
    else {set_env("KIMODO_BACKEND",context->backend_env.value_or(""));context->kimodo_gpu=kimodo_on_gpu();}
    // Models are verified once, after the device is settled (a refused --device gpu skips it).
    if(!context->checked_versions.contains(version)) {check_models(models,version=="v1.2"?"kimodo-v1.2":"kimodo");context->checked_versions.insert(version);}
    context->kimodo_version=version;
    load_kimodo(models,context);
    context->kimodo_device=device;context->kimodo_threads=threads;
}
// One unconditioned prompt keeps the original sampler; sequences and constraints use
// the patched conditioned one, which matches upstream _multiprompt.
kimodo::motion_data generate_motion(const fs::path& models,RunContext* context,const std::vector<kimodo::prompt_segment>& segments,unsigned transition,unsigned steps,uint64_t seed,const Guidance& guidance,const kimodo::motion_condition* condition) {
    std::string failure;
    try {
        auto result=segments.size()==1&&!condition?context->kimodo->generate_text(segments[0].prompt,segments[0].frames,steps,seed,guidance.text,guidance.constraint)
            :context->kimodo->generate_text_conditioned(segments,transition,steps,seed,guidance.text,guidance.constraint,condition?*condition:kimodo::motion_condition{});
        if(result)return std::move(*result);
        failure=result.error();
    } catch(const std::exception& e) {failure=e.what();}
    if(!context->kimodo_gpu||context->kimodo_strict)throw std::runtime_error(failure);
    context->kimodo.reset();context->kimodo_gpu=false;context->gpu_failed=true;use_cpu("GPU generation failed ("+failure+")");
    load_kimodo(models,context);
    return generate_motion(models,context,segments,transition,steps,seed,guidance,condition);
}
// FILE.glb[@F]: F is an integer (negative from the end), first or last.
struct PoseRef { std::string file; std::optional<long> frame; };
PoseRef pose_ref(const std::string& text) {
    PoseRef ref{text,std::nullopt};
    auto at=text.rfind('@');
    if(at!=std::string::npos&&at>0) {
        auto suffix=text.substr(at+1);
        if(suffix=="first")ref.frame=0;
        else if(suffix=="last")ref.frame=-1;
        else if(!suffix.empty()&&suffix.find_first_not_of("-0123456789")==std::string::npos&&suffix.find('-',1)==std::string::npos&&suffix!="-")
            ref.frame=static_cast<long>(integer(suffix,-18000,17999,"pose frame"));
        if(ref.frame)ref.file=text.substr(0,at);
    }
    if(ref.file.empty())throw std::runtime_error("Pose needs a SOMA .glb file: "+text);
    return ref;
}
long frame_index(const std::string& text,const char* label) { return static_cast<long>(integer(text,-18000,17999,label)); }
PathConstraint path_point(const std::string& text,long* frame) {
    auto at=text.rfind('@');
    if(at==std::string::npos)throw std::runtime_error("Use --path X,Z[,YAW]@FRAME");
    std::vector<std::string> parts;size_t start=0;auto values=text.substr(0,at);
    while(true) {auto comma=values.find(',',start);parts.push_back(values.substr(start,comma-start));if(comma==std::string::npos)break;start=comma+1;}
    if(parts.size()<2||parts.size()>3)throw std::runtime_error("Use --path X,Z[,YAW]@FRAME");
    PathConstraint p;
    p.x=static_cast<float>(number(parts[0],-10000,10000,"path X"));p.z=static_cast<float>(number(parts[1],-10000,10000,"path Z"));
    if(parts.size()==3) {p.has_heading=true;p.heading=static_cast<float>(number(parts[2],-3600,3600,"path yaw"));}
    *frame=frame_index(text.substr(at+1),"path frame");
    return p;
}
int main_impl(const std::vector<std::string>& args,RunContext* context=nullptr);
int batch(const fs::path& manifest,const fs::path& models,RunContext* context) {
    if(fs::file_size(manifest)>1024*1024)throw std::runtime_error("Batch manifest exceeds 1 MiB");
    std::ifstream stream(manifest);nlohmann::json doc;stream>>doc;
    if(!doc.is_object()||doc.size()!=1||!doc.contains("actions")||!doc["actions"].is_array()||doc["actions"].empty()||doc["actions"].size()>1000)
        throw std::runtime_error("Batch file must contain an actions array with 1..1000 actions");
    const auto base=fs::absolute(manifest).parent_path();
    std::vector<std::vector<std::string>> commands;
    std::set<fs::path> outputs;
    auto path=[&](const nlohmann::json& value) {
        auto p=utf8path(value.get<std::string>());if(p.empty())throw std::runtime_error("Empty batch path");
        return fs::absolute(p.is_absolute()?p:base/p).lexically_normal();
    };
    for(size_t index=0;index<doc["actions"].size();++index)try {
        const auto& action=doc["actions"][index];
        if(!action.is_object())throw std::runtime_error("Action must be an object");
        auto type=action.at("action").get<std::string>();
        std::set<std::string> keys{"action","output"};
        std::vector<std::string> argv{"tide-motion",type};
        unsigned variants=1;bool sheet_gif=false;
        if(type=="generate") {
            keys.insert({"prompt","prompts","duration","seed","steps","transition","startPose","endPose","keyframes","effectors","path","placement","correct",
                "variants","variantsGif","tide","guidance","constraintGuidance","device","threads","correctionFrames","inPlace","model"});
            if(action.contains("placement")) {
                auto value=action["placement"].get<std::string>();
                if(value!="auto"&&value!="world"&&value!="free")throw std::runtime_error("placement is auto, world or free");
                argv.insert(argv.end(),{"--placement",value});
            }
            if(action.contains("correct")&&!action["correct"].get<bool>())argv.push_back("--no-correction");
            auto numeric=[](const nlohmann::json& value,const char* label) {
                if(!value.is_number())throw std::runtime_error(std::string(label)+" must be numeric");
                return value.dump();
            };
            if(action.contains("variants")) {auto text=numeric(action["variants"],"variants");variants=static_cast<unsigned>(integer(text,1,64,"variants"));argv.insert(argv.end(),{"--variants",text});}
            if(action.contains("variantsGif")&&action["variantsGif"].get<bool>()) {sheet_gif=true;argv.push_back("--variants-gif");}
            if(action.contains("inPlace")&&action["inPlace"].get<bool>())argv.push_back("--in-place");
            for(auto pair:{std::pair{"guidance","--guidance"},std::pair{"constraintGuidance","--constraint-guidance"}})if(action.contains(pair.first)) {
                auto text=numeric(action[pair.first],pair.first);number(text,0,10,pair.first);argv.insert(argv.end(),{pair.second,text});
            }
            if(action.contains("correctionFrames")) {auto text=numeric(action["correctionFrames"],"correctionFrames");integer(text,0,60,"correctionFrames");argv.insert(argv.end(),{"--correction-frames",text});}
            if(action.contains("model")) {auto value=action["model"].get<std::string>();if(value!="v1.1"&&value!="v1.2")throw std::runtime_error("model is v1.1 or v1.2");argv.insert(argv.end(),{"--model",value});}
            if(action.contains("device")) {auto value=action["device"].get<std::string>();if(value!="auto"&&value!="gpu"&&value!="cpu")throw std::runtime_error("device is auto, gpu or cpu");argv.insert(argv.end(),{"--device",value});}
            if(action.contains("tide")) {auto value=action["tide"].get<std::string>();if(value.empty())throw std::runtime_error("tide must not be empty");argv.insert(argv.end(),{"--tide",value});}
            if(action.contains("prompts")) {
                const auto& prompts=action["prompts"];
                if(action.contains("prompt")||action.contains("duration"))throw std::runtime_error("Use prompts or prompt/duration, not both");
                if(!prompts.is_array()||prompts.empty()||prompts.size()>16)throw std::runtime_error("prompts must hold 1..16 objects");
                for(size_t i=0;i<prompts.size();++i) {
                    const auto& item=prompts[i];
                    if(!item.is_object())throw std::runtime_error("Each prompts entry must be an object");
                    for(const auto& field:item.items())if(field.key()!="prompt"&&field.key()!="duration")throw std::runtime_error("Unknown prompts field: "+field.key());
                    if(i)argv.push_back("--then");
                    argv.push_back(item.at("prompt").get<std::string>());
                    if(item.contains("duration")) {auto text=numeric(item["duration"],"duration");number(text,0.2,10,"duration");argv.insert(argv.end(),{"--duration",text});}
                }
            } else argv.push_back(action.at("prompt").get<std::string>());
            // Pose files may be outputs of earlier actions, so only their paths are checked here.
            auto pose=[&](const nlohmann::json& value,const char* frame_key) {
                if(value.is_string())return path(value).string();
                if(!value.is_object())throw std::runtime_error("A pose is a file path or an object with file");
                auto text=path(value.at("file")).string();
                if(value.contains(frame_key))text+="@"+std::to_string(frame_index(numeric(value[frame_key],frame_key),frame_key));
                return text;
            };
            auto fields=[](const nlohmann::json& item,std::set<std::string> allowed) {
                if(!item.is_object())throw std::runtime_error("Constraint entries must be objects");
                for(const auto& field:item.items())if(!allowed.contains(field.key()))throw std::runtime_error("Unknown constraint field: "+field.key());
            };
            if(action.contains("transition")) {auto text=numeric(action["transition"],"transition");integer(text,1,60,"transition");argv.insert(argv.end(),{"--transition",text});}
            for(auto pair:{std::pair{"startPose","--start-pose"},std::pair{"endPose","--end-pose"}})if(action.contains(pair.first)) {
                if(action[pair.first].is_object())fields(action[pair.first],{"file","frame"});
                argv.insert(argv.end(),{pair.second,pose(action[pair.first],"frame")});
            }
            for(const char* key:{"keyframes","effectors","path"})if(action.contains(key)&&(!action[key].is_array()||action[key].empty()))throw std::runtime_error(std::string(key)+" must be a nonempty array");
            if(action.contains("keyframes"))for(const auto& item:action["keyframes"]) {
                fields(item,{"frame","file","sourceFrame"});
                argv.insert(argv.end(),{"--keyframe",std::to_string(frame_index(numeric(item.at("frame"),"frame"),"frame")),pose(item,"sourceFrame")});
            }
            if(action.contains("effectors"))for(const auto& item:action["effectors"]) {
                fields(item,{"effectors","frame","file","sourceFrame"});
                std::string parts;
                if(item.at("effectors").is_array()) {for(const auto& part:item["effectors"]) {if(!parts.empty())parts+=',';parts+=part.get<std::string>();}}
                else parts=item["effectors"].get<std::string>();
                (void)parse_effectors(parts);
                argv.insert(argv.end(),{"--effector",parts,std::to_string(frame_index(numeric(item.at("frame"),"frame"),"frame")),pose(item,"sourceFrame")});
            }
            if(action.contains("path"))for(const auto& item:action["path"]) {
                fields(item,{"frame","x","z","yaw"});
                std::string text=numeric(item.at("x"),"x")+","+numeric(item.at("z"),"z");
                if(item.contains("yaw"))text+=","+numeric(item["yaw"],"yaw");
                text+="@"+numeric(item.at("frame"),"frame");
                long frame=0;(void)path_point(text,&frame);
                argv.insert(argv.end(),{"--path",text});
            }
        }
        else if(type=="connect") { argv.push_back(path(action.at("from")).string());argv.push_back(path(action.at("to")).string());keys.insert({"from","to","tokens","joined","offset","fromClip","toClip","threads"}); }
        else if(type=="pose-sequence") {
            auto poses=path(action.at("recipe"));(void)load_pose_sequence(poses);
            argv.push_back(poses.string());keys.insert({"recipe","threads"});
        }
        else if(type=="cleanup") {
            keys.insert({"source","clip","contacts","map"});argv.push_back(path(action.at("source")).string());
            auto plan=path(action.at("contacts"));load_cleanup_plan(plan);
            argv.insert(argv.end(),{"--contacts",plan.string()});
            if(action.contains("map")) {auto m=path(action["map"]);load_mapping(m);argv.insert(argv.end(),{"--map",m.string()});}
            if(action.contains("clip"))argv.insert(argv.end(),{"--clip",action["clip"].get<std::string>()});
        }
        else if(type=="mirror") {
            argv.push_back(path(action.at("source")).string());keys.insert({"source","clip"});
            if(action.contains("clip"))argv.insert(argv.end(),{"--clip",action["clip"].get<std::string>()});
        }
        else if(type=="retarget") {
            argv.push_back(path(action.at("source")).string());
            auto mapping=path(action.at("map"));(void)load_mapping(mapping);
            argv.insert(argv.end(),{"--map",mapping.string()});
            bool to_soma=action.contains("toSoma")&&action["toSoma"].get<bool>();
            if(to_soma) {
                if(action.contains("target"))throw std::runtime_error("toSoma does not take a target");
                argv.push_back("--to-soma");
            } else argv.insert(argv.end(),{"--target",path(action.at("target")).string()});
            if(action.contains("clip"))argv.insert(argv.end(),{"--clip",action["clip"].get<std::string>()});
            keys.insert({"source","target","map","toSoma","clip"});
        }
        else throw std::runtime_error("Action must be generate, connect, pose-sequence, retarget, cleanup or mirror");
        for(const auto& item:action.items())if(!keys.contains(item.key()))throw std::runtime_error("Unknown action field: "+item.key());
        auto output=path(action.at("output"));
        if(output.extension()!=".glb")throw std::runtime_error("Output must be a .glb path: "+output.string());
        // Variants write CLIP.seedS.glb for each seed and a contact sheet beside them.
        std::vector<fs::path> written;
        uint64_t first_seed=0;
        if(action.contains("seed")&&action["seed"].is_number())first_seed=static_cast<uint64_t>(integer(action["seed"].dump(),0,4294967295.0,"seed"));
        if(variants>1) {
            for(unsigned v=0;v<variants;++v)written.push_back(variant_path(output,first_seed+v));
            written.push_back(variants_path(output,".png"));if(sheet_gif)written.push_back(variants_path(output,".gif"));
        } else written.push_back(output);
        for(const auto& file:written)if(fs::exists(file)||fs::exists(file.string()+".part")||fs::exists(file.string()+".json")||!outputs.insert(file).second)throw std::runtime_error("Output must be a new, distinct path: "+file.string());
        argv.insert(argv.end(),{"-o",output.string()});
        if(type=="generate"||type=="connect"||type=="pose-sequence")argv.insert(argv.end(),{"--models",models.string()});
        for(const char* key:{"duration","seed","steps","tokens","threads"})if(action.contains(key)) {
            if(!action[key].is_number())throw std::runtime_error(std::string(key)+" must be numeric");
            auto text=action[key].dump();
            if(std::string(key)=="duration")number(text,0.2,10,key);
            else if(std::string(key)=="tokens")integer(text,6,16,key);
            else if(std::string(key)=="steps")integer(text,1,1000,key);
            else if(std::string(key)=="threads")integer(text,1,1024,key);
            else integer(text,0,4294967295.0,key);
            argv.insert(argv.end(),{"--"+std::string(key),text});
        }
        if(action.contains("joined")&&action["joined"].get<bool>())argv.push_back("--joined");
        if(action.contains("offset")) {
            const auto& offset=action["offset"];
            if(!offset.is_array()||offset.size()!=3)throw std::runtime_error("offset requires three numbers");
            argv.push_back("--offset");
            for(const auto& value:offset) { if(!value.is_number())throw std::runtime_error("offset must be numeric");auto text=value.dump();number(text,-10000,10000,"offset");argv.push_back(text); }
        }
        for(auto pair:{std::pair{"fromClip","--from-clip"},std::pair{"toClip","--to-clip"}})if(action.contains(pair.first))argv.insert(argv.end(),{pair.second,action[pair.first].get<std::string>()});
        commands.push_back(std::move(argv));
    } catch(const std::exception& e) {throw std::runtime_error("Batch action "+std::to_string(index+1)+": "+e.what());}
    // Keep the 8 GB text model resident for repeated prompts rather than reloading
    // every layer per action. Explicit operator environment settings still win.
    if(!std::getenv("KIMODO_TEXT_LAYER_CHUNK")) {
#ifdef _WIN32
        _putenv_s("KIMODO_TEXT_LAYER_CHUNK","32");
#else
        setenv("KIMODO_TEXT_LAYER_CHUNK","32",0);
#endif
    }
    for(size_t i=0;i<commands.size();++i) {
        std::cerr<<"Batch action "<<i+1<<"/"<<commands.size()<<"\n";
        try {int code=main_impl(commands[i],context);if(code)return code;}
        catch(const std::exception& e){throw std::runtime_error("Batch action "+std::to_string(i+1)+" failed: "+e.what()+"; completed outputs have been kept");}
    }
    return 0;
}
int main_impl(const std::vector<std::string>& args,RunContext* context) {
    RunContext owned;if(!context)context=&owned;
    if(args.size()<2||args[1]=="--help"||args[1]=="-h") { help();return 0; }
    auto unknown=[](const std::string& name){return std::runtime_error("Unknown command: "+name+"; tide-motion --help lists the commands");};
    if(args[1]=="help") { if(args.size()>2&&!command_help(args[2]))throw unknown(args[2]);help(args.size()>2?args[2]:"");return 0; }
    if(args[1]=="--version") { std::cout<<"tide-motion "<<MOTION_VERSION<<"\n";return 0; }
    if(!command_help(args[1]))throw unknown(args[1]);
    std::string command=args[1],clip_name,from_name,to_name,tide="tide";
    for(size_t i=2;i<args.size();++i)if(args[i]=="--help"||args[i]=="-h") {help(command);return 0;}
    fs::path models,output,capture,gif,target_file,map_file,character_file,texture_file,contacts_file;bool mannequin=false;
    if(command=="generate"||command=="connect"||command=="pose-sequence"||command=="models"||command=="batch")models=model_directory();
    double duration=3;uint64_t seed=0;unsigned steps=50,transition=5;int frames=0,tokens=0;
    bool accept=false,joined=false,skeleton=false,to_soma=false,correct=true;Vec3 offset{};
    Placement placement=Placement::automatic;std::string placement_name="auto";
    unsigned variants=1,threads=0;int correction_frames=10;bool variants_gif=false,in_place=false;
    Guidance guidance;std::string device="auto",model_version="v1.1";
    std::vector<std::string> pos;
    std::vector<std::string> flags;
    // Prompt durations; index 0 is the positional prompt, later ones come from --then.
    std::vector<std::pair<std::string,double>> segments{{"",3.0}};
    struct PoseSpec { std::string kind; long frame=0; std::string ref; unsigned effectors=0; };
    std::vector<PoseSpec> pose_specs;
    std::vector<std::pair<PathConstraint,long>> path_specs;
    for(size_t i=2;i<args.size();++i) {
        auto value=[&]() { if(++i>=args.size()||args[i].empty())throw std::runtime_error("Missing value for "+args[i-1]);return args[i]; };
        const auto& a=args[i];
        if(a.starts_with('-'))flags.push_back(a);
        if(a=="--models")models=utf8path(value());
        else if(a=="-o"||a=="--output")output=utf8path(value());
        else if(a=="--duration") {duration=number(value(),0.2,10,"duration");segments.back().second=duration;}
        else if(a=="--then")segments.push_back({value(),3.0});
        else if(a=="--transition")transition=static_cast<unsigned>(integer(value(),1,60,"transition"));
        else if(a=="--start-pose")pose_specs.push_back({"start-pose",0,value()});
        else if(a=="--end-pose")pose_specs.push_back({"end-pose",-1,value()});
        else if(a=="--keyframe") {long f=frame_index(value(),"keyframe frame");pose_specs.push_back({"keyframe",f,value()});}
        else if(a=="--effector") {unsigned bits=parse_effectors(value());long f=frame_index(value(),"effector frame");pose_specs.push_back({"effector",f,value(),bits});}
        else if(a=="--path") {long f=0;auto p=path_point(value(),&f);path_specs.push_back({p,f});}
        else if(a=="--gif")gif=utf8path(value());
        else if(a=="--placement") {
            placement_name=value();
            if(placement_name=="auto")placement=Placement::automatic;
            else if(placement_name=="world")placement=Placement::world;
            else if(placement_name=="free")placement=Placement::free;
            else throw std::runtime_error("--placement is auto, world or free");
        }
        else if(a=="--no-correction")correct=false;
        else if(a=="--correction-frames")correction_frames=static_cast<int>(integer(value(),0,60,"correction frames"));
        else if(a=="--in-place")in_place=true;
        else if(a=="--variants")variants=static_cast<unsigned>(integer(value(),1,64,"variants"));
        else if(a=="--variants-gif")variants_gif=true;
        else if(a=="--guidance")guidance.text=static_cast<float>(number(value(),0,10,"guidance"));
        else if(a=="--constraint-guidance")guidance.constraint=static_cast<float>(number(value(),0,10,"constraint guidance"));
        else if(a=="--model") {model_version=value();if(model_version!="v1.1"&&model_version!="v1.2")throw std::runtime_error("--model is v1.1 or v1.2 (smooth-root SOMA-RP)");}
        else if(a=="--device") {device=value();if(device!="auto"&&device!="gpu"&&device!="cpu")throw std::runtime_error("--device is auto, gpu or cpu");}
        else if(a=="--threads")threads=static_cast<unsigned>(integer(value(),1,1024,"threads"));
        else if(a=="--steps")steps=static_cast<unsigned>(integer(value(),1,1000,"steps"));
        else if(a=="--seed")seed=static_cast<uint64_t>(integer(value(),0,4294967295.0,"seed"));
        else if(a=="--frames")frames=static_cast<int>(integer(value(),1,18000,"frames"));
        else if(a=="--tokens")tokens=static_cast<int>(integer(value(),6,16,"tokens"));
        else if(a=="--offset") {offset.x=static_cast<float>(number(value(),-10000,10000,"offset"));offset.y=static_cast<float>(number(value(),-10000,10000,"offset"));offset.z=static_cast<float>(number(value(),-10000,10000,"offset"));}
        else if(a=="--contacts")contacts_file=utf8path(value());
        else if(a=="--clip")clip_name=value();
        else if(a=="--from-clip")from_name=value();
        else if(a=="--to-clip")to_name=value();
        else if(a=="--tide")tide=value();
        else if(a=="--capture")capture=utf8path(value());
        else if(a=="--accept-licenses")accept=true;
        else if(a=="--joined")joined=true;
        else if(a=="--skeleton")skeleton=true;
        else if(a=="--mannequin")mannequin=true;
        else if(a=="--character")character_file=utf8path(value());
        else if(a=="--texture")texture_file=utf8path(value());
        else if(a=="--to-soma")to_soma=true;
        else if(a=="--target")target_file=utf8path(value());
        else if(a=="--map")map_file=utf8path(value());
        else if(a.starts_with('-'))throw std::runtime_error("Unknown option: "+a);
        else pos.push_back(a);
    }
    std::set<std::string> allowed;
    if(command=="generate")allowed={"--models","-o","--output","--duration","--seed","--steps","--then","--transition","--start-pose","--end-pose","--keyframe","--effector","--path","--placement","--no-correction",
        "--correction-frames","--in-place","--variants","--variants-gif","--tide","--guidance","--constraint-guidance","--device","--threads","--model"};
    else if(command=="connect")allowed={"--models","-o","--output","--tokens","--joined","--offset","--from-clip","--to-clip","--threads"};
    else if(command=="mirror")allowed={"-o","--output","--clip"};
    else if(command=="retarget")allowed={"-o","--output","--target","--map","--to-soma","--clip"};
    else if(command=="inspect")allowed={"--clip","--skeleton"};
    else if(command=="preview")allowed={"--clip","--tide","--capture","--gif","--frames","--mannequin","--skeleton","--character","--map","--texture"};
    else if(command=="cleanup")allowed={"-o","--output","--contacts","--map","--clip"};
    else if(command=="models")allowed={"--models","--accept-licenses"};
    else if(command=="pose-sequence")allowed={"-o","--output","--models","--threads"};
    else if(command=="batch")allowed={"--models"};
    for(const auto& flag:flags)if(!allowed.contains(flag))throw std::runtime_error(command+" does not take "+flag);
    if(command=="batch") {
        if(pos.size()!=1)throw std::runtime_error("batch takes one jobs.json manifest");
        return batch(utf8path(pos[0]),models,context);
    }
    if(command=="models") {
        if(pos.empty()||pos.size()>2||(pos[0]!="install"&&pos[0]!="check"))throw std::runtime_error("Use models install or models check [all|kimodo|kimodo-v1.2|bricks]");
        std::string selection=pos.size()==2?pos[1]:"all";
        if(pos[0]=="check")check_models(models,selection);
        else {
            if(!accept)throw std::runtime_error("Read licenses/NVIDIA-KIMODO-MODEL.txt, NVIDIA-OPEN-MODEL.txt and META-LLAMA-3.txt, then pass --accept-licenses to download models");
            setup_models(models,selection);
        }
        return 0;
    }
    if(command=="inspect"||command=="preview") {
        if(pos.size()!=1)throw std::runtime_error(command+" takes one GLB or FBX file");
        auto path=utf8path(pos[0]);
        if(command=="preview") {
            PreviewOptions options;options.tide=tide;options.capture=capture;options.gif=gif;options.frames=frames;
            auto clip=load_glb(path,clip_name,false);
            bool canonical=true;try{require_soma(clip);}catch(const std::exception&){canonical=false;}
            if(canonical)options.facing=kimodo_heading(clip,clip.frame(0));
            if(!character_file.empty()) {
                if(skeleton||mannequin)throw std::runtime_error("Use one of --character, --mannequin and --skeleton");
                options.body=PreviewBody::character;options.character=character_file;options.texture=texture_file;
                if(!map_file.empty()) {
                    // A SOMA clip is retargeted onto the character here, as retarget --target does.
                    if(!canonical)throw std::runtime_error("preview --map retargets SOMA-30 clips; this clip already uses another rig, so leave out --map");
                    auto result=retarget(clip,load_rig_glb(character_file),load_mapping(map_file));
                    if(result.discarded_translations)std::cerr<<"Note: source has non-root translation animation that retargeting does not copy.\n";
                    clip=std::move(result.clip);
                } else if(canonical)throw std::runtime_error("Give --map BONES.json to retarget this SOMA clip onto the character");
            } else if(!map_file.empty()||!texture_file.empty())throw std::runtime_error("--map and --texture go with --character");
            else if(skeleton&&mannequin)throw std::runtime_error("Use one of --mannequin and --skeleton");
            else if(skeleton)options.body=PreviewBody::skeleton;
            else if(!canonical) {
                if(mannequin)throw std::runtime_error("The mannequin is SOMA-30; preview this clip with --character MODEL or --skeleton");
                std::cerr<<"Not a SOMA-30 clip: drawing its skeleton (give --character MODEL to see it on a body).\n";
                options.body=PreviewBody::skeleton;
            }
            preview(clip,path,options);
            return 0;
        }
        if(skeleton&&!clip_name.empty())throw std::runtime_error("inspect --skeleton does not take --clip");
        auto clip=skeleton?load_rig_glb(path):load_glb(path,clip_name,false);
        {
            bool canonical=true;try{require_soma(clip);}catch(const std::exception&){canonical=false;}
            nlohmann::json info={{"name",clip.name},{"rig",canonical?"soma30":"custom"},{"joints",clip.joints.size()},{"frames",clip.frames()},{"fps",clip.fps},{"duration",clip.duration()}};
            if(skeleton) {
                info["bones"]=nlohmann::json::array();
                for(const auto& joint:clip.joints)info["bones"].push_back({{"name",joint.name},{"parent",joint.parent},{"translation",{joint.rest.p.x,joint.rest.p.y,joint.rest.p.z}},{"rotation",{joint.rest.q.x,joint.rest.q.y,joint.rest.q.z,joint.rest.q.w}},{"scale",joint.scale}});
            }
            std::cout<<info.dump(2)<<"\n";
        }
        return 0;
    }
    if(output.empty()||output.extension()!=".glb")throw std::runtime_error("Choose a .glb output with -o");
    auto refuse_existing=[](const fs::path& file) {
        if(fs::exists(file)||fs::exists(file.string()+".part")||fs::exists(file.string()+".json"))throw std::runtime_error("Output, partial output or recipe already exists: "+file.string());
    };
    if(variants<2)refuse_existing(output);
    if(command=="pose-sequence") {
        if(pos.size()!=1)throw std::runtime_error("pose-sequence takes one JSON recipe");
        auto plan=load_pose_sequence(utf8path(pos[0]));
        if(!context->bricks||context->bricks_threads!=threads) {
            if(!context->bricks)check_models(models,"bricks");
            context->bricks=std::make_unique<BricksModel>(models/"motionbricks.onnx",threads);context->bricks_threads=threads;
        }
        auto made=generate_pose_sequence(plan,*context->bricks);made.name=output.stem().string();
        save_glb(made,output);
        nlohmann::json recipe={{"tool","tide-motion"},{"version",MOTION_VERSION},{"command",command},{"authoring",plan.recipe},{"threads",threads},{"fps",30},{"frames",made.frames()},
            {"motionbricksRevision","54fc563aeaaf819a753d72cc9fdc6713989cdd43"}};
        std::ofstream metadata(output.string()+".json");metadata<<recipe.dump(2)<<"\n";metadata.close();
        if(!metadata)throw std::runtime_error("Animation saved but could not write its recipe");
        std::cout<<"Saved "<<output.string()<<" ("<<made.frames()<<" frames)\n";return 0;
    }
    Clip clip;
    nlohmann::json retarget_recipe,generate_recipe,mirror_recipe;
    // Store generation settings alongside the asset; no need to rerun inference at build time.
    auto recipe_for=[&](const Clip& made,uint64_t used_seed) {
        return nlohmann::json{{"tool","tide-motion"},{"version",MOTION_VERSION},{"command",command},{"inputs",pos},{"rig","soma30"},{"fps",30},{"frames",made.frames()},{"requestedDuration",duration},{"seed",used_seed},{"steps",steps},{"tokens",tokens},{"joined",joined},{"offset",{offset.x,offset.y,offset.z}},
            {"fromClip",from_name},{"toClip",to_name},{"kimodoNativeRevision","ed8a931aedb132f768190f6931c365855caba74c"},{"motionbricksRevision","54fc563aeaaf819a753d72cc9fdc6713989cdd43"}};
    };
    auto finish=[](const Clip& made,const fs::path& path,const nlohmann::json& recipe) {
        save_glb(made,path);
        std::ofstream metadata(path.string()+".json");metadata<<recipe.dump(2)<<"\n";metadata.close();
        if(!metadata)throw std::runtime_error("Animation saved but could not write its recipe");
        std::cout<<"Saved "<<path.string()<<" ("<<made.frames()<<" frames, "<<made.duration()<<" seconds)\n";
    };
    if(command=="cleanup") {
        if(pos.size()!=1||contacts_file.empty())throw std::runtime_error("cleanup needs one source and --contacts PLAN.json");
        auto plan=load_cleanup_plan(contacts_file);
        clip=load_glb(utf8path(pos[0]),clip_name,false);
        std::optional<RetargetMapping> mapping;if(!map_file.empty())mapping=load_mapping(map_file);
        auto source_recipe=source_metadata(utf8path(pos[0]));
        auto protected_frames=protected_recipe_frames(source_recipe,clip.frames());
        plan.protected_frames.insert(plan.protected_frames.end(),protected_frames.begin(),protected_frames.end());
        auto reports=cleanup_contacts(clip,plan,mapping?&*mapping:nullptr);
        mirror_recipe={{"tool","tide-motion"},{"version",MOTION_VERSION},{"command","cleanup"},{"source",pos[0]},{"clip",clip_name},
            {"fps",clip.fps},{"frames",clip.frames()},{"solver","tide-two-bone-v1"},{"fadeFrames",plan.fade_frames},{"maxCorrectionMetres",plan.max_correction},{"protectedFrames",plan.protected_frames},
            {"sourceRecipe",source_recipe},{"rootMotionPreserved",true},{"contacts",nlohmann::json::array()}};
        if(source_recipe.contains("constraints"))mirror_recipe["constraints"]=source_recipe["constraints"];
        if(mapping) {std::ifstream input(map_file);nlohmann::json m;input>>m;mirror_recipe["mapping"]=m;}
        for(size_t i=0;i<reports.size();++i) {
            const auto& r=reports[i];const auto& contact=plan.contacts[i];
            mirror_recipe["contacts"].push_back({{"foot",contact.foot?"right-foot":"left-foot"},{"start",contact.start},{"end",contact.end},{"samples",r.samples},{"skipped",r.skipped},{"skipReason",r.skip_reason},
                {"horizontalMeanStepBeforeMetres",r.drift_before},{"horizontalMeanStepAfterMetres",r.drift_after},{"anchorResidualMaxMetres",r.residual_max},{"jointCorrectionMaxDegrees",r.rotation_max}});
        }
    } else if(command=="mirror") {
        if(pos.size()!=1)throw std::runtime_error("mirror takes one SOMA-30 .glb clip");
        clip=load_glb(utf8path(pos[0]),clip_name,true);
        mirror_soma(clip);
        mirror_recipe={{"tool","tide-motion"},{"version",MOTION_VERSION},{"command",command},{"source",pos[0]},{"clip",clip_name},{"rig","soma30"},{"fps",clip.fps},{"frames",clip.frames()},
            {"plane","the first frame's sagittal plane (through its root, up and its facing)"}};
    } else if(command=="retarget") {
        if(pos.size()!=1||map_file.empty()||(!to_soma&&target_file.empty())||(to_soma&&!target_file.empty()))
            throw std::runtime_error("Use retarget SOURCE.glb --map BONES.json with either --target CHARACTER.glb or --to-soma, and -o CLIP.glb");
        auto mapping=load_mapping(map_file);
        auto source=load_glb(utf8path(pos[0]),clip_name,!to_soma);
        Clip target;
        if(to_soma) {target.joints=soma();for(const auto& joint:target.joints)target.poses.push_back(joint.rest);}
        else target=load_rig_glb(target_file);
        auto result=retarget(source,target,mapping,to_soma);
        if(result.discarded_translations)std::cerr<<"Note: source has non-root translation animation; target bone lengths are preserved, so those translations are not copied.\n";
        clip=std::move(result.clip);
        auto protected_frames=protected_recipe_frames(source_metadata(utf8path(pos[0])),source.frames());
        std::ifstream mapping_stream(map_file);nlohmann::json mapping_json;mapping_stream>>mapping_json;
        retarget_recipe={{"tool","tide-motion"},{"version",MOTION_VERSION},{"command",command},{"source",pos[0]},{"target",target_file.string()},{"toSoma",to_soma},{"clip",clip_name},{"mapping",mapping_json},{"rootScale",result.scale},{"headingDegrees",result.heading_degrees},{"discardedNonRootTranslations",result.discarded_translations},{"protectedFrames",protected_frames},{"fps",clip.fps},{"frames",clip.frames()}};
    } else if(command=="generate") {
        if(pos.size()!=1||pos[0].empty())throw std::runtime_error("generate takes one quoted prompt");
        if(segments.size()>16)throw std::runtime_error("generate takes at most 16 prompts");
        segments[0].first=pos[0];
        std::vector<kimodo::prompt_segment> parts;size_t total=0;
        for(size_t i=0;i<segments.size();++i) {
            auto count=static_cast<unsigned>(std::round(segments[i].second*30));
            if(i&&transition>=count)throw std::runtime_error("--transition must be shorter than every following prompt ("+std::to_string(count)+" frames)");
            parts.push_back({segments[i].first,count});total+=count;
        }
        // Resolve and load every constraint before any model is loaded.
        auto resolve=[&](long frame,const char* label) {
            long resolved=frame<0?static_cast<long>(total)+frame:frame;
            if(resolved<0||resolved>=static_cast<long>(total))throw std::runtime_error(std::string(label)+" "+std::to_string(frame)+" is outside the "+std::to_string(total)+"-frame output");
            return static_cast<size_t>(resolved);
        };
        MotionConditions conditions;
        std::map<std::string,Clip> pose_files;
        for(const auto& spec:pose_specs) {
            auto ref=pose_ref(spec.ref);
            auto& source=pose_files[ref.file];
            if(source.joints.empty()) {source=load_glb(utf8path(ref.file),"",true);require_soma(source);}
            long wanted=ref.frame?*ref.frame:(spec.kind=="start-pose"?-1:0);
            long index=wanted<0?static_cast<long>(source.frames())+wanted:wanted;
            if(index<0||index>=static_cast<long>(source.frames()))throw std::runtime_error(ref.file+" has no frame "+std::to_string(wanted)+" ("+std::to_string(source.frames())+" frames)");
            PoseConstraint pose{spec.kind,resolve(spec.frame,spec.kind.c_str()),ref.file,index,{},spec.effectors};
            auto frame=source.frame(static_cast<size_t>(index));pose.pose.assign(frame.begin(),frame.end());
            conditions.poses.push_back(std::move(pose));
        }
        for(auto [point,frame]:path_specs) {point.frame=resolve(frame,"path frame");conditions.path.push_back(point);}
        bool shift=place_constraints(conditions,placement);
        std::optional<KimodoCondition> condition;
        kimodo::motion_condition native;
        if(!conditions.empty()) {
            condition=build_condition(conditions,total);
            native={condition->observed,condition->mask,condition->first_heading};
        }
        if(in_place&&!conditions.path.empty())throw std::runtime_error("--in-place removes the root travel --path asks for; use one or the other");
        if(variants_gif&&variants<2)throw std::runtime_error("--variants-gif goes with --variants 2 or more");
        if(seed+variants-1>4294967295ULL)throw std::runtime_error("--seed plus --variants passes 4294967295");
        // Every output is checked before the model loads.
        std::vector<fs::path> outputs;
        for(unsigned v=0;v<variants;++v)outputs.push_back(variants>1?variant_path(output,seed+v):output);
        VariantSheet sheet;sheet.tide=tide;
        if(variants>1) {sheet.png=variants_path(output,".png");if(variants_gif)sheet.gif=variants_path(output,".gif");}
        for(const auto& file:outputs)refuse_existing(file);
        for(const auto& file:{sheet.png,sheet.gif})if(!file.empty()&&fs::exists(file))throw std::runtime_error("Contact sheet already exists: "+file.string());
        ensure_kimodo(models,context,device,threads,model_version);
        if(condition&&context->kimodo->motion_dim()!=kimodo_features)throw std::runtime_error("Constraints need the SOMA-30 Kimodo model");
        std::vector<Clip> made;std::vector<std::string> labels;
        for(unsigned v=0;v<variants;++v) {
        const uint64_t used_seed=seed+v;
        clip=Clip{};generate_recipe=nlohmann::json::object();
        if(variants>1)std::cerr<<"Generating variant "<<v+1<<"/"<<variants<<" (seed "<<used_seed<<")...\n";
        else std::cerr<<"Generating motion...\n";
        auto result=generate_motion(models,context,parts,transition,steps,used_seed,guidance,condition?&native:nullptr);
        if(result.joints!=30||result.local_rotations_xyzw.size()!=static_cast<size_t>(result.frames)*30*4||result.root_positions.size()!=static_cast<size_t>(result.frames)*3)throw std::runtime_error("Kimodo returned an unexpected SOMA output layout");
        clip.joints=soma();clip.name=outputs[v].stem().string();
        // Constraints were expressed relative to the frame-0 origin; return to their world.
        Vec3 origin=condition?condition->origin:Vec3{};
        for(size_t f=0;f<result.frames;++f)for(size_t j=0;j<30;++j) {
            size_t index=(f*30+j)*4;Transform t=clip.joints[j].rest;
            t.q=normalized({result.local_rotations_xyzw[index],result.local_rotations_xyzw[index+1],result.local_rotations_xyzw[index+2],result.local_rotations_xyzw[index+3]});
            if(j==0)t.p=Vec3{result.root_positions[f*3],result.root_positions[f*3+1],result.root_positions[f*3+2]}+origin;
            clip.poses.push_back(t);
        }
        if(shift)shift_to_last_pose(clip,conditions);
        generate_recipe["prompts"]=nlohmann::json::array();
        for(size_t i=0;i<parts.size();++i)generate_recipe["prompts"].push_back({{"prompt",parts[i].prompt},{"duration",segments[i].second},{"frames",parts[i].frames}});
        generate_recipe["transition"]=parts.size()>1?transition:0;
        generate_recipe["placement"]=placement_name;generate_recipe["corrected"]=correct&&!conditions.poses.empty();
        generate_recipe["correctionFrames"]=correction_frames;generate_recipe["inPlace"]=in_place;
        generate_recipe["guidance"]=guidance.text;generate_recipe["constraintGuidance"]=guidance.constraint;
        generate_recipe["model"]="nvidia/Kimodo-SOMA-RP-"+model_version;
        generate_recipe["modelGgufSha256"]=model_version=="v1.2"?"9c24c5ee9b6b3683947ab2796727ffb2fbd69b2ebc7683214809013687d87a19":"3bf1229f4c1eff1d28f5196a854113da2df9a11a5e21c60694630903bf948ee4";
        if(model_version=="v1.2")generate_recipe["checkpointRevision"]="0e3c89024d896064b6851201373f7834784a386b";
        generate_recipe["device"]=context->kimodo_gpu?"gpu":"cpu";generate_recipe["threads"]=threads;
        if(variants>1)generate_recipe["variants"]={{"count",variants},{"firstSeed",seed},{"sheet",sheet.png.filename().string()}};
        generate_recipe["constraints"]=nlohmann::json::array();
        auto round=[](float value,float unit){return std::round(value/unit)*unit;};
        // Errors are the model's own; correction then makes full-body poses exact.
        for(const auto& pose:conditions.poses) {
            auto e=constraint_error(clip,pose);
            nlohmann::json item={{"kind",pose.kind},{"frame",pose.frame},{"file",pose.file},{"sourceFrame",pose.source_frame},{"anchored",pose.anchored},
                {"modelError",{{"rootMetres",round(e.root,1e-4f)},{"positionMeanMetres",round(e.position_mean,1e-4f)},{"positionMaxMetres",round(e.position_max,1e-4f)},{"rotationMeanDegrees",round(e.rotation_mean,0.01f)},{"rotationMaxDegrees",round(e.rotation_max,0.01f)}}}};
            if(pose.effectors)item["effectors"]=effector_list(pose.effectors);
            generate_recipe["constraints"].push_back(item);
            std::cerr<<pose.kind<<(pose.effectors?" "+effector_list(pose.effectors):std::string())<<" at frame "<<pose.frame<<(pose.anchored?"":" (free on the ground)")<<": root "<<round(e.root*100,0.1f)<<" cm; "<<(pose.effectors?"effector":"joint")<<" positions mean "<<round(e.position_mean*100,0.1f)<<" / max "<<round(e.position_max*100,0.1f)<<" cm; rotations mean "<<round(e.rotation_mean,0.1f)<<" / max "<<round(e.rotation_max,0.1f)<<" degrees"<<(correct&&!pose.effectors?"; corrected to exact":"")<<"\n";
        }
        if(correct) {
            correct_poses(clip,conditions,correction_frames);
            size_t index=0;
            for(const auto& pose:conditions.poses) {
                auto e=constraint_error(clip,pose);
                generate_recipe["constraints"][index++]["error"]={{"rootMetres",round(e.root,1e-4f)},{"positionMaxMetres",round(e.position_max,1e-4f)},{"rotationMaxDegrees",round(e.rotation_max,0.01f)}};
            }
        }
        for(const auto& point:conditions.path) {
            float e=path_error(clip,point);
            nlohmann::json item={{"kind","path"},{"frame",point.frame},{"x",point.x},{"z",point.z},{"error",{{"hipsGroundMetres",round(e,1e-4f)}}}};
            if(point.has_heading)item["yaw"]=point.heading;
            generate_recipe["constraints"].push_back(item);
            std::cerr<<"path at frame "<<point.frame<<": hips "<<round(e*100,0.1f)<<" cm from "<<point.x<<", "<<point.z<<"\n";
        }
        // Recorded errors are measured before the travel is removed.
        if(in_place)remove_root_travel(clip);
        auto recipe=recipe_for(clip,used_seed);
        double requested=0;for(const auto& segment:segments)requested+=segment.second;
        recipe["requestedDuration"]=requested;
        for(const auto& item:generate_recipe.items())recipe[item.key()]=item.value();
        finish(clip,outputs[v],recipe);
        if(variants>1) {made.push_back(clip);labels.push_back("seed "+std::to_string(used_seed));}
        }
        if(variants>1) {
            // The clips are kept whatever happens to the sheet.
            try {preview_variants(made,labels,sheet);std::cout<<"Saved "<<sheet.png.string()<<(sheet.gif.empty()?"":" and "+sheet.gif.string())<<"\n";}
            catch(const std::exception& e) {std::cerr<<"Contact sheet skipped: "<<e.what()<<"\n";}
        }
        return 0;
    } else {
        if(pos.size()!=2)throw std::runtime_error("connect takes source and target GLB files");
        auto a=load_glb(utf8path(pos[0]),from_name),b=load_glb(utf8path(pos[1]),to_name);
        for(size_t f=0;f<b.frames();++f)b.frame(f)[0].p=b.frame(f)[0].p+offset;
        if(!context->bricks||context->bricks_threads!=threads) {
            if(!context->bricks)check_models(models,"bricks");
            context->bricks=std::make_unique<BricksModel>(models/"motionbricks.onnx",threads);context->bricks_threads=threads;
        }
        clip=context->bricks->generate(a,b,tokens);
        if(joined)clip=join(a,clip,b);
    }
    auto recipe=recipe_for(clip,seed);
    if(command=="connect")recipe["threads"]=threads;
    if(command=="retarget")recipe=std::move(retarget_recipe);
    if(command=="mirror"||command=="cleanup")recipe=std::move(mirror_recipe);
    finish(clip,output,recipe);return 0;
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
    try {
        SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
        std::vector<std::string> args;
        for(int i=0;i<argc;++i) { int n=WideCharToMultiByte(CP_UTF8,0,argv[i],-1,nullptr,0,nullptr,nullptr);std::string s(static_cast<size_t>(n),0);WideCharToMultiByte(CP_UTF8,0,argv[i],-1,s.data(),n,nullptr,nullptr);s.pop_back();args.push_back(s); }
        return main_impl(args);
    } catch(const std::exception& e) { std::cerr<<"tide-motion: "<<e.what()<<"\n";return 1; }
}
#else
int main(int argc,char** argv) {
    try { return main_impl({argv,argv+argc}); }
    catch(const std::exception& e) {std::cerr<<"tide-motion: "<<e.what()<<"\n";return 1;}
}
#endif
