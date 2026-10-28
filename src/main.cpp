#include "motion.hpp"
#include "equipment.hpp"
#include "author.hpp"
#include "ardy.hpp"
#include "upgrade.hpp"
#include "momask.hpp"
#include "ardy_gpu.hpp"
#include "dart.hpp"
#include "dart_gpu.hpp"
#include "review.hpp"
#include <kimodo/kimodo.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
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
void use_cpu(const std::string& reason) {
    std::cerr<<reason<<"; generating on the CPU.\n";
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
    if(command=="review")return R"(tide-motion review MANIFEST.json -o REVIEW.html

Build a local, self-contained advisory review and REVIEW.html.json observations.
Version-1 manifest: {version:1, cases:[{id,label,file,clip?,map?,context?,settings?}]}.
Paths are relative to the manifest. Up to 24 cases; explicit map retargets to SOMA.
No inference, renderer, server, network or new installed dependency is needed.
Watch before revealing hints; record inaccurate, acceptable, intentional or
actionable observations and timestamped missed issues. Export feedback as JSON.
Capsule/ground checks are sampled SOMA proxies, not skin collision or human
reachability judgments. No findings is not a quality pass. No assets are changed.
See notes/advisory-review.md for settings, coverage and feedback contracts.
)";
    if(command=="equipment")return equipment_help();
    if(command=="momask"||command=="inpaint"||command=="extend")return momask_help();
    if(command=="dart")return R"(tide-motion dart PLAN.json -o OUT.glb [--models DIR] [--threads N]

Native offline DART free-text timelines, source continuation, goals and inbetweening.
Install explicitly with models install dart --accept-licenses. Canonical SOMA, 30 Hz.
Version-1 plan: seed, warmupChunks, segments [{text, chunks, target?: [x,z], guidance?}].
Any valid UTF-8 text is accepted; an empty prompt is unconditional. Goal targets are
world metres. Policy training skills are walk/run/hop on left leg; other texts
remain available with measured misses. Targets stop within 0.3 m or at the budget.
Optional history/historyState or source/sourceFrame/sourceMap seed continuation;
seedMode history or repeat supports moving history or a single pose.
Sampler controls: guidance, sampler ddim|ddpm, timesteps (increasing 0..9), eta,
noiseScale, zeroNoise, policyMode sample|mean|none, fixFloor and groundHeight.
optimization enables native analytic gradients/Adam over the whole latent timeline:
constraints [{frame,joint,position,axes?,weight?}], target/targetFrame/targetJoints,
iterations, learningRate, unitGradient, anneal, jerkWeight, floorWeight, and scene SDF.
Default jointSpace soma optimizes exported FK; predicted uses raw model joints.
These are soft objectives: recipes report actual errors, not guaranteed endpoints.
See notes/dart-backend.md for input schemas, examples and measured limitations.
auto prefers native Windows DirectML and restarts on CPU if GPU execution fails.
gpu is strict; cpu forces CPU. Text/model conversion stay native CPU operations.
Consecutive forward batch actions share neural calls (batchSize 1..8, default 8).
Optimization jobs reuse sessions and run individually; no Python runtime required.
)";
    if(command=="generate")return R"(tide-motion generate "PROMPT" -o CLIP.glb [options]

Generates a SOMA-30 clip from text (30 fps).

  --backend kimodo|ardy|momask Backend (default Kimodo); MoMask weights are manual-only
                              MoMask controls/limits: tide-motion help momask
  --history FILE.glb          Imported SOMA context for ARDY or MoMask
  --history-frames N          ARDY visible history, 4..240, multiple of four
                              ARDY uses 1..10 steps (default 10), switches --then
                              prompts on 60-frame chunk boundaries, without blending.
                              Converted models live in MODELS/ardy; see tools/ardy/.
                              Windows ARDY uses DirectML GPU with --device gpu;
                              auto tries GPU and falls back to CPU. Text stays local CPU.

      --model v1.1|v1.2      Smooth-root SOMA-RP checkpoint (default v1.1)
  -o, --output FILE.glb       New output (a FILE.glb.json recipe is written beside it)
  --duration SECONDS          Prompt length: Kimodo 0.2..10, ARDY/MoMask 0.2..600
                              Default: Kimodo/ARDY 3, MoMask 2; see help momask
  --then "PROMPT"             Continue with another prompt (up to 16 in total); a
                              --duration after it sets that prompt's length
  --transition FRAMES         Frames blended between prompts, 1..60 (default 5)
  --start-pose FILE.glb[@F]   Begin in frame F of a SOMA clip (default: its last
                              frame, so the output continues from it)
  --end-pose FILE.glb[@F]     End in frame F of a SOMA clip (default: its first
                              frame, so the output leads into it)
  --end-at X,Z[,YAW]         Place --end-pose on the ground, optionally turn it
  --keyframe FRAME FILE.glb[@F]
                              Reach frame F (default first) at output FRAME
  --effector PARTS FRAME FILE.glb[@F]
                              Reach only the hands/feet of frame F (default first)
                              at output FRAME; PARTS is a comma list of left-hand,
                              right-hand, left-foot and right-foot
  --path X,Z[,YAW]@FRAME      Put the root over ground point X,Z (metres) at output
                              FRAME, optionally facing YAW degrees (0 faces +Z,
                              90 faces +X); repeat for waypoints
  --follow world|ends         Where --keyframe and --effector targets are: world
                              (default) as in their file; ends: also turned and
                              moved by the share of the end pose's --end-at (or
                              final --path) placement their frame has reached,
                              counted from the start pose
  --dense-path                Kimodo: constrain the root on every frame from frame 0
                              to the last waypoint, interpolated and smoothed as
                              upstream's dense 2D root path (see below)
  --pose-map BONES.json       Retarget pose files on another mapped humanoid rig
                              (GLB or FBX) to SOMA first, as retarget --to-soma
  --placement auto|world|free
                              Ground placement of poses after frame 0 (see below)
  --fit-poses                 Select the lowest raw pose rotation error across seeds
  --footsteps N               Select a clip with N measured travelling footfalls
  --attempts N                Seed budget for either selection, 1..64 (default 8)
  --correction tide|kimodo|none
                              tide (default) makes full-body poses exact with a
                              fade; kimodo runs the native port of upstream's
                              MotionCorrection (root warps, effector IK and foot
                              contact IK from the model's contacts); none keeps the
                              model's output
  --no-correction             Same as --correction none
  --rig soma30|soma77         Output skeleton (default soma30); soma77 adds
                              upstream's relaxed fingers, toe ends and head end
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
  --steps N                   Kimodo 1..1000 (default 50); ARDY 1..10 (default 10)
  --device auto|gpu|cpu       Where generation runs (default auto: GPU, or CPU
                              when it is unavailable or fails; gpu never falls back)
  --threads N                 CPU threads, 1..1024 (default: all)
  --models DIR                Model directory (default: TIDE_MOTION_MODELS or the
                              OS cache)

F is a frame of the pose file; negative counts from its end (-1 is the last),
and "first" or "last" also work. Output FRAME is 0-based at 30 fps; negative
counts from the end. Pose files are SOMA-30 or SOMA-77 GLBs; give --pose-map for
another mapped humanoid (retargeted as --to-soma: rotations and root only, with its
calibration limits, so the constraint is the retargeted SOMA pose).

Placement: a start pose (or a path point at frame 0) fixes where the output
begins, in that clip's world. world also pins later poses to their ground
positions in their files; free keeps their body, height and facing but lets the
motion choose where on the ground they happen using two learned passes.
auto (default) is world when the
first frame or a path fixes the world, otherwise free, and then moves the whole
output so its last pose lands where it is in its file (so an end pose leads
straight into that clip). Without a start pose the output begins facing +Z.

Kimodo reaches poses closely but not exactly; the model's error at each
constraint is printed and stored in the recipe. Full-body poses are then made
exact, fading the correction out over up to --correction-frames frames either
side (effector targets and path points are left as generated). --in-place is
applied last: frame 0 stays put and later poses move with the removed travel.

--correction kimodo instead warps the root and every rotation to full-body poses
(exact), pins effector targets with two-bone IK and the root within 4 cm of path
points, then locks feet and toes where Kimodo predicts contacts. Its counts are
stored in the recipe; it is an alternative to the default, not added to it.

Dense paths: waypoints (plus frame 0 where the output starts) are interpolated
linearly to every frame up to the last waypoint and, with three or more, smoothed
within 6 cm as upstream; every frame then constrains the root, and poses on the
path measure their joints from it. Waypoint yaws are kept. The recipe records the
output's smooth-root distance from the path.

--rig soma77 writes the 77-joint presentation skeleton: the same 30 joints plus
constant relaxed fingers. Every command reads SOMA-77 clips as their exact
SOMA-30 subset; mirror and cleanup keep the 77 joints.

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
  --tokens N|MIN..MAX         Allowed transition length in tokens, 6..16; the
                              model picks within a range (default: 8..14)
  --target-poses N            Target frames that condition the model, 1..4:
                              1 stops at the target pose, 2 adds its velocity,
                              3-4 acceleration (default: 4)
  --contacts-out PLAN.json    Write the model's predicted planted-heel intervals
                              as a cleanup contact plan (frames of the output)
  --contact-threshold T       Heel contact confidence for --contacts-out, 0..1
                              (default: 0.5)
  --joined                    Export source, bridge and target as one animation
  --offset X Y Z              Move the target clip before connecting (metres)
  --from-clip NAME            Animation to use from a multi-animation FROM.glb
  --to-clip NAME              Animation to use from a multi-animation TO.glb
  --threads N                 CPU threads, 1..1024 (default: half)
  --models DIR                Model directory

Bridge endpoints are protected from later cleanup. --joined also carries authored
source/target protected frames with the correct offsets after seam omission.
)";
    if(command=="pose-sequence")return R"(tide-motion pose-sequence POSES.json -o CLIP.glb [--models DIR] [--threads N]
    [--contacts-out PLAN.json] [--contact-threshold T]

Native MotionBricks offline authoring through 1..128 explicit SOMA pose targets.
Poses use schema tide-motion-pose-sequence-1, start and targets. Each pose has
file, optional clip/frame/hold, required position [root X,Y,Z metres] and yaw
(degrees, 0 faces +Z, 90 +X). Each target requires tokens 6..16 or [min, max]
(the model picks); it adds 4*tokens-7-early intervals at 30 Hz. Optional poses
1..4 (default 4) sets how many target frames condition the model: 1 stops there.
Optional early N (not on the final target) hands over to the next target N frames
before arrival, so that target is a guide passed near, not reached exactly.
Start defaults to the last source frame, targets
to the first. Negative frame indices count from the end. Windows sample the last
four context / first four target samples, clamped at clip ends; hold:true repeats
the selected pose. Later contexts use the last four generated samples.
Paths are relative to POSES.json. All source clips must exist before the run
(including in batch). Output is ordinary GLB plus an effective JSON recipe.
--contacts-out writes predicted planted-heel intervals as a cleanup plan.
No scene understanding, IK, arbitrary timing or velocity seam guarantee.
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
The plan also bounds joint correction (maxJointCorrectionDegrees, default 30),
extra local joint rotation per frame (maxJointStepIncreaseDegrees, default 5),
and extra 3D ankle travel per frame (maxAnkleStepIncrease, default 0.005 metres).
Step guards include the transitions into/out of each contact. Rejected intervals
are restored in full; recipes distinguish attempted corrections from final motion.
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

Runs ordered generate, inpaint, extend, connect, pose-sequence, dart, retarget, equipment, cleanup and mirror actions, loading each model
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
  --prop MESH.glb|.fbx        Draw a mesh rigidly on --prop-bone; repeat for more props
  --prop-bone BONE            The bone it follows: one of --character's bones
                              (after retargeting), or a SOMA bone such as
                              RightHand on the mannequin. The bone's scale is
                              ignored (Synty bones carry 0.01)
  --prop-offset X,Y,Z         Move it within the bone, in metres (default 0)
  --prop-rotation X,Y,Z       Turn it within the bone, Euler degrees (default 0)
                            Prop options after --prop apply until the next --prop
  --first-person              One camera on --camera-bone (default Head)
  --camera-bone BONE          Explicit eye/head node on the drawn rig
  --camera-mode bone|root     Follow bone, or carry its initial view on root
  --camera-offset X,Y,Z       Eye offset in metres (default 0,0.08,0.12)
  --camera-rotation X,Y,Z     Bone-local Euler degrees (default 0)
  --camera-fov DEGREES        Vertical field of view (10..150; default 75)
  --camera-near METRES        Near clip distance (0.001..1; default 0.01)
  --setup FILE.json           Load versioned props/camera setup; relative paths
  --save-setup FILE.json      Save resolved setup; refuses an existing file
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
    if(command=="upgrade")return R"(tide-motion upgrade [--check] [--version X.Y.Z] [--force] [--root DIR]
tide-motion upgrade --archive PACKAGE.zip|.tar.gz [--check] [--force] [--root DIR]

Install the latest published stable release for this platform, keeping models,
settings and PATH. --check only reports the selected release. --version selects
an exact stable release (including an older one). --force reinstalls the selected
release even when this executable is already as new or newer.

--archive uses a local official-format package and adjacent .sha256, with no
network access. --root overrides the installation directory (otherwise the root
of this running installed executable). Source builds require an explicit root.
Downloads use HTTPS curl; Windows extraction uses system PowerShell, Unix tar.
Checksums, safe archive paths, package version and runnable executable are checked
before a transactional replacement. Close other running Tide Motion processes
if Windows reports locked files. No model downloads or engine updates occur.
)";
    if(command=="models")return R"(tide-motion models install [all|kimodo|kimodo-v1.2|text|bricks|dart] --accept-licenses [--models DIR]
tide-motion models check [all|kimodo|kimodo-v1.2|text|bricks|dart] [--models DIR]
tide-motion models import-momask BUNDLE_DIRECTORY [--models DIR]
tide-motion models check momask [--models DIR]

Downloads (resumably) and verifies the pinned model files by SHA-256.
kimodo-v1.2 downloads pinned official safetensors/statistics and converts them
to verified GGUF natively, then adds the shared text files. No Python is needed.
all retains v1.1 and MotionBricks.
MoMask is manual-only: import its supported prepared ONNX bundle into DIR/momask.
Raw .tar/.pt files are not that bundle. No MoMask weights are downloaded or bundled;
manual import does not establish commercial rights for the supplied checkpoints.
dart is an explicit opt-in: pinned DART/CLIP files are converted to native ONNX
graphs locally. It adds no SMPL body assets, Python or hosted inference.

  --accept-licenses           Accept the model licences in licenses/ (install)
  --models DIR                Model directory (default: TIDE_MOTION_MODELS or the
                              OS cache)
)";
    return nullptr;
}
void help(std::string_view command={}) {
    if(const char* text=command_help(command)) {std::cout<<text;return;}
    std::cout<<R"(Tide Motion - native offline animation authoring

  tide-motion models install [all|kimodo|kimodo-v1.2|text|bricks|dart] --accept-licenses [--models DIR]
  tide-motion models check [all|kimodo|kimodo-v1.2|text|bricks|ardy|dart] [--models DIR]
  tide-motion generate "PROMPT" -o CLIP.glb [--duration 3] [--seed 0] [--steps 50]
      [--variants N [--variants-gif]] [--guidance 2] [--constraint-guidance 2]
      [--then "PROMPT" [--duration S]]... [--transition 5]
      [--start-pose FILE.glb[@F]] [--end-pose FILE.glb[@F]]
      [--keyframe FRAME FILE.glb[@F]]... [--effector PARTS FRAME FILE.glb[@F]]...
      [--path X,Z[,YAW]@FRAME]... [--correction-frames 10] [--in-place]
      [--device auto|gpu|cpu] [--threads N]
  tide-motion generate "PROMPT" --backend momask -o CLIP.glb [--duration 2] [--seed 0]
  tide-motion inpaint "PROMPT" --source SOMA.glb --begin F --end F -o EDIT.glb
  tide-motion extend "PROMPT" --source SOMA.glb --duration SECONDS -o LONGER.glb
  tide-motion models import-momask BUNDLE_DIRECTORY [--models DIR]
  tide-motion connect FROM.glb TO.glb -o BRIDGE.glb [--tokens N|MIN..MAX] [--joined]
      [--offset X Y Z] [--from-clip NAME] [--to-clip NAME] [--threads N]
  tide-motion pose-sequence POSES.json -o CLIP.glb [--threads N]
  tide-motion dart PLAN.json -o CLIP.glb [--threads N]
  tide-motion equipment inspect|bake|preview CLIP.glb --plan EQUIPMENT.json
  tide-motion cleanup CLIP.glb --contacts PLAN.json -o CLEAN.glb [--map BONES.json]
  tide-motion mirror CLIP.glb -o MIRRORED.glb [--clip NAME]
  tide-motion retarget SOURCE.glb --target CHARACTER.glb|.fbx --map BONES.json -o CLIP.glb
  tide-motion retarget SOURCE.glb|.fbx --to-soma --map BONES.json -o SOMA.glb [--clip NAME]
  tide-motion batch JOBS.json [--models DIR]
  tide-motion inspect FILE.glb|FILE.fbx [--clip NAME | --skeleton]
  tide-motion review MANIFEST.json -o REVIEW.html
  tide-motion preview CLIP.glb [--mannequin | --skeleton | --character MODEL.glb|.fbx
      [--map BONES.json] [--texture IMAGE.png]] [--prop MESH --prop-bone BONE]
      [--tide PATH] [--capture IMAGE.png]
      [--gif OUT.gif] [--frames N] [--clip NAME]
  tide-motion help [COMMAND]       one command's options (also COMMAND --help)
  tide-motion --version

Downloadable models are fetched once and SHA-256 verified. MoMask weights are
manual-only; use help momask for its model import, controls and limits.
Generation and transitions run
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
        // Early (guide) targets are passed near, not reached exactly.
        for(const auto& target:recipe.at("authoring").at("targets"))if(target.value("early",0)==0)add(target.at("endFrame"));
    }
    std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return out;
}
struct RunContext {
    std::unique_ptr<MomaskContext> momask;
    std::unique_ptr<ArdyModel> ardy;unsigned ardy_threads=0;std::string ardy_device;
    std::unique_ptr<kimodo::model> kimodo;
    // strict: an explicit --device, which never falls back. gpu_failed: auto fell back
    // in this run, so later auto actions stay on the CPU.
    bool kimodo_gpu=false,kimodo_strict=false,gpu_failed=false;
    std::string kimodo_version="v1.1";std::set<std::string> checked_versions;
    std::string kimodo_device;unsigned kimodo_threads=0;
    std::unique_ptr<BricksModel> bricks;unsigned bricks_threads=0;
    // Batch connect runs: prepare records each request without inference; the
    // following normal execution of the same actions takes its bridge in order.
    struct PendingBridge { Clip from,to; BricksRequest options; fs::path output; };
    struct ReadyBridge { fs::path output; Bridge bridge; size_t requests=1,index=0; };
    bool bricks_prepare=false;std::vector<PendingBridge> bricks_pending;std::deque<ReadyBridge> bricks_ready;
    std::unique_ptr<DartModel> dart;unsigned dart_threads=0;std::string dart_device;
    struct ReadyDart {fs::path output;DartResult result;};std::deque<ReadyDart> dart_ready;
};
DartModel& dart_session(const fs::path& models,unsigned threads,const std::string& device,RunContext* context) {
    if(!context->dart||context->dart_threads!=threads||context->dart_device!=device) {
        context->dart.reset();check_models(models,"dart");context->dart=std::make_unique<DartModel>(models/"dart",threads,device);context->dart_threads=threads;context->dart_device=device;
    }
    return *context->dart;
}
BricksModel& bricks_session(const fs::path& models,unsigned threads,RunContext* context) {
    if(!context->bricks||context->bricks_threads!=threads) {
        if(!context->bricks)check_models(models,"bricks");
        context->bricks=std::make_unique<BricksModel>(models/"motionbricks.onnx",threads);context->bricks_threads=threads;
    }
    return *context->bricks;
}
struct Guidance { float text=2,constraint=2; };
void load_kimodo(const fs::path& models,RunContext* context) {
    std::cerr<<"Loading Kimodo SOMA and text encoder (native GGML)...\n";
    std::string failure;
    try {
        // The device and threads go to Kimodo as runtime options (kimodo_runtime_options),
        // not through KIMODO_BACKEND/KIMODO_THREADS; threads 0 still honours KIMODO_THREADS.
        kimodo::runtime_options options;
        options.device=context->kimodo_gpu?kimodo::runtime_options::device_kind::vulkan:kimodo::runtime_options::device_kind::cpu;
        options.threads=context->kimodo_threads;
        auto model=kimodo::model::load((models/(context->kimodo_version=="v1.2"?"kimodo-v1.2.gguf":"kimodo.gguf")).string(),(models/"text/encoder.gguf").string(),options);
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
    if(context->kimodo) {std::cerr<<"Reloading Kimodo for another model, --device or --threads.\n";context->kimodo.reset();}
    context->kimodo_threads=threads;
    context->kimodo_strict=device!="auto";
    if(device=="cpu") {context->kimodo_gpu=false;std::cerr<<"Generating on the CPU (--device cpu).\n";}
    else if(device=="gpu") {
#ifdef MOTION_VULKAN
        std::string reason=vulkan_unavailable();
        if(!reason.empty())throw std::runtime_error("--device gpu: Vulkan is unavailable ("+reason+")");
        context->kimodo_gpu=true;
#else
        throw std::runtime_error("--device gpu: this build has no GPU support; use --device auto or cpu");
#endif
    } else if(context->gpu_failed) {context->kimodo_gpu=false;std::cerr<<"Generating on the CPU (the GPU failed earlier in this run).\n";}
    else context->kimodo_gpu=kimodo_on_gpu();
    // Models are verified once, after the device is settled (a refused --device gpu skips it).
    if(!context->checked_versions.contains(version)) {check_models(models,version=="v1.2"?"kimodo-v1.2":"kimodo");context->checked_versions.insert(version);}
    context->kimodo_version=version;
    load_kimodo(models,context);
    context->kimodo_device=device;
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
// Three comma-separated numbers, as --prop-offset X,Y,Z.
Vec3 triplet(const std::string& text,double limit,const char* label) {
    std::vector<std::string> parts;size_t start=0;
    while(true) {auto comma=text.find(',',start);parts.push_back(text.substr(start,comma-start));if(comma==std::string::npos)break;start=comma+1;}
    if(parts.size()!=3)throw std::runtime_error(std::string("Use ")+label+" X,Y,Z");
    return {static_cast<float>(number(parts[0],-limit,limit,label)),static_cast<float>(number(parts[1],-limit,limit,label)),static_cast<float>(number(parts[2],-limit,limit,label))};
}
int main_impl(const std::vector<std::string>& args,RunContext* context=nullptr);
// Public CLI uses the shared cache; the native MoMask parser consumes the
// concrete CACHE/momask directory and owns no automatic download path.
std::vector<std::string> momask_arguments(const std::vector<std::string>& args) {
    bool prefix=args.size()>1&&args[1]=="momask";
    std::string action=prefix?(args.size()>2?args[2]:"--help"):args.at(1);
    std::vector<std::string> out{args[0],"momask",action};
    if(action=="--help"||action=="help")return out;
    for(size_t i=prefix?3:2;i<args.size();++i)if(args[i]=="--help"||args[i]=="-h")return {args[0],"momask","--help"};
    fs::path root=model_directory();
    bool backend_seen=false,models_seen=false;
    for(size_t i=prefix?3:2;i<args.size();++i) {
        if(args[i]=="--backend") {if(backend_seen||i+1==args.size()||args[i+1]!="momask")throw std::runtime_error("This action requires backend momask");backend_seen=true;++i;}
        else if(args[i]=="--models") {if(models_seen||i+1==args.size())throw std::runtime_error("Missing or duplicate --models");models_seen=true;root=utf8path(args[++i]);}
        else if(args[i]=="--help"||args[i]=="-h")return {args[0],"momask","--help"};
        else out.push_back(args[i]=="--output"?"-o":args[i]);
    }
    out.insert(out.end(),{"--models",(root/"momask").string()});return out;
}
int run_momask(const std::vector<std::string>& args,RunContext* context,bool preflight=false,bool defer_source=false) {
    auto routed=momask_arguments(args);
    if(!preflight&&!context->momask)context->momask=std::make_unique<MomaskContext>();
    return momask_main(routed,preflight?nullptr:context->momask.get(),preflight,defer_source);
}
struct BatchPathLess {
    bool operator()(const fs::path& a,const fs::path& b) const {
#ifdef _WIN32
        int result=CompareStringOrdinal(a.c_str(),-1,b.c_str(),-1,TRUE);
        if(!result)throw std::runtime_error("Cannot compare batch output paths");
        return result==CSTR_LESS_THAN;
#else
        return a<b;
#endif
    }
};
int batch(const fs::path& manifest,const fs::path& models,RunContext* context) {
    if(fs::file_size(manifest)>1024*1024)throw std::runtime_error("Batch manifest exceeds 1 MiB");
    std::ifstream stream(manifest);nlohmann::json doc;stream>>doc;
    if(!doc.is_object()||doc.size()!=1||!doc.contains("actions")||!doc["actions"].is_array()||doc["actions"].empty()||doc["actions"].size()>1000)
        throw std::runtime_error("Batch file must contain an actions array with 1..1000 actions");
    const auto base=fs::absolute(manifest).parent_path();
    std::vector<std::vector<std::string>> commands;
    std::set<fs::path,BatchPathLess> outputs;
    auto path=[&](const nlohmann::json& value) {
        auto p=utf8path(value.get<std::string>());if(p.empty())throw std::runtime_error("Empty batch path");
        return fs::weakly_canonical(fs::absolute(p.is_absolute()?p:base/p));
    };
    for(size_t index=0;index<doc["actions"].size();++index)try {
        const auto& action=doc["actions"][index];
        if(!action.is_object())throw std::runtime_error("Action must be an object");
        auto type=action.at("action").get<std::string>();
        std::set<std::string> keys{"action","output"};
        std::vector<std::string> argv{"tide-motion",type};
        unsigned variants=1;bool sheet_gif=false;
        bool momask_action=type=="inpaint"||type=="extend"||(type=="generate"&&action.value("backend",std::string("kimodo"))=="momask");
        bool deferred_source=false;
        if(momask_action) {
            if(action.value("backend",std::string("momask"))!="momask")throw std::runtime_error("inpaint backend must be momask");
            keys.insert({"backend","prompt","prompts","duration","seed","steps","guidance","residualGuidance","temperature","topkr","source","begin","end","ranges","variants","variantsGif","tide","threads","device","rig","inPlace","sampling","residualLayers","residualTemperature","residualTopkr","unconditional","history","historyFrames","chunkFrames","startPose","endPose","keyframes","effectors","path","endAt","poseMap","placement","follow","correction","correctionFrames","densePath","pathMargin","fitPoses","footsteps","attempts","loop","contactsOut","contextOverlapTokens","preserveContextResiduals","preserveRoot","editMode"});
            if(action.contains("prompts")) {
                if(action.contains("prompt")||action.contains("duration"))throw std::runtime_error("Use prompts or prompt/duration, not both");
                const auto& prompts=action["prompts"];if(!prompts.is_array()||prompts.empty()||prompts.size()>16)throw std::runtime_error("MoMask prompts must contain 1..16 segments");
                for(size_t i=0;i<prompts.size();++i){const auto& part=prompts[i];if(!part.is_object())throw std::runtime_error("MoMask prompt segment must be an object");for(const auto& item:part.items())if(item.key()!="text"&&item.key()!="prompt"&&item.key()!="duration")throw std::runtime_error("Unknown MoMask segment field");if(part.contains("text")&&part.contains("prompt"))throw std::runtime_error("Use prompt or text for a segment, not both");if(i)argv.push_back("--then");argv.push_back(part.at(part.contains("prompt")?"prompt":"text").get<std::string>());if(part.contains("duration")){if(!part["duration"].is_number())throw std::runtime_error("Segment duration must be numeric");argv.insert(argv.end(),{"--duration",part["duration"].dump()});}}
            } else argv.push_back(action.at("prompt").get<std::string>());
            argv.insert(argv.end(),{"--backend","momask"});
            if(action.contains("source")) {
                auto source=path(action["source"]);deferred_source=outputs.contains(source);
                if(!fs::exists(source)&&!deferred_source)throw std::runtime_error("MoMask source does not exist and is not an earlier batch output");
                argv.insert(argv.end(),{"--source",source.string()});
            }
            if(action.contains("history")){auto history=path(action["history"]);bool deferred=outputs.contains(history);if(!fs::exists(history)&&!deferred)throw std::runtime_error("MoMask history is not an existing file or earlier output");deferred_source|=deferred;argv.insert(argv.end(),{"--history",history.string()});}
            if(action.contains("ranges")){const auto& ranges=action["ranges"];if(!ranges.is_array()||ranges.empty()||ranges.size()>1000)throw std::runtime_error("MoMask ranges must contain 1..1000 pairs");for(const auto& range:ranges){if(!range.is_array()||range.size()!=2||!range[0].is_number()||!range[1].is_number())throw std::runtime_error("MoMask range must be [begin,end]");argv.insert(argv.end(),{"--range",range[0].dump(),range[1].dump()});}}
            for(auto pair:{std::pair{"guidance","--guidance"},std::pair{"residualGuidance","--residual-guidance"},std::pair{"temperature","--temperature"},std::pair{"topkr","--topkr"},std::pair{"begin","--begin"},std::pair{"end","--end"},std::pair{"residualLayers","--residual-layers"},std::pair{"residualTemperature","--residual-temperature"},std::pair{"residualTopkr","--residual-topkr"},std::pair{"historyFrames","--history-frames"},std::pair{"chunkFrames","--chunk-frames"},std::pair{"contextOverlapTokens","--context-overlap-tokens"}})if(action.contains(pair.first)) {
                if(!action[pair.first].is_number())throw std::runtime_error(std::string(pair.first)+" must be numeric");
                argv.insert(argv.end(),{pair.second,action[pair.first].dump()});
            }
            if(action.contains("variants")) {if(!action["variants"].is_number())throw std::runtime_error("variants must be numeric");variants=static_cast<unsigned>(integer(action["variants"].dump(),1,64,"variants"));argv.insert(argv.end(),{"--variants",std::to_string(variants)});}
            if(action.value("variantsGif",false)){sheet_gif=true;argv.push_back("--variants-gif");}
            if(action.value("inPlace",false))argv.push_back("--in-place");
            if(action.value("unconditional",false))argv.push_back("--unconditional");
            if(action.value("preserveContextResiduals",false))argv.push_back("--preserve-context-residuals");
            if(action.value("preserveRoot",false))argv.push_back("--preserve-root");
            for(auto pair:{std::pair{"device","--device"},std::pair{"rig","--rig"},std::pair{"tide","--tide"},std::pair{"sampling","--sampling"},std::pair{"editMode","--edit-mode"}})if(action.contains(pair.first))argv.insert(argv.end(),{pair.second,action[pair.first].get<std::string>()});
            auto numeric=[](const nlohmann::json& value){if(!value.is_number())throw std::runtime_error("MoMask constraint value must be numeric");return value.dump();};
            auto reference=[&](const nlohmann::json& value,const char* source_frame){
                std::string text;if(value.is_string())text=value.get<std::string>();else {if(!value.is_object())throw std::runtime_error("MoMask pose is a path or file/frame object");for(const auto& field:value.items())if(field.key()!="file"&&field.key()!=source_frame&&field.key()!="frame"&&field.key()!="effectors")throw std::runtime_error("Unknown MoMask pose field");text=value.at("file").get<std::string>();if(value.contains(source_frame))text+="@"+numeric(value.at(source_frame));}
                auto ref=pose_ref(text);auto file=path(ref.file);bool deferred=outputs.contains(file);if(!fs::exists(file)&&!deferred)throw std::runtime_error("MoMask pose file is not an earlier output or existing file");deferred_source|=deferred;return file.string()+(ref.frame?"@"+std::to_string(*ref.frame):"");
            };
            for(auto pair:{std::pair{"startPose","--start-pose"},std::pair{"endPose","--end-pose"}})if(action.contains(pair.first))argv.insert(argv.end(),{pair.second,reference(action[pair.first],"frame")});
            for(const char* key:{"keyframes","effectors","path"})if(action.contains(key)&&(!action[key].is_array()||action[key].empty()))throw std::runtime_error(std::string(key)+" must be a nonempty array");
            if(action.contains("keyframes"))for(const auto& item:action["keyframes"])argv.insert(argv.end(),{"--keyframe",numeric(item.at("frame")),reference(item,"sourceFrame")});
            if(action.contains("effectors"))for(const auto& item:action["effectors"]){std::string parts;if(item.at("effectors").is_array()){for(const auto& part:item["effectors"]){if(!parts.empty())parts+=',';parts+=part.get<std::string>();}}else parts=item["effectors"].get<std::string>();argv.insert(argv.end(),{"--effector",parts,numeric(item.at("frame")),reference(item,"sourceFrame")});}
            if(action.contains("path"))for(const auto& item:action["path"]){for(const auto& field:item.items())if(field.key()!="x"&&field.key()!="z"&&field.key()!="yaw"&&field.key()!="frame")throw std::runtime_error("Unknown MoMask path field");std::string point=numeric(item.at("x"))+","+numeric(item.at("z"));if(item.contains("yaw"))point+=","+numeric(item.at("yaw"));point+="@"+numeric(item.at("frame"));argv.insert(argv.end(),{"--path",point});}
            if(action.contains("endAt")){const auto& point=action["endAt"];if(!point.is_array()||point.size()<2||point.size()>3)throw std::runtime_error("MoMask endAt is [x,z,yaw?]");std::string text;for(const auto& v:point){if(!text.empty())text+=',';text+=numeric(v);}argv.insert(argv.end(),{"--end-at",text});}
            for(auto pair:{std::pair{"placement","--placement"},std::pair{"follow","--follow"},std::pair{"correction","--correction"}})if(action.contains(pair.first))argv.insert(argv.end(),{pair.second,action[pair.first].get<std::string>()});
            for(auto pair:{std::pair{"correctionFrames","--correction-frames"},std::pair{"pathMargin","--path-margin"},std::pair{"footsteps","--footsteps"},std::pair{"attempts","--attempts"}})if(action.contains(pair.first))argv.insert(argv.end(),{pair.second,numeric(action[pair.first])});
            if(action.value("densePath",false))argv.push_back("--dense-path");
            if(action.value("fitPoses",false))argv.push_back("--fit-poses");
            if(action.value("loop",false))argv.push_back("--loop");
            if(action.contains("poseMap"))argv.insert(argv.end(),{"--pose-map",path(action["poseMap"]).string()});
        } else if(type=="dart") {
            keys.insert({"plan","threads","device","batchSize"});auto plan=path(action.at("plan"));auto p=load_dart_plan(plan,true);
            for(const auto& input:{p.history_file,p.source_file,p.optimization?p.optimization->target_file:fs::path{}})if(!input.empty()&&!fs::exists(input)){auto parent=input;bool known=outputs.contains(parent);if(parent.extension()==".json"){parent.replace_extension();known=known||outputs.contains(parent);}if(!known)throw std::runtime_error("DART input does not exist and is not an earlier batch output: "+input.string());}
            argv.push_back(plan.string());
            if(action.contains("device")){auto d=action["device"].get<std::string>();if(d!="auto"&&d!="cpu"&&d!="gpu")throw std::runtime_error("DART device is auto, cpu or gpu");argv.insert(argv.end(),{"--device",d});}
            if(action.value("device",std::string("auto"))=="gpu"&&!dart_gpu_compiled())throw std::runtime_error("DART GPU: this build has no DirectML support");
            if(action.contains("batchSize")){if(!action["batchSize"].is_number_integer())throw std::runtime_error("batchSize must be integer");auto v=integer(action["batchSize"].dump(),1,8,"DART batchSize");argv.insert(argv.end(),{"--batch-size",std::to_string(static_cast<unsigned>(v))});}
        } else if(type=="generate") {
            auto selected_backend=action.value("backend",std::string("kimodo"));
            if(selected_backend=="ardy") {
                if(action.contains("model"))throw std::runtime_error("model selects Kimodo; omit it for backend ardy");
                if(action.contains("steps")&&(!action["steps"].is_number_integer()||action["steps"].get<int>()<1||action["steps"].get<int>()>10))throw std::runtime_error("ARDY steps must be 1..10");
                if(action.value("device",std::string("auto"))=="gpu"&&!ardy_gpu_compiled())throw std::runtime_error("ARDY GPU: this build has no DirectML support");
                if(action.contains("transition"))throw std::runtime_error("ARDY does not take transition");
            } else if(action.contains("history")||action.contains("historyFrames"))throw std::runtime_error("history requires backend ardy");
            keys.insert({"prompt","prompts","duration","seed","steps","transition","startPose","endPose","endAt","keyframes","effectors","path","placement","follow","correct","fitPoses","footsteps","attempts",
                "variants","variantsGif","tide","guidance","constraintGuidance","device","threads","correctionFrames","inPlace","model","rig","densePath","correction","poseMap","backend","history","historyFrames"});
            if(action.contains("backend")) {auto b=action["backend"].get<std::string>();if(b!="kimodo"&&b!="ardy")throw std::runtime_error("backend is kimodo or ardy");argv.insert(argv.end(),{"--backend",b});}
            if(action.contains("history"))argv.insert(argv.end(),{"--history",path(action["history"]).string()});
            if(action.contains("historyFrames")) {if(!action["historyFrames"].is_number_integer())throw std::runtime_error("historyFrames must be integer");auto h=action["historyFrames"].get<int>();if(h<4||h>240||h%4)throw std::runtime_error("historyFrames must be 4..240, multiple of four");argv.insert(argv.end(),{"--history-frames",std::to_string(h)});}
            if(action.contains("fitPoses")&&action["fitPoses"].get<bool>())argv.push_back("--fit-poses");
            if(action.contains("placement")) {
                auto value=action["placement"].get<std::string>();
                if(value!="auto"&&value!="world"&&value!="free")throw std::runtime_error("placement is auto, world or free");
                argv.insert(argv.end(),{"--placement",value});
            }
            if(action.contains("follow")) {
                auto value=action["follow"].get<std::string>();
                if(value!="world"&&value!="ends")throw std::runtime_error("follow is world or ends");
                argv.insert(argv.end(),{"--follow",value});
            }
            if(action.contains("correct")&&!action["correct"].get<bool>())argv.push_back("--no-correction");
            if(action.contains("correction")) {auto value=action["correction"].get<std::string>();if(value!="tide"&&value!="kimodo"&&value!="none")throw std::runtime_error("correction is tide, kimodo or none");argv.insert(argv.end(),{"--correction",value});}
            if(action.contains("rig")) {auto value=action["rig"].get<std::string>();if(value!="soma30"&&value!="soma77")throw std::runtime_error("rig is soma30 or soma77");argv.insert(argv.end(),{"--rig",value});}
            if(action.contains("densePath")&&action["densePath"].get<bool>())argv.push_back("--dense-path");
            if(action.contains("poseMap")) {auto m=path(action["poseMap"]);(void)load_mapping(m);argv.insert(argv.end(),{"--pose-map",m.string()});}
            auto numeric=[](const nlohmann::json& value,const char* label) {
                if(!value.is_number())throw std::runtime_error(std::string(label)+" must be numeric");
                return value.dump();
            };
            for(auto pair:{std::pair{"footsteps","--footsteps"},std::pair{"attempts","--attempts"}})if(action.contains(pair.first)) {
                auto text=numeric(action[pair.first],pair.first);integer(text,pair.first==std::string("attempts")?1:0,64,pair.first);argv.insert(argv.end(),{pair.second,text});
            }
            if(action.contains("variants")) {auto text=numeric(action["variants"],"variants");variants=static_cast<unsigned>(integer(text,1,64,"variants"));argv.insert(argv.end(),{"--variants",text});}
            if(action.contains("attempts")&&!action.contains("footsteps")&&!action.value("fitPoses",false))throw std::runtime_error("attempts requires footsteps or fitPoses");
            if(action.contains("footsteps")||action.value("fitPoses",false)) {
                if(variants>1)throw std::runtime_error("selection takes one clip; use variants separately");
                const auto first=action.contains("seed")?static_cast<uint64_t>(integer(numeric(action["seed"],"seed"),0,4294967295.0,"seed")):uint64_t{0};
                const auto limit=action.value("attempts",unsigned{8});
                if(first+limit-1>4294967295ULL)throw std::runtime_error("seed plus attempts passes 4294967295");
            }
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
                    if(item.contains("duration")) {auto text=numeric(item["duration"],"duration");number(text,0.2,selected_backend=="ardy"?600:10,"duration");argv.insert(argv.end(),{"--duration",text});}
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
            if(action.contains("endAt")) {
                if(!action.contains("endPose"))throw std::runtime_error("endAt requires endPose");
                const auto& at=action["endAt"];
                if(!at.is_array()||(at.size()!=2&&at.size()!=3))throw std::runtime_error("endAt is [X,Z] or [X,Z,YAW]");
                std::string text;
                for(const auto& v:at) {if(!text.empty())text+=',';text+=numeric(v,"endAt");}
                long f=0;(void)path_point(text+"@-1",&f);
                argv.insert(argv.end(),{"--end-at",text});
            }
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
        else if(type=="connect") { argv.push_back(path(action.at("from")).string());argv.push_back(path(action.at("to")).string());keys.insert({"from","to","tokens","targetPoses","contactsOut","contactThreshold","joined","offset","fromClip","toClip","threads"}); }
        else if(type=="pose-sequence") {
            auto poses=path(action.at("recipe"));(void)load_pose_sequence(poses);
            argv.push_back(poses.string());keys.insert({"recipe","threads","contactsOut","contactThreshold"});
        }
        else if(type=="equipment") {
            keys.insert({"source","plan","correctGrips","strict","clip","bodyOutput"});
            argv.push_back("bake");argv.push_back(path(action.at("source")).string());
            auto equipment=path(action.at("plan"));(void)load_equipment_plan(equipment);
            argv.insert(argv.end(),{"--plan",equipment.string()});
            if(action.contains("bodyOutput"))argv.insert(argv.end(),{"--body-output",path(action["bodyOutput"]).string()});
            for(const char* flag:{"correctGrips","strict"})if(action.contains(flag)) {
                if(!action[flag].is_boolean())throw std::runtime_error(std::string(flag)+" must be boolean");
                if(action[flag].get<bool>())argv.push_back(std::string(flag)=="strict"?"--strict":"--correct-grips");
            }
            if(action.contains("clip"))argv.insert(argv.end(),{"--clip",action["clip"].get<std::string>()});
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
        else throw std::runtime_error("Action must be generate, inpaint, extend, connect, pose-sequence, dart, retarget, equipment, cleanup or mirror");
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
        if(action.contains("contactsOut")) {
            auto plan=path(action["contactsOut"]);
            if(plan.extension()!=".json")throw std::runtime_error("contactsOut must be a .json path");
            written.push_back(plan);argv.insert(argv.end(),{"--contacts-out",plan.string()});
        }
        if(type=="equipment"&&action.contains("bodyOutput")) {
            auto body=path(action["bodyOutput"]);if(body.extension()!=".glb")throw std::runtime_error("Equipment bodyOutput must be .glb");written.push_back(body);
        }
        for(const auto& file:written){if(fs::exists(file)||fs::exists(file.string()+".part")||fs::exists(file.string()+".json")||!outputs.insert(file).second)throw std::runtime_error("Output must be a new, distinct path: "+file.string());if(file.extension()==".glb"&&!outputs.insert(fs::path(file.string()+".json")).second)throw std::runtime_error("Output recipe must have a distinct path: "+file.string());}
        argv.insert(argv.end(),{"-o",output.string()});
        if(type=="generate"||type=="inpaint"||type=="extend"||type=="connect"||type=="pose-sequence"||type=="dart")argv.insert(argv.end(),{"--models",models.string()});
        if(action.contains("tokens")&&action["tokens"].is_array()) {
            const auto& range=action["tokens"];
            if(range.size()!=2||!range[0].is_number()||!range[1].is_number())throw std::runtime_error("tokens is an integer or [min, max]");
            auto text=range[0].dump()+".."+range[1].dump();(void)parse_bricks_tokens(text);
            argv.insert(argv.end(),{"--tokens",text});
        }
        for(const char* key:{"targetPoses","contactThreshold"})if(action.contains(key)) {
            if(!action[key].is_number())throw std::runtime_error(std::string(key)+" must be numeric");
            auto text=action[key].dump();
            if(std::string(key)=="targetPoses") {integer(text,1,4,key);argv.insert(argv.end(),{"--target-poses",text});}
            else {number(text,0,1,key);argv.insert(argv.end(),{"--contact-threshold",text});}
        }
        for(const char* key:{"duration","seed","steps","tokens","threads"})if(action.contains(key)&&!action[key].is_array()) {
            if(!action[key].is_number())throw std::runtime_error(std::string(key)+" must be numeric");
            auto text=action[key].dump();
            if(std::string(key)=="duration")number(text,0.2,(action.value("backend",std::string("kimodo"))=="ardy"||momask_action)?600:10,key);
            else if(std::string(key)=="tokens")(void)parse_bricks_tokens(text);
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
        if(momask_action)run_momask(argv,context,true,deferred_source);
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
    // Consecutive connects share one MotionBricks run (up to bricks_batch_limit) when
    // they use the same threads and none reads an output of an earlier one in the run.
    auto option=[](const std::vector<std::string>& argv,const char* name) {
        for(size_t k=0;k+1<argv.size();++k)if(argv[k]==name)return argv[k+1];
        return std::string{};
    };
    size_t batched_until=0;
    for(size_t i=0;i<commands.size();++i) {
        if(commands[i][1]=="dart"&&context->dart_ready.empty()) {
            auto threads=option(commands[i],"--threads"),device=option(commands[i],"--device"),limit=option(commands[i],"--batch-size");
            size_t end=i;std::vector<DartPlan> plans;size_t maximum=limit.empty()?8:static_cast<size_t>(std::stoul(limit));
            for(;end<commands.size()&&end-i<maximum&&commands[end][1]=="dart"&&option(commands[end],"--threads")==threads&&option(commands[end],"--device")==device&&option(commands[end],"--batch-size")==limit;++end){auto p=load_dart_plan(utf8path(commands[end][2]),true);if((!p.history_file.empty()&&!fs::exists(p.history_file))||(!p.source_file.empty()&&!fs::exists(p.source_file))||(p.optimization&&!p.optimization->target_file.empty()&&!fs::exists(p.optimization->target_file)))break;if(p.optimization&&!plans.empty())break;plans.push_back(std::move(p));if(plans.back().optimization){++end;break;}}
            if(plans.empty())throw std::runtime_error("DART input is not available at this batch action");
            std::cerr<<"DART neural batch actions "<<i+1<<"-"<<end<<" ("<<plans.size()<<" independent streams)\n";
            auto results=dart_session(models,threads.empty()?0:static_cast<unsigned>(std::stoul(threads)),device.empty()?"auto":device,context).generate(plans);
            for(size_t k=0;k<results.size();++k)context->dart_ready.push_back({fs::absolute(utf8path(option(commands[i+k],"-o"))),std::move(results[k])});
        }
        if(i>=batched_until&&commands[i][1]=="connect") {
            std::set<fs::path> run_outputs;const auto threads=option(commands[i],"--threads");size_t j=i;
            context->bricks_prepare=true;context->bricks_pending.clear();
            for(;j<commands.size()&&j-i<bricks_batch_limit&&commands[j][1]=="connect"&&option(commands[j],"--threads")==threads;++j) {
                if(run_outputs.contains(fs::absolute(commands[j][2]))||run_outputs.contains(fs::absolute(commands[j][3])))break;
                // A failing action runs normally later, after the earlier outputs exist.
                try {main_impl(commands[j],context);}catch(const std::exception&){break;}
                run_outputs.insert(fs::absolute(option(commands[j],"-o")));
            }
            context->bricks_prepare=false;
            if(context->bricks_pending.size()>1) {
                std::vector<BricksModel::Request> requests;
                for(const auto& p:context->bricks_pending)requests.push_back({&p.from,&p.to,p.options});
                unsigned t=threads.empty()?0u:static_cast<unsigned>(std::stoul(threads));
                std::cerr<<"Batch actions "<<i+1<<"-"<<j<<": one MotionBricks run\n";
                try {
                    auto bridges=bricks_session(models,t,context).generate(requests);
                    for(size_t k=0;k<bridges.size();++k)context->bricks_ready.push_back({context->bricks_pending[k].output,std::move(bridges[k]),bridges.size(),k});
                } catch(const std::exception& e) {throw std::runtime_error("Batch actions "+std::to_string(i+1)+"-"+std::to_string(j)+" failed: "+e.what()+"; completed outputs have been kept");}
                batched_until=j;
            }
            context->bricks_pending.clear();
        }
        std::cerr<<"Batch action "<<i+1<<"/"<<commands.size()<<"\n";
        try {int code=main_impl(commands[i],context);if(code)return code;}
        catch(const std::exception& e){throw std::runtime_error("Batch action "+std::to_string(i+1)+" failed: "+e.what()+"; completed outputs have been kept");}
    }
    return 0;
}
int main_impl(const std::vector<std::string>& args,RunContext* context) {
    if(args.size()>1&&args[1]=="equipment")return equipment_command(args);
    RunContext owned;if(!context)context=&owned;
    if(args.size()>1) {
        bool selected=false;
        if(args[1]=="generate")for(size_t i=2;i+1<args.size();++i)if(args[i]=="--backend"&&args[i+1]=="momask")selected=true;
        if(selected||args[1]=="inpaint"||args[1]=="extend"||args[1]=="momask")return run_momask(args,context);
    }
    if(args.size()<2||args[1]=="--help"||args[1]=="-h") { help();return 0; }
    auto unknown=[](const std::string& name){return std::runtime_error("Unknown command: "+name+"; tide-motion --help lists the commands");};
    if(args[1]=="help") { if(args.size()>2&&!command_help(args[2]))throw unknown(args[2]);help(args.size()>2?args[2]:"");return 0; }
    if(args[1]=="--version") { std::cout<<"tide-motion "<<MOTION_VERSION<<"\n";return 0; }
    if(args[1]=="review") {
        for(size_t i=2;i<args.size();++i)if(args[i]=="--help"||args[i]=="-h"){help("review");return 0;}
        return review_command(args);
    }
    if(args[1]=="upgrade") {
        for(size_t i=2;i<args.size();++i)if(args[i]=="--help"||args[i]=="-h") {help("upgrade");return 0;}
        return upgrade(args,MOTION_VERSION);
    }
    if(!command_help(args[1]))throw unknown(args[1]);
    std::string command=args[1],clip_name,from_name,to_name,tide="tide";
    for(size_t i=2;i<args.size();++i)if(args[i]=="--help"||args[i]=="-h") {help(command);return 0;}
    fs::path models,output,capture,gif,target_file,map_file,character_file,texture_file,contacts_file;bool mannequin=false;
    PreviewProp prop;std::vector<PreviewProp> props;PreviewCamera camera;fs::path setup_file,save_setup;
    if(command=="generate"||command=="connect"||command=="pose-sequence"||command=="models"||command=="batch"||command=="dart")models=model_directory();
    double duration=3;uint64_t seed=0;unsigned steps=50,transition=5;int frames=0;
    BricksRequest bricks_request;std::string tokens_text;fs::path contacts_out;float contact_threshold=0.5f;
    bool accept=false,joined=false,skeleton=false,to_soma=false,correct=true,dense_path=false;Vec3 offset{};
    std::string correction,rig="soma30";fs::path pose_map_file;
    Placement placement=Placement::automatic;std::string placement_name="auto",follow="world";
    unsigned variants=1,threads=0;int correction_frames=10;bool variants_gif=false,in_place=false;
    Guidance guidance;std::string device="auto",backend="kimodo",model_version="v1.1";fs::path history_file;unsigned history_frames=240;
    bool fit_poses=false;int wanted_footsteps=-1;unsigned attempts=8;
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
        else if(a=="--backend") {backend=value();if(backend!="kimodo"&&backend!="ardy")throw std::runtime_error("--backend is kimodo or ardy");}
        else if(a=="--history")history_file=utf8path(value());
        else if(a=="--history-frames") {history_frames=static_cast<unsigned>(integer(value(),4,240,"history frames"));if(history_frames%4)throw std::runtime_error("history frames must be a multiple of four");}
        else if(a=="-o"||a=="--output")output=utf8path(value());
        else if(a=="--duration") {duration=number(value(),0.2,600,"duration");segments.back().second=duration;}
        else if(a=="--then")segments.push_back({value(),3.0});
        else if(a=="--transition")transition=static_cast<unsigned>(integer(value(),1,60,"transition"));
        else if(a=="--start-pose")pose_specs.push_back({"start-pose",0,value()});
        else if(a=="--end-pose")pose_specs.push_back({"end-pose",-1,value()});
        else if(a=="--end-at") {long f=0;auto p=path_point(value()+"@-1",&f);path_specs.push_back({p,f});}
        else if(a=="--keyframe") {long f=frame_index(value(),"keyframe frame");pose_specs.push_back({"keyframe",f,value()});}
        else if(a=="--effector") {unsigned bits=parse_effectors(value());long f=frame_index(value(),"effector frame");pose_specs.push_back({"effector",f,value(),bits});}
        else if(a=="--path") {long f=0;auto p=path_point(value(),&f);path_specs.push_back({p,f});}
        else if(a=="--gif")gif=utf8path(value());
        else if(a=="--follow") {follow=value();if(follow!="world"&&follow!="ends")throw std::runtime_error("--follow is world or ends");}
        else if(a=="--placement") {
            placement_name=value();
            if(placement_name=="auto")placement=Placement::automatic;
            else if(placement_name=="world")placement=Placement::world;
            else if(placement_name=="free")placement=Placement::free;
            else throw std::runtime_error("--placement is auto, world or free");
        }
        else if(a=="--fit-poses")fit_poses=true;
        else if(a=="--footsteps")wanted_footsteps=static_cast<int>(integer(value(),0,64,"footsteps"));
        else if(a=="--attempts")attempts=static_cast<unsigned>(integer(value(),1,64,"attempts"));
        else if(a=="--no-correction")correct=false;
        else if(a=="--correction") {correction=value();if(correction!="tide"&&correction!="kimodo"&&correction!="none")throw std::runtime_error("--correction is tide, kimodo or none");}
        else if(a=="--rig") {rig=value();if(rig!="soma30"&&rig!="soma77")throw std::runtime_error("--rig is soma30 or soma77");}
        else if(a=="--dense-path")dense_path=true;
        else if(a=="--pose-map")pose_map_file=utf8path(value());
        else if(a=="--correction-frames")correction_frames=static_cast<int>(integer(value(),0,60,"correction frames"));
        else if(a=="--in-place")in_place=true;
        else if(a=="--variants")variants=static_cast<unsigned>(integer(value(),1,64,"variants"));
        else if(a=="--variants-gif")variants_gif=true;
        else if(a=="--guidance")guidance.text=static_cast<float>(number(value(),0,10,"guidance"));
        else if(a=="--constraint-guidance")guidance.constraint=static_cast<float>(number(value(),0,10,"constraint guidance"));
        else if(a=="--model") {model_version=value();if(model_version!="v1.1"&&model_version!="v1.2")throw std::runtime_error("--model is v1.1 or v1.2 (smooth-root SOMA-RP)");}
        else if(a=="--device") {device=value();if(device!="auto"&&device!="gpu"&&device!="cpu")throw std::runtime_error("--device is auto, gpu or cpu");}
        else if(a=="--threads")threads=static_cast<unsigned>(integer(value(),1,1024,"threads"));
        else if(a=="--batch-size")(void)integer(value(),1,8,"DART batch size");
        else if(a=="--steps")steps=static_cast<unsigned>(integer(value(),1,1000,"steps"));
        else if(a=="--seed")seed=static_cast<uint64_t>(integer(value(),0,4294967295.0,"seed"));
        else if(a=="--frames")frames=static_cast<int>(integer(value(),1,18000,"frames"));
        else if(a=="--tokens") {tokens_text=value();auto r=parse_bricks_tokens(tokens_text);bricks_request.min_tokens=r.min_tokens;bricks_request.max_tokens=r.max_tokens;}
        else if(a=="--target-poses")bricks_request.target_poses=static_cast<int>(integer(value(),1,4,"target poses"));
        else if(a=="--contacts-out")contacts_out=utf8path(value());
        else if(a=="--contact-threshold")contact_threshold=static_cast<float>(number(value(),0,1,"contact threshold"));
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
        else if(a=="--prop") {
            if(!prop.mesh.empty()) {props.push_back(prop);prop={};}
            prop.mesh=utf8path(value());
        }
        else if(a=="--prop-bone")prop.bone=value();
        else if(a=="--prop-offset")prop.offset=triplet(value(),100,"--prop-offset");
        else if(a=="--prop-rotation")prop.rotation=triplet(value(),3600,"--prop-rotation");
        else if(a=="--first-person")camera.enabled=true;
        else if(a=="--camera-bone")camera.bone=value();
        else if(a=="--camera-mode")camera.mode=value();
        else if(a=="--camera-offset")camera.offset=triplet(value(),100,"--camera-offset");
        else if(a=="--camera-rotation")camera.rotation=triplet(value(),3600,"--camera-rotation");
        else if(a=="--camera-fov")camera.fov=static_cast<float>(number(value(),10,150,"camera fov"));
        else if(a=="--camera-near")camera.near_plane=static_cast<float>(number(value(),0.001,1,"camera near"));
        else if(a=="--setup")setup_file=utf8path(value());
        else if(a=="--save-setup")save_setup=utf8path(value());
        else if(a=="--to-soma")to_soma=true;
        else if(a=="--target")target_file=utf8path(value());
        else if(a=="--map")map_file=utf8path(value());
        else if(a.starts_with('-'))throw std::runtime_error("Unknown option: "+a);
        else pos.push_back(a);
    }
    if(!prop.mesh.empty())props.push_back(prop);
    std::set<std::string> allowed;
    if(command=="generate")allowed={"--models","-o","--output","--duration","--seed","--steps","--then","--transition","--start-pose","--end-pose","--end-at","--keyframe","--effector","--path","--placement","--follow","--fit-poses","--footsteps","--attempts","--no-correction",
        "--correction","--rig","--dense-path","--pose-map",
        "--correction-frames","--in-place","--variants","--variants-gif","--tide","--guidance","--constraint-guidance","--device","--threads","--model","--backend","--history","--history-frames"};
    else if(command=="connect")allowed={"--models","-o","--output","--tokens","--target-poses","--contacts-out","--contact-threshold","--joined","--offset","--from-clip","--to-clip","--threads"};
    else if(command=="mirror")allowed={"-o","--output","--clip"};
    else if(command=="retarget")allowed={"-o","--output","--target","--map","--to-soma","--clip"};
    else if(command=="inspect")allowed={"--clip","--skeleton"};
    else if(command=="preview")allowed={"--clip","--tide","--capture","--gif","--frames","--mannequin","--skeleton","--character","--map","--texture","--prop","--prop-bone","--prop-offset","--prop-rotation","--first-person","--camera-bone","--camera-mode","--camera-offset","--camera-rotation","--camera-fov","--camera-near","--setup","--save-setup"};
    else if(command=="cleanup")allowed={"-o","--output","--contacts","--map","--clip"};
    else if(command=="models")allowed={"--models","--accept-licenses"};
    else if(command=="pose-sequence")allowed={"-o","--output","--models","--threads","--contacts-out","--contact-threshold"};
    else if(command=="dart")allowed={"-o","--output","--models","--threads","--device","--batch-size"};
    else if(command=="batch")allowed={"--models"};
    for(const auto& flag:flags)if(!allowed.contains(flag))throw std::runtime_error(command+" does not take "+flag);
    if(command=="batch") {
        if(pos.size()!=1)throw std::runtime_error("batch takes one jobs.json manifest");
        return batch(utf8path(pos[0]),models,context);
    }
    if(command=="models") {
        if(pos.size()==2&&pos[0]=="import-momask")return momask_main({args[0],"momask","install",pos[1],"--models",(models/"momask").string()});
        if(pos.size()==2&&pos[1]=="momask") {
            if(pos[0]=="check")return momask_main({args[0],"momask","check","--models",(models/"momask").string()});
            if(pos[0]=="install")throw std::runtime_error("MoMask weights are manual-only; use tide-motion models import-momask BUNDLE_DIRECTORY");
        }
        if(pos.size()==2&&pos[0]=="import-ardy") {
            if(!accept)throw std::runtime_error("Read licenses/NVIDIA-ARDY-MODEL.txt and META-LLAMA-3.txt, then pass --accept-licenses");
            auto source=utf8path(pos[1]);check_ardy_models(source);
            if(!fs::exists(source/"LICENSE")||!fs::exists(source/"NOTICE"))throw std::runtime_error("ARDY conversion must retain LICENSE and NOTICE");
            auto destination=models/"ardy";
            if(fs::exists(destination))throw std::runtime_error("ARDY model destination already exists");
            fs::create_directories(destination);
            for(const char* name:{"denoiser.onnx","encoder.onnx","decoder.onnx","manifest.json","LICENSE","NOTICE"})fs::copy_file(source/name,destination/name);
            check_ardy_models(destination);std::cout<<"ARDY models imported into "<<destination.string()<<"\n";return 0;
        }
        if(pos.empty()||pos.size()>2||(pos[0]!="install"&&pos[0]!="check"))throw std::runtime_error("Use models install/check [all|kimodo|kimodo-v1.2|text|bricks|ardy|dart], or models import-ardy DIR --accept-licenses");
        std::string selection=pos.size()==2?pos[1]:"all";
        if(pos[0]=="check"&&selection=="ardy")check_ardy_models(models/"ardy");
        else if(pos[0]=="check")check_models(models,selection);
        else {
            if(!accept)throw std::runtime_error(selection=="dart"?"Read licenses/DART-SOURCE.txt and OPENAI-CLIP.txt, then pass --accept-licenses to download DART models":"Read licenses/NVIDIA-KIMODO-MODEL.txt, NVIDIA-OPEN-MODEL.txt and META-LLAMA-3.txt, then pass --accept-licenses to download models");
            setup_models(models,selection);
        }
        return 0;
    }
    if(command=="inspect"||command=="preview") {
        if(pos.size()!=1)throw std::runtime_error(command+" takes one GLB or FBX file");
        auto path=utf8path(pos[0]);
        if(command=="preview") {
            PreviewOptions options;
            if(!setup_file.empty()) {
                for(const auto& flag:flags)if(flag.starts_with("--prop")||flag.starts_with("--camera")||flag=="--first-person")
                    throw std::runtime_error("--setup cannot be combined with prop or camera options; edit the setup JSON");
                options=load_preview_setup(setup_file);props=options.props;
            } else {
                for(const auto& flag:flags)if(flag.starts_with("--camera")&&!camera.enabled)
                    throw std::runtime_error("Camera options go with --first-person");
                options.camera=camera;
            }
            options.tide=tide;options.capture=capture;options.gif=gif;options.frames=frames;
            auto clip=load_glb(path,clip_name,false);
            // SOMA-77 previews as its exact SOMA-30 subset on bodies; --skeleton draws all 77.
            if(is_soma77(clip)&&!skeleton)clip=soma77_to_30(clip);
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
            if(!props.empty()) {
                if(options.body==PreviewBody::skeleton)throw std::runtime_error("--prop is drawn on a body; use --mannequin or --character");
                // Resolve names on the drawn rig, including nodes without skin weights.
                std::vector<std::string> bones;
                if(options.body==PreviewBody::character)bones=node_names(character_file);
                else for(const auto& joint:clip.joints)bones.push_back(joint.name);
                for(const auto& attachment:props) {
                    if(attachment.bone.empty())throw std::runtime_error("--prop needs --prop-bone BONE (a bone of --character, or a SOMA bone such as RightHand on the mannequin)");
                    if(std::find(bones.begin(),bones.end(),attachment.bone)==bones.end())
                        throw std::runtime_error("--prop-bone '"+attachment.bone+"' is not a bone of "+(options.body==PreviewBody::character?character_file.string():std::string("the SOMA mannequin")));
                }
                options.props=props;
            } else for(const char* flag:{"--prop-bone","--prop-offset","--prop-rotation"})
                if(std::find(flags.begin(),flags.end(),flag)!=flags.end())throw std::runtime_error(std::string(flag)+" goes with --prop");
            validate_preview_setup(options);
            if(options.camera.enabled) {
                auto rig=preview_rig(clip,options);bool found=false;
                for(const auto& joint:rig.joints)if(joint.name==options.camera.bone)found=true;
                if(!found)throw std::runtime_error("Camera bone not found: "+options.camera.bone);
            }
            if(!save_setup.empty())save_preview_setup(options,save_setup);
            preview(clip,path,options);
            return 0;
        }
        if(skeleton&&!clip_name.empty())throw std::runtime_error("inspect --skeleton does not take --clip");
        auto clip=skeleton?load_rig_glb(path):load_glb(path,clip_name,false);
        {
            bool canonical=true;try{require_soma(clip);}catch(const std::exception&){canonical=false;}
            nlohmann::json info={{"name",clip.name},{"rig",canonical?"soma30":is_soma77(clip)?"soma77":"custom"},{"joints",clip.joints.size()},{"frames",clip.frames()},{"fps",clip.fps},{"duration",clip.duration()}};
            if(!skeleton) {
                auto travel=root_travel(clip);
                if(canonical) {
                    const auto events=footfalls(clip);
                    info["footfalls"]={{"rule","soma-travelling-footfalls-v1"},{"count",events.size()},{"events",nlohmann::json::array()}};
                    for(const auto& e:events)info["footfalls"]["events"].push_back({{"foot",e.foot?"right-foot":"left-foot"},{"lift",e.lift},{"landing",e.landing},{"distanceMetres",e.distance},{"heightMetres",e.height}});
                }
                info["rootTravel"]={{"displacementMetres",travel.displacement},{"pathMetres",travel.distance},{"peakSpeedMetresPerSecond",travel.peak_speed}};
            }
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
    if(!contacts_out.empty()) {
        if(contacts_out.extension()!=".json")throw std::runtime_error("--contacts-out must be a .json path");
        if(fs::exists(contacts_out)||fs::absolute(contacts_out)==fs::absolute(output.string()+".json"))throw std::runtime_error("Contact plan must be a new path, not the output recipe: "+contacts_out.string());
    }
    // Model-predicted contacts, recorded in the recipe and optionally as a cleanup plan.
    auto contact_record=[&](const FootContacts& predicted,size_t offset,nlohmann::json& recipe) {
        auto planted=planted_contacts(predicted,contact_threshold,offset);
        nlohmann::json frames=nlohmann::json::array(),intervals=nlohmann::json::array();
        for(const auto& c:predicted)frames.push_back({std::round(c[0]*1000)/1000,std::round(c[1]*1000)/1000,std::round(c[2]*1000)/1000,std::round(c[3]*1000)/1000});
        for(const auto& c:planted)intervals.push_back({{"foot",c.foot?"right-foot":"left-foot"},{"start",c.start},{"end",c.end}});
        recipe["predictedFootContacts"]={{"order",{"left-heel","left-toe","right-heel","right-toe"}},{"firstFrame",offset},{"frames",frames},
            {"threshold",contact_threshold},{"plantedHeels",intervals}};
        if(contacts_out.empty())return;
        if(planted.empty()) {std::cerr<<"No predicted planted heel lasts three frames; no contact plan written\n";recipe["predictedFootContacts"]["plan"]=nullptr;return;}
        std::ofstream plan(contacts_out);plan<<nlohmann::json{{"contacts",intervals}}.dump(2)<<"\n";plan.close();
        if(!plan)throw std::runtime_error("Could not write the contact plan: "+contacts_out.string());
        recipe["predictedFootContacts"]["plan"]=contacts_out.string();
        std::cout<<"Saved "<<contacts_out.string()<<" ("<<planted.size()<<" predicted contacts)\n";
    };
    if(command=="dart") {
        if(pos.size()!=1)throw std::runtime_error("dart takes one JSON plan");
        auto plan=load_dart_plan(utf8path(pos[0]));
        if(device=="gpu"&&!dart_gpu_compiled())throw std::runtime_error("DART GPU: this build has no DirectML support");
        DartResult result;
        if(!context->dart_ready.empty()){auto ready=std::move(context->dart_ready.front());context->dart_ready.pop_front();if(ready.output!=fs::absolute(output))throw std::runtime_error("DART batch output order mismatch");result=std::move(ready.result);}
        else result=dart_session(models,threads,device,context).generate(plan);
        result.clip.name=output.stem().string();save_glb(result.clip,output);
        nlohmann::json recipe={{"tool","tide-motion"},{"version",MOTION_VERSION},{"command","dart"},{"authoring",plan.recipe},{"threads",threads?threads:4},{"device",result.metrics.at("device")},{"accelerator",result.metrics.at("device")=="gpu"?"directml":"cpu"},{"fps",30},{"frames",result.clip.frames()},{"metrics",result.metrics},{"dartState",result.history},{"dartInitialState",result.initial_history},
            {"sourceRevision","bb67ae1ed6ce051080468bf15bc6e54a6c3f8417"},{"weightsSha256","6c186ea7807efe7cef02e639726bb94040f1c44fb8beef0621c667440d7b7877"},{"sampler",{{"prediction","x0"},{"schedule","cosine10"},{"method",plan.sampler},{"timesteps",plan.timesteps},{"eta",plan.eta},{"guidance",plan.guidance},{"noiseScale",plan.noise_scale},{"zeroNoise",plan.zero_noise},{"policyMode",plan.policy_mode},{"fixFloor",plan.fix_floor},{"groundHeight",plan.ground_height},{"seedMode",plan.seed_mode}}},
            {"randomGenerator","std::mt19937 + std::normal_distribution<float>; compiler-dependent normal transform"},
            {"checkpoints",{{"denoiser","e6c550641deaaccb6313f3529a4ca6602eb00d675dcff3aa47217825b5a43615"},{"decoder","b818cc8704ad9b408c745cac48943c99bf5402ffca521b25ffaf6639a5d5d276"},{"policy","81af5cda56e519afadb7bdf74ba82322cc79f903c1cbea151b05fdc71f460a0e"},{"text","40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"},{"normalizer","cb442539380d15a5763b799ba063d9b0b6574a751a03ae0c86d4a6f24bff5bd4"}}}};
        recipe["inputs"]=nlohmann::json::array();for(const auto& input:{plan.history_file,plan.source_file,plan.source_map_file,plan.optimization?plan.optimization->target_file:fs::path{},plan.optimization?plan.optimization->target_map_file:fs::path{},plan.optimization&&plan.optimization->scene?plan.optimization->scene->file:fs::path{}})if(!input.empty())recipe["inputs"].push_back({{"path",input.string()},{"bytes",fs::file_size(input)},{"sha256",sha256_file(input)}});
        std::ofstream metadata(output.string()+".json");metadata<<recipe.dump(2)<<"\n";metadata.close();if(!metadata)throw std::runtime_error("Animation saved but could not write DART recipe");
        std::cout<<"Saved "<<output.string()<<" ("<<result.clip.frames()<<" frames)\n";return 0;
    }
    if(command=="pose-sequence") {
        if(pos.size()!=1)throw std::runtime_error("pose-sequence takes one JSON recipe");
        auto plan=load_pose_sequence(utf8path(pos[0]));
        if(!context->bricks||context->bricks_threads!=threads) {
            if(!context->bricks)check_models(models,"bricks");
            context->bricks=std::make_unique<BricksModel>(models/"motionbricks.onnx",threads);context->bricks_threads=threads;
        }
        auto authored=generate_pose_sequence(plan,*context->bricks);auto& made=authored.clip;made.name=output.stem().string();
        save_glb(made,output);
        nlohmann::json recipe={{"tool","tide-motion"},{"version",MOTION_VERSION},{"command",command},{"authoring",authored.recipe},{"threads",threads},{"fps",30},{"frames",made.frames()},
            {"motionbricksRevision","54fc563aeaaf819a753d72cc9fdc6713989cdd43"}};
        contact_record(authored.contacts,0,recipe);
        std::ofstream metadata(output.string()+".json");metadata<<recipe.dump(2)<<"\n";metadata.close();
        if(!metadata)throw std::runtime_error("Animation saved but could not write its recipe");
        std::cout<<"Saved "<<output.string()<<" ("<<made.frames()<<" frames)\n";return 0;
    }
    Clip clip;Bridge bridge;size_t contact_offset=0,batch_requests=1,batch_index=0;
    nlohmann::json retarget_recipe,generate_recipe,mirror_recipe;
    std::vector<size_t> connect_pins;
    // Store generation settings alongside the asset; no need to rerun inference at build time.
    auto recipe_for=[&](const Clip& made,uint64_t used_seed) {
        return nlohmann::json{{"tool","tide-motion"},{"version",MOTION_VERSION},{"command",command},{"inputs",pos},{"rig","soma30"},{"fps",30},{"frames",made.frames()},{"requestedDuration",duration},{"seed",used_seed},{"steps",steps},{"tokens",tokens_text.empty()?nlohmann::json(nullptr):nlohmann::json(tokens_text)},{"joined",joined},{"offset",{offset.x,offset.y,offset.z}},
            {"fromClip",from_name},{"toClip",to_name},{"kimodoNativeRevision","ed8a931aedb132f768190f6931c365855caba74c"},{"motionbricksRevision","54fc563aeaaf819a753d72cc9fdc6713989cdd43"}};
    };
    auto finish=[](const Clip& made,const fs::path& path,const nlohmann::json& recipe) {
        save_glb(made,path);
        auto measured=recipe;const auto travel=root_travel(made);
        bool canonical=true;try{require_soma(made);}catch(const std::exception&){canonical=false;}
        if(canonical) {
            auto events=footfalls(made);
            measured["footfalls"]={{"rule","soma-travelling-footfalls-v1"},{"count",events.size()},{"events",nlohmann::json::array()}};
            for(const auto& e:events)measured["footfalls"]["events"].push_back({{"foot",e.foot?"right-foot":"left-foot"},{"lift",e.lift},{"landing",e.landing},{"distanceMetres",e.distance},{"heightMetres",e.height}});
        }
        measured["rootTravel"]={{"displacementMetres",travel.displacement},{"pathMetres",travel.distance},{"peakSpeedMetresPerSecond",travel.peak_speed}};
        std::ofstream metadata(path.string()+".json");metadata<<measured.dump(2)<<"\n";metadata.close();
        if(!metadata)throw std::runtime_error("Animation saved but could not write its recipe");
        std::cout<<"Saved "<<path.string()<<" ("<<made.frames()<<" frames, "<<made.duration()<<" seconds)\n";
    };
    if(command=="cleanup") {
        if(pos.size()!=1||contacts_file.empty())throw std::runtime_error("cleanup needs one source and --contacts PLAN.json");
        auto plan=load_cleanup_plan(contacts_file);
        clip=load_glb(utf8path(pos[0]),clip_name,false);
        std::optional<RetargetMapping> mapping;if(!map_file.empty())mapping=load_mapping(map_file);
        // A SOMA-77 clip is cleaned as its SOMA-30 subset; its own finger rotations are kept.
        std::optional<Clip> full;
        if(!mapping&&is_soma77(clip)) {full=clip;clip=soma77_to_30(clip);}
        auto source_recipe=source_metadata(utf8path(pos[0]));
        auto protected_frames=protected_recipe_frames(source_recipe,clip.frames());
        plan.protected_frames.insert(plan.protected_frames.end(),protected_frames.begin(),protected_frames.end());
        auto reports=cleanup_contacts(clip,plan,mapping?&*mapping:nullptr);
        if(full)clip=soma30_to_77(clip,&*full);
        mirror_recipe={{"tool","tide-motion"},{"version",MOTION_VERSION},{"command","cleanup"},{"source",pos[0]},{"clip",clip_name},
            {"fps",clip.fps},{"frames",clip.frames()},{"solver","tide-two-bone-v1"},{"fadeFrames",plan.fade_frames},{"maxCorrectionMetres",plan.max_correction},{"maxJointCorrectionDegrees",plan.max_joint_correction},
            {"maxJointStepIncreaseDegrees",plan.max_joint_step_increase},{"maxAnkleStepIncreaseMetres",plan.max_ankle_step_increase},{"safeguards","tide-contact-bounds-v1"},{"protectedFrames",plan.protected_frames},
            {"sourceRecipe",source_recipe},{"rootMotionPreserved",true},{"contacts",nlohmann::json::array()}};
        if(source_recipe.contains("constraints"))mirror_recipe["constraints"]=source_recipe["constraints"];
        if(mapping) {std::ifstream input(map_file);nlohmann::json m;input>>m;mirror_recipe["mapping"]=m;}
        for(size_t i=0;i<reports.size();++i) {
            const auto& r=reports[i];const auto& contact=plan.contacts[i];
            mirror_recipe["contacts"].push_back({{"foot",contact.foot?"right-foot":"left-foot"},{"start",contact.start},{"end",contact.end},{"samples",r.samples},{"skipped",r.skipped},{"skipReason",r.skip_reason},
                {"horizontalMeanStepBeforeMetres",r.drift_before},{"horizontalMeanStepAfterMetres",r.drift_after},{"anchorResidualMaxMetres",r.residual_max},{"jointCorrectionMaxDegrees",r.rotation_max},
                {"jointStepMaxBeforeDegrees",r.joint_step_before},{"jointStepMaxAfterDegrees",r.joint_step_after},
                {"ankleStepMaxBeforeMetres",r.ankle_step_before},{"ankleStepMaxAfterMetres",r.ankle_step_after},
                {"attemptedJointCorrectionMaxDegrees",r.attempted_rotation_max},{"attemptedJointStepIncreaseMaxDegrees",r.joint_step_increase},{"attemptedAnkleStepIncreaseMaxMetres",r.ankle_step_increase}});
        }
    } else if(command=="mirror") {
        if(pos.size()!=1)throw std::runtime_error("mirror takes one SOMA-30 .glb clip");
        auto raw=load_glb(utf8path(pos[0]),clip_name,false);
        const bool full=is_soma77(raw);
        if(full)clip=mirror_soma77(raw);
        else {clip=load_glb(utf8path(pos[0]),clip_name,true);mirror_soma(clip);}
        auto protected_frames=protected_recipe_frames(source_metadata(utf8path(pos[0])),clip.frames());
        mirror_recipe={{"tool","tide-motion"},{"version",MOTION_VERSION},{"command",command},{"source",pos[0]},{"clip",clip_name},{"rig",full?"soma77":"soma30"},{"fps",clip.fps},{"frames",clip.frames()},
            {"protectedFrames",protected_frames},{"plane","the first frame's sagittal plane (through its root, up and its facing)"}};
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
        if(backend=="ardy") {
            if(std::find(flags.begin(),flags.end(),"--model")!=flags.end())throw std::runtime_error("--model selects Kimodo; omit it for --backend ardy");
            if(std::find(flags.begin(),flags.end(),"--steps")==flags.end())steps=10;
            if(steps>10)throw std::runtime_error("ARDY steps must be 1..10");
            if(device=="gpu"&&!ardy_gpu_compiled())throw std::runtime_error("ARDY GPU: this build has no DirectML support");
            if(std::find(flags.begin(),flags.end(),"--transition")!=flags.end())throw std::runtime_error("ARDY continues learned history; --transition applies to Kimodo");
        } else if(!history_file.empty()||std::find(flags.begin(),flags.end(),"--history-frames")!=flags.end())throw std::runtime_error("--history and --history-frames require --backend ardy");
        if(!correction.empty()&&!correct)throw std::runtime_error("Use --correction none or --no-correction, not both");
        if(correction.empty())correction=correct?"tide":"none";
        correct=correction=="tide";
        if(correction=="kimodo"&&backend!="kimodo")throw std::runtime_error("--correction kimodo uses Kimodo's own foot contacts; it needs --backend kimodo");
        if(correction!="tide"&&std::find(flags.begin(),flags.end(),"--correction-frames")!=flags.end())throw std::runtime_error("--correction-frames goes with --correction tide");
        if(dense_path&&backend!="kimodo")throw std::runtime_error("--dense-path follows Kimodo's root representation; it needs --backend kimodo");
        if(dense_path&&path_specs.empty())throw std::runtime_error("--dense-path needs --path waypoints");
        std::optional<RetargetMapping> pose_map;if(!pose_map_file.empty())pose_map=load_mapping(pose_map_file);
        if(pos.size()!=1||pos[0].empty())throw std::runtime_error("generate takes one quoted prompt");
        if(segments.size()>16)throw std::runtime_error("generate takes at most 16 prompts");
        for(const auto& s:segments)if(backend=="kimodo"&&s.second>10)throw std::runtime_error("Kimodo duration is out of range (0.2..10)");
        segments[0].first=pos[0];
        std::vector<kimodo::prompt_segment> parts;size_t total=0;
        for(size_t i=0;i<segments.size();++i) {
            auto count=static_cast<unsigned>(std::round(segments[i].second*30));
            if(i&&transition>=count)throw std::runtime_error("--transition must be shorter than every following prompt ("+std::to_string(count)+" frames)");
            parts.push_back({segments[i].first,count});total+=count;
        }
        if(total>18000)throw std::runtime_error("Generation exceeds 18000 frames");
        // Resolve and load every constraint before any model is loaded.
        auto resolve=[&](long frame,const char* label) {
            long resolved=frame<0?static_cast<long>(total)+frame:frame;
            if(resolved<0||resolved>=static_cast<long>(total))throw std::runtime_error(std::string(label)+" "+std::to_string(frame)+" is outside the "+std::to_string(total)+"-frame output");
            return static_cast<size_t>(resolved);
        };
        MotionConditions conditions;
        std::map<std::string,Clip> pose_files;std::set<std::string> mapped_sources;
        for(const auto& spec:pose_specs) {
            auto ref=pose_ref(spec.ref);
            auto& source=pose_files[ref.file];
            if(source.joints.empty()) {
                // SOMA-30 and SOMA-77 files are used directly; another mapped humanoid is
                // retargeted to SOMA first (as retarget --to-soma) when --pose-map is given.
                try {source=load_glb(utf8path(ref.file),"",true);}
                catch(const std::exception& e) {
                    if(!pose_map)throw std::runtime_error(ref.file+": "+e.what()+" (give --pose-map BONES.json to retarget a mapped humanoid pose source)");
                    Clip target;target.joints=soma();for(const auto& joint:target.joints)target.poses.push_back(joint.rest);
                    auto mapped=retarget(load_glb(utf8path(ref.file),"",false),target,*pose_map,true);
                    if(mapped.discarded_translations)std::cerr<<"Note: "<<ref.file<<" has non-root translation animation that retargeting does not copy.\n";
                    source=std::move(mapped.clip);mapped_sources.insert(ref.file);
                }
                require_soma(source);
            }
            long wanted=ref.frame?*ref.frame:(spec.kind=="start-pose"?-1:0);
            long index=wanted<0?static_cast<long>(source.frames())+wanted:wanted;
            if(index<0||index>=static_cast<long>(source.frames()))throw std::runtime_error(ref.file+" has no frame "+std::to_string(wanted)+" ("+std::to_string(source.frames())+" frames)");
            PoseConstraint pose{spec.kind,resolve(spec.frame,spec.kind.c_str()),ref.file,index,{},spec.effectors};
            auto frame=source.frame(static_cast<size_t>(index));pose.pose.assign(frame.begin(),frame.end());
            conditions.poses.push_back(std::move(pose));
        }
        for(auto [point,frame]:path_specs) {point.frame=resolve(frame,"path frame");conditions.path.push_back(point);}
        const auto waypoints=conditions.path;
        if(dense_path) {
            // Frame 0 of the dense path is where the output starts: a start pose's hips, else the origin.
            Vec3 start{};for(const auto& pose:conditions.poses)if(pose.frame==0)start=pose.pose[0].p;
            for(const auto& point:waypoints)if(point.frame==0)start={point.x,0,point.z};
            densify_path(conditions,start);
        }
        if(std::find(flags.begin(),flags.end(),"--end-at")!=flags.end()&&
           std::none_of(conditions.poses.begin(),conditions.poses.end(),[](const auto& p){return p.kind=="end-pose";}))
            throw std::runtime_error("--end-at requires --end-pose");
        if(fit_poses&&std::none_of(conditions.poses.begin(),conditions.poses.end(),[](const auto& p){return !p.effectors;}))
            throw std::runtime_error("--fit-poses requires a full-body pose");
        const bool selecting=wanted_footsteps>=0||fit_poses;
        if(selecting&&variants>1)throw std::runtime_error("selection takes one clip; use --variants separately to inspect candidates");
        if(!selecting&&std::find(flags.begin(),flags.end(),"--attempts")!=flags.end())throw std::runtime_error("--attempts requires --footsteps or --fit-poses");
        if(selecting&&seed+attempts-1>4294967295ULL)throw std::runtime_error("--seed plus --attempts passes 4294967295");
        if(follow=="ends") {
            if(std::none_of(conditions.poses.begin(),conditions.poses.end(),[](const auto& p){return p.kind=="keyframe"||p.kind=="effector";}))
                throw std::runtime_error("--follow ends turns --keyframe and --effector targets; give at least one");
            if(std::none_of(conditions.poses.begin(),conditions.poses.end(),[](const auto& p){return p.kind=="end-pose";}))
                throw std::runtime_error("--follow ends needs --end-pose");
        }
        std::optional<PoseConstraint> end_source;
        for(const auto& pose:conditions.poses)if(pose.kind=="end-pose")end_source=pose;
        bool shift=place_constraints(conditions,placement);
        // Targets take their share of the end pose's placement before encoding, so the
        // model, the recorded errors and correction all see them turned.
        nlohmann::json followed=nlohmann::json::array();
        if(follow=="ends") {
            auto turns=follow_ends(conditions,*end_source);size_t k=0;
            for(const auto& pose:conditions.poses)if(pose.kind=="keyframe"||pose.kind=="effector")
                followed.push_back({{"frame",pose.frame},{"kind",pose.kind},{"turnDegrees",turns[k++]*180/std::numbers::pi_v<float>}});
        }
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
        std::optional<Clip> imported;
        if(backend=="ardy") {
            if(!history_file.empty()) {
                imported=load_glb(history_file,"",true);require_soma(*imported);
                size_t count=std::min(size_t(history_frames),imported->frames()/4*4);
                if(count<4)throw std::runtime_error("ARDY history needs at least four frames");
                imported->poses.erase(imported->poses.begin(),imported->poses.end()-static_cast<ptrdiff_t>(count*30));
            }
            if(!context->ardy||context->ardy_threads!=threads||context->ardy_device!=device){context->ardy.reset();context->ardy=std::make_unique<ArdyModel>(models/"ardy",threads,device);context->ardy_threads=threads;context->ardy_device=device;}
        } else {
            ensure_kimodo(models,context,device,threads,model_version);
            if(condition&&context->kimodo->motion_dim()!=kimodo_features)throw std::runtime_error("Constraints need the SOMA-30 Kimodo model");
        }
        std::vector<Clip> made;std::vector<std::string> labels;
        nlohmann::json searched=nlohmann::json::array();
        const unsigned candidates=selecting?attempts:variants;
        std::optional<Clip> best_clip; nlohmann::json best_recipe;
        float best_score=std::numeric_limits<float>::infinity();
        for(unsigned v=0;v<candidates;++v) {
        const uint64_t used_seed=seed+v;
        const size_t output_index=selecting?0:v;
        clip=Clip{};generate_recipe=nlohmann::json::object();
        if(variants>1)std::cerr<<"Generating variant "<<v+1<<"/"<<variants<<" (seed "<<used_seed<<")...\n";
        else std::cerr<<"Generating motion...\n";
        auto effective=conditions;bool floating=false;std::vector<float> contacts;
        if(backend=="ardy") {
            std::vector<ArdyPrompt> prompts;for(const auto& p:parts)prompts.push_back({p.prompt,p.frames});
            clip=context->ardy->generate(prompts,ArdyOptions{steps,threads,history_frames,used_seed,guidance.text,guidance.constraint},conditions,imported?&*imported:nullptr);
            clip.name=outputs[output_index].stem().string();
            generate_recipe["backend"]="ardy";generate_recipe["checkpointRevision"]="e6c6804fb6b55636ad927145d74959fd46882ccd";
            generate_recipe["history"]=history_file.string();generate_recipe["historyFrames"]=history_frames;
            generate_recipe["modelManifestSha256"]="aac3b966ed1368c4ec126c969f0763d59126b047a1053a2394c20983acb917d2";
            generate_recipe["sampler"]="DDIM";generate_recipe["chunkFrames"]=60;
        } else {
        // Partial root masks are outside the upstream full-pose conditioning contract.
        // Choose floating locations with a learned motion pass, then condition complete
        // root coordinates in a second learned pass. Never derive travel by blending.
        floating=std::any_of(effective.poses.begin(),effective.poses.end(),[](const auto& p){return !p.anchored;});
        std::optional<KimodoCondition> placed;
        kimodo::motion_condition placed_native;
        if(floating) {
            auto guide=conditions;
            std::erase_if(guide.poses,[](const auto& p){return !p.anchored;});
            auto gc=build_condition(guide,total);
            kimodo::motion_condition gn{gc.observed,gc.mask,gc.first_heading};
            std::cerr<<"Choosing free pose locations with a learned motion pass...\n";
            auto travel=generate_motion(models,context,parts,transition,steps,used_seed,guidance,guide.empty()?nullptr:&gn);
            if(travel.frames!=total||travel.root_positions.size()!=total*3)throw std::runtime_error("Unexpected free placement guide layout");
            for(auto& p:effective.poses)if(!p.anchored) {
                p.pose[0].p.x=travel.root_positions[p.frame*3]+gc.origin.x;
                p.pose[0].p.z=travel.root_positions[p.frame*3+2]+gc.origin.z;
                p.anchored=true;
            }
            placed=build_condition(effective,total);
            placed_native={placed->observed,placed->mask,placed->first_heading};
        }
        auto result=generate_motion(models,context,parts,transition,steps,used_seed,guidance,floating?&placed_native:condition?&native:nullptr);
        if(result.joints!=30||result.local_rotations_xyzw.size()!=static_cast<size_t>(result.frames)*30*4||result.root_positions.size()!=static_cast<size_t>(result.frames)*3)throw std::runtime_error("Kimodo returned an unexpected SOMA output layout");
        contacts=std::move(result.foot_contacts);
        clip.joints=soma();clip.name=outputs[output_index].stem().string();
        // Constraints were expressed relative to the frame-0 origin; return to their world.
        Vec3 origin=condition?condition->origin:Vec3{};
        for(size_t f=0;f<result.frames;++f)for(size_t j=0;j<30;++j) {
            size_t index=(f*30+j)*4;Transform t=clip.joints[j].rest;
            t.q=normalized({result.local_rotations_xyzw[index],result.local_rotations_xyzw[index+1],result.local_rotations_xyzw[index+2],result.local_rotations_xyzw[index+3]});
            if(j==0)t.p=Vec3{result.root_positions[f*3],result.root_positions[f*3+1],result.root_positions[f*3+2]}+origin;
            clip.poses.push_back(t);
        }
        }
        if(shift)shift_to_last_pose(clip,conditions);
        generate_recipe["prompts"]=nlohmann::json::array();
        for(size_t i=0;i<parts.size();++i)generate_recipe["prompts"].push_back({{"prompt",parts[i].prompt},{"duration",segments[i].second},{"frames",parts[i].frames}});
        generate_recipe["transition"]=backend=="kimodo"&&parts.size()>1?transition:0;
        generate_recipe["freePlacementPasses"]=floating?2:1;
        if(floating) {
            generate_recipe["inferredPoseLocations"]=nlohmann::json::array();
            for(size_t i=0;i<conditions.poses.size();++i)if(!conditions.poses[i].anchored)
                generate_recipe["inferredPoseLocations"].push_back({{"frame",effective.poses[i].frame},{"x",effective.poses[i].pose[0].p.x},{"z",effective.poses[i].pose[0].p.z}});
        }
        generate_recipe["fitPoses"]=fit_poses;
        generate_recipe["follow"]=follow;if(follow!="world")generate_recipe["followedTargets"]=followed;
        generate_recipe["placement"]=placement_name;generate_recipe["corrected"]=correction=="kimodo"||(correct&&!conditions.poses.empty());
        generate_recipe["correction"]=correction;generate_recipe["rig"]=rig;
        if(!contacts.empty()) {
            // Kimodo's own foot contacts (decoded, thresholded at 0.5), as [start, end) frames.
            nlohmann::json intervals;const char* names[4]={"leftFoot","leftToe","rightFoot","rightToe"};
            for(size_t k=0;k<4;++k) {intervals[names[k]]=nlohmann::json::array();for(auto [first,end]:contact_intervals(contacts,k))intervals[names[k]].push_back({first,end});}
            generate_recipe["footContacts"]=intervals;
        }
        generate_recipe["correctionFrames"]=correction_frames;generate_recipe["inPlace"]=in_place;
        generate_recipe["guidance"]=guidance.text;generate_recipe["constraintGuidance"]=guidance.constraint;
        if(backend=="kimodo") {
        generate_recipe["model"]="nvidia/Kimodo-SOMA-RP-"+model_version;
        generate_recipe["modelGgufSha256"]=model_version=="v1.2"?"9c24c5ee9b6b3683947ab2796727ffb2fbd69b2ebc7683214809013687d87a19":"3bf1229f4c1eff1d28f5196a854113da2df9a11a5e21c60694630903bf948ee4";
        if(model_version=="v1.2")generate_recipe["checkpointRevision"]="0e3c89024d896064b6851201373f7834784a386b";
        }
        generate_recipe["device"]=backend=="ardy"?context->ardy->device():context->kimodo_gpu?"gpu":"cpu";generate_recipe["threads"]=backend=="ardy"&&!threads?8:threads;
        if(backend=="ardy")generate_recipe["accelerator"]=context->ardy->device()=="gpu"?"directml":"cpu";
        if(variants>1)generate_recipe["variants"]={{"count",variants},{"firstSeed",seed},{"sheet",sheet.png.filename().string()}};
        generate_recipe["constraints"]=nlohmann::json::array();
        auto round=[](float value,float unit){return std::round(value/unit)*unit;};
        float pose_score=0;
        // Errors are the model's own; correction then makes full-body poses exact.
        for(const auto& pose:conditions.poses) {
            auto e=constraint_error(clip,pose);
            if(!pose.effectors)pose_score=std::max(pose_score,e.rotation_mean);
            nlohmann::json item={{"kind",pose.kind},{"frame",pose.frame},{"file",pose.file},{"sourceFrame",pose.source_frame},{"anchored",pose.anchored},
                {"modelError",{{"rootMetres",round(e.root,1e-4f)},{"positionMeanMetres",round(e.position_mean,1e-4f)},{"positionMaxMetres",round(e.position_max,1e-4f)},{"rotationMeanDegrees",round(e.rotation_mean,0.01f)},{"rotationMaxDegrees",round(e.rotation_max,0.01f)},{"rotationMaxJoint",e.rotation_max_joint}}}};
            if(pose.effectors)item["effectors"]=effector_list(pose.effectors);
            generate_recipe["constraints"].push_back(item);
            std::cerr<<pose.kind<<(pose.effectors?" "+effector_list(pose.effectors):std::string())<<" at frame "<<pose.frame<<(pose.anchored?"":" (free on the ground)")<<": root "<<round(e.root*100,0.1f)<<" cm; "<<(pose.effectors?"effector":"joint")<<" positions mean "<<round(e.position_mean*100,0.1f)<<" / max "<<round(e.position_max*100,0.1f)<<" cm; rotations mean "<<round(e.rotation_mean,0.1f)<<" / max "<<round(e.rotation_max,0.1f)<<" degrees"<<(correct&&!pose.effectors?"; corrected to exact":correction=="kimodo"?"; Kimodo correction follows":"")<<"\n";
        }
        if(correction=="kimodo") {
            auto report=correct_kimodo(clip,conditions,contacts);
            generate_recipe["kimodoCorrection"]={{"source","nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954 MotionCorrection"},{"rootMarginMetres",0.04},{"contactThreshold",0.5},
                {"contactFrames",{{"leftFoot",report.contact_frames[0]},{"leftToe",report.contact_frames[1]},{"rightFoot",report.contact_frames[2]},{"rightToe",report.contact_frames[3]}}},
                {"effectorIkFrames",report.effector_frames},{"ankleIkFrames",report.ankle_frames},{"toeIkFrames",report.toe_frames},{"milliseconds",report.milliseconds}};
            std::cerr<<"Kimodo correction: "<<report.ankle_frames<<" ankle and "<<report.toe_frames<<" toe contact frames, "<<report.effector_frames<<" effector frames ("<<round(static_cast<float>(report.milliseconds),0.1f)<<" ms)\n";
        }
        if(correct||correction=="kimodo") {
            if(correct)correct_poses(clip,conditions,correction_frames);
            size_t index=0;
            for(const auto& pose:conditions.poses) {
                auto e=constraint_error(clip,pose);
                generate_recipe["constraints"][index++]["error"]={{"rootMetres",round(e.root,1e-4f)},{"positionMaxMetres",round(e.position_max,1e-4f)},{"rotationMaxDegrees",round(e.rotation_max,0.01f)}};
            }
        }
        if(dense_path) {
            // Measured as the representation defines it: the output's smooth root against
            // every dense sample, and the hips for reference.
            auto smooth=smooth_root_of(clip);double sm=0,sx=0,hm=0,hx=0;
            for(const auto& point:conditions.path) {
                double d=std::hypot(smooth[point.frame][0]-point.x,smooth[point.frame][1]-point.z),h=path_error(clip,point);
                sm+=d;sx=std::max(sx,d);hm+=h;hx=std::max(hx,h);
            }
            const double count=static_cast<double>(conditions.path.size());
            generate_recipe["densePath"]={{"firstFrame",conditions.path.front().frame},{"lastFrame",conditions.path.back().frame},{"frames",conditions.path.size()},{"smoothed",waypoints.size()+(waypoints.front().frame==0?0:1)>=3},
                {"smoothRootErrorMeanMetres",round(static_cast<float>(sm/count),1e-4f)},{"smoothRootErrorMaxMetres",round(static_cast<float>(sx),1e-4f)},
                {"hipsErrorMeanMetres",round(static_cast<float>(hm/count),1e-4f)},{"hipsErrorMaxMetres",round(static_cast<float>(hx),1e-4f)}};
            std::cerr<<"dense path frames "<<conditions.path.front().frame<<".."<<conditions.path.back().frame<<": smooth root mean "<<round(static_cast<float>(sm/count*100),0.1f)<<" / max "<<round(static_cast<float>(sx*100),0.1f)<<" cm from the path\n";
        }
        if(!pose_map_file.empty()) {std::ifstream input(pose_map_file);nlohmann::json m;input>>m;generate_recipe["poseMap"]=m;generate_recipe["mappedPoseSources"]=mapped_sources;}
        for(const auto& point:dense_path?waypoints:conditions.path) {
            float e=path_error(clip,point);
            nlohmann::json item={{"kind","path"},{"frame",point.frame},{"x",point.x},{"z",point.z},{"error",{{"hipsGroundMetres",round(e,1e-4f)}}}};
            if(point.has_heading)item["yaw"]=point.heading;
            generate_recipe["constraints"].push_back(item);
            std::cerr<<"path at frame "<<point.frame<<": hips "<<round(e*100,0.1f)<<" cm from "<<point.x<<", "<<point.z<<"\n";
        }
        // Recorded errors are measured before the travel is removed.
        if(in_place)remove_root_travel(clip);
        const auto events=footfalls(clip);
        if(selecting) {
            searched.push_back({{"seed",used_seed},{"footfalls",events.size()},{"poseMeanRotationDegrees",pose_score}});
            std::cerr<<"Seed "<<used_seed<<": "<<events.size()<<" travelling footfalls; worst pose mean rotation "<<pose_score<<" degrees\n";
            if(wanted_footsteps>=0&&events.size()!=static_cast<size_t>(wanted_footsteps))continue;
            generate_recipe["selection"]={{"firstSeed",seed},{"attemptLimit",attempts},{"attempted",searched}};
            if(wanted_footsteps>=0)generate_recipe["selection"]["requestedFootsteps"]=wanted_footsteps;
        }
        auto recipe=recipe_for(clip,used_seed);recipe["rig"]=rig;
        double requested=0;for(const auto& segment:segments)requested+=segment.second;
        recipe["requestedDuration"]=requested;
        for(const auto& item:generate_recipe.items())recipe[item.key()]=item.value();
        if(fit_poses) {
            if(pose_score<best_score){best_score=pose_score;best_clip=std::move(clip);best_recipe=std::move(recipe);}
            continue;
        }
        finish(rig=="soma77"?soma30_to_77(clip):clip,outputs[output_index],recipe);
        if(variants>1) {made.push_back(clip);labels.push_back("seed "+std::to_string(used_seed));}
        if(selecting)return 0;
        }
        if(best_clip) {
            best_recipe["selection"]["attempted"]=searched;
            best_recipe["selection"]["rankBy"]="worstPoseMeanRotationDegrees";
            best_recipe["selection"]["scoreDegrees"]=best_score;
            std::cerr<<"Selected seed "<<best_recipe["seed"]<<" with worst pose mean rotation "<<best_score<<" degrees from "<<attempts<<" attempts\n";
            finish(rig=="soma77"?soma30_to_77(*best_clip):*best_clip,output,best_recipe);return 0;
        }
        if(selecting)throw std::runtime_error("No candidate matched --footsteps after "+std::to_string(attempts)+" attempts; no output saved. Counts: "+searched.dump());
        if(variants>1) {
            // The clips are kept whatever happens to the sheet.
            try {preview_variants(made,labels,sheet);std::cout<<"Saved "<<sheet.png.string()<<(sheet.gif.empty()?"":" and "+sheet.gif.string())<<"\n";}
            catch(const std::exception& e) {std::cerr<<"Contact sheet skipped: "<<e.what()<<"\n";}
        }
        return 0;
    } else {
        if(pos.size()!=2)throw std::runtime_error("connect takes source and target GLB files");
        auto a=load_glb(utf8path(pos[0]),from_name),b=load_glb(utf8path(pos[1]),to_name);
        // Validate inherited metadata before loading an inference session.
        auto source_pins=protected_recipe_frames(source_metadata(utf8path(pos[0])),a.frames());
        auto target_pins=protected_recipe_frames(source_metadata(utf8path(pos[1])),b.frames());
        for(size_t f=0;f<b.frames();++f)b.frame(f)[0].p=b.frame(f)[0].p+offset;
        if(context->bricks_prepare) {context->bricks_pending.push_back({std::move(a),std::move(b),bricks_request,fs::absolute(output)});return 0;}
        if(!context->bricks_ready.empty()) {
            auto ready=std::move(context->bricks_ready.front());context->bricks_ready.pop_front();
            if(ready.output!=fs::absolute(output))throw std::runtime_error("Internal error: batched bridge out of order");
            bridge=std::move(ready.bridge);batch_requests=ready.requests;batch_index=ready.index;
        } else bridge=bricks_session(models,threads,context).generate(a,b,bricks_request);
        clip=bridge.clip;
        connect_pins=connected_protected_frames(a.frames(),clip.frames(),b.frames(),source_pins,target_pins,joined);
        if(joined) {clip=join(a,clip,b);contact_offset=a.frames()-1;}
    }
    auto recipe=recipe_for(clip,seed);
    if(command=="connect") {
        recipe["threads"]=threads;recipe["protectedFrames"]=connect_pins;
        recipe["motionbricksBatch"]={{"requests",batch_requests},{"index",batch_index}};
        recipe["allowedTokens"]={bricks_request.min_tokens,bricks_request.max_tokens};
        recipe["chosenTokens"]=bridge.tokens;recipe["targetPoses"]=bricks_request.target_poses;
        recipe["bridgeFrames"]={contact_offset,contact_offset+bridge.clip.frames()-1};
        const auto& rc=bridge.correction;
        recipe["seamCorrection"]={{"rootMetres",{{rc.root[0].x,rc.root[0].y,rc.root[0].z},{rc.root[1].x,rc.root[1].y,rc.root[1].z}}},
            {"maxJointDegrees",{rc.joint_degrees[0],rc.joint_degrees[1]}},{"maxJoint",{bridge.clip.joints[rc.joint[0]].name,bridge.clip.joints[rc.joint[1]].name}}};
        contact_record(bridge.contacts,contact_offset,recipe);
    }
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
