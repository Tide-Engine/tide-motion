#include "equipment.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <set>
#include <stdexcept>

namespace motion {
const char* equipment_help(){return R"(tide-motion equipment inspect BODY.glb --plan EQUIPMENT.json [--correct-grips] [--strict] [--report NEW.json]
tide-motion equipment bake BODY.glb --plan EQUIPMENT.json -o NEW.glb [--body-output BODY-ONLY.glb] [--correct-grips] [--strict]
tide-motion equipment preview BODY.glb --plan EQUIPMENT.json [--correct-grips] [--strict]
    [--first-person] [--setup CAMERA.json] [--frames N] [--capture NEW.png] [--gif NEW.gif] [--tide PATH]
    [--character MODEL.glb|.fbx --map BONES.json --texture IMAGE.png] [--clip NAME]

Native offline equipment authoring; no model/session/cache required. Version-1
plan: objects with contiguous inclusive segments and absolute-frame keys;
space world, bone or object; position in metres, rotation in Euler degrees,
meshScale for geometry only. Independent baked object roots retain these tracks.
Grips name an object-local target and a body bone/palm offset over [start,end].
--correct-grips needs an explicit direct upper-arm/forearm/hand chain per grip.
Only chain rotations change; translations, root, fingers and protected frames stay.
Object trajectories are resolved from the original body, then held fixed during IK.
Unreachable/excessive/discontinuous corrections are rejected and reported.
--strict refuses a bake/preview when any final sampled grip misses its tolerance.
Strings author two rigid segments from bow anchors through keyed nock positions;
events (including release) carry frame/time in GLB extras and JSON. Arrow flight
must be authored as object keys; no projectile simulation or collision guarantee.
Bake embeds equipment meshes and body skeleton animation, not a character mesh.
It writes a JSON recipe/report and refuses existing output/recipe/partial files.
See notes/equipment-authoring.md for schema and sword/shield/bow examples.
)";}
namespace {
int count(const std::string& s) {
    size_t used=0;int n=std::stoi(s,&used);if(used!=s.size()||n<1||n>360000)throw std::runtime_error("Equipment preview frames must be 1..360000");return n;
}
void new_file(const fs::path& path) {
    if(path.empty()||fs::exists(path)||fs::exists(path.string()+".part"))throw std::runtime_error("Equipment output/partial already exists or is empty: "+path.string());
}
std::string normalized_path(const fs::path& path) {
    auto n=fs::weakly_canonical(fs::absolute(path)).generic_string();
#ifdef _WIN32
    for(auto& c:n)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
#endif
    return n;
}
void write_json_new(const fs::path& path,const nlohmann::json& j) {
    if(path.has_parent_path())fs::create_directories(path.parent_path());
    fs::path part=path.string()+".part";
#ifdef _WIN32
    FILE* file=_wfopen(part.c_str(),L"wbx");
#else
    FILE* file=std::fopen(part.c_str(),"wbx");
#endif
    if(!file)throw std::runtime_error("Cannot create equipment JSON partial exclusively: "+part.string());
    try {
        auto text=j.dump(2)+"\n";bool ok=std::fwrite(text.data(),1,text.size(),file)==text.size();
        if(std::fclose(file))ok=false;
        file=nullptr;
        if(!ok)throw std::runtime_error("Cannot write equipment JSON");
        fs::create_hard_link(part,path);std::error_code ignored;fs::remove(part,ignored);
    }catch(...){if(file)std::fclose(file);std::error_code ignored;fs::remove(part,ignored);throw;}
}
void protect(const fs::path& source,EquipmentPlan& plan) {
    fs::path path=source.string()+".json";if(!fs::exists(path))return;
    if(fs::file_size(path)>16*1024*1024)throw std::runtime_error("Source recipe exceeds 16 MiB");
    std::ifstream in(path);nlohmann::json j;in>>j;
    if(!j.contains("protectedFrames"))return;
    if(!j["protectedFrames"].is_array()||j["protectedFrames"].size()>18000)throw std::runtime_error("Invalid source protectedFrames");
    for(const auto& f:j["protectedFrames"]) {
        if(!f.is_number_integer()||f.get<double>()<0||f.get<double>()>17999)throw std::runtime_error("Invalid source protected frame");
        plan.protected_frames.push_back(f.get<size_t>());
    }
}
}
int equipment_command(const std::vector<std::string>& args) {
    for(size_t i=2;i<args.size();++i)if(args[i]=="--help"||args[i]=="-h"){std::cout<<equipment_help();return 0;}
    if(args.size()<4)throw std::runtime_error(equipment_help());
    const auto& action=args[2];if(action!="inspect"&&action!="bake"&&action!="preview")throw std::runtime_error("Equipment action must be inspect, bake or preview");
    fs::path source=utf8path(args[3]),plan_path,output,body_output,report,map,setup;std::string clip_name;bool correct=false,strict=false,first=false;
    PreviewOptions options;std::set<std::string> seen;
    for(size_t i=4;i<args.size();++i) {
        const auto& a=args[i];if(!seen.insert(a).second)throw std::runtime_error("Duplicate equipment option: "+a);
        auto value=[&]()->const std::string&{if(++i>=args.size())throw std::runtime_error("Missing value for "+a);return args[i];};
        if(a=="--plan")plan_path=utf8path(value());else if(a=="-o"||a=="--output")output=utf8path(value());
        else if(a=="--body-output")body_output=utf8path(value());
        else if(a=="--report")report=utf8path(value());else if(a=="--correct-grips")correct=true;else if(a=="--strict")strict=true;
        else if(a=="--clip")clip_name=value();else if(a=="--map")map=utf8path(value());
        else if(a=="--character"){options.body=PreviewBody::character;options.character=utf8path(value());}
        else if(a=="--texture")options.texture=utf8path(value());else if(a=="--setup")setup=utf8path(value());else if(a=="--first-person")first=true;
        else if(a=="--frames")options.frames=count(value());else if(a=="--capture")options.capture=utf8path(value());
        else if(a=="--gif")options.gif=utf8path(value());else if(a=="--tide")options.tide=value();
        else throw std::runtime_error("Unknown equipment option: "+a);
    }
    if(plan_path.empty())throw std::runtime_error("Equipment needs --plan FILE.json");
    if(action!="preview")for(const char* flag:{"--setup","--first-person","--frames","--capture","--gif","--tide","--texture"})if(seen.contains(flag))throw std::runtime_error(std::string(flag)+" is equipment preview only");
    if(action=="bake") {
        if(output.extension()!=".glb")throw std::runtime_error("Equipment bake needs -o NEW.glb");
        new_file(output);
        new_file(output.string()+".json");
    }else if(!output.empty())throw std::runtime_error("Equipment -o is bake only; inspect uses --report");
    if(!body_output.empty()) {
        if(action!="bake"||body_output.extension()!=".glb")throw std::runtime_error("--body-output needs equipment bake and a new .glb path");
        new_file(body_output);new_file(body_output.string()+".json");
        if(normalized_path(body_output)==normalized_path(output))throw std::runtime_error("Body-only output must differ from the equipment GLB");
    }
    if(!report.empty()){if(report.extension()!=".json")throw std::runtime_error("Equipment report needs .json");new_file(report);}
    if(!report.empty()&&!output.empty()&&normalized_path(report)==normalized_path(output.string()+".json"))throw std::runtime_error("Equipment report must differ from the bake recipe path");
    if(!report.empty()&&!body_output.empty()&&normalized_path(report)==normalized_path(body_output.string()+".json"))throw std::runtime_error("Equipment report must differ from the body recipe path");
    if(!setup.empty()) {
        if(first)throw std::runtime_error("Equipment --setup cannot be combined with --first-person");
        auto settings=load_preview_setup(setup);if(!settings.props.empty())throw std::runtime_error("Equipment setup must contain camera only; put dynamic props in --plan");options.camera=settings.camera;
    }else options.camera.enabled=first;
    auto source_clip=load_glb(source,clip_name,false);
    if(options.body==PreviewBody::character) {
        bool canonical=true;try{require_soma(source_clip);}catch(const std::exception&){canonical=false;}
        if(!map.empty()) {if(!canonical)throw std::runtime_error("Equipment --map expects a SOMA source");source_clip=retarget(source_clip,load_rig_glb(options.character),load_mapping(map)).clip;}
        else if(canonical)throw std::runtime_error("Equipment --character needs --map for a SOMA source");
    }else if(!map.empty()||!options.texture.empty())throw std::runtime_error("Equipment --map/--texture go with --character");
    auto plan=load_equipment_plan(plan_path);protect(source,plan);
    // Include explicit unweighted anchor nodes while sampling the drawn character.
    if(options.body==PreviewBody::character) {
        auto bind=options;for(const auto& o:plan.objects)for(const auto& s:o.segments)if(s.space=="bone")bind.props.push_back({{},s.anchor,{},{}});
        source_clip=preview_rig(source_clip,bind);
    }
    auto result=author_equipment(source_clip,plan,correct);
    result.report["source"]={{"path",fs::absolute(source).generic_string()},{"sha256",sha256_file(source)},{"clip",clip_name}};
    result.report["toolVersion"]=MOTION_VERSION;result.report["planSha256"]=sha256_file(plan_path);
    result.report["planPath"]=fs::absolute(plan_path).generic_string();result.report["protectedFrames"]=plan.protected_frames;
    result.report["meshes"]=nlohmann::json::array();
    for(const auto& object:result.objects)if(!object.mesh.empty())result.report["meshes"].push_back({{"object",object.name},{"path",fs::absolute(object.mesh).generic_string()},{"sha256",sha256_file(object.mesh)}});
    if(!options.character.empty())result.report["character"]={{"path",fs::absolute(options.character).generic_string()},{"sha256",sha256_file(options.character)}};
    if(!map.empty())result.report["map"]={{"path",fs::absolute(map).generic_string()},{"sha256",sha256_file(map)}};
    if(strict&&!result.report.at("allGripsSatisfied").get<bool>())throw std::runtime_error("Equipment strict validation failed: one or more sampled grips exceed tolerance; inspect without --strict for residuals");
    if(action!="inspect"&&!result.report.at("allGripsSatisfied").get<bool>())
        std::cerr<<"Equipment has unsatisfied sampled grips; see the report or use --strict to refuse the result.\n";
    if(action=="inspect")std::cout<<result.report.dump(2)<<'\n';
    else if(action=="bake") {
        save_equipment_glb(result,output);
        write_json_new(output.string()+".json",nlohmann::json{{"tool","tide-motion"},{"command","equipment bake"},{"report",result.report},{"events",result.events},{"protectedFrames",plan.protected_frames}});
        if(!body_output.empty()) {
            EquipmentResult body_only;body_only.body=result.body;body_only.report=result.report;body_only.report["bodyOnly"]=true;body_only.events=nlohmann::json::array();
            save_equipment_glb(body_only,body_output);
            write_json_new(body_output.string()+".json",nlohmann::json{{"tool","tide-motion"},{"command","equipment bake body-only"},{"equipmentOutput",fs::absolute(output).generic_string()},{"report",result.report},{"protectedFrames",plan.protected_frames}});
        }
        std::cout<<"Saved "<<output.string()<<" ("<<result.body.frames()<<" frames, "<<result.objects.size()<<" equipment tracks)\n";
    } else {
        Clip display=result.body;if(options.body!=PreviewBody::character&&is_soma77(display))display=soma77_to_30(display);
        bool soma=true;try{require_soma(display);}catch(const std::exception&){soma=false;}
        if(options.body!=PreviewBody::character&&!soma)throw std::runtime_error("Equipment custom-rig preview needs --character");
        if(soma)options.facing=kimodo_heading(display,display.frame(0));
        options.equipment=&result;preview(display,source,options);
    }
    if(!report.empty())write_json_new(report,result.report);
    return 0;
}
}
