#include "dex2hoi_preview.hpp"
#include "dex2hoi_motion.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

namespace motion::dex2hoi {
namespace {
using nlohmann::json;
void require(bool ok,const std::string& reason) {
    if(!ok)throw std::runtime_error("Dex2HOI preview: "+reason);
}
json read_json(const fs::path& path) {
    require(fs::is_regular_file(path)&&fs::file_size(path)<=16*1024*1024,"missing or oversized JSON: "+path.string());
    std::ifstream in(path);std::vector<std::set<std::string>> keys;
    auto unique=[&](int,json::parse_event_t event,json& value) {
        if(event==json::parse_event_t::object_start)keys.emplace_back();
        else if(event==json::parse_event_t::key)require(keys.back().insert(value.get<std::string>()).second,"duplicate JSON key");
        else if(event==json::parse_event_t::object_end)keys.pop_back();
        return true;
    };
    return json::parse(in,unique);
}
void keys(const json& value,std::initializer_list<const char*> allowed) {
    require(value.is_object(),"manifest entries must be objects");
    for(auto it=value.begin();it!=value.end();++it)
        require(std::any_of(allowed.begin(),allowed.end(),[&](const char* key){return it.key()==key;}),"unknown manifest field: "+it.key());
}
template<size_t N> std::array<float,N> array(const json& value) {
    require(value.is_array()&&value.size()==N,"wrong numeric array length");std::array<float,N> out{};
    for(size_t i=0;i<N;++i){out[i]=value[i].get<float>();require(std::isfinite(out[i]),"nonfinite manifest number");}
    return out;
}
Transform transform(const json& value) {
    keys(value,{"position","rotation"});auto p=array<3>(value.at("position"));auto q=array<4>(value.at("rotation"));
    double norm=0;for(float v:q)norm+=v*v;
    require(std::abs(norm-1)<1e-3,"manifest pose quaternion must be unit xyzw");
    return {{p[0],p[1],p[2]},normalized(Quat{q[0],q[1],q[2],q[3]})};
}
fs::path relative(const fs::path& input,const json& value) {
    auto path=utf8path(value.get<std::string>());require(!path.empty(),"empty manifest path");
    return fs::absolute(path.is_absolute()?path:input.parent_path()/path);
}
std::map<std::string,size_t> names(const Clip& clip) {
    std::map<std::string,size_t> out;
    for(size_t j=0;j<clip.joints.size();++j)
        require(!clip.joints[j].name.empty()&&out.emplace(clip.joints[j].name,j).second,"ambiguous or empty node name: "+clip.joints[j].name);
    return out;
}
float rotation_error(Quat a,Quat b) {
    a=normalized(a);b=normalized(b);return 1-std::abs(a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
}
bool same(Transform a,Transform b) {
    auto d=a.p-b.p;return dot(d,d)<=1e-10f&&rotation_error(a.q,b.q)<=1e-6f;
}
void compatible(const Clip& expected,const Clip& actual,const std::map<std::string,size_t>& at) {
    for(const auto& joint:expected.joints) {
        require(at.contains(joint.name),"missing body node: "+joint.name);
        const auto& other=actual.joints[at.at(joint.name)];
        std::string parent=joint.parent<0?"":expected.joints[static_cast<size_t>(joint.parent)].name;
        std::string other_parent=other.parent<0?"":actual.joints[static_cast<size_t>(other.parent)].name;
        require(parent==other_parent&&same(joint.rest,other.rest)&&std::abs(joint.scale-other.scale)<1e-6f,"incompatible rest rig or hierarchy at "+joint.name);
    }
}
json glb_document(const fs::path& source) {
    std::ifstream in(source,std::ios::binary);require(static_cast<bool>(in),"cannot open output GLB");
    auto integer=[&](){unsigned char bytes[4]{};require(static_cast<bool>(in.read(reinterpret_cast<char*>(bytes),4)),"truncated GLB header");return static_cast<uint32_t>(bytes[0])|(static_cast<uint32_t>(bytes[1])<<8)|(static_cast<uint32_t>(bytes[2])<<16)|(static_cast<uint32_t>(bytes[3])<<24);};
    require(integer()==0x46546c67&&integer()==2,"output must be binary glTF 2");
    require(integer()==fs::file_size(source),"GLB byte length mismatch");auto count=integer();
    require(integer()==0x4e4f534a&&count<=16*1024*1024,"invalid or oversized GLB JSON");
    std::string text(count,'\0');require(static_cast<bool>(in.read(text.data(),count)),"truncated GLB JSON");return json::parse(text);
}
std::string path_key(const fs::path& path) {
    auto result=fs::weakly_canonical(fs::absolute(path)).generic_string();
#ifdef _WIN32
    for(char& c:result)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
#endif
    return result;
}
void new_output(const fs::path& path) {
    auto present=[](const fs::path& candidate) {
        std::error_code error;auto status=fs::symlink_status(candidate,error);
        require(!error||error==std::errc::no_such_file_or_directory,"cannot inspect output path: "+candidate.string());
        return fs::exists(status)||fs::is_symlink(status);
    };
    require(!path.empty()&&!present(path)&&!present(path.string()+".part"),"capture/GIF or partial already exists: "+path.string());
}
float number(const std::string& text,float low,float high) {
    size_t used=0;float value=std::stof(text,&used);
    require(used==text.size()&&std::isfinite(value)&&value>=low&&value<=high,"numeric option is out of range");return value;
}
Vec3 triplet(const std::string& text,float bound) {
    auto first=text.find(','),second=first==std::string::npos?first:text.find(',',first+1);
    require(first!=std::string::npos&&second!=std::string::npos&&text.find(',',second+1)==std::string::npos,"camera vector needs X,Y,Z");
    return {number(text.substr(0,first),-bound,bound),number(text.substr(first+1,second-first-1),-bound,bound),number(text.substr(second+1),-bound,bound)};
}
}

EquipmentResult load_preview_result(const fs::path& source,const fs::path& input) {
    auto config=read_json(input);keys(config,{"version","history","bones","objects"});
    require(config.at("version").is_number_integer()&&config.at("version")==1,"history manifest version must be 1");
    auto history_path=relative(input,config.at("history"));auto history=load_glb(history_path,"",false);auto history_names=names(history);
    require(history.frames()==10&&std::abs(history.fps-30)<1e-4f,"history must contain ten frames at 30 Hz");
    const auto& binding=config.at("bones");require(binding.is_array()&&binding.size()==65,"history needs 65 explicit bone names");
    Dex2HoiRig rig;rig.joints=history.joints;
    for(size_t j=0;j<65;++j){auto name=binding[j].get<std::string>();require(history_names.contains(name),"history lacks mapped bone: "+name);rig.bones[j]=history_names.at(name);}
    rig.validate();
    const auto& objects=config.at("objects");require(objects.is_array()&&objects.size()==2,"history needs two explicit object slots");
    std::array<Dex2HoiObject,2> original;
    for(size_t s=0;s<2;++s) {
        const auto& value=objects[s];keys(value,{"name","mesh","present","exportTrack","points","poses"});auto& object=original[s];
        object.name=value.at("name").get<std::string>();object.present=value.value("present",true);object.export_track=value.value("exportTrack",true);
        if(value.contains("mesh")){object.mesh=relative(input,value.at("mesh"));require(fs::is_regular_file(object.mesh),"missing prop mesh: "+object.mesh.string());}
        const auto& points=value.at("points");require(points.is_array()&&points.size()==128,"object needs 128 local surface points");
        for(const auto& p:points){auto v=array<3>(p);object.surface_points.push_back({v[0],v[1],v[2]});}
        const auto& poses=value.at("poses");require(poses.is_array()&&poses.size()==10,"object needs ten prefix transforms");
        for(const auto& p:poses)object.poses.push_back(transform(p));
    }
    // Reuse native source representation validation, without weights or sessions.
    (void)encode_dex2hoi_history(rig,history,original);
    std::vector<std::string> requested;
    for(const auto& joint:history.joints)requested.push_back(joint.name);
    for(const auto& object:original)if(object.present&&object.export_track)requested.push_back(object.name);
    // Geometry descendants may legitimately have nonuniform static transforms.
    // Decode only explicitly bound body/object nodes and their ancestors. Their
    // original mesh hierarchies remain in the unmodified source asset.
    auto raw=load_glb_nodes(source,requested);auto at=names(raw);auto doc=glb_document(source);
    const auto& metadata=doc.at("asset").at("extras").at("tideMotion");
    require(metadata.at("equipmentVersion")==1&&metadata.at("fps")==30&&metadata.at("frames")==70&&raw.frames()==70&&std::abs(raw.fps-30)<1e-4f,"generated output must contain 70 frames at 30 Hz");
    compatible(history,raw,at);
    EquipmentResult result;result.body=history;result.body.name=raw.name;result.body.fps=raw.fps;result.body.poses.clear();result.events=json::array();
    for(size_t f=0;f<raw.frames();++f)for(const auto& joint:history.joints)result.body.poses.push_back(raw.frame(f)[at.at(joint.name)]);
    for(size_t f=0;f<10;++f)for(size_t j=0;j<history.joints.size();++j)require(same(result.body.frame(f)[j],history.frame(f)[j]),"body prefix differs from supplied history");
    std::map<std::string,size_t> doc_nodes;
    for(size_t j=0;j<doc.at("nodes").size();++j)if(doc["nodes"][j].contains("name"))doc_nodes.emplace(doc["nodes"][j]["name"].get<std::string>(),j);
    std::map<std::pair<size_t,std::string>,size_t> channels;
    require(doc.at("animations").size()==1,"output must contain one animation");
    for(const auto& channel:doc["animations"][0].at("channels"))++channels[{channel.at("target").at("node").get<size_t>(),channel.at("target").at("path").get<std::string>()}];
    auto tracks=[&](const std::string& name) {
        require(doc_nodes.contains(name),"missing exported node: "+name);
        for(const char* path:{"translation","rotation"})require(channels[{doc_nodes.at(name),path}]==1,"missing or ambiguous animation track: "+name+"/"+path);
    };
    for(const auto& joint:history.joints)tracks(joint.name);
    std::set<std::string> exported;
    for(const auto& object:original)if(object.present&&object.export_track) {
        require(exported.insert(object.name).second&&!history_names.contains(object.name),"ambiguous body/object name: "+object.name);
        require(at.contains(object.name),"missing exported object: "+object.name);auto index=at.at(object.name);tracks(object.name);
        const auto& node=doc["nodes"][doc_nodes.at(object.name)];
        require(raw.joints[index].parent<0&&node.at("extras").at("tideMotion").at("equipmentObject")==true,"object must be an independent exported world track: "+object.name);
        require(std::abs(raw.joints[index].scale-1)<1e-6f,"generated prop scale must remain one");
        EquipmentTrack track;track.name=object.name;track.mesh=object.mesh;
        for(size_t f=0;f<raw.frames();++f){auto world=world_pose(raw,raw.frame(f));track.poses.push_back(world[index]);track.scales.push_back({1,1,1});}
        for(size_t f=0;f<10;++f)require(same(track.poses[f],object.poses[f]),"object prefix differs from supplied history: "+object.name);
        result.objects.push_back(std::move(track));
    }
    auto recipe_path=fs::path(source.string()+".json");bool recipe_verified=false;
    if(fs::exists(recipe_path)) {
        auto recipe=read_json(recipe_path);require(recipe.value("backend","")=="dex2hoi","sidecar is not a Dex2HOI recipe");
        require(recipe.at("inputSha256")==sha256_file(input)&&recipe.at("historySha256")==sha256_file(history_path),"supplied input/history differs from generation recipe");
        const auto& meshes=recipe.at("meshSha256");require(meshes.is_array()&&meshes.size()==2,"invalid recipe mesh pins");
        for(size_t s=0;s<2;++s)require(original[s].mesh.empty()?meshes[s].is_null():meshes[s]==sha256_file(original[s].mesh),"prop mesh differs from generation recipe");
        recipe_verified=true;
    }
    result.body.validate();result.report={{"backend","dex2hoi"},{"previewOnly",true},{"recipeVerified",recipe_verified},{"defaultCameraBone",binding[5]},{"frames",70}};
    return result;
}

PreviewRequest parse_preview_request(const std::vector<std::string>& args) {
    require(args.size()>=4&&args[2]=="preview","use dex2hoi preview OUTPUT.glb --input HISTORY.json --character CHARACTER.glb");
    PreviewRequest request;request.source=utf8path(args[3]);request.options.body=PreviewBody::character;std::set<std::string> seen;bool camera_option=false;
    for(size_t i=4;i<args.size();++i) {
        const auto& key=args[i];require(seen.insert(key).second,"duplicate option: "+key);
        auto value=[&]()->const std::string&{require(++i<args.size()&&!args[i].empty(),"missing value for "+key);return args[i];};
        if(key=="--input")request.input=utf8path(value());
        else if(key=="--character")request.options.character=utf8path(value());
        else if(key=="--tide")request.options.tide=value();
        else if(key=="--gif")request.options.gif=utf8path(value());
        else if(key=="--capture")request.options.capture=utf8path(value());
        else if(key=="--frames") {
            const auto& text=value();require(!text.empty()&&text.find_first_not_of("0123456789")==std::string::npos,"frames must be 1..360000");
            auto count=std::stoull(text);require(count>=1&&count<=360000,"frames must be 1..360000");request.options.frames=static_cast<int>(count);
        } else if(key=="--first-person")request.options.camera.enabled=true;
        else if(key.starts_with("--camera-")) {
            camera_option=true;auto& camera=request.options.camera;
            if(key=="--camera-mode")camera.mode=value();
            else if(key=="--camera-bone"){camera.bone=value();request.camera_bone_explicit=true;}
            else if(key=="--camera-offset")camera.offset=triplet(value(),100);
            else if(key=="--camera-rotation")camera.rotation=triplet(value(),3600);
            else if(key=="--camera-fov")camera.fov=number(value(),10,150);
            else if(key=="--camera-near")camera.near_plane=number(value(),.001f,1);
            else throw std::runtime_error("Dex2HOI preview: unknown option: "+key);
        } else throw std::runtime_error("Dex2HOI preview: unknown option: "+key);
    }
    require(request.source.extension()==".glb"&&!request.input.empty()&&!request.options.character.empty(),"preview needs OUTPUT.glb, --input and --character");
    require(!camera_option||request.options.camera.enabled,"camera options require --first-person");validate_preview_setup(request.options);
    std::set<std::string> paths{path_key(request.source),path_key(request.input),path_key(request.options.character)};
    for(const auto& path:{request.options.capture,request.options.gif})if(!path.empty()) {
        require(paths.insert(path_key(path)).second,"capture/GIF paths collide with inputs or each other");new_output(path);
    }
    require(request.options.capture.empty()||request.options.capture.extension()==".png","capture needs a .png path");
    require(request.options.gif.empty()||request.options.gif.extension()==".gif","GIF needs a .gif path");return request;
}
int preview_command(const std::vector<std::string>& args) {
    auto request=parse_preview_request(args);auto result=load_preview_result(request.source,request.input);
    if(request.options.camera.enabled&&!request.camera_bone_explicit)request.options.camera.bone=result.report.at("defaultCameraBone").get<std::string>();
    std::vector<std::string> body_names;for(const auto& joint:result.body.joints)body_names.push_back(joint.name);
    auto character=load_rig_glb(request.options.character,body_names);compatible(result.body,character,names(character));
    request.options.equipment=&result;preview(result.body,request.source,request.options);return 0;
}
}
