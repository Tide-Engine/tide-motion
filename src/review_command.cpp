#include "review.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace motion {
int review_command(const std::vector<std::string>& args) {
    using Json=nlohmann::json;fs::path plan,output;
    for(size_t i=2;i<args.size();++i){if(args[i]=="-o"||args[i]=="--output"){if(++i==args.size()||!output.empty())throw std::runtime_error("review needs one output");output=utf8path(args[i]);}else if(args[i].starts_with("-"))throw std::runtime_error("Unknown review option: "+args[i]);else {if(!plan.empty())throw std::runtime_error("review takes one versioned manifest");plan=utf8path(args[i]);}}
    if(plan.empty()||output.extension()!=".html")throw std::runtime_error("Use review MANIFEST.json -o REVIEW.html");
    fs::path report=output;report+=".json";
    for(const auto& p:{output,report})if(fs::exists(p)||fs::exists(fs::path(p.string()+".part")))throw std::runtime_error("Review output or partial already exists: "+p.string());
    if(fs::file_size(plan)>1024*1024)throw std::runtime_error("Review manifest exceeds 1 MiB");
    std::ifstream in(plan);Json doc;in>>doc;
    if(!doc.is_object())throw std::runtime_error("Review manifest must be an object");
    for(auto it=doc.begin();it!=doc.end();++it)if(it.key()!="version"&&it.key()!="cases"&&it.key()!="title")throw std::runtime_error("Unknown review manifest field: "+it.key());
    if(doc.at("version")!=1||!doc.at("cases").is_array()||doc.at("cases").empty()||doc.at("cases").size()>24)throw std::runtime_error("Review version 1 needs 1..24 cases");
    auto resolve=[&](const std::string& s){auto p=utf8path(s);return fs::absolute(p.is_absolute()?p:fs::absolute(plan).parent_path()/p).lexically_normal();};
    Json bundle={{"schemaVersion",1},{"title",doc.value("title",std::string("Tide motion review"))},{"cases",Json::array()},{"identity",Json::array()}};
    std::set<std::string> ids;size_t frames=0;
    for(const auto& entry:doc.at("cases")){
        if(!entry.is_object())throw std::runtime_error("Review case must be an object");
        for(auto it=entry.begin();it!=entry.end();++it)if(it.key()!="id"&&it.key()!="label"&&it.key()!="file"&&it.key()!="clip"&&it.key()!="map"&&it.key()!="context"&&it.key()!="settings")throw std::runtime_error("Unknown review case field: "+it.key());
        std::string id=entry.at("id").get<std::string>();if(id.empty()||id.size()>128||!ids.insert(id).second)throw std::runtime_error("Review case needs a unique id (1..128 bytes)");
        auto source=resolve(entry.at("file").get<std::string>());auto sha=sha256_file(source);auto clip=load_glb(source,entry.value("clip",std::string{}),false);Json provenance={{"file",source.string()},{"sha256",sha},{"animation",clip.name},{"presentation","source skeleton"}};
        if(entry.contains("map")){auto map=resolve(entry.at("map").get<std::string>());Clip target;target.joints=soma();for(const auto& j:target.joints)target.poses.push_back(j.rest);auto result=retarget(clip,target,load_mapping(map),true);clip=std::move(result.clip);provenance["mapSha256"]=sha256_file(map);provenance["map"]=map.string();provenance["presentation"]="retargeted SOMA proxy; not original mesh or exact source poses";provenance["discardedNonRootTranslations"]=result.discarded_translations;}
        if(is_soma77(clip))clip=soma77_to_30(clip);
        frames+=clip.frames();if(frames>60000)throw std::runtime_error("Review set exceeds 60000 total frames");
        auto settings=entry.value("settings",Json::object());auto data=review_motion(clip,settings);data["id"]=id;data["label"]=entry.value("label",id);data["context"]=entry.value("context",std::string{});data["source"]=provenance;
        bundle["identity"].push_back({{"id",id},{"source",provenance},{"settings",settings},{"rulesVersion",1}});bundle["cases"].push_back(std::move(data));
    }
    auto html=review_html(bundle);if(html.size()>64*1024*1024)throw std::runtime_error("Review HTML exceeds 64 MiB; use a smaller set");
    auto reportData=bundle;for(auto& c:reportData["cases"]){c.erase("positions");for(auto& p:c["props"])p.erase("positions");}
    if(!output.parent_path().empty())fs::create_directories(output.parent_path());
    // No output is written until every source, mapping and setting has been checked.
    auto write=[](const fs::path& p,const std::string& text){std::ofstream out(p,std::ios::binary);out<<text;out.close();if(!out)throw std::runtime_error("Could not write review: "+p.string());};
    write(report,reportData.dump(2));write(output,html);
    std::cout<<"Saved advisory review "<<output.string()<<" and "<<report.string()<<" ("<<bundle["cases"].size()<<" cases). No quality verdict or clip changes.\n";return 0;
}
}
