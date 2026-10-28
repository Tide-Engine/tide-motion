#pragma once
#include "motion.hpp"
#include <nlohmann/json.hpp>
namespace motion {
struct UpgradeRelease { std::string version,archive,url,checksum_url; uint64_t bytes=0; };
std::array<unsigned,3> upgrade_version(std::string_view);
std::string upgrade_platform();
UpgradeRelease upgrade_release(const nlohmann::json&,const std::string& platform);
int upgrade(const std::vector<std::string>& args,const std::string& current);
}
