#pragma once
#include "motion.hpp"
#include <nlohmann/json.hpp>

namespace motion::dex2hoi {
inline constexpr const char* upstream_revision="a92aca6b48b7a0c9a45a1465aa984213d3c8ceda";

struct Bundle {
    fs::path root;
    nlohmann::json manifest;
    std::string manifest_sha256;
};

// Checks a manually prepared native bundle. This never downloads weights or
// executes a converter, and verification does not establish rights to artifacts.
Bundle inspect_bundle(const fs::path& root);

// Copies only verified, explicitly listed files into a new private destination.
// Existing destinations and staging directories are never overwritten.
Bundle import_bundle(const fs::path& source,const fs::path& destination);
}
