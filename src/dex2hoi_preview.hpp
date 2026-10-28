#pragma once
#include "equipment.hpp"

namespace motion::dex2hoi {
struct PreviewRequest {
    fs::path source,input;
    PreviewOptions options;
    bool camera_bone_explicit=false;
};
// Read-only reconstruction of a generated body and independent learned object
// tracks. This opens no model cache/session and never authors replacement motion.
EquipmentResult load_preview_result(const fs::path& source,const fs::path& input);
// Parser/output preflight is exposed for renderer-free contract checks.
PreviewRequest parse_preview_request(const std::vector<std::string>& args);
int preview_command(const std::vector<std::string>& args);
}
