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
// Export only the validated source body, with embedded source/input provenance,
// for an explicit equipment-authoring stage. Does not run inference or IK.
void extract_body(const fs::path& source,const fs::path& input,const fs::path& output);
int extract_body_command(const std::vector<std::string>& args);
// Parser/output preflight is exposed for renderer-free contract checks.
PreviewRequest parse_preview_request(const std::vector<std::string>& args);
int preview_command(const std::vector<std::string>& args);
}
