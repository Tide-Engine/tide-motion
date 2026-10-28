#pragma once
#include "motion.hpp"
#include <nlohmann/json.hpp>
namespace motion {
// Advisory observations only. No clip is modified or assigned a quality verdict.
nlohmann::json review_motion(const Clip&, const nlohmann::json& settings = nlohmann::json::object());
std::string review_html(const nlohmann::json& bundle);
int review_command(const std::vector<std::string>& args);
}
