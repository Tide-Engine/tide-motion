#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
namespace motion {
struct DartSlice {const char* file;uint64_t source,bytes,destination;bool half;};
std::span<const DartSlice> dart_tensor_slices();
std::string dart_graph_schema(std::string_view file);
}
