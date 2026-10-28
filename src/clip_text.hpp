#pragma once
#include "motion.hpp"
namespace motion {
// OpenAI CLIP byte BPE, Unicode NFC/lowercase and HTML entity preprocessing.
// Raw token data remains a separately hash-verified model installation file.
class ClipTokenizer {
    struct Impl;std::unique_ptr<Impl> impl_;
public:
    explicit ClipTokenizer(const fs::path& gzip_merges);
    ~ClipTokenizer();
    std::array<int64_t,77> encode(const std::string&) const;
};
}
