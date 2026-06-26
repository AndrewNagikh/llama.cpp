#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

// Single layer blob metadata stored in the local Layer Store (Task 9.7).

struct layer_blob {
    int32_t                 layer_index   = -1;
    uint64_t                offset_begin  = 0;
    uint64_t                offset_end    = 0;
    uint64_t                size_bytes    = 0;
    std::string             checksum;
    std::filesystem::path   path;
};
