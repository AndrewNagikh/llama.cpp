#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Fetch a byte range from http(s):// or file:// URL.
bool dist_http_get_range(
        const std::string & url,
        uint64_t offset,
        uint64_t length,
        std::vector<uint8_t> & out);

// Download a whole http(s):// URL straight to a local file (curl -o), no
// range/manifest/layer-store machinery -- for small whole-file fetches like
// a speculative-decoding draft model (Task 19 Phase 3) where the caller
// already knows exactly which file it wants and where it should land.
bool dist_http_download_file(
        const std::string & url,
        const std::string & dest_path,
        std::string & err);
