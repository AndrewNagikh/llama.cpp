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
