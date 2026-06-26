#include "layer_checksum.h"

#include <cstdio>
#include <cstring>

static uint64_t fnv1a64(const uint8_t * data, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(data[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

std::string compute_blob_checksum(const uint8_t * data, size_t len) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(fnv1a64(data, len)));
    return std::string("fnv1a:") + buf;
}

bool checksum_matches(
        const std::string & expected,
        const uint8_t * data,
        size_t len,
        int32_t layer_index,
        uint64_t size_bytes) {
    if (expected.empty()) {
        return len > 0;
    }

    const std::string stub = "manifest:layer:" + std::to_string(layer_index);
    if (expected == stub) {
        return len == size_bytes;
    }

    return expected == compute_blob_checksum(data, len);
}
