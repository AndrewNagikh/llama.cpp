#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Lightweight checksum helpers for layer blobs (independent of GGUF layout).

std::string compute_blob_checksum(const uint8_t * data, size_t len);

bool checksum_matches(
        const std::string & expected,
        const uint8_t * data,
        size_t len,
        int32_t layer_index,
        uint64_t size_bytes);
