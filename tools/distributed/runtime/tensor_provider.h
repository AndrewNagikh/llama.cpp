#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Task 11.7 — abstract tensor access for layer-first runtime (no full GGUF required).

class tensor_provider {
public:
    virtual ~tensor_provider() = default;

    virtual bool tensor_exists(const std::string & tensor_name) const = 0;
    virtual bool load_tensor(const std::string & tensor_name, std::vector<uint8_t> & out) const = 0;
    virtual bool verify_tensor(const std::string & tensor_name, const std::string & expected_checksum) const = 0;
    virtual uint64_t tensor_size_bytes(const std::string & tensor_name) const = 0;
};
