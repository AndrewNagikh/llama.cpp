#pragma once

#include <cstdint>
#include <string>
#include <vector>

std::string sha256_hex(const uint8_t * data, size_t len);
std::string sha256_file_hex(const std::string & path);

bool read_file_bytes(const std::string & path, std::vector<uint8_t> & out);
bool read_file_range(const std::string & path, uint64_t offset, uint64_t length, std::vector<uint8_t> & out);

bool float_vectors_near(
        const float * a,
        const float * b,
        size_t n,
        float rel_tol = 1e-4f,
        float abs_tol = 1e-5f);
