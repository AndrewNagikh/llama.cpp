#include "verification_common.h"

extern "C" {
#include "sha256.h"
}

#include <cmath>
#include <fstream>

std::string sha256_hex(const uint8_t * data, const size_t len) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, data, len);
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(SHA256_DIGEST_SIZE * 2);
    for (int i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        out.push_back(hex[digest[i] >> 4]);
        out.push_back(hex[digest[i] & 0x0f]);
    }
    return out;
}

std::string sha256_file_hex(const std::string & path) {
    std::vector<uint8_t> data;
    if (!read_file_bytes(path, data) || data.empty()) {
        return {};
    }
    return sha256_hex(data.data(), data.size());
}

bool read_file_bytes(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return false;
    }
    const auto size = in.tellg();
    if (size < 0) {
        return false;
    }
    in.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(size));
    return static_cast<bool>(in);
}

bool read_file_range(
        const std::string & path,
        const uint64_t offset,
        const uint64_t length,
        std::vector<uint8_t> & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    out.resize(static_cast<size_t>(length));
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(length));
    return static_cast<bool>(in) || in.gcount() == static_cast<std::streamsize>(length);
}

bool float_vectors_near(
        const float * a,
        const float * b,
        const size_t n,
        const float rel_tol,
        const float abs_tol) {
    for (size_t i = 0; i < n; ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        const float scale = std::max(std::fabs(a[i]), std::fabs(b[i]));
        if (diff > abs_tol && diff > rel_tol * scale) {
            return false;
        }
    }
    return true;
}
