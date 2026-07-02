#include "dist_http_fetch.h"

#include "dist_common.h"
#include "dist_process.h"

#include "httplib.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

struct parsed_http_url {
    bool        https = false;
    std::string host;
    int         port  = 80;
    std::string path  = "/";
};

static bool parse_http_url(const std::string & url, parsed_http_url & out) {
    std::string rest;
    if (url.rfind("https://", 0) == 0) {
        out.https = true;
        out.port  = 443;
        rest      = url.substr(8);
    } else if (url.rfind("http://", 0) == 0) {
        out.https = false;
        out.port  = 80;
        rest      = url.substr(7);
    } else {
        return false;
    }

    if (rest.empty()) {
        return false;
    }

    const auto slash = rest.find('/');
    std::string hostport = slash == std::string::npos ? rest : rest.substr(0, slash);
    out.path = slash == std::string::npos ? "/" : rest.substr(slash);
    if (out.path.empty()) {
        out.path = "/";
    }

    const auto colon = hostport.find(':');
    if (colon != std::string::npos) {
        out.host = hostport.substr(0, colon);
        try {
            out.port = std::stoi(hostport.substr(colon + 1));
        } catch (...) {
            return false;
        }
    } else {
        out.host = hostport;
    }

    return !out.host.empty();
}

static bool read_file_range(
        const std::string & path,
        const uint64_t offset,
        const uint64_t length,
        std::vector<uint8_t> & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        return false;
    }
    out.resize(static_cast<size_t>(length));
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(length));
    return static_cast<bool>(in) || in.gcount() == static_cast<std::streamoff>(length);
}

static httplib::Headers default_http_headers() {
    httplib::Headers headers = {
        { "User-Agent", "distributed-llama-node-agent/0.1" },
        { "Accept", "*/*" },
    };
    const std::string token = dist_hf_token();
    if (!token.empty()) {
        headers.emplace("Authorization", "Bearer " + token);
    }
    return headers;
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
template<typename ClientT>
static bool httplib_range_get_impl(
        ClientT & cli,
        const parsed_http_url & parsed,
        const uint64_t offset,
        const uint64_t length,
        std::vector<uint8_t> & out) {
    if (length == 0) {
        return false;
    }

    cli.set_connection_timeout(30, 0);
    cli.set_read_timeout(900, 0);
    cli.set_follow_location(true);

    const httplib::Headers base_headers = default_http_headers();
    constexpr uint64_t chunk_size = 32 * 1024 * 1024;
    out.clear();
    out.reserve(static_cast<size_t>(length));

    for (uint64_t pos = 0; pos < length; pos += chunk_size) {
        const uint64_t chunk_len = std::min(chunk_size, length - pos);
        const uint64_t chunk_end = offset + pos + chunk_len - 1;

        char range_buf[80];
        snprintf(range_buf, sizeof(range_buf), "bytes=%llu-%llu",
                static_cast<unsigned long long>(offset + pos),
                static_cast<unsigned long long>(chunk_end));

        httplib::Headers chunk_headers = base_headers;
        chunk_headers.emplace("Range", range_buf);

        const auto res = cli.Get(parsed.path.c_str(), chunk_headers);
        if (!res || (res->status != 200 && res->status != 206)) {
            return false;
        }

        out.insert(out.end(), res->body.begin(), res->body.end());
        if (res->body.size() < chunk_len && res->status == 206) {
            return false;
        }
    }

    if (out.size() < length) {
        out.resize(static_cast<size_t>(length));
    }
    return out.size() >= length;
}

static bool httplib_range_get(
        const parsed_http_url & parsed,
        const uint64_t offset,
        const uint64_t length,
        std::vector<uint8_t> & out) {
    if (parsed.https) {
        httplib::SSLClient ssl(parsed.host.c_str(), parsed.port);
        ssl.enable_server_certificate_verification(true);
        return httplib_range_get_impl(ssl, parsed, offset, length, out);
    }
    httplib::Client cli(parsed.host.c_str(), parsed.port);
    return httplib_range_get_impl(cli, parsed, offset, length, out);
}
#endif

#if !defined(_WIN32)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

static bool curl_range_get(
        const std::string & url,
        const uint64_t offset,
        const uint64_t length,
        std::vector<uint8_t> & out) {
    if (length == 0) {
        return false;
    }

    char range_buf[80];
    snprintf(range_buf, sizeof(range_buf), "%llu-%llu",
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(offset + length - 1));

    const std::string token = dist_hf_token();
    std::vector<std::string> argv = {
        "curl", "-sfL", "--max-time", "900", "--range", range_buf,
        "-H", "User-Agent: distributed-llama-node-agent/0.1",
    };
    if (!token.empty()) {
        argv.push_back("-H");
        argv.push_back("Authorization: Bearer " + token);
    }
    argv.push_back(url);

#if defined(_WIN32)
    argv[0] = "curl.exe";
#endif

    std::string err;
    const int rc = dist_process_run_capture_stdout(argv, out, err);
    if (rc != 0) {
        out.clear();
        return false;
    }
    return out.size() >= length;
}

} // namespace

bool dist_http_get_range(
        const std::string & url,
        const uint64_t offset,
        const uint64_t length,
        std::vector<uint8_t> & out) {
    if (length == 0) {
        return false;
    }
    if (url.rfind("file://", 0) == 0) {
        return read_file_range(url.substr(7), offset, length, out);
    }

    parsed_http_url parsed;
    if (!parse_http_url(url, parsed)) {
        return false;
    }

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (httplib_range_get(parsed, offset, length, out)) {
        return true;
    }
#endif

    fprintf(stderr, "node_agent: HTTPS via curl fallback (%s)\n", parsed.host.c_str());
    return curl_range_get(url, offset, length, out);
}
