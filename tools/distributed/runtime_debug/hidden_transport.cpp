#include "hidden_transport.h"

#include "runtime_debug.h"
#include "verification/verification_common.h"

#include "llama-distributed.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

bool dist_debug_transport_dump_enabled() {
    dist_debug_load_config();
    return dist_debug_enabled() ||
           dist_debug_env_truthy("LLAMA_RUNTIME_STATE") ||
           dist_debug_env_truthy("LLAMA_DIST_TRANSPORT_DUMP");
}

static std::string transport_dump_path(
        const char * tag,
        int32_t step,
        const char * link,
        const char * suffix) {
    const dist_debug_config & cfg = dist_debug_config_get();
    std::ostringstream os;
    os << cfg.trace_dir << "/transport_s" << step << "_" << link << "_" << tag << "_" << suffix << ".bin";
    return os.str();
}

bool hidden_write_bin(const std::string & path, const float * data, const size_t nfloats) {
    if (data == nullptr || nfloats == 0) {
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(
            std::filesystem::path(path).parent_path(), ec);
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const size_t wrote = fwrite(data, sizeof(float), nfloats, f);
    fclose(f);
    return wrote == nfloats;
}

bool hidden_memcmp_buffers(
        const float * a,
        const float * b,
        const size_t nfloats,
        size_t * diff_offset) {
    if (a == nullptr || b == nullptr) {
        if (diff_offset) {
            *diff_offset = 0;
        }
        return false;
    }
    for (size_t i = 0; i < nfloats; ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            if (diff_offset) {
                *diff_offset = i;
            }
            return false;
        }
    }
    return true;
}

bool hidden_compare_bins(
        const std::string & path_a,
        const std::string & path_b,
        size_t * diff_offset) {
    FILE * fa = fopen(path_a.c_str(), "rb");
    FILE * fb = fopen(path_b.c_str(), "rb");
    if (!fa || !fb) {
        if (fa) {
            fclose(fa);
        }
        if (fb) {
            fclose(fb);
        }
        return false;
    }

    fseek(fa, 0, SEEK_END);
    fseek(fb, 0, SEEK_END);
    const long sa = ftell(fa);
    const long sb = ftell(fb);
    rewind(fa);
    rewind(fb);

    if (sa != sb || sa <= 0) {
        fclose(fa);
        fclose(fb);
        if (diff_offset) {
            *diff_offset = 0;
        }
        return false;
    }

    std::vector<uint8_t> ba((size_t) sa);
    std::vector<uint8_t> bb((size_t) sb);
    const bool ok_read = fread(ba.data(), 1, ba.size(), fa) == ba.size() &&
                         fread(bb.data(), 1, bb.size(), fb) == bb.size();
    fclose(fa);
    fclose(fb);
    if (!ok_read) {
        return false;
    }

    for (size_t i = 0; i < ba.size(); ++i) {
        if (ba[i] != bb[i]) {
            if (diff_offset) {
                *diff_offset = i / sizeof(float);
            }
            return false;
        }
    }
    return true;
}

std::string hidden_transport_trace_json(const hidden_transport_trace & tr) {
    std::ostringstream os;
    os << "{"
       << "\"step\":" << tr.step << ","
       << "\"phase\":\"" << tr.phase << "\","
       << "\"direction\":\"" << tr.direction << "\","
       << "\"link\":\"" << tr.link << "\","
       << "\"n_tokens\":" << tr.n_tokens << ","
       << "\"n_embd\":" << tr.n_embd << ","
       << "\"layer_end\":" << tr.layer_end << ","
       << "\"pos_start\":" << tr.pos_start << ","
       << "\"payload_bytes\":" << tr.payload_bytes << ","
       << "\"header_bytes\":" << tr.header_bytes << ","
       << "\"alignment\":" << tr.alignment << ","
       << "\"stride\":" << tr.stride << ","
       << "\"dtype\":\"" << tr.dtype << "\","
       << "\"latency_ms\":" << tr.latency_ms << ","
       << "\"sha256\":\"" << tr.sha256 << "\","
       << "\"memcmp_ok\":" << (tr.memcmp_ok ? "true" : "false") << ","
       << "\"dump_before\":\"" << tr.dump_path_before << "\","
       << "\"dump_after\":\"" << tr.dump_path_after << "\"}";
    return os.str();
}

hidden_transport_trace dist_debug_transport_send(
        const int32_t step,
        const char * phase,
        const char * link,
        const int32_t n_tokens,
        const int32_t n_embd,
        const int32_t layer_end,
        const int32_t pos_start,
        const float * data,
        const double latency_ms) {
    hidden_transport_trace tr{};
    tr.step          = step;
    tr.phase         = phase ? phase : "";
    tr.direction     = "send";
    tr.link          = link ? link : "";
    tr.n_tokens      = n_tokens;
    tr.n_embd        = n_embd;
    tr.layer_end     = layer_end;
    tr.pos_start     = pos_start;
    tr.latency_ms    = latency_ms;
    tr.header_bytes  = 20 + 8; // split_tcp_header + split_gen_hidden_meta
    tr.payload_bytes = (size_t) n_tokens * (size_t) n_embd * sizeof(float);

    if (!dist_debug_transport_dump_enabled() || data == nullptr || n_tokens <= 0) {
        return tr;
    }

    const size_t nfloats = (size_t) n_tokens * (size_t) n_embd;
    tr.sha256            = sha256_hex(
            reinterpret_cast<const uint8_t *>(data),
            nfloats * sizeof(float));
    tr.dump_path_before  = transport_dump_path("hidden", step, link, "before_tcp");
    hidden_write_bin(tr.dump_path_before, data, nfloats);
    return tr;
}

hidden_transport_trace dist_debug_transport_recv(
        const int32_t step,
        const char * phase,
        const char * link,
        const int32_t n_tokens,
        const int32_t n_embd,
        const int32_t layer_end,
        const int32_t pos_start,
        const float * data,
        const double latency_ms) {
    hidden_transport_trace tr{};
    tr.step          = step;
    tr.phase         = phase ? phase : "";
    tr.direction     = "recv";
    tr.link          = link ? link : "";
    tr.n_tokens      = n_tokens;
    tr.n_embd        = n_embd;
    tr.layer_end     = layer_end;
    tr.pos_start     = pos_start;
    tr.latency_ms    = latency_ms;
    tr.header_bytes  = 20 + 8;
    tr.payload_bytes = (size_t) n_tokens * (size_t) n_embd * sizeof(float);

    if (!dist_debug_transport_dump_enabled() || data == nullptr || n_tokens <= 0) {
        return tr;
    }

    const size_t nfloats = (size_t) n_tokens * (size_t) n_embd;
    tr.sha256            = sha256_hex(
            reinterpret_cast<const uint8_t *>(data),
            nfloats * sizeof(float));
    tr.dump_path_after   = transport_dump_path("hidden", step, link, "after_tcp");
    hidden_write_bin(tr.dump_path_after, data, nfloats);

    const std::string before = transport_dump_path("hidden", step, link, "before_tcp");
    size_t diff              = SIZE_MAX;
    tr.memcmp_ok             = hidden_compare_bins(before, tr.dump_path_after, &diff);
    return tr;
}

hidden_state_api_check verify_hidden_state_roundtrip(
        llama_context * ctx,
        const float * data,
        const int32_t n_tokens,
        const int32_t n_embd) {
    hidden_state_api_check result{};
    if (!ctx || !data || n_tokens <= 0 || n_embd <= 0) {
        result.message = "invalid args";
        return result;
    }

    const size_t nfloats = (size_t) n_tokens * (size_t) n_embd;
    llama_set_hidden_state(ctx, data, n_tokens);

    std::vector<float> readback(nfloats);
    const int32_t got = llama_get_hidden_state(ctx, readback.data(), (int32_t) nfloats);
    if (got != (int32_t) nfloats) {
        result.message = "get_hidden_state size mismatch got=" + std::to_string(got) +
                         " expected=" + std::to_string(nfloats);
        return result;
    }

    size_t diff = SIZE_MAX;
    result.ok   = hidden_memcmp_buffers(data, readback.data(), nfloats, &diff);
    if (!result.ok) {
        result.diff_offset = diff;
        result.message     = "hidden state roundtrip mismatch at float index " + std::to_string(diff);
    } else {
        result.message = "hidden state roundtrip OK";
    }
    return result;
}
