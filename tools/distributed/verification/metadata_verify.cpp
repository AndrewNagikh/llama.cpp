#include "metadata_verify.h"

#include "gguf.h"

#include <cstring>

static bool key_has_prefix(const char * key, const std::string & prefix) {
    return std::strncmp(key, prefix.c_str(), prefix.size()) == 0;
}

static std::string kv_value_string(const gguf_context * ctx, const int64_t key_id) {
    const auto type = gguf_get_kv_type(ctx, key_id);
    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx, key_id);
        case GGUF_TYPE_UINT32:
            return std::to_string(gguf_get_val_u32(ctx, key_id));
        case GGUF_TYPE_INT32:
            return std::to_string(gguf_get_val_i32(ctx, key_id));
        case GGUF_TYPE_FLOAT32:
            return std::to_string(gguf_get_val_f32(ctx, key_id));
        case GGUF_TYPE_BOOL:
            return gguf_get_val_bool(ctx, key_id) ? "true" : "false";
        default:
            return "<type:" + std::string(gguf_type_name(type)) + ">";
    }
}

metadata_verify_result metadata_verify_kv_prefixes(
        const std::string & original,
        const std::string & materialized,
        const std::vector<std::string> & prefixes) {
    metadata_verify_result result;
    result.summary.name = "metadata";

    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = nullptr;

    gguf_context * octx = gguf_init_from_file(original.c_str(), params);
    gguf_context * mctx = gguf_init_from_file(materialized.c_str(), params);
    if (!octx || !mctx) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "failed to open GGUF for metadata compare";
        if (octx) {
            gguf_free(octx);
        }
        if (mctx) {
            gguf_free(mctx);
        }
        return result;
    }

    for (int64_t i = 0; i < gguf_get_n_kv(octx); ++i) {
        const char * key = gguf_get_key(octx, i);
        bool match_prefix = false;
        for (const auto & p : prefixes) {
            if (key_has_prefix(key, p)) {
                match_prefix = true;
                break;
            }
        }
        if (!match_prefix) {
            continue;
        }

        const int64_t mid = gguf_find_key(mctx, key);
        if (mid < 0) {
            result.differences.push_back(std::string("missing key in materialized: ") + key);
            continue;
        }
        if (gguf_get_kv_type(octx, i) != gguf_get_kv_type(mctx, mid)) {
            result.differences.push_back(std::string("type mismatch for key: ") + key);
            continue;
        }
        const std::string ov = kv_value_string(octx, i);
        const std::string mv = kv_value_string(mctx, mid);
        if (ov != mv) {
            result.differences.push_back(std::string("value mismatch for ") + key + ": " + ov + " vs " + mv);
        }
    }

    gguf_free(octx);
    gguf_free(mctx);

    result.summary.status  = result.differences.empty() ? verify_status::ok : verify_status::fail;
    result.summary.message = result.differences.empty() ? "metadata KV prefixes match" : "metadata differs";
    return result;
}
