#include "gguf_inspector.h"

#include "gguf.h"

#include <cstdio>
#include <vector>

static nlohmann::json dims_json(const gguf_context * ctx, const int64_t tensor_id) {
    nlohmann::json dims = nlohmann::json::array();
    // gguf without ggml context: parse from file only gives size, not dims via API without ctx
    // Use gguf_init with ctx for dimensions when available
    (void) ctx;
    (void) tensor_id;
    return dims;
}

nlohmann::json gguf_inspect_file(const std::string & path) {
    nlohmann::json out;
    out["path"] = path;

    gguf_init_params params{};
    params.no_alloc = true;
    ggml_context * ggml_ctx = nullptr;
    params.ctx = &ggml_ctx;

    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        out["error"] = "failed to open GGUF";
        return out;
    }

    out["version"]        = gguf_get_version(ctx);
    out["alignment"]      = gguf_get_alignment(ctx);
    out["metadata_size"]  = gguf_get_meta_size(ctx);
    out["data_offset"]    = gguf_get_data_offset(ctx);
    out["tensor_count"]   = gguf_get_n_tensors(ctx);
    out["kv_count"]       = gguf_get_n_kv(ctx);

    const uint64_t data_base = gguf_get_data_offset(ctx);
    nlohmann::json tensors  = nlohmann::json::array();

    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        const uint64_t rel_off = gguf_get_tensor_offset(ctx, i);
        const uint64_t abs_off = data_base + rel_off;
        nlohmann::json tj = {
            { "name", name },
            { "index", i },
            { "offset", abs_off },
            { "offset_relative", rel_off },
            { "size_bytes", gguf_get_tensor_size(ctx, i) },
            { "ggml_type", ggml_type_name(gguf_get_tensor_type(ctx, i)) },
        };
        if (ggml_ctx) {
            const ggml_tensor * t = ggml_get_tensor(ggml_ctx, name);
            if (t) {
                nlohmann::json dims = nlohmann::json::array();
                for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                    if (t->ne[d] > 1 || d == 0) {
                        dims.push_back(t->ne[d]);
                    }
                }
                tj["dims"] = dims;
            }
        }
        tensors.push_back(std::move(tj));
    }
    out["tensors"] = std::move(tensors);

    if (ggml_ctx) {
        ggml_free(ggml_ctx);
    }
    gguf_free(ctx);
    return out;
}

std::string gguf_inspect_file_json(const std::string & path) {
    return gguf_inspect_file(path).dump(2);
}
