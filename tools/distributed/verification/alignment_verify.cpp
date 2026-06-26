#include "alignment_verify.h"

#include "gguf.h"

alignment_verify_result alignment_verify_file(const std::string & path) {
    alignment_verify_result result;
    result.summary.name = "alignment";

    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = nullptr;

    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "failed to open GGUF";
        return result;
    }

    const size_t alignment   = gguf_get_alignment(ctx);
    const size_t data_offset = gguf_get_data_offset(ctx);

    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        result.issues.push_back("alignment is not a power of two: " + std::to_string(alignment));
    }
    if (data_offset % alignment != 0) {
        result.issues.push_back("data_offset not aligned to " + std::to_string(alignment));
    }

    const size_t data_base = gguf_get_data_offset(ctx);
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        const uint64_t rel = gguf_get_tensor_offset(ctx, i);
        const uint64_t abs = data_base + rel;
        if (abs % alignment != 0) {
            result.issues.push_back(std::string("tensor ") + gguf_get_tensor_name(ctx, i) +
                    " absolute offset " + std::to_string(abs) + " not aligned");
            if (result.issues.size() >= 10) {
                break;
            }
        }
    }

    gguf_free(ctx);
    result.summary.status  = result.issues.empty() ? verify_status::ok : verify_status::fail;
    result.summary.message = result.issues.empty() ? "alignment rules satisfied" : "alignment issues";
    return result;
}
