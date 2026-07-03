#include "runtime_role_descriptor.h"

#include <algorithm>

nlohmann::json runtime_role_descriptor::to_json() const {
    return {
        { "role", runtime_role_name(role) },
        { "supports_tokenizer", supports_tokenizer },
        { "supports_embedding", supports_embedding },
        { "supports_sampling", supports_sampling },
        { "prefers_gpu", prefers_gpu },
        { "required_memory_bytes", required_memory_bytes },
        { "estimated_compute_bytes", estimated_compute_bytes },
    };
}

runtime_role_descriptor default_descriptor_for_role(
        const runtime_role role,
        const uint64_t     model_weight_bytes) {
    runtime_role_descriptor d{};
    d.role = role;

    const uint64_t w = model_weight_bytes > 0 ? model_weight_bytes : 256ULL * 1024ULL * 1024ULL;

    switch (role) {
        case runtime_role::tokenizer:
            d.supports_tokenizer = true;
            d.prefers_gpu        = false;
            d.required_memory_bytes = std::min<uint64_t>(w / 32, 512ULL * 1024ULL * 1024ULL);
            d.estimated_compute_bytes = 64ULL * 1024ULL * 1024ULL;
            break;
        case runtime_role::embedding:
            d.supports_embedding = true;
            d.prefers_gpu        = true;
            d.required_memory_bytes = std::max<uint64_t>(w / 16, 512ULL * 1024ULL * 1024ULL);
            d.estimated_compute_bytes = w / 32;
            break;
        case runtime_role::pipeline_stage:
            d.prefers_gpu = true;
            d.required_memory_bytes = w / 8;
            d.estimated_compute_bytes = w / 16;
            break;
        case runtime_role::output_head:
            d.prefers_gpu = true;
            d.required_memory_bytes = std::max<uint64_t>(w / 32, 256ULL * 1024ULL * 1024ULL);
            d.estimated_compute_bytes = w / 64;
            break;
        case runtime_role::sampler:
            d.supports_sampling = true;
            d.prefers_gpu       = false;
            d.required_memory_bytes = 128ULL * 1024ULL * 1024ULL;
            d.estimated_compute_bytes = 32ULL * 1024ULL * 1024ULL;
            break;
        default:
            break;
    }
    return d;
}
