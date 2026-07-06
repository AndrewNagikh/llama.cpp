#include "runtime/runtime_descriptor.h"

#include <cstdio>

static int require(const bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "test-runtime-service: %s\n", message);
        return 1;
    }
    return 0;
}

static bool has_dependency(
        const runtime_service_descriptor & service,
        const runtime_role role) {
    for (const runtime_role dependency : service.dependencies) {
        if (dependency == role) {
            return true;
        }
    }
    return false;
}

int main() {
    const runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("qwen-test", "qwen.arch", 32);

    const runtime_service_descriptor * tokenizer =
            find_runtime_service(desc, runtime_role::tokenizer);
    const runtime_service_descriptor * embedding =
            find_runtime_service(desc, runtime_role::embedding);
    const runtime_service_descriptor * pipeline =
            find_runtime_service(desc, runtime_role::pipeline_stage);
    const runtime_service_descriptor * output =
            find_runtime_service(desc, runtime_role::output_head);
    const runtime_service_descriptor * sampler =
            find_runtime_service(desc, runtime_role::sampler);

    if (require(tokenizer != nullptr, "missing tokenizer") != 0 ||
            require(embedding != nullptr, "missing embedding") != 0 ||
            require(pipeline != nullptr, "missing pipeline") != 0 ||
            require(output != nullptr, "missing output") != 0 ||
            require(sampler != nullptr, "missing sampler") != 0) {
        return 1;
    }

    if (require(tokenizer->output_type == "token_ids", "tokenizer output mismatch") != 0 ||
            require(embedding->input_type == "token_ids", "embedding input mismatch") != 0 ||
            require(embedding->output_type == "hidden_states", "embedding output mismatch") != 0 ||
            require(pipeline->input_type == "hidden_states", "pipeline input mismatch") != 0 ||
            require(pipeline->output_type == "hidden_states", "pipeline output mismatch") != 0 ||
            require(output->input_type == "hidden_states", "output input mismatch") != 0 ||
            require(output->output_type == "logits", "output output mismatch") != 0 ||
            require(sampler->input_type == "logits", "sampler input mismatch") != 0) {
        return 1;
    }

    if (require(has_dependency(*embedding, runtime_role::tokenizer), "embedding dependency mismatch") != 0 ||
            require(has_dependency(*pipeline, runtime_role::embedding), "pipeline dependency mismatch") != 0 ||
            require(has_dependency(*output, runtime_role::pipeline_stage), "output dependency mismatch") != 0 ||
            require(has_dependency(*sampler, runtime_role::output_head), "sampler dependency mismatch") != 0) {
        return 1;
    }

    if (require(tokenizer->supports_replication, "tokenizer replication disabled") != 0 ||
            require(sampler->supports_replication, "sampler replication disabled") != 0 ||
            require(pipeline->supports_parallel, "pipeline parallel disabled") != 0 ||
            require(pipeline->capabilities.can_pipeline, "pipeline capability disabled") != 0) {
        return 1;
    }

    printf("test-runtime-service: OK\n");
    return 0;
}
