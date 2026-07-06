#include "runtime/runtime_descriptor.h"

#include <cstdio>

static int require(const bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "test-runtime-descriptor: %s\n", message);
        return 1;
    }
    return 0;
}

int main() {
    runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("tinyllama", "llama.arch", 22);

    const runtime_descriptor_validation valid = validate_runtime_descriptor(desc);
    if (require(valid.ok(), "valid descriptor rejected") != 0) {
        return 1;
    }

    if (require(find_runtime_service(desc, runtime_role::tokenizer) != nullptr, "missing tokenizer") != 0 ||
            require(find_runtime_service(desc, runtime_role::embedding) != nullptr, "missing embedding") != 0 ||
            require(find_runtime_service(desc, runtime_role::pipeline_stage) != nullptr, "missing pipeline") != 0 ||
            require(find_runtime_service(desc, runtime_role::output_head) != nullptr, "missing output") != 0 ||
            require(find_runtime_service(desc, runtime_role::sampler) != nullptr, "missing sampler") != 0) {
        return 1;
    }

    if (require(find_runtime_resource(desc, "embedding.token") != nullptr, "missing token embedding") != 0 ||
            require(find_runtime_resource(desc, "pipeline.blocks") != nullptr, "missing pipeline blocks") != 0 ||
            require(find_runtime_resource(desc, "output.lm_head") != nullptr, "missing lm head") != 0 ||
            require(find_runtime_resource(desc, "sampler.metadata") != nullptr, "missing sampler metadata") != 0) {
        return 1;
    }

    desc.services.erase(desc.services.begin());
    const runtime_descriptor_validation missing_service = validate_runtime_descriptor(desc);
    if (require(!missing_service.ok(), "missing service accepted") != 0 ||
            require(!missing_service.errors.empty(), "missing service produced no error") != 0) {
        return 1;
    }

    printf("test-runtime-descriptor: OK\n");
    return 0;
}
