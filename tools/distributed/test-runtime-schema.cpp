#include "runtime/runtime_descriptor.h"

#include <cstdio>

static int require(const bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "test-runtime-schema: %s\n", message);
        return 1;
    }
    return 0;
}

int main() {
    runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("gemma-test", "gemma.arch", 26);

    const nlohmann::json j = desc.to_json();
    if (require(j.at("schema_version") == "1.0", "schema version mismatch") != 0 ||
            require(j.at("model_id") == "gemma-test", "model id mismatch") != 0 ||
            require(j.at("architecture_descriptor_id") == "gemma.arch", "architecture id mismatch") != 0 ||
            require(j.at("services").is_array(), "services is not array") != 0 ||
            require(j.at("resources").is_array(), "resources is not array") != 0 ||
            require(j.at("dependencies").is_array(), "dependencies is not array") != 0 ||
            require(j.at("services").size() == 5, "service count mismatch") != 0 ||
            require(j.at("dependencies").size() == 4, "dependency count mismatch") != 0) {
        return 1;
    }

    const runtime_resource_kind kind =
            runtime_resource_kind_from_string("token_embedding");
    if (require(kind == runtime_resource_kind::token_embedding, "resource kind parse failed") != 0 ||
            require(runtime_resource_kind_name(kind) == "token_embedding", "resource kind name failed") != 0) {
        return 1;
    }

    desc.schema_version = "2.0";
    const runtime_descriptor_validation unsupported =
            validate_runtime_descriptor(desc);
    if (require(!unsupported.ok(), "unsupported schema accepted") != 0) {
        return 1;
    }

    desc = make_basic_text_generation_runtime_descriptor("bad-resource", "llama.arch", 4);
    runtime_service_descriptor * embedding = nullptr;
    for (runtime_service_descriptor & service : desc.services) {
        if (service.role == runtime_role::embedding) {
            embedding = &service;
            break;
        }
    }
    if (require(embedding != nullptr, "embedding service missing") != 0) {
        return 1;
    }
    embedding->required_resources.push_back("missing.resource");
    const runtime_descriptor_validation missing_resource =
            validate_runtime_descriptor(desc);
    if (require(!missing_resource.ok(), "missing resource accepted") != 0) {
        return 1;
    }

    printf("test-runtime-schema: OK\n");
    return 0;
}
