#include "runtime_compat_materialize.h"

#include "node_agent/layer_store/worker_builder.h"

bool runtime_compat_materialize_worker_gguf(
        const layer_store & store,
        const model_manifest & manifest,
        const semantic_runtime_descriptor & rt,
        const worker_role role,
        const int32_t layer_start,
        const int32_t layer_end,
        const std::string & output_path,
        std::string & err) {
    return materialize_worker_gguf(
            store, manifest, rt, role, layer_start, layer_end, output_path, err);
}
