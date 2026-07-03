#pragma once

#include "tensor_provider.h"

#include "node_agent/layer_store/layer_store.h"
#include "architecture/semantic_runtime_descriptor.h"

#include <memory>
#include <string>

// Layer Store backed tensor provider.

class layer_store_tensor_provider : public tensor_provider {
public:
    layer_store_tensor_provider(layer_store store, semantic_runtime_descriptor rt);

    bool tensor_exists(const std::string & tensor_name) const override;
    bool load_tensor(const std::string & tensor_name, std::vector<uint8_t> & out) const override;
    bool verify_tensor(const std::string & tensor_name, const std::string & expected_checksum) const override;
    uint64_t tensor_size_bytes(const std::string & tensor_name) const override;

private:
    layer_store                  store_;
    semantic_runtime_descriptor  rt_;

    bool resolve_blob_tensor(
            const std::string & tensor_name,
            std::string & blob_id,
            std::string & blob_tensor) const;
};

std::unique_ptr<layer_store_tensor_provider> make_layer_store_tensor_provider(
        const std::string & models_dir,
        const std::string & model_id,
        const model_manifest & manifest);
