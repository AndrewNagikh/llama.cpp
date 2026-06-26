#pragma once

#include "layer_store/layer_store.h"
#include "manifest_builder/manifest_builder.h"

#include <string>

// Populate a layer store from a local GGUF (for verification tests).
bool populate_layer_store_from_gguf(
        layer_store & store,
        const model_manifest & manifest,
        const std::string & source_path);
