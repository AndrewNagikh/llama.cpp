#pragma once

#include "architecture_descriptor.h"
#include "orchestrator/manifest_builder/manifest_builder.h"

const tensor_descriptor * manifest_find_tensor(
        const model_manifest & manifest,
        const std::string & name);

const tensor_descriptor * manifest_find_role(
        const model_manifest & manifest,
        tensor_role role);

bool manifest_has_tensor(const model_manifest & manifest, const std::string & name);

semantic_blob * ensure_blob(
        architecture_descriptor & desc,
        const std::string & id,
        tensor_semantic_role role,
        blob_deploy_target deploy);

void add_tensor_to_blob(
        semantic_blob & blob,
        const tensor_descriptor & tensor);

void populate_layer_blobs(architecture_descriptor & desc, const model_manifest & manifest);

void set_dense_worker_requirements(architecture_descriptor & desc);

bool arch_prefix_matches(const std::string & architecture, const char * prefix);
