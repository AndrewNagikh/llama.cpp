#include "architecture_descriptor.h"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace {

static bool ends_with(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

static bool contains(const std::string & s, const char * needle) {
    return s.find(needle) != std::string::npos;
}

static bool arch_is_moe(const std::string & arch) {
    return arch.find("moe") != std::string::npos;
}

static std::vector<tensor_role_requirement> default_role_requirements(
        const architecture_descriptor & desc) {
    std::vector<tensor_role_requirement> reqs;

    auto add = [&](tensor_semantic_role semantic,
            bool entry, bool middle, bool final, bool replicate) {
        reqs.push_back({ semantic, entry, middle, final, replicate });
    };

    add(tensor_semantic_role::embedding, true, false, desc.tied_embeddings, desc.tied_embeddings);
    add(tensor_semantic_role::output_head, false, false, true, false);
    add(tensor_semantic_role::output_norm, false, false, true, false);
    add(tensor_semantic_role::input_norm, true, false, false, false);
    add(tensor_semantic_role::transformer_layer, false, true, false, false);

    if (desc.is_moe) {
        add(tensor_semantic_role::router, false, true, false, false);
        add(tensor_semantic_role::expert, false, true, false, false);
        add(tensor_semantic_role::shared_expert, false, true, false, false);
    }

    return reqs;
}

} // namespace

std::string tensor_semantic_role_to_string(const tensor_semantic_role role) {
    switch (role) {
        case tensor_semantic_role::embedding:         return "embedding";
        case tensor_semantic_role::output_head:       return "output_head";
        case tensor_semantic_role::output_norm:       return "output_norm";
        case tensor_semantic_role::input_norm:        return "input_norm";
        case tensor_semantic_role::transformer_layer: return "transformer_layer";
        case tensor_semantic_role::attention:         return "attention";
        case tensor_semantic_role::ffn:               return "ffn";
        case tensor_semantic_role::expert:            return "expert";
        case tensor_semantic_role::router:            return "router";
        case tensor_semantic_role::shared_expert:     return "shared_expert";
        case tensor_semantic_role::rotary:            return "rotary";
        case tensor_semantic_role::bias:              return "bias";
        case tensor_semantic_role::gate:              return "gate";
        case tensor_semantic_role::metadata:          return "metadata";
        case tensor_semantic_role::other:             return "other";
        case tensor_semantic_role::unknown:           return "unknown";
    }
    return "unknown";
}

tensor_semantic_role tensor_semantic_role_from_string(const std::string & s) {
    if (s == "embedding")         return tensor_semantic_role::embedding;
    if (s == "output_head")     return tensor_semantic_role::output_head;
    if (s == "output_norm")     return tensor_semantic_role::output_norm;
    if (s == "input_norm")      return tensor_semantic_role::input_norm;
    if (s == "transformer_layer") return tensor_semantic_role::transformer_layer;
    if (s == "attention")       return tensor_semantic_role::attention;
    if (s == "ffn")             return tensor_semantic_role::ffn;
    if (s == "expert")          return tensor_semantic_role::expert;
    if (s == "router")          return tensor_semantic_role::router;
    if (s == "shared_expert")   return tensor_semantic_role::shared_expert;
    if (s == "rotary")          return tensor_semantic_role::rotary;
    if (s == "bias")            return tensor_semantic_role::bias;
    if (s == "gate")            return tensor_semantic_role::gate;
    if (s == "metadata")        return tensor_semantic_role::metadata;
    if (s == "other")           return tensor_semantic_role::other;
    return tensor_semantic_role::unknown;
}

std::string worker_deploy_role_to_string(const worker_deploy_role role) {
    switch (role) {
        case worker_deploy_role::entry:  return "ENTRY";
        case worker_deploy_role::middle: return "MIDDLE";
        case worker_deploy_role::final:  return "FINAL";
        case worker_deploy_role::full:   return "FULL";
    }
    return "UNKNOWN";
}

void classify_tensor_enhanced(const std::string & name, int32_t & layer, tensor_role & role) {
    layer = -1;
    role  = tensor_role::other;

    if (name == "token_embd.weight") {
        role = tensor_role::embedding;
        return;
    }
    if (name == "output.weight") {
        role = tensor_role::lm_head;
        return;
    }
    if (name == "output_norm.weight") {
        role = tensor_role::output_norm;
        return;
    }
    if (name == "norm.weight" || name == "token_embd_norm.weight") {
        role = tensor_role::norm;
        return;
    }

    if (name.rfind("blk.", 0) == 0) {
        const size_t dot = name.find('.', 4);
        if (dot != std::string::npos) {
            try {
                layer = std::stoi(name.substr(4, dot - 4));
                role  = tensor_role::layer;
            } catch (...) {
                role = tensor_role::other;
            }
        }
    }
}

tensor_semantic_role classify_tensor_semantic(
        const std::string & name,
        const tensor_role manifest_role,
        const int32_t layer) {
    switch (manifest_role) {
        case tensor_role::embedding:    return tensor_semantic_role::embedding;
        case tensor_role::lm_head:      return tensor_semantic_role::output_head;
        case tensor_role::output_norm:  return tensor_semantic_role::output_norm;
        case tensor_role::norm:         return tensor_semantic_role::input_norm;
        case tensor_role::layer:        break;
        case tensor_role::other:
        case tensor_role::unknown:
            break;
    }

    if (contains(name, "rope") || contains(name, "rot_embd")) {
        return tensor_semantic_role::rotary;
    }
    if (contains(name, "ffn_gate_inp") || contains(name, ".gate.")) {
        return tensor_semantic_role::router;
    }
    if (contains(name, "_exps.") || contains(name, "expert")) {
        return tensor_semantic_role::expert;
    }
    if (contains(name, "shared_expert")) {
        return tensor_semantic_role::shared_expert;
    }
    if (layer >= 0) {
        if (contains(name, ".attn_")) {
            return tensor_semantic_role::attention;
        }
        if (contains(name, ".ffn_")) {
            return tensor_semantic_role::ffn;
        }
        return tensor_semantic_role::transformer_layer;
    }
    if (ends_with(name, ".bias")) {
        return tensor_semantic_role::bias;
    }
    return tensor_semantic_role::other;
}

architecture_descriptor build_architecture_descriptor(const model_manifest & manifest) {
    architecture_descriptor desc;
    desc.architecture     = manifest.architecture;
    desc.n_layer          = manifest.n_layer;
    desc.special_tensors  = manifest.special_tensors;
    desc.is_moe           = arch_is_moe(manifest.architecture);

    desc.has_separate_output = manifest.special_tensors.count("lm_head") > 0;
    desc.tied_embeddings     = !desc.has_separate_output;

    for (const auto & kv : manifest.special_tensors) {
        if (kv.first == "output_norm") {
            desc.has_output_norm = true;
        }
    }

    desc.tensors.reserve(manifest.tensors.size());
    for (const auto & t : manifest.tensors) {
        semantic_tensor_entry entry;
        entry.name          = t.name;
        entry.manifest_role = t.role;
        entry.layer         = t.layer;
        entry.semantic      = classify_tensor_semantic(t.name, t.role, t.layer);
        if (entry.semantic == tensor_semantic_role::output_head) {
            desc.has_separate_output = true;
            desc.tied_embeddings     = false;
        }
        desc.tensors.push_back(std::move(entry));
    }

    if (!desc.has_separate_output) {
        desc.tied_embeddings = true;
    }

    desc.role_requirements = default_role_requirements(desc);
    return desc;
}

bool architecture_output_satisfied_by_embedding(const architecture_descriptor & desc) {
    return desc.tied_embeddings;
}

bool architecture_worker_needs_embedding_blob(
        const architecture_descriptor & desc,
        const worker_deploy_role role) {
    if (role == worker_deploy_role::entry || role == worker_deploy_role::full) {
        return true;
    }
    if (desc.tied_embeddings &&
            (role == worker_deploy_role::final || role == worker_deploy_role::full)) {
        return true;
    }
    return false;
}

bool architecture_should_replicate_embedding(const architecture_descriptor & desc) {
    return desc.tied_embeddings;
}

bool architecture_materialize_needs_embedding_for_output(
        const architecture_descriptor & desc,
        const bool include_embedding,
        const bool include_output) {
    return include_output && desc.tied_embeddings && !include_embedding;
}

bool architecture_tensor_included(
        const architecture_descriptor & desc,
        const tensor_descriptor & tensor,
        const int32_t layer_start,
        const int32_t layer_end,
        const bool include_embedding,
        const bool include_output) {
    const tensor_semantic_role semantic =
            classify_tensor_semantic(tensor.name, tensor.role, tensor.layer);

    if (semantic == tensor_semantic_role::embedding) {
        if (include_embedding) {
            return true;
        }
        if (include_output && desc.tied_embeddings) {
            return true;
        }
        return false;
    }

    if (include_embedding && tensor.layer < layer_start &&
            semantic != tensor_semantic_role::output_norm &&
            semantic != tensor_semantic_role::output_head) {
        return true;
    }

    if (include_output) {
        if (semantic == tensor_semantic_role::output_norm ||
                semantic == tensor_semantic_role::output_head) {
            return true;
        }
    }

    return tensor.layer >= layer_start && tensor.layer < layer_end;
}
