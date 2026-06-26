#include "semantic_roles.h"

#include <algorithm>
#include <cstring>

namespace {

bool ends_with(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

bool contains(const std::string & s, const char * needle) {
    return s.find(needle) != std::string::npos;
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
    if (s == "embedding")          return tensor_semantic_role::embedding;
    if (s == "output_head")        return tensor_semantic_role::output_head;
    if (s == "output_norm")        return tensor_semantic_role::output_norm;
    if (s == "input_norm")         return tensor_semantic_role::input_norm;
    if (s == "transformer_layer") return tensor_semantic_role::transformer_layer;
    if (s == "attention")          return tensor_semantic_role::attention;
    if (s == "ffn")                return tensor_semantic_role::ffn;
    if (s == "expert")             return tensor_semantic_role::expert;
    if (s == "router")             return tensor_semantic_role::router;
    if (s == "shared_expert")      return tensor_semantic_role::shared_expert;
    if (s == "rotary")             return tensor_semantic_role::rotary;
    if (s == "bias")               return tensor_semantic_role::bias;
    if (s == "gate")               return tensor_semantic_role::gate;
    if (s == "metadata")           return tensor_semantic_role::metadata;
    if (s == "other")              return tensor_semantic_role::other;
    return tensor_semantic_role::unknown;
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
