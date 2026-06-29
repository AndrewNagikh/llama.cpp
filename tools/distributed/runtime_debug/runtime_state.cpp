#include "runtime_state.h"

#include "runtime_debug.h"
#include "trace_recorder.h"
#include "verification/verification_common.h"

#include "llama-distributed.h"

#include <cstdint>
#include <sstream>

bool dist_debug_runtime_state_enabled() {
    dist_debug_load_config();
    return dist_debug_enabled() || dist_debug_env_truthy("LLAMA_RUNTIME_STATE");
}

runtime_state_snapshot capture_runtime_state(
        llama_context * ctx,
        const int32_t step,
        const char * phase,
        const char * worker,
        const int32_t batch_n_tokens,
        const int32_t * token_ids,
        const int32_t * positions,
        const int32_t seq_id,
        const int32_t prev_token,
        const int32_t current_token) {
    runtime_state_snapshot snap{};
    snap.step            = step;
    snap.phase           = phase ? phase : "";
    snap.worker          = worker ? worker : "";
    snap.batch_n_tokens  = batch_n_tokens;
    snap.seq_id          = seq_id;
    snap.prev_token      = prev_token;
    snap.current_token   = current_token;

    if (!ctx) {
        return snap;
    }

    const llama_model * model = llama_get_model(ctx);
    snap.n_embd               = llama_model_n_embd(model);
    snap.layer_start          = 0; // exposed via cparams in future
    snap.layer_end            = llama_model_n_layer(model);

    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        const llama_pos seq_max = llama_memory_seq_pos_max(mem, seq_id);
        snap.kv_entries         = seq_max >= 0 ? static_cast<int32_t>(seq_max + 1) : 0;
        snap.n_past             = snap.kv_entries;
        snap.kv_n_layer         = llama_model_n_layer(model);
    }

    if (token_ids && batch_n_tokens > 0) {
        snap.token_ids.assign(token_ids, token_ids + batch_n_tokens);
    }
    if (positions && batch_n_tokens > 0) {
        snap.positions.assign(positions, positions + batch_n_tokens);
    }

    const float * hidden = llama_get_embeddings(ctx);
    if (hidden != nullptr && batch_n_tokens > 0 && snap.n_embd > 0) {
        snap.hidden_n_tokens = batch_n_tokens;
        snap.hidden_ptr      = hidden;
        snap.hidden_bytes    = (size_t) batch_n_tokens * (size_t) snap.n_embd * sizeof(float);
        snap.hidden_sha256   = sha256_hex(
                reinterpret_cast<const uint8_t *>(hidden),
                snap.hidden_bytes);
    }

    return snap;
}

void dist_debug_emit_runtime_state(
        trace_recorder * rec,
        const runtime_state_snapshot & snap) {
    if (rec == nullptr || !dist_debug_runtime_state_enabled()) {
        return;
    }
    rec->emit_runtime_state(snap);
}

std::string runtime_state_json(const runtime_state_snapshot & snap) {
    std::ostringstream os;
    os << "{"
       << "\"step\":" << snap.step << ","
       << "\"phase\":\"" << snap.phase << "\","
       << "\"worker\":\"" << snap.worker << "\","
       << "\"n_past\":" << snap.n_past << ","
       << "\"batch_n_tokens\":" << snap.batch_n_tokens << ","
       << "\"seq_id\":" << snap.seq_id << ","
       << "\"kv_entries\":" << snap.kv_entries << ","
       << "\"kv_n_layer\":" << snap.kv_n_layer << ","
       << "\"hidden_n_tokens\":" << snap.hidden_n_tokens << ","
       << "\"n_embd\":" << snap.n_embd << ","
       << "\"hidden_ptr\":" << reinterpret_cast<uintptr_t>(snap.hidden_ptr) << ","
       << "\"hidden_bytes\":" << snap.hidden_bytes << ","
       << "\"hidden_sha256\":\"" << snap.hidden_sha256 << "\","
       << "\"prev_token\":" << snap.prev_token << ","
       << "\"current_token\":" << snap.current_token << ","
       << "\"token_ids\":[";
    for (size_t i = 0; i < snap.token_ids.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        os << snap.token_ids[i];
    }
    os << "],\"positions\":[";
    for (size_t i = 0; i < snap.positions.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        os << snap.positions[i];
    }
    os << "]}";
    return os.str();
}
