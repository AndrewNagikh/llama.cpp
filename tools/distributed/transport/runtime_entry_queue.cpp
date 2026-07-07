#include "runtime_entry_queue.h"

#include "runtime_protocol.h"
#include "split_wave_wire.h"
#include "../workers/wave_inbound_queue.h"

#include <cstdio>

bool runtime_entry_queue_client_enabled(const uint32_t agreed_protocol) {
    return agreed_protocol >= DIST_RUNTIME_PROTOCOL_V2 && runtime_entry_queue_enabled();
}

bool runtime_stage_queue_client_enabled(const uint32_t agreed_protocol) {
    return agreed_protocol >= DIST_RUNTIME_PROTOCOL_V2 && runtime_stage_queue_enabled();
}

bool runtime_client_pipeline_client_enabled(const uint32_t agreed_protocol) {
    return agreed_protocol >= DIST_RUNTIME_PROTOCOL_V2 && runtime_client_pipeline_enabled();
}

bool pipeline_send_gen_req(
        const int ctrl_fd,
        const split_gen_cmd cmd,
        const int32_t n_tokens,
        const int32_t pos_start,
        const int32_t layer_end,
        const int32_t * tokens) {
    return split_gen_send_req(ctrl_fd, cmd, n_tokens, pos_start, layer_end, 0, tokens);
}

bool pipeline_send_gen_hidden_req(
        const int ctrl_fd,
        const split_gen_cmd cmd,
        const int32_t n_tokens,
        const int32_t n_embd,
        const int32_t pos_start,
        const int32_t layer_end,
        const float * hidden) {
    return split_gen_send_hidden_req(ctrl_fd, cmd, n_tokens, n_embd, pos_start, layer_end, hidden);
}

bool pipeline_recv_queue_ack(const int ctrl_fd, int32_t & queue_depth, int32_t & wave_id) {
    split_gen_queue_ack ack{};
    if (!split_gen_recv_queue_ack(ctrl_fd, ack)) {
        return false;
    }
    if (ack.status != 0) {
        fprintf(stderr, "pipeline: queue ack status=%d\n", ack.status);
        return false;
    }
    queue_depth = ack.queue_depth;
    wave_id     = ack.wave_id;
    return true;
}

bool pipeline_recv_token_ready(const int ctrl_fd, int32_t & token_id, int32_t & wave_id) {
    split_gen_token_ready ready{};
    if (!split_gen_recv_token_ready(ctrl_fd, ready)) {
        return false;
    }
    token_id = ready.token_id;
    wave_id  = ready.wave_id;
    return true;
}

bool pipeline_recv_gen_complete(const int ctrl_fd, split_gen3_a_resp & resp) {
    return split_gen3_recv_a_resp(ctrl_fd, resp, nullptr);
}

bool pipeline_send_drain_pending(const int ctrl_fd, const int32_t layer_end) {
    return pipeline_send_gen_req(ctrl_fd, SPLIT_GEN_CMD_DRAIN_PENDING, 0, 0, layer_end, nullptr);
}

static bool recv_token_then_complete(
        const int ctrl_fd,
        split_gen3_a_resp & resp,
        int32_t * out_token_early) {
    int32_t token_id = -1;
    int32_t wave_id  = -1;
    if (!pipeline_recv_token_ready(ctrl_fd, token_id, wave_id)) {
        return false;
    }
    if (out_token_early != nullptr) {
        *out_token_early = token_id;
    }
    if (!pipeline_recv_gen_complete(ctrl_fd, resp)) {
        return false;
    }
    if (resp.token_id < 0) {
        resp.token_id = token_id;
    }
    return true;
}

bool pipeline_gen3_decode_token_queued_step(
        const int ctrl_fd,
        int32_t & cur,
        const int32_t pos,
        const int32_t layer_end,
        split_gen3_a_resp & resp,
        const bool pipeline_next,
        const int32_t next_pos) {
    if (!pipeline_send_gen_req(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, layer_end, &cur)) {
        return false;
    }
    int32_t depth = 0;
    int32_t wave_id = -1;
    if (!pipeline_recv_queue_ack(ctrl_fd, depth, wave_id)) {
        return false;
    }
    int32_t token_id = -1;
    int32_t ready_wave = -1;
    if (!pipeline_recv_token_ready(ctrl_fd, token_id, ready_wave)) {
        return false;
    }
    cur = token_id;
    resp.token_id = token_id;
    if (pipeline_next) {
        if (!pipeline_send_gen_req(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, next_pos, layer_end, &cur)) {
            return false;
        }
        int32_t depth2 = 0;
        int32_t wave2 = -1;
        if (!pipeline_recv_queue_ack(ctrl_fd, depth2, wave2)) {
            return false;
        }
    }
    return pipeline_recv_gen_complete(ctrl_fd, resp);
}

bool pipeline_gen3_decode_hidden_queued_pipelined(
        const int ctrl_fd,
        const int32_t n_embd,
        const int32_t pos,
        const int32_t layer_end,
        const float * hidden,
        split_gen3_a_resp & resp,
        pipeline_embedding_next_fn prepare_next,
        const int32_t next_pos) {
    if (!pipeline_send_gen_hidden_req(
                ctrl_fd, SPLIT_GEN_CMD_DECODE_HIDDEN, 1, n_embd, pos, layer_end, hidden)) {
        return false;
    }
    int32_t depth = 0;
    int32_t wave_id = -1;
    if (!pipeline_recv_queue_ack(ctrl_fd, depth, wave_id)) {
        return false;
    }
    int32_t token_id = -1;
    int32_t ready_wave = -1;
    if (!pipeline_recv_token_ready(ctrl_fd, token_id, ready_wave)) {
        return false;
    }
    resp.token_id = token_id;
    if (prepare_next) {
        std::vector<float> next_hidden;
        int32_t next_n_embd = 0;
        if (!prepare_next(token_id, next_pos, next_hidden, next_n_embd)) {
            return false;
        }
        if (!pipeline_send_gen_hidden_req(
                    ctrl_fd, SPLIT_GEN_CMD_DECODE_HIDDEN, 1, next_n_embd, next_pos, layer_end,
                    next_hidden.data())) {
            return false;
        }
        int32_t depth2 = 0;
        int32_t wave2 = -1;
        if (!pipeline_recv_queue_ack(ctrl_fd, depth2, wave2)) {
            return false;
        }
    }
    return pipeline_recv_gen_complete(ctrl_fd, resp);
}

bool pipeline_gen3_decode_hidden_pipelined(
        const int ctrl_fd,
        const int32_t n_embd,
        const int32_t pos,
        const int32_t layer_end,
        const float * hidden,
        split_gen3_a_resp & resp,
        const bool send_next,
        const int32_t next_pos,
        const float * next_hidden) {
    if (!pipeline_send_gen_hidden_req(
                ctrl_fd, SPLIT_GEN_CMD_DECODE_HIDDEN, 1, n_embd, pos, layer_end, hidden)) {
        return false;
    }
    int32_t depth = 0;
    int32_t wave_id = -1;
    if (!pipeline_recv_queue_ack(ctrl_fd, depth, wave_id)) {
        return false;
    }
    int32_t token_id = -1;
    int32_t ready_wave = -1;
    if (!pipeline_recv_token_ready(ctrl_fd, token_id, ready_wave)) {
        return false;
    }
    resp.token_id = token_id;
    if (send_next && next_hidden != nullptr) {
        if (!pipeline_send_gen_hidden_req(
                    ctrl_fd, SPLIT_GEN_CMD_DECODE_HIDDEN, 1, n_embd, next_pos, layer_end, next_hidden)) {
            return false;
        }
        int32_t depth2 = 0;
        int32_t wave2 = -1;
        if (!pipeline_recv_queue_ack(ctrl_fd, depth2, wave2)) {
            return false;
        }
    }
    return pipeline_recv_gen_complete(ctrl_fd, resp);
}

bool pipeline_gen3_roundtrip_queued(
        const int ctrl_fd,
        const split_gen_cmd cmd,
        const int32_t n_tokens,
        const int32_t pos_start,
        const int32_t layer_end,
        const int32_t * tokens,
        split_gen3_a_resp & resp,
        const bool pipeline_next_allowed,
        int32_t * out_token_early) {
    (void) pipeline_next_allowed;
    if (!pipeline_send_gen_req(ctrl_fd, cmd, n_tokens, pos_start, layer_end, tokens)) {
        return false;
    }
    int32_t depth = 0;
    int32_t wave_id = -1;
    if (!pipeline_recv_queue_ack(ctrl_fd, depth, wave_id)) {
        return false;
    }
    return recv_token_then_complete(ctrl_fd, resp, out_token_early);
}

bool pipeline_gen3_roundtrip_hidden_queued(
        const int ctrl_fd,
        const split_gen_cmd cmd,
        const int32_t n_tokens,
        const int32_t n_embd,
        const int32_t pos_start,
        const int32_t layer_end,
        const float * hidden,
        split_gen3_a_resp & resp,
        const bool pipeline_next_allowed,
        int32_t * out_token_early) {
    if (cmd == SPLIT_GEN_CMD_DECODE_HIDDEN && pipeline_next_allowed) {
        (void) out_token_early;
        fprintf(stderr, "pipeline: decode hidden pipelining requires next_hidden; use pipeline_gen3_decode_hidden_pipelined\n");
        return false;
    }
    if (!pipeline_send_gen_hidden_req(ctrl_fd, cmd, n_tokens, n_embd, pos_start, layer_end, hidden)) {
        return false;
    }
    int32_t depth = 0;
    int32_t wave_id = -1;
    if (!pipeline_recv_queue_ack(ctrl_fd, depth, wave_id)) {
        return false;
    }
    return recv_token_then_complete(ctrl_fd, resp, out_token_early);
}
