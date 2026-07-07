#pragma once

#include "split_tcp_wire.h"

#include <cstdint>
#include <functional>
#include <vector>

bool runtime_entry_queue_client_enabled(uint32_t agreed_protocol);
bool runtime_stage_queue_client_enabled(uint32_t agreed_protocol);
bool runtime_client_pipeline_client_enabled(uint32_t agreed_protocol);

using pipeline_embedding_next_fn = std::function<bool(
        int32_t output_token,
        int32_t next_pos,
        std::vector<float> & hidden,
        int32_t & n_embd)>;

bool pipeline_send_gen_req(
        int ctrl_fd,
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t pos_start,
        int32_t layer_end,
        const int32_t * tokens);

bool pipeline_send_gen_hidden_req(
        int ctrl_fd,
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t pos_start,
        int32_t layer_end,
        const float * hidden);

bool pipeline_recv_queue_ack(int ctrl_fd, int32_t & queue_depth, int32_t & wave_id);

bool pipeline_recv_token_ready(int ctrl_fd, int32_t & token_id, int32_t & wave_id);

bool pipeline_recv_gen_complete(int ctrl_fd, split_gen3_a_resp & resp);

bool pipeline_send_drain_pending(int ctrl_fd, int32_t layer_end);

bool pipeline_gen3_decode_token_queued_step(
        int ctrl_fd,
        int32_t & cur,
        int32_t pos,
        int32_t layer_end,
        split_gen3_a_resp & resp,
        bool pipeline_next,
        int32_t next_pos);

bool pipeline_gen3_decode_hidden_queued_pipelined(
        int ctrl_fd,
        int32_t n_embd,
        int32_t pos,
        int32_t layer_end,
        const float * hidden,
        split_gen3_a_resp & resp,
        pipeline_embedding_next_fn prepare_next,
        int32_t next_pos);

bool pipeline_gen3_decode_hidden_pipelined(
        int ctrl_fd,
        int32_t n_embd,
        int32_t pos,
        int32_t layer_end,
        const float * hidden,
        split_gen3_a_resp & resp,
        bool send_next,
        int32_t next_pos,
        const float * next_hidden);

bool pipeline_gen3_roundtrip_queued(
        int ctrl_fd,
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t pos_start,
        int32_t layer_end,
        const int32_t * tokens,
        split_gen3_a_resp & resp,
        bool pipeline_next_allowed,
        int32_t * out_token_early);

bool pipeline_gen3_roundtrip_hidden_queued(
        int ctrl_fd,
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t pos_start,
        int32_t layer_end,
        const float * hidden,
        split_gen3_a_resp & resp,
        bool pipeline_next_allowed,
        int32_t * out_token_early);
