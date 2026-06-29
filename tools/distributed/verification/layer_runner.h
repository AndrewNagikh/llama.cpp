#pragma once

#include "runtime_debug/layer_trace.h"
#include "runtime_debug/tensor_stats.h"

#include "llama.h"

#include <string>
#include <vector>

struct layer_boundary_sample {
    int32_t layer_index = -1;
    int32_t token_index = -1;
    int32_t n_embd      = 0;
    tensor_stats input_stats{};
    tensor_stats output_stats{};
    std::vector<float> input;
    std::vector<float> output;
    std::vector<int32_t> positions;
    int32_t kv_seq_max = -1;
    bool has_input     = false;
    bool has_output    = false;
};

struct layer_runner_ctx {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
};

layer_runner_ctx layer_runner_load(const std::string & model_path, int32_t n_ctx = 512);
void layer_runner_free(layer_runner_ctx & rt);

std::vector<llama_token> layer_runner_tokenize(const layer_runner_ctx & rt, const std::string & prompt);

layer_boundary_sample layer_runner_capture_boundary(
        layer_runner_ctx & rt,
        const std::vector<llama_token> & tokens,
        int32_t layer_index,
        bool capture_input);
