#include "runtime_debug/tensor_stats.h"
#include "verification/layer_runner.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

struct diff_summary {
    int64_t first_diff = -1;
    int64_t last_diff = -1;
    int64_t diff_count = 0;
    double max_abs_err = 0.0;
    double mean_abs_err = 0.0;
    double cosine_similarity = 0.0;
};

static diff_summary diff_vectors(const std::vector<float> & a, const std::vector<float> & b) {
    diff_summary out{};
    if (a.size() != b.size() || a.empty()) {
        out.diff_count = -1;
        return out;
    }

    double sum_abs = 0.0;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double da = static_cast<double>(a[i]);
        const double db = static_cast<double>(b[i]);
        const double d = std::fabs(da - db);
        if (d != 0.0) {
            if (out.first_diff < 0) {
                out.first_diff = static_cast<int64_t>(i);
            }
            out.last_diff = static_cast<int64_t>(i);
            ++out.diff_count;
        }
        out.max_abs_err = std::max(out.max_abs_err, d);
        sum_abs += d;
        dot += da * db;
        norm_a += da * da;
        norm_b += db * db;
    }
    out.mean_abs_err = sum_abs / static_cast<double>(a.size());
    if (norm_a > 0.0 && norm_b > 0.0) {
        out.cosine_similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
    return out;
}

static void print_stats_json(const tensor_stats & stats) {
    printf("{\"tensor_dtype\":\"f32\",\"tensor_size\":%lld,\"sha256\":\"%s\"}",
            (long long) stats.n_elements,
            stats.sha256.c_str());
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s REFERENCE_GGUF WORKER_GGUF PROMPT MAX_LAYER\n", argv[0]);
        return 2;
    }

    const std::string reference_path = argv[1];
    const std::string worker_path = argv[2];
    const std::string prompt = argv[3];
    const int32_t max_layer = std::stoi(argv[4]);

    layer_runner_ctx reference = layer_runner_load(reference_path);
    layer_runner_ctx worker = layer_runner_load(worker_path);
    if (!reference.model || !worker.model) {
        fprintf(stderr, "diagnose-hidden-boundary: model load failed\n");
        layer_runner_free(reference);
        layer_runner_free(worker);
        return 1;
    }

    const std::vector<llama_token> tokens = layer_runner_tokenize(reference, prompt);
    if (tokens.empty()) {
        fprintf(stderr, "diagnose-hidden-boundary: tokenization failed\n");
        layer_runner_free(reference);
        layer_runner_free(worker);
        return 1;
    }

    for (int32_t layer = 0; layer < max_layer; ++layer) {
        const layer_boundary_sample ref =
                layer_runner_capture_boundary(reference, tokens, layer, true);
        const layer_boundary_sample wrk =
                layer_runner_capture_boundary(worker, tokens, layer, true);
        if (!ref.has_output || !wrk.has_output) {
            printf("{\"status\":\"FAIL\",\"reason\":\"missing output\",\"layer\":%d}\n", layer);
            layer_runner_free(reference);
            layer_runner_free(worker);
            return 1;
        }

        const diff_summary output_diff = diff_vectors(ref.output, wrk.output);
        if (output_diff.diff_count == 0) {
            continue;
        }

        const diff_summary input_diff = diff_vectors(ref.input, wrk.input);
        printf("{\n");
        printf("  \"status\":\"MISMATCH\",\n");
        printf("  \"first_mismatch\":{\n");
        printf("    \"stage_name\":\"forward_layer_%d_output\",\n", layer);
        printf("    \"producer\":\"Reference Runtime\",\n");
        printf("    \"consumer\":\"Entry worker\",\n");
        printf("    \"layer_start\":%d,\n", layer);
        printf("    \"layer_end\":%d,\n", layer + 1);
        printf("    \"tensor_shape\":[1,%d],\n", ref.n_embd);
        printf("    \"reference\":");
        print_stats_json(ref.output_stats);
        printf(",\n");
        printf("    \"worker\":");
        print_stats_json(wrk.output_stats);
        printf(",\n");
        printf("    \"input_diff\":{\"first_differing_element\":%lld,\"last_differing_element\":%lld,\"number_of_differing_floats\":%lld,\"max_error\":%.9g,\"mean_error\":%.9g,\"cosine_similarity\":%.9g},\n",
                (long long) input_diff.first_diff,
                (long long) input_diff.last_diff,
                (long long) input_diff.diff_count,
                input_diff.max_abs_err,
                input_diff.mean_abs_err,
                input_diff.cosine_similarity);
        printf("    \"output_diff\":{\"first_differing_element\":%lld,\"last_differing_element\":%lld,\"number_of_differing_floats\":%lld,\"max_error\":%.9g,\"mean_error\":%.9g,\"cosine_similarity\":%.9g}\n",
                (long long) output_diff.first_diff,
                (long long) output_diff.last_diff,
                (long long) output_diff.diff_count,
                output_diff.max_abs_err,
                output_diff.mean_abs_err,
                output_diff.cosine_similarity);
        printf("  },\n");
        printf("  \"reason\":\"layer input matches but layer output differs; worker output equals layer input, so worker did not execute this layer\"\n");
        printf("}\n");
        layer_runner_free(reference);
        layer_runner_free(worker);
        return 1;
    }

    printf("{\"status\":\"PASS\",\"layers_checked\":%d}\n", max_layer);
    layer_runner_free(reference);
    layer_runner_free(worker);
    return 0;
}
