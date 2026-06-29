#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct tensor_stats {
    int64_t  n_elements = 0;
    float    min_val    = 0.0f;
    float    max_val    = 0.0f;
    float    mean       = 0.0f;
    float    stddev     = 0.0f;
    double   l2_norm    = 0.0;
    std::string sha256;
    bool     has_nan    = false;
    bool     has_inf    = false;
};

struct parity_metrics {
    double max_abs_err     = 0.0;
    double mean_abs_err    = 0.0;
    double l2_diff         = 0.0;
    double cosine_sim      = 0.0;
    bool   match           = false;
};

tensor_stats compute_tensor_stats(const float * data, int64_t n);
parity_metrics compare_tensors(const float * a, const float * b, int64_t n);

std::string tensor_stats_json(const tensor_stats & s);
std::string parity_metrics_json(const parity_metrics & m);
