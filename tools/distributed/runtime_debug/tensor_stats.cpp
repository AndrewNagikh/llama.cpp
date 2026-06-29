#include "tensor_stats.h"

#include "verification/verification_common.h"

#include <cmath>
#include <sstream>

tensor_stats compute_tensor_stats(const float * data, const int64_t n) {
    tensor_stats stats{};
    stats.n_elements = n;
    if (data == nullptr || n <= 0) {
        return stats;
    }

    stats.sha256 = sha256_hex(
            reinterpret_cast<const uint8_t *>(data),
            static_cast<size_t>(n) * sizeof(float));

    double sum  = 0.0;
    double sum2 = 0.0;
    stats.min_val = data[0];
    stats.max_val = data[0];

    for (int64_t i = 0; i < n; ++i) {
        const float v = data[i];
        if (std::isnan(v)) {
            stats.has_nan = true;
        }
        if (std::isinf(v)) {
            stats.has_inf = true;
        }
        stats.min_val = std::min(stats.min_val, v);
        stats.max_val = std::max(stats.max_val, v);
        sum  += v;
        sum2 += static_cast<double>(v) * v;
    }

    stats.mean = static_cast<float>(sum / static_cast<double>(n));
    const double var = sum2 / static_cast<double>(n) -
                       static_cast<double>(stats.mean) * stats.mean;
    stats.stddev = static_cast<float>(std::sqrt(std::max(0.0, var)));
    stats.l2_norm = std::sqrt(sum2);
    return stats;
}

parity_metrics compare_tensors(const float * a, const float * b, const int64_t n) {
    parity_metrics m{};
    if (a == nullptr || b == nullptr || n <= 0) {
        return m;
    }

    double dot_ab = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    double mae    = 0.0;

    for (int64_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        mae += std::fabs(d);
        m.max_abs_err = std::max(m.max_abs_err, std::fabs(d));
        dot_ab += static_cast<double>(a[i]) * b[i];
        norm_a += static_cast<double>(a[i]) * a[i];
        norm_b += static_cast<double>(b[i]) * b[i];
    }

    m.mean_abs_err = mae / static_cast<double>(n);
    m.l2_diff      = std::sqrt(mae * mae * static_cast<double>(n) / static_cast<double>(n));
    {
        double l2 = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            l2 += d * d;
        }
        m.l2_diff = std::sqrt(l2);
    }

    if (norm_a > 0.0 && norm_b > 0.0) {
        m.cosine_sim = dot_ab / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    m.match = float_vectors_near(a, b, static_cast<size_t>(n));
    return m;
}

static void json_escape(std::ostringstream & os, const std::string & s) {
    os << '"';
    for (char c : s) {
        if (c == '"') {
            os << "\\\"";
        } else if (c == '\\') {
            os << "\\\\";
        } else {
            os << c;
        }
    }
    os << '"';
}

std::string tensor_stats_json(const tensor_stats & s) {
    std::ostringstream os;
    os << "{"
       << "\"n_elements\":" << s.n_elements << ","
       << "\"min\":" << s.min_val << ","
       << "\"max\":" << s.max_val << ","
       << "\"mean\":" << s.mean << ","
       << "\"std\":" << s.stddev << ","
       << "\"l2_norm\":" << s.l2_norm << ","
       << "\"has_nan\":" << (s.has_nan ? "true" : "false") << ","
       << "\"has_inf\":" << (s.has_inf ? "true" : "false") << ","
       << "\"sha256\":";
    json_escape(os, s.sha256);
    os << "}";
    return os.str();
}

std::string parity_metrics_json(const parity_metrics & m) {
    std::ostringstream os;
    os << "{"
       << "\"max_abs_err\":" << m.max_abs_err << ","
       << "\"mean_abs_err\":" << m.mean_abs_err << ","
       << "\"l2_diff\":" << m.l2_diff << ","
       << "\"cosine_sim\":" << m.cosine_sim << ","
       << "\"match\":" << (m.match ? "true" : "false")
       << "}";
    return os.str();
}
