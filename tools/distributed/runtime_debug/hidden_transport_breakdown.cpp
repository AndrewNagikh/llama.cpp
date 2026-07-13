#include "hidden_transport_breakdown.h"

#include "perf_trace.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

bool hidden_gather_sync_split_enabled() {
    static const bool enabled = [] {
        const char * v = std::getenv("DIST_RUNTIME_GATHER_SYNC_SPLIT");
        if (v == nullptr || v[0] == '\0') {
            return false;
        }
        return std::strcmp(v, "0") != 0 &&
               std::strcmp(v, "false") != 0 &&
               std::strcmp(v, "FALSE") != 0;
    }();
    return enabled;
}

static void emit_pack_span(
        const char * begin_event,
        const char * end_event,
        int32_t token_idx,
        int64_t dur_us,
        const char * attrs_json) {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_trace_set_component("entry");
    perf_emit_instant(begin_event, perf_category::SERIALIZATION, "entry", token_idx, attrs_json);
    perf_emit_span(end_event, perf_category::SERIALIZATION, "entry", token_idx, dur_us, attrs_json);
}

bool hidden_pack_gather_stage_hidden(
        llama_context * ctx,
        const int32_t n_tokens,
        const int32_t n_embd,
        std::vector<float> & out,
        hidden_pack_gather_stats & stats) {
    stats = hidden_pack_gather_stats{};
    stats.payload_bytes = n_tokens * n_embd * static_cast<int32_t>(sizeof(float));

    if (n_tokens <= 0) {
        const int64_t t0 = ggml_time_us();
        out.clear();
        stats.alloc_us = ggml_time_us() - t0;
        stats.capacity_before = out.capacity();
        stats.capacity_after  = out.capacity();
        stats.size_before     = out.size();
        stats.size_after      = out.size();
        return true;
    }

    stats.capacity_before = out.capacity();
    stats.size_before     = out.size();

    const int64_t t_alloc0 = ggml_time_us();
    out.resize(static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd));
    stats.alloc_us = ggml_time_us() - t_alloc0;

    stats.capacity_after = out.capacity();
    stats.size_after     = out.size();
    stats.capacity_grew  = stats.capacity_after > stats.capacity_before;

    char alloc_attrs[256];
    std::snprintf(
            alloc_attrs,
            sizeof(alloc_attrs),
            "{\"op\":\"vector_resize\",\"capacity_before\":%zu,\"capacity_after\":%zu,"
            "\"size_before\":%zu,\"size_after\":%zu,\"capacity_grew\":%s,"
            "\"bytes_requested\":%d}",
            stats.capacity_before,
            stats.capacity_after,
            stats.size_before,
            stats.size_after,
            stats.capacity_grew ? "true" : "false",
            stats.payload_bytes);

    if (hidden_gather_sync_split_enabled()) {
        const int64_t t_sync0 = ggml_time_us();
        llama_synchronize(ctx);
        stats.sync_us = ggml_time_us() - t_sync0;
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        const int64_t t_gather0 = ggml_time_us();
        const float * h = n_tokens > 1 ? llama_get_embeddings_ith(ctx, i) : llama_get_embeddings(ctx);
        stats.gather_us += ggml_time_us() - t_gather0;

        if (h == nullptr) {
            return false;
        }

        const int64_t t_copy0 = ggml_time_us();
        std::memcpy(
                out.data() + static_cast<size_t>(i) * static_cast<size_t>(n_embd),
                h,
                static_cast<size_t>(n_embd) * sizeof(float));
        stats.copy_us += ggml_time_us() - t_copy0;
        stats.copy_count += 1;
    }

    // No separate serialization buffer between copy and wire in current path.
    stats.serialize_us = 0;
    return true;
}

bool hidden_pack_send_ab_hidden(
        const int fd,
        const int32_t n_tokens,
        const int32_t n_embd,
        const int32_t layer_end,
        const int32_t pos_start,
        const int32_t include_logits,
        const float * data,
        hidden_pack_send_stats & stats) {
    stats = hidden_pack_send_stats{};

    const int64_t t_frame0 = ggml_time_us();
    if (!split_ab_send_cmd(fd, SPLIT_AB_CMD_HIDDEN)) {
        return false;
    }

    split_gen_hidden_meta meta{};
    meta.pos_start      = pos_start;
    meta.include_logits = include_logits;

    if (!split_tcp_send_all(fd, &meta, sizeof(meta))) {
        return false;
    }

    split_tcp_header hdr{};
    hdr.magic     = SPLIT_TCP_MAGIC;
    hdr.version   = SPLIT_TCP_VERSION;
    hdr.n_tokens  = n_tokens;
    hdr.n_embd    = n_embd;
    hdr.layer_end = layer_end;

    const int64_t t_send_hdr0 = ggml_time_us();
    stats.frame_us = t_send_hdr0 - t_frame0;

    if (!split_tcp_send_all(fd, &hdr, sizeof(hdr))) {
        return false;
    }
    stats.send_hdr_us = ggml_time_us() - t_send_hdr0;

    const size_t payload = static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd) * sizeof(float);
    const int64_t t_payload0 = ggml_time_us();
    if (!split_tcp_send_all(fd, data, payload)) {
        return false;
    }
    stats.send_payload_us = ggml_time_us() - t_payload0;
    stats.send_us           = stats.send_hdr_us + stats.send_payload_us;
    return true;
}

void hidden_pack_emit_breakdown_spans(
        const hidden_pack_stats & stats,
        const int32_t token_idx,
        const int32_t payload_bytes) {
    if (!perf_trace_enabled()) {
        return;
    }

    const hidden_pack_gather_stats & g = stats.gather;
    const hidden_pack_send_stats & s   = stats.send;

    char alloc_attrs[256];
    std::snprintf(
            alloc_attrs,
            sizeof(alloc_attrs),
            "{\"op\":\"vector_resize\",\"capacity_before\":%zu,\"capacity_after\":%zu,"
            "\"capacity_grew\":%s,\"bytes_requested\":%d}",
            g.capacity_before,
            g.capacity_after,
            g.capacity_grew ? "true" : "false",
            payload_bytes);
    emit_pack_span("ALLOC_BEGIN", "ALLOC_END", token_idx, g.alloc_us, alloc_attrs);

    if (hidden_gather_sync_split_enabled()) {
        emit_pack_span(
                "GATHER_SYNC_BEGIN",
                "GATHER_SYNC_END",
                token_idx,
                g.sync_us,
                "{\"op\":\"llama_synchronize\",\"placement\":\"before_gather_loop\"}");
    }

    char gather_attrs[128];
    std::snprintf(gather_attrs, sizeof(gather_attrs), "{\"n_tokens\":%d}", g.copy_count > 0 ? g.copy_count : 1);
    emit_pack_span("GATHER_BEGIN", "GATHER_END", token_idx, g.gather_us, gather_attrs);

    char copy_attrs[160];
    std::snprintf(
            copy_attrs,
            sizeof(copy_attrs),
            "{\"copy_count\":%d,\"bytes_per_copy\":%d,\"source\":\"ggml_embeddings\","
            "\"dest\":\"std::vector\"}",
            g.copy_count,
            g.copy_count > 0 ? payload_bytes / g.copy_count : payload_bytes);
    emit_pack_span("COPY_BEGIN", "COPY_END", token_idx, g.copy_us, copy_attrs);

    emit_pack_span(
            "SERIALIZE_BEGIN",
            "SERIALIZE_END",
            token_idx,
            g.serialize_us,
            "{\"present\":false,\"note\":\"no separate serialize buffer; payload is raw float[]\"}");

    char frame_attrs[128];
    std::snprintf(
            frame_attrs,
            sizeof(frame_attrs),
            "{\"parts\":[\"split_ab_cmd\",\"hidden_meta\",\"tcp_header\"]}");
    emit_pack_span("FRAME_BEGIN", "FRAME_END", token_idx, s.frame_us, frame_attrs);

    char send_attrs[160];
    std::snprintf(
            send_attrs,
            sizeof(send_attrs),
            "{\"tcp_header_us\":%lld,\"tcp_payload_us\":%lld,\"dest\":\"kernel_socket_buffer\"}",
            static_cast<long long>(s.send_hdr_us),
            static_cast<long long>(s.send_payload_us));
    emit_pack_span("SEND_BEGIN", "SEND_END", token_idx, s.send_us, send_attrs);

    const int64_t pack_total =
            g.alloc_us + g.sync_us + g.gather_us + g.copy_us + g.serialize_us + s.frame_us + s.send_us;

    char summary_attrs[560];
    std::snprintf(
            summary_attrs,
            sizeof(summary_attrs),
            "{\"payload_bytes\":%d,\"pack_total_us\":%lld,"
            "\"alloc_us\":%lld,\"sync_us\":%lld,\"gather_us\":%lld,\"copy_us\":%lld,"
            "\"serialize_us\":%lld,\"frame_us\":%lld,\"send_us\":%lld,"
            "\"heap_copy_count\":%d,\"wire_read_count\":1,"
            "\"copy_path\":\"ggml_embeddings->std::vector->kernel_tcp\","
            "\"alloc_per_token\":%s,\"vector_capacity_grew\":%s}",
            payload_bytes,
            static_cast<long long>(pack_total),
            static_cast<long long>(g.alloc_us),
            static_cast<long long>(g.sync_us),
            static_cast<long long>(g.gather_us),
            static_cast<long long>(g.copy_us),
            static_cast<long long>(g.serialize_us),
            static_cast<long long>(s.frame_us),
            static_cast<long long>(s.send_us),
            g.copy_count + 1,
            "true",
            g.capacity_grew ? "true" : "false");
    perf_emit_instant("HIDDEN_PACK_SUMMARY", perf_category::SERIALIZATION, "entry", token_idx, summary_attrs);
    emit_pack_span("HIDDEN_PACK_TOTAL_BEGIN", "HIDDEN_PACK_TOTAL_END", token_idx, pack_total, summary_attrs);

    // Legacy aggregate span (pre-15.1) for regression compatibility.
    const int64_t legacy_serialize_us = g.alloc_us + g.sync_us + g.gather_us + g.copy_us + g.serialize_us;
    perf_emit_span(
            "SERIALIZE_HIDDEN_END",
            perf_category::SERIALIZATION,
            "entry",
            token_idx,
            legacy_serialize_us,
            summary_attrs);
    perf_emit_hidden_transfer(
            "entry",
            token_idx,
            "ab",
            payload_bytes,
            legacy_serialize_us,
            s.send_us,
            0,
            0);
    perf_emit_span("ENTRY_SEND_END", perf_category::NETWORK, "entry", token_idx, s.send_us, send_attrs);
}
