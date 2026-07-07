#include "runtime_protocol.h"

#include "split_tcp_wire.h"
#include "split_wave_wire.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>

namespace {

bool env_truthy(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0;
}

uint32_t clamp_protocol(const uint32_t value) {
    if (value < DIST_RUNTIME_PROTOCOL_V1) {
        return DIST_RUNTIME_PROTOCOL_V1;
    }
    if (value > DIST_RUNTIME_PROTOCOL_V2) {
        return DIST_RUNTIME_PROTOCOL_V2;
    }
    return value;
}

} // namespace

bool runtime_protocol_v1_forced() {
    if (env_truthy("DIST_RUNTIME_PROTOCOL_V1")) {
        return true;
    }
    const char * v2 = std::getenv("DIST_RUNTIME_PROTOCOL_V2");
    if (v2 != nullptr && v2[0] != '\0' && !env_truthy("DIST_RUNTIME_PROTOCOL_V2")) {
        return true;
    }
    return false;
}

bool runtime_protocol_v2_enabled() {
    return !runtime_protocol_v1_forced();
}

void runtime_protocol_log_v1_deprecation(const char * component) {
    static std::atomic<bool> logged{ false };
    if (logged.exchange(true)) {
        return;
    }
    fprintf(stderr,
            "runtime_protocol: DEPRECATED v1 RPC path active (%s); "
            "set DIST_RUNTIME_PROTOCOL_V1=0 or unset DIST_RUNTIME_PROTOCOL_V2=0 for v2. "
            "v1 rollback support ends after migration window (RFC-0013 Phase 6).\n",
            component != nullptr ? component : "unknown");
}

uint32_t runtime_protocol_server_max() {
    return runtime_protocol_v2_enabled() ? DIST_RUNTIME_PROTOCOL_V2 : DIST_RUNTIME_PROTOCOL_V1;
}

bool runtime_protocol_negotiate(
        const int ctrl_fd,
        const std::string & session_id,
        const uint32_t requested_protocol,
        uint32_t & agreed_protocol,
        uint32_t & server_max_protocol) {
    (void) session_id;
    const uint32_t requested = clamp_protocol(requested_protocol);
    server_max_protocol = runtime_protocol_server_max();

    split_gen_a_req req{};
    req.magic          = SPLIT_GEN_MAGIC;
    req.version        = SPLIT_GEN_VERSION;
    req.cmd            = SPLIT_GEN_CMD_PROTO_NEGOTIATE;
    req.n_tokens       = (int32_t) requested;
    req.pos_start      = (int32_t) SPLIT_WAVE_ENVELOPE_VERSION;
    req.layer_end      = 0;
    req.include_logits = 0;

    if (!split_tcp_send_all(ctrl_fd, &req, sizeof(req))) {
        return false;
    }

    split_proto_negotiate_resp resp{};
    if (!split_tcp_recv_all(ctrl_fd, &resp, sizeof(resp))) {
        return false;
    }
    if (resp.magic != SPLIT_GEN_MAGIC || resp.version != SPLIT_GEN_VERSION) {
        fprintf(stderr, "runtime_protocol: bad negotiate response magic/version\n");
        return false;
    }

    agreed_protocol     = resp.agreed_protocol;
    server_max_protocol = resp.server_max_protocol;
    return resp.status == 0;
}

bool runtime_protocol_handle_negotiate_server(const int ctrl_fd, const split_gen_a_req & req) {
    const uint32_t requested = clamp_protocol((uint32_t) req.n_tokens);
    const uint32_t server_max = runtime_protocol_server_max();
    const uint32_t agreed = std::min(requested, server_max);

    split_proto_negotiate_resp resp{};
    resp.magic               = SPLIT_GEN_MAGIC;
    resp.version             = SPLIT_GEN_VERSION;
    resp.agreed_protocol     = agreed;
    resp.server_max_protocol = server_max;
    resp.status              = 0;

    if (!split_tcp_send_all(ctrl_fd, &resp, sizeof(resp))) {
        return false;
    }

    if (agreed < DIST_RUNTIME_PROTOCOL_V2) {
        runtime_protocol_log_v1_deprecation("worker");
    }

    if (agreed >= DIST_RUNTIME_PROTOCOL_V2 && runtime_protocol_v2_enabled()) {
        return runtime_protocol_handle_v2_handshake_server(ctrl_fd);
    }
    return true;
}

bool runtime_protocol_handle_v2_handshake_server(const int ctrl_fd) {
    split_wave_envelope_hdr hdr{};
    std::vector<uint8_t> payload;
    if (!split_wave_recv_envelope(ctrl_fd, hdr, payload)) {
        return false;
    }
    if (hdr.event_type != SPLIT_WAVE_EVENT_PROTO_VERSION) {
        fprintf(stderr, "runtime_protocol: expected PROTO_VERSION event, got %u\n", hdr.event_type);
        return false;
    }

    split_wave_proto_version_payload body{};
    if (!split_wave_decode_proto_version(payload, body)) {
        return false;
    }

    const uint32_t agreed = std::min(body.protocol_version, runtime_protocol_server_max());
    std::string session_id(hdr.session_id);
    const size_t end = session_id.find('\0');
    if (end != std::string::npos) {
        session_id.resize(end);
    }

    return split_wave_send_proto_ack(
            ctrl_fd,
            session_id,
            hdr.sequence,
            agreed,
            runtime_protocol_server_max(),
            0);
}

bool runtime_protocol_exchange_v2_handshake(
        const int ctrl_fd,
        const std::string & session_id,
        const uint32_t agreed_protocol,
        const uint32_t server_max_protocol) {
    if (agreed_protocol < DIST_RUNTIME_PROTOCOL_V2) {
        return true;
    }

    if (!split_wave_send_proto_version(
                ctrl_fd,
                session_id,
                0,
                agreed_protocol,
                SPLIT_WAVE_ENVELOPE_VERSION)) {
        return false;
    }

    split_wave_envelope_hdr hdr{};
    std::vector<uint8_t> payload;
    if (!split_wave_recv_envelope(ctrl_fd, hdr, payload)) {
        return false;
    }
    if (hdr.event_type != SPLIT_WAVE_EVENT_PROTO_ACK) {
        fprintf(stderr, "runtime_protocol: expected PROTO_ACK, got event %u\n", hdr.event_type);
        return false;
    }

    split_wave_proto_ack_payload ack{};
    if (!split_wave_decode_proto_ack(payload, ack)) {
        return false;
    }
    if (ack.status != 0) {
        fprintf(stderr, "runtime_protocol: PROTO_ACK status=%u\n", ack.status);
        return false;
    }
    if (ack.agreed_protocol < DIST_RUNTIME_PROTOCOL_V2) {
        fprintf(stderr, "runtime_protocol: server downgraded to v%u after v2 negotiate\n", ack.agreed_protocol);
        return false;
    }
    (void) server_max_protocol;
    return true;
}
