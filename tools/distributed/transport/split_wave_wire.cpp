#include "split_wave_wire.h"

#include "split_tcp_wire.h"

#include <cstdio>
#include <cstring>

namespace {

bool check_wave_magic(uint32_t magic) {
    if (magic != SPLIT_WAVE_MAGIC) {
        fprintf(stderr, "split_wave: bad magic 0x%08x (expected 0x%08x)\n", magic, SPLIT_WAVE_MAGIC);
        return false;
    }
    return true;
}

} // namespace

void split_wave_session_id_copy(char dst[36], const std::string & session_id) {
    std::memset(dst, 0, 36);
    if (session_id.empty()) {
        return;
    }
    const size_t n = session_id.size() < 35 ? session_id.size() : 35;
    std::memcpy(dst, session_id.data(), n);
}

bool split_wave_send_envelope(
        const int fd,
        const uint32_t event_type,
        const uint32_t flags,
        const int32_t wave_id,
        const int32_t sequence,
        const std::string & session_id,
        const void * payload,
        const uint32_t payload_len) {
    split_wave_envelope_hdr hdr{};
    hdr.magic             = SPLIT_WAVE_MAGIC;
    hdr.envelope_version  = SPLIT_WAVE_ENVELOPE_VERSION;
    hdr.event_type        = event_type;
    hdr.flags             = flags;
    hdr.wave_id           = wave_id;
    hdr.sequence          = sequence;
    hdr.payload_len       = payload_len;
    split_wave_session_id_copy(hdr.session_id, session_id);

    if (!split_tcp_send_all(fd, &hdr, sizeof(hdr))) {
        return false;
    }
    if (payload_len > 0 && payload != nullptr) {
        return split_tcp_send_all(fd, payload, payload_len);
    }
    return true;
}

bool split_wave_recv_envelope(
        const int fd,
        split_wave_envelope_hdr & hdr,
        std::vector<uint8_t> & payload) {
    if (!split_tcp_recv_all(fd, &hdr, sizeof(hdr))) {
        return false;
    }
    if (!check_wave_magic(hdr.magic) || hdr.envelope_version != SPLIT_WAVE_ENVELOPE_VERSION) {
        return false;
    }
    payload.clear();
    if (hdr.payload_len == 0) {
        return true;
    }
    payload.resize(hdr.payload_len);
    return split_tcp_recv_all(fd, payload.data(), payload.size());
}

bool split_wave_send_proto_version(
        const int fd,
        const std::string & session_id,
        const int32_t sequence,
        const uint32_t requested_protocol,
        const uint32_t envelope_version) {
    split_wave_proto_version_payload body{};
    body.protocol_version  = requested_protocol;
    body.envelope_version  = envelope_version;
    body.capabilities      = 0;
    return split_wave_send_envelope(
            fd,
            SPLIT_WAVE_EVENT_PROTO_VERSION,
            SPLIT_WAVE_FLAG_NONE,
            -1,
            sequence,
            session_id,
            &body,
            sizeof(body));
}

bool split_wave_send_proto_ack(
        const int fd,
        const std::string & session_id,
        const int32_t sequence,
        const uint32_t agreed_protocol,
        const uint32_t server_max_protocol,
        const uint32_t status) {
    split_wave_proto_ack_payload body{};
    body.agreed_protocol     = agreed_protocol;
    body.server_max_protocol = server_max_protocol;
    body.status              = status;
    return split_wave_send_envelope(
            fd,
            SPLIT_WAVE_EVENT_PROTO_ACK,
            SPLIT_WAVE_FLAG_NONE,
            -1,
            sequence,
            session_id,
            &body,
            sizeof(body));
}

bool split_wave_decode_proto_version(
        const std::vector<uint8_t> & payload,
        split_wave_proto_version_payload & out) {
    if (payload.size() < sizeof(out)) {
        return false;
    }
    std::memcpy(&out, payload.data(), sizeof(out));
    return true;
}

bool split_wave_decode_proto_ack(
        const std::vector<uint8_t> & payload,
        split_wave_proto_ack_payload & out) {
    if (payload.size() < sizeof(out)) {
        return false;
    }
    std::memcpy(&out, payload.data(), sizeof(out));
    return true;
}
