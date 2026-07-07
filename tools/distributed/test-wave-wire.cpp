#include "transport/split_wave_wire.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(bool ok, const char * msg) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    }
}

int main() {
    char sid[36];
    split_wave_session_id_copy(sid, "sess-test-001");
    check(std::strncmp(sid, "sess-test-001", 13) == 0, "session_id copy");
    check(sid[13] == '\0', "session_id null terminated");

    split_wave_proto_version_payload in{};
    in.protocol_version = DIST_RUNTIME_PROTOCOL_V2;
    in.envelope_version = SPLIT_WAVE_ENVELOPE_VERSION;
    in.capabilities     = 0;

    std::vector<uint8_t> buf(sizeof(in));
    std::memcpy(buf.data(), &in, sizeof(in));

    split_wave_proto_version_payload out{};
    check(split_wave_decode_proto_version(buf, out), "decode proto version");
    check(out.protocol_version == DIST_RUNTIME_PROTOCOL_V2, "protocol version roundtrip");
    check(out.envelope_version == SPLIT_WAVE_ENVELOPE_VERSION, "envelope version roundtrip");

    split_wave_proto_ack_payload ack_in{};
    ack_in.agreed_protocol     = DIST_RUNTIME_PROTOCOL_V2;
    ack_in.server_max_protocol = DIST_RUNTIME_PROTOCOL_V2;
    ack_in.status              = 0;
    std::vector<uint8_t> ack_buf(sizeof(ack_in));
    std::memcpy(ack_buf.data(), &ack_in, sizeof(ack_in));
    split_wave_proto_ack_payload ack_out{};
    check(split_wave_decode_proto_ack(ack_buf, ack_out), "decode proto ack");
    check(ack_out.agreed_protocol == DIST_RUNTIME_PROTOCOL_V2, "ack agreed");

    return failures == 0 ? 0 : 1;
}
