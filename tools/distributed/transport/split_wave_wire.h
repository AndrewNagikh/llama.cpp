#pragma once

#include <cstdint>
#include <string>
#include <vector>

// RFC-0013 Runtime Protocol v2 — asynchronous event envelope (Task 13.2).
// Transport binding (TCP fd) is an implementation detail; framing is normative.

constexpr uint32_t SPLIT_WAVE_MAGIC             = 0x45564157; // 'WAVE'
constexpr uint32_t SPLIT_WAVE_ENVELOPE_VERSION  = 1;

constexpr uint32_t DIST_RUNTIME_PROTOCOL_V1 = 1;
constexpr uint32_t DIST_RUNTIME_PROTOCOL_V2 = 2;

enum split_wave_event_type : uint32_t {
    SPLIT_WAVE_EVENT_PROTO_VERSION = 1,
    SPLIT_WAVE_EVENT_PROTO_ACK     = 2,
    SPLIT_WAVE_EVENT_WAVE_CREATED  = 3,
    SPLIT_WAVE_EVENT_WAVE_QUEUED   = 4,
    SPLIT_WAVE_EVENT_WAVE_COMPLETE = 5,
    SPLIT_WAVE_EVENT_WAVE_CANCEL   = 6,
    SPLIT_WAVE_EVENT_WAVE_ERROR    = 7,
    SPLIT_WAVE_EVENT_BACKPRESSURE  = 8,
};

enum split_wave_event_flags : uint32_t {
    SPLIT_WAVE_FLAG_NONE   = 0,
    SPLIT_WAVE_FLAG_REPLAY = 1u << 0,
};

#pragma pack(push, 1)
struct split_wave_envelope_hdr {
    uint32_t magic;
    uint32_t envelope_version;
    uint32_t event_type;
    uint32_t flags;
    int32_t  wave_id;
    int32_t  sequence;
    char     session_id[36];
    uint32_t payload_len;
};

struct split_wave_proto_version_payload {
    uint32_t protocol_version;
    uint32_t envelope_version;
    uint32_t capabilities;
};

struct split_wave_proto_ack_payload {
    uint32_t agreed_protocol;
    uint32_t server_max_protocol;
    uint32_t status; // 0 = ok
};
#pragma pack(pop)

static_assert(sizeof(split_wave_envelope_hdr) == 64, "split_wave_envelope_hdr size mismatch");
static_assert(sizeof(split_wave_proto_version_payload) == 12, "split_wave_proto_version_payload size mismatch");
static_assert(sizeof(split_wave_proto_ack_payload) == 12, "split_wave_proto_ack_payload size mismatch");

void split_wave_session_id_copy(char dst[36], const std::string & session_id);

bool split_wave_send_envelope(
        int fd,
        uint32_t event_type,
        uint32_t flags,
        int32_t wave_id,
        int32_t sequence,
        const std::string & session_id,
        const void * payload,
        uint32_t payload_len);

bool split_wave_recv_envelope(
        int fd,
        split_wave_envelope_hdr & hdr,
        std::vector<uint8_t> & payload);

bool split_wave_send_proto_version(
        int fd,
        const std::string & session_id,
        int32_t sequence,
        uint32_t requested_protocol,
        uint32_t envelope_version);

bool split_wave_send_proto_ack(
        int fd,
        const std::string & session_id,
        int32_t sequence,
        uint32_t agreed_protocol,
        uint32_t server_max_protocol,
        uint32_t status = 0);

bool split_wave_decode_proto_version(
        const std::vector<uint8_t> & payload,
        split_wave_proto_version_payload & out);

bool split_wave_decode_proto_ack(
        const std::vector<uint8_t> & payload,
        split_wave_proto_ack_payload & out);
