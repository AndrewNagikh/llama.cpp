#pragma once

#include <cstdint>
#include <string>

#include "split_tcp_wire.h"

// RFC-0013 Phase 2 — protocol version negotiation (v1 ctrl channel) + runtime flags.

bool runtime_protocol_v2_enabled();

bool runtime_protocol_v1_forced();

void runtime_protocol_log_v1_deprecation(const char * component);

uint32_t runtime_protocol_server_max();

bool runtime_protocol_negotiate(
        int ctrl_fd,
        const std::string & session_id,
        uint32_t requested_protocol,
        uint32_t & agreed_protocol,
        uint32_t & server_max_protocol);

bool runtime_protocol_handle_negotiate_server(int ctrl_fd, const split_gen_a_req & req);

bool runtime_protocol_handle_v2_handshake_server(int ctrl_fd);

bool runtime_protocol_exchange_v2_handshake(
        int ctrl_fd,
        const std::string & session_id,
        uint32_t agreed_protocol,
        uint32_t server_max_protocol);
