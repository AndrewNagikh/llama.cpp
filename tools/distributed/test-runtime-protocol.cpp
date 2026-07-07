#include "transport/runtime_protocol.h"
#include "transport/split_tcp_wire.h"
#include "transport/split_wave_wire.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <sys/socket.h>

static int failures = 0;

static void check(bool ok, const char * msg) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    }
}

static void clear_protocol_env() {
    unsetenv("DIST_RUNTIME_PROTOCOL_V1");
    unsetenv("DIST_RUNTIME_PROTOCOL_V2");
}

static void run_server(const int fd, const bool v2_enabled) {
    clear_protocol_env();
    if (v2_enabled) {
        setenv("DIST_RUNTIME_PROTOCOL_V2", "1", 1);
    } else {
        setenv("DIST_RUNTIME_PROTOCOL_V2", "0", 1);
    }

    split_gen_a_req req{};
    if (!split_tcp_recv_all(fd, &req, sizeof(req))) {
        fprintf(stderr, "server: recv negotiate req failed\n");
        failures++;
        return;
    }
    if (req.cmd != SPLIT_GEN_CMD_PROTO_NEGOTIATE) {
        fprintf(stderr, "server: unexpected cmd %u\n", req.cmd);
        failures++;
        return;
    }
    if (!runtime_protocol_handle_negotiate_server(fd, req)) {
        fprintf(stderr, "server: handle negotiate failed\n");
        failures++;
    }
}

static void test_negotiate_v2_default() {
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
        fprintf(stderr, "socketpair failed\n");
        failures++;
        return;
    }

    clear_protocol_env();
    std::thread server(run_server, fds[1], true);
    check(runtime_protocol_v2_enabled(), "v2 enabled by default (Phase 6)");
    check(!runtime_protocol_v1_forced(), "v1 not forced by default");

    uint32_t agreed = 0;
    uint32_t server_max = 0;
    const bool ok = runtime_protocol_negotiate(
            fds[0], "sess-v2", DIST_RUNTIME_PROTOCOL_V2, agreed, server_max);
    check(ok, "negotiate returns ok");
    check(agreed == DIST_RUNTIME_PROTOCOL_V2, "default agreed v2");
    check(server_max == DIST_RUNTIME_PROTOCOL_V2, "server max v2 by default");

    if (ok && agreed >= DIST_RUNTIME_PROTOCOL_V2) {
        check(
                runtime_protocol_exchange_v2_handshake(fds[0], "sess-v2", agreed, server_max),
                "v2 envelope handshake");
    }

    server.join();
    close(fds[0]);
    close(fds[1]);
}

static void test_negotiate_v1_rollback() {
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
        fprintf(stderr, "socketpair failed\n");
        failures++;
        return;
    }

    clear_protocol_env();
    setenv("DIST_RUNTIME_PROTOCOL_V2", "0", 1);
    std::thread server(run_server, fds[1], false);
    check(runtime_protocol_v1_forced(), "v2=0 forces v1 rollback");
    check(!runtime_protocol_v2_enabled(), "v2 disabled when forced v1");

    uint32_t agreed = 0;
    uint32_t server_max = 0;
    const bool ok = runtime_protocol_negotiate(
            fds[0], "sess-v1", DIST_RUNTIME_PROTOCOL_V2, agreed, server_max);
    check(ok, "v1 rollback negotiate ok");
    check(agreed == DIST_RUNTIME_PROTOCOL_V1, "agreed v1 on rollback");
    check(server_max == DIST_RUNTIME_PROTOCOL_V1, "server max v1 on rollback");

    server.join();
    close(fds[0]);
    close(fds[1]);
    clear_protocol_env();
}

int main() {
    test_negotiate_v2_default();
    test_negotiate_v1_rollback();
    return failures == 0 ? 0 : 1;
}
