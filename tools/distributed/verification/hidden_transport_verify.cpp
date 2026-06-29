#include "hidden_transport_verify.h"

#include "../transport/split_tcp_wire.h"

#include "ggml.h"

#include <cstdio>

#if !defined(_WIN32)
#include <sys/socket.h>
#include <unistd.h>
#endif

hidden_transport_verify_result verify_hidden_tcp_loopback(
        const int32_t n_tokens,
        const int32_t n_embd,
        const float * data) {
    hidden_transport_verify_result result{};
    if (n_tokens <= 0 || n_embd <= 0 || data == nullptr) {
        result.message = "invalid dimensions";
        return result;
    }

#if defined(_WIN32)
    result.message = "loopback not implemented on Windows";
    return result;
#else
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
        result.message = "socketpair failed";
        return result;
    }

    const int32_t layer_end = 1;
    const int32_t pos_start = 0;
    result.send_trace = dist_debug_transport_send(
            0, "loopback", "loop", n_tokens, n_embd, layer_end, pos_start, data, 0.0);

    const int64_t t0 = ggml_time_us();
    const bool sent  = split_ab_send_hidden(fds[0], n_tokens, n_embd, layer_end, pos_start, 0, data);
    split_ab_cmd cmd{};
    const bool got_cmd = split_ab_recv_cmd(fds[1], cmd);
    split_tcp_hidden_msg msg{};
    const bool recv  = got_cmd && cmd == SPLIT_AB_CMD_HIDDEN && split_ab_recv_hidden(fds[1], msg);
    const double ms  = (ggml_time_us() - t0) / 1000.0;

    close(fds[0]);
    close(fds[1]);

    if (!sent || !recv) {
        result.message = "tcp send/recv failed";
        return result;
    }

    result.recv_trace = dist_debug_transport_recv(
            0, "loopback", "loop", msg.header.n_tokens, msg.header.n_embd,
            msg.header.layer_end, msg.meta.pos_start, msg.data.data(), ms);

    size_t diff = SIZE_MAX;
    result.tcp_memcmp_ok = hidden_memcmp_buffers(
            data, msg.data.data(), msg.data.size(), &diff);
    result.tcp_diff_offset = diff;
    result.ok              = result.tcp_memcmp_ok;
    result.message         = result.ok ? "TCP hidden transport bit-exact"
                                         : "TCP hidden mismatch at float " + std::to_string(diff);
    return result;
#endif
}

hidden_transport_verify_result verify_hidden_transport_files(
        const std::string & before_path,
        const std::string & after_path) {
    hidden_transport_verify_result result{};
    size_t diff              = SIZE_MAX;
    result.tcp_memcmp_ok     = hidden_compare_bins(before_path, after_path, &diff);
    result.tcp_diff_offset   = diff;
    result.ok                = result.tcp_memcmp_ok;
    result.message           = result.ok ? "hidden.bin memcmp OK"
                                         : "hidden.bin memcmp FAIL at byte offset " + std::to_string(diff);
    return result;
}
