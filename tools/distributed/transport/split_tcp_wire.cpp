#include "split_tcp_wire.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef int socklen_t;
static bool g_wsa_started = false;
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

// The ctrl/AB/BC protocol is small-message request/response ping-pong;
// with Nagle enabled that pattern hits the delayed-ACK timer (~40 ms per
// round trip: Task 12 measured the v1 bubble at 40.8 ms and Task 17.1A
// attributed the v2 period to two ~41 ms waits per token). Opt out with
// DIST_TCP_NODELAY=0.
static bool split_tcp_nodelay_enabled() {
    static const bool enabled = [] {
        const char * v = std::getenv("DIST_TCP_NODELAY");
        if (v == nullptr || v[0] == '\0') {
            return true;
        }
        return strcmp(v, "0") != 0 && strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0;
    }();
    return enabled;
}

static void split_tcp_set_nodelay(const int fd) {
    if (fd < 0 || !split_tcp_nodelay_enabled()) {
        return;
    }
    int yes = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *) &yes, sizeof(yes));
}

void split_tcp_init() {
#if defined(_WIN32)
    if (!g_wsa_started) {
        WSADATA wsa{};
        if (WSAStartup(MAKEWORD(2, 2), &wsa) == 0) {
            g_wsa_started = true;
        }
    }
#endif
}

void split_tcp_close(const int fd) {
    if (fd < 0) {
        return;
    }
#if defined(_WIN32)
    closesocket(fd);
#else
    close(fd);
#endif
}

static bool io_send_all(int fd, const void * data, size_t size) {
    auto * base = (const uint8_t *) data;
    size_t done = 0;
    while (done < size) {
#if defined(_WIN32)
        const int n = send(fd, (const char *) (base + done), (int) (size - done), 0);
#else
        const ssize_t n = send(fd, base + done, size - done, 0);
#endif
        if (n <= 0) {
            return false;
        }
        done += (size_t) n;
    }
    return true;
}

static bool io_recv_all(int fd, void * data, size_t size) {
    auto * base = (uint8_t *) data;
    size_t done = 0;
    while (done < size) {
#if defined(_WIN32)
        const int n = recv(fd, (char *) (base + done), (int) (size - done), 0);
#else
        const ssize_t n = recv(fd, base + done, size - done, 0);
#endif
        if (n <= 0) {
            return false;
        }
        done += (size_t) n;
    }
    return true;
}

void split_tcp_set_timeouts(int fd, int timeout_ms) {
    if (fd < 0 || timeout_ms <= 0) {
        return;
    }
#if defined(_WIN32)
    const DWORD tv = (DWORD) timeout_ms;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *) &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *) &tv, sizeof(tv));
#else
    timeval tv{};
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

bool split_tcp_send_all(int fd, const void * data, size_t size) {
    return io_send_all(fd, data, size);
}

bool split_tcp_recv_all(int fd, void * data, size_t size) {
    return io_recv_all(fd, data, size);
}

int split_tcp_listen_host(const char * host, int port) {
    split_tcp_init();
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }

    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *) &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) port);

    if (host == nullptr || host[0] == '\0' || strcmp(host, "127.0.0.1") == 0 || strcmp(host, "localhost") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    } else if (strcmp(host, "0.0.0.0") == 0 || strcmp(host, "*") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        split_tcp_close(fd);
        return -1;
    }

    if (bind(fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        split_tcp_close(fd);
        return -1;
    }

    if (listen(fd, 8) != 0) {
        split_tcp_close(fd);
        return -1;
    }

    return fd;
}

int split_tcp_listen(int port) {
    return split_tcp_listen_host("127.0.0.1", port);
}

int split_tcp_accept(int listen_fd) {
    sockaddr_in client{};
    socklen_t len = sizeof(client);
    const int fd = accept(listen_fd, (sockaddr *) &client, &len);
    split_tcp_set_nodelay(fd);
    return fd;
}

int split_tcp_connect(const char * host, int port) {
    if (port <= 0 || port > 65535) {
        return -1;
    }

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    const char * connect_host = host;
    if (connect_host == nullptr || connect_host[0] == '\0') {
        connect_host = "127.0.0.1";
    }

    addrinfo hints{};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo * res = nullptr;
    if (getaddrinfo(connect_host, port_str, &hints, &res) != 0 || res == nullptr) {
        return -1;
    }

    int fd = -1;
    for (addrinfo * ai = res; ai != nullptr; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            continue;
        }

#if !defined(_WIN32)
        const int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }
#endif

        if (connect(fd, ai->ai_addr, (socklen_t) ai->ai_addrlen) == 0) {
#if !defined(_WIN32)
            if (flags >= 0) {
                fcntl(fd, F_SETFL, flags);
            }
#endif
            split_tcp_set_timeouts(fd, 30000);
            split_tcp_set_nodelay(fd);
            freeaddrinfo(res);
            return fd;
        }

#if !defined(_WIN32)
        if (errno == EINPROGRESS) {
            fd_set wfds;
            FD_ZERO(&wfds);
            FD_SET(fd, &wfds);
            timeval tv{};
            tv.tv_sec = 1;
            const int sel = select(fd + 1, nullptr, &wfds, nullptr, &tv);
            if (sel > 0 && FD_ISSET(fd, &wfds)) {
                int so_error = 0;
                socklen_t so_len = sizeof(so_error);
                if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &so_len) == 0 && so_error == 0) {
                    if (flags >= 0) {
                        fcntl(fd, F_SETFL, flags);
                    }
                    split_tcp_set_timeouts(fd, 30000);
                    split_tcp_set_nodelay(fd);
                    freeaddrinfo(res);
                    return fd;
                }
            }
        }
#endif

        split_tcp_close(fd);
        fd = -1;
    }

    freeaddrinfo(res);
    return -1;
}

int split_tcp_connect_retry(const char * host, int port, int retries, int delay_ms) {
    split_tcp_init();
    for (int i = 0; i < retries; ++i) {
        const int fd = split_tcp_connect(host, port);
        if (fd >= 0) {
            return fd;
        }
        if (i + 1 < retries) {
#if defined(_WIN32)
            Sleep((DWORD) delay_ms);
#else
            usleep((useconds_t) delay_ms * 1000);
#endif
        }
    }
    return -1;
}

bool split_tcp_send_hidden(int fd, int32_t n_tokens, int32_t n_embd, int32_t layer_end, const float * data) {
    split_tcp_header hdr{};
    hdr.magic     = SPLIT_TCP_MAGIC;
    hdr.version   = SPLIT_TCP_VERSION;
    hdr.n_tokens  = n_tokens;
    hdr.n_embd    = n_embd;
    hdr.layer_end = layer_end;

    const size_t payload = (size_t) n_tokens * (size_t) n_embd * sizeof(float);

    if (!split_tcp_send_all(fd, &hdr, sizeof(hdr))) {
        return false;
    }

    return split_tcp_send_all(fd, data, payload);
}

bool split_tcp_recv_hidden(int fd, split_tcp_hidden_msg & msg) {
    if (!split_tcp_recv_all(fd, &msg.header, sizeof(msg.header))) {
        return false;
    }

    if (msg.header.magic != SPLIT_TCP_MAGIC || msg.header.version != SPLIT_TCP_VERSION) {
        fprintf(stderr, "split_tcp: invalid header magic/version\n");
        return false;
    }

    if (msg.header.n_tokens <= 0 || msg.header.n_embd <= 0) {
        fprintf(stderr, "split_tcp: invalid dimensions\n");
        return false;
    }

    const size_t nfloats = (size_t) msg.header.n_tokens * (size_t) msg.header.n_embd;
    msg.data.resize(nfloats);

    return split_tcp_recv_all(fd, msg.data.data(), nfloats * sizeof(float));
}

bool split_tcp_write_result(const char * path, const split_tcp_result_file & meta, const float * logits) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }

    const bool ok =
        fwrite(&meta, sizeof(meta), 1, f) == 1 &&
        fwrite(logits, sizeof(float), (size_t) meta.n_vocab, f) == (size_t) meta.n_vocab;

    fclose(f);
    return ok;
}

bool split_tcp_read_result(const char * path, split_tcp_result_file & meta, std::vector<float> & logits) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }

    if (fread(&meta, sizeof(meta), 1, f) != 1 || meta.n_vocab <= 0) {
        fclose(f);
        return false;
    }

    logits.resize((size_t) meta.n_vocab);
    const bool ok = fread(logits.data(), sizeof(float), (size_t) meta.n_vocab, f) == (size_t) meta.n_vocab;
    fclose(f);
    return ok;
}

static bool split_gen_check_magic(uint32_t magic, uint32_t expected) {
    if (magic != expected) {
        fprintf(stderr, "split_gen: bad magic 0x%08x (expected 0x%08x)\n", magic, expected);
        return false;
    }
    return true;
}

bool split_gen_send_req(int fd, split_gen_cmd cmd, int32_t n_tokens, int32_t pos_start,
        int32_t layer_end, int32_t include_logits, const int32_t * tokens) {
    split_gen_a_req req{};
    req.magic           = SPLIT_GEN_MAGIC;
    req.version         = SPLIT_GEN_VERSION;
    req.cmd             = cmd;
    req.n_tokens        = n_tokens;
    req.pos_start       = pos_start;
    req.layer_end       = layer_end;
    req.include_logits  = include_logits;

    if (!split_tcp_send_all(fd, &req, sizeof(req))) {
        return false;
    }

    if (n_tokens > 0 && tokens != nullptr) {
        return split_tcp_send_all(fd, tokens, (size_t) n_tokens * sizeof(int32_t));
    }

    return true;
}

bool split_gen_send_hidden_req(int fd, split_gen_cmd cmd, int32_t n_tokens, int32_t n_embd,
        int32_t pos_start, int32_t layer_end, const float * hidden) {
    split_gen_a_req req{};
    req.magic          = SPLIT_GEN_MAGIC;
    req.version        = SPLIT_GEN_VERSION;
    req.cmd            = cmd;
    req.n_tokens       = n_tokens;
    req.pos_start      = pos_start;
    req.layer_end      = layer_end;
    req.include_logits = 0;

    if (!split_tcp_send_all(fd, &req, sizeof(req))) {
        return false;
    }
    if (!split_tcp_send_all(fd, &n_embd, sizeof(n_embd))) {
        return false;
    }
    if (n_tokens > 0 && n_embd > 0 && hidden != nullptr) {
        return split_tcp_send_all(
                fd, hidden, (size_t) n_tokens * (size_t) n_embd * sizeof(float));
    }
    return true;
}

bool split_gen_recv_req(int fd, split_gen_a_req & req, std::vector<int32_t> & tokens,
        std::vector<float> * hidden, int32_t * n_embd_out) {
    if (!split_tcp_recv_all(fd, &req, sizeof(req))) {
        return false;
    }

    if (!split_gen_check_magic(req.magic, SPLIT_GEN_MAGIC) || req.version != SPLIT_GEN_VERSION) {
        return false;
    }

    tokens.clear();
    if (hidden != nullptr) {
        hidden->clear();
    }
    if (n_embd_out != nullptr) {
        *n_embd_out = 0;
    }

    if (req.n_tokens > 0 &&
            (req.cmd == SPLIT_GEN_CMD_PREFILL || req.cmd == SPLIT_GEN_CMD_DECODE ||
             req.cmd == SPLIT_GEN_CMD_VERIFY)) {
        tokens.resize((size_t) req.n_tokens);
        if (!split_tcp_recv_all(fd, tokens.data(), tokens.size() * sizeof(int32_t))) {
            return false;
        }
    }

    if (req.n_tokens > 0 &&
            (req.cmd == SPLIT_GEN_CMD_PREFILL_HIDDEN || req.cmd == SPLIT_GEN_CMD_DECODE_HIDDEN)) {
        int32_t n_embd = 0;
        if (!split_tcp_recv_all(fd, &n_embd, sizeof(n_embd))) {
            return false;
        }
        if (n_embd_out != nullptr) {
            *n_embd_out = n_embd;
        }
        if (hidden != nullptr && n_embd > 0) {
            hidden->resize((size_t) req.n_tokens * (size_t) n_embd);
            if (!split_tcp_recv_all(fd, hidden->data(), hidden->size() * sizeof(float))) {
                return false;
            }
        }
    }

    return true;
}

bool split_gen_send_resp(int fd, const split_gen_a_resp & resp, const float * logits, int32_t n_vocab) {
    if (!split_tcp_send_all(fd, &resp, sizeof(resp))) {
        return false;
    }

    if (resp.include_logits && logits != nullptr && n_vocab > 0) {
        return split_tcp_send_all(fd, logits, (size_t) n_vocab * sizeof(float));
    }

    return true;
}

bool split_gen_recv_resp(int fd, split_gen_a_resp & resp, std::vector<float> * logits) {
    if (!split_tcp_recv_all(fd, &resp, sizeof(resp))) {
        return false;
    }

    if (!split_gen_check_magic(resp.magic, SPLIT_GEN_MAGIC)) {
        return false;
    }

    if (logits != nullptr) {
        logits->clear();
        if (resp.include_logits && resp.n_vocab > 0) {
            logits->resize((size_t) resp.n_vocab);
            if (!split_tcp_recv_all(fd, logits->data(), logits->size() * sizeof(float))) {
                return false;
            }
        }
    } else if (resp.include_logits && resp.n_vocab > 0) {
        std::vector<float> skip((size_t) resp.n_vocab);
        if (!split_tcp_recv_all(fd, skip.data(), skip.size() * sizeof(float))) {
            return false;
        }
    }

    return true;
}

bool split_ab_send_cmd(int fd, split_ab_cmd cmd) {
    const uint32_t msg[2] = { SPLIT_AB_MAGIC, cmd };
    return split_tcp_send_all(fd, msg, sizeof(msg));
}

bool split_ab_send_hidden(int fd, int32_t n_tokens, int32_t n_embd, int32_t layer_end,
        int32_t pos_start, int32_t include_logits, const float * data) {
    if (!split_ab_send_cmd(fd, SPLIT_AB_CMD_HIDDEN)) {
        return false;
    }

    split_gen_hidden_meta meta{};
    meta.pos_start      = pos_start;
    meta.include_logits = include_logits;

    if (!split_tcp_send_all(fd, &meta, sizeof(meta))) {
        return false;
    }

    return split_tcp_send_hidden(fd, n_tokens, n_embd, layer_end, data);
}

bool split_ab_send_reset(int fd) {
    return split_ab_send_cmd(fd, SPLIT_AB_CMD_RESET);
}

bool split_ab_send_shutdown(int fd) {
    return split_ab_send_cmd(fd, SPLIT_AB_CMD_SHUTDOWN);
}

bool split_ab_send_verify_ids(int fd, int32_t pos_start, const int32_t * ids, int32_t n) {
    if (!split_ab_send_cmd(fd, SPLIT_AB_CMD_VERIFY_IDS)) {
        return false;
    }
    const int32_t hdr[2] = { pos_start, n };
    if (!split_tcp_send_all(fd, hdr, sizeof(hdr))) {
        return false;
    }
    if (n > 0) {
        return split_tcp_send_all(fd, ids, (size_t) n * sizeof(int32_t));
    }
    return true;
}

bool split_ab_recv_verify_ids(int fd, int32_t & pos_start, std::vector<int32_t> & ids) {
    int32_t hdr[2];
    if (!split_tcp_recv_all(fd, hdr, sizeof(hdr))) {
        return false;
    }
    pos_start = hdr[0];
    const int32_t n = hdr[1];
    ids.clear();
    if (n < 0 || n > 1024) {
        return false;
    }
    if (n > 0) {
        ids.resize((size_t) n);
        if (!split_tcp_recv_all(fd, ids.data(), ids.size() * sizeof(int32_t))) {
            return false;
        }
    }
    return true;
}

bool split_ab_recv_cmd(int fd, split_ab_cmd & cmd) {
    uint32_t msg[2];
    if (!split_tcp_recv_all(fd, msg, sizeof(msg))) {
        return false;
    }
    if (!split_gen_check_magic(msg[0], SPLIT_AB_MAGIC)) {
        return false;
    }
    cmd = (split_ab_cmd) msg[1];
    return true;
}

bool split_ab_recv_hidden(int fd, split_tcp_hidden_msg & msg) {
    if (!split_tcp_recv_all(fd, &msg.meta, sizeof(msg.meta))) {
        return false;
    }
    return split_tcp_recv_hidden(fd, msg);
}

bool split_ab_send_b_resp(int fd, const split_gen_b_resp & resp, const float * logits, int32_t n_vocab) {
    if (!split_tcp_send_all(fd, &resp, sizeof(resp))) {
        return false;
    }
    if (resp.include_logits && logits != nullptr && n_vocab > 0) {
        return split_tcp_send_all(fd, logits, (size_t) n_vocab * sizeof(float));
    }
    return true;
}

bool split_ab_recv_b_resp(int fd, split_gen_b_resp & resp, std::vector<float> * logits) {
    if (!split_tcp_recv_all(fd, &resp, sizeof(resp))) {
        return false;
    }
    if (!split_gen_check_magic(resp.magic, SPLIT_GEN_MAGIC)) {
        return false;
    }
    if (logits != nullptr) {
        logits->clear();
        if (resp.include_logits && resp.n_vocab > 0) {
            logits->resize((size_t) resp.n_vocab);
            if (!split_tcp_recv_all(fd, logits->data(), logits->size() * sizeof(float))) {
                return false;
            }
        }
    } else if (resp.include_logits && resp.n_vocab > 0) {
        std::vector<float> skip((size_t) resp.n_vocab);
        if (!split_tcp_recv_all(fd, skip.data(), skip.size() * sizeof(float))) {
            return false;
        }
    }
    return true;
}

bool split_gen3_send_a_resp(int fd, const split_gen3_a_resp & resp, const float * logits, int32_t n_vocab) {
    if (!split_tcp_send_all(fd, &resp, sizeof(resp))) {
        return false;
    }
    if (resp.include_logits && logits != nullptr && n_vocab > 0) {
        return split_tcp_send_all(fd, logits, (size_t) n_vocab * sizeof(float));
    }
    return true;
}

bool split_gen3_recv_a_resp(int fd, split_gen3_a_resp & resp, std::vector<float> * logits) {
    if (!split_tcp_recv_all(fd, &resp, sizeof(resp))) {
        return false;
    }
    if (!split_gen_check_magic(resp.magic, SPLIT_GEN_MAGIC)) {
        return false;
    }
    if (logits != nullptr) {
        logits->clear();
        if (resp.include_logits && resp.n_vocab > 0) {
            logits->resize((size_t) resp.n_vocab);
            if (!split_tcp_recv_all(fd, logits->data(), logits->size() * sizeof(float))) {
                return false;
            }
        }
    } else if (resp.include_logits && resp.n_vocab > 0) {
        std::vector<float> skip((size_t) resp.n_vocab);
        if (!split_tcp_recv_all(fd, skip.data(), skip.size() * sizeof(float))) {
            return false;
        }
    }
    return true;
}

bool split_gen3_send_mid_resp(int fd, const split_gen3_mid_resp & resp) {
    return split_tcp_send_all(fd, &resp, sizeof(resp));
}

bool split_gen3_recv_mid_resp(int fd, split_gen3_mid_resp & resp) {
    if (!split_tcp_recv_all(fd, &resp, sizeof(resp))) {
        return false;
    }
    return split_gen_check_magic(resp.magic, SPLIT_GEN_MAGIC);
}

bool split_gen3_send_c_resp(int fd, const split_gen3_c_resp & resp) {
    return split_tcp_send_all(fd, &resp, sizeof(resp));
}

bool split_gen3_recv_c_resp(int fd, split_gen3_c_resp & resp) {
    if (!split_tcp_recv_all(fd, &resp, sizeof(resp))) {
        return false;
    }
    return split_gen_check_magic(resp.magic, SPLIT_GEN_MAGIC);
}

bool split_gen_send_queue_ack(const int fd, const int32_t queue_depth, const int32_t wave_id, const int32_t status) {
    split_gen_queue_ack ack{};
    ack.magic       = SPLIT_GENQ_MAGIC;
    ack.version     = SPLIT_GEN_VERSION;
    ack.queue_depth = queue_depth;
    ack.wave_id     = wave_id;
    ack.status      = status;
    return split_tcp_send_all(fd, &ack, sizeof(ack));
}

bool split_gen_recv_queue_ack(const int fd, split_gen_queue_ack & ack) {
    if (!split_tcp_recv_all(fd, &ack, sizeof(ack))) {
        return false;
    }
    if (!split_gen_check_magic(ack.magic, SPLIT_GENQ_MAGIC)) {
        return false;
    }
    return true;
}

bool split_gen_send_token_ready(const int fd, const int32_t token_id, const int32_t pos_start, const int32_t wave_id) {
    split_gen_token_ready ready{};
    ready.magic     = SPLIT_GENT_MAGIC;
    ready.version   = SPLIT_GEN_VERSION;
    ready.token_id  = token_id;
    ready.pos_start = pos_start;
    ready.wave_id   = wave_id;
    return split_tcp_send_all(fd, &ready, sizeof(ready));
}

bool split_gen_recv_token_ready(const int fd, split_gen_token_ready & ready) {
    if (!split_tcp_recv_all(fd, &ready, sizeof(ready))) {
        return false;
    }
    if (!split_gen_check_magic(ready.magic, SPLIT_GENT_MAGIC)) {
        return false;
    }
    return true;
}

bool split_gen_peek_ctrl_magic(const int fd, uint32_t & magic_out) {
    uint32_t magic = 0;
#if defined(_WIN32)
    const int n = recv(fd, (char *) &magic, sizeof(magic), MSG_PEEK);
#else
    const ssize_t n = recv(fd, &magic, sizeof(magic), MSG_PEEK);
#endif
    if (n != (int) sizeof(magic)) {
        return false;
    }
    magic_out = magic;
    return true;
}
