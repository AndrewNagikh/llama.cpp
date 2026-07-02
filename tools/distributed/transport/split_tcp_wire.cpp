#include "split_tcp_wire.h"

#include <cerrno>
#include <cstdio>
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
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

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
    return accept(listen_fd, (sockaddr *) &client, &len);
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

        if (connect(fd, ai->ai_addr, (socklen_t) ai->ai_addrlen) == 0) {
            split_tcp_set_timeouts(fd, 30000);
            freeaddrinfo(res);
            return fd;
        }

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

bool split_gen_recv_req(int fd, split_gen_a_req & req, std::vector<int32_t> & tokens) {
    if (!split_tcp_recv_all(fd, &req, sizeof(req))) {
        return false;
    }

    if (!split_gen_check_magic(req.magic, SPLIT_GEN_MAGIC) || req.version != SPLIT_GEN_VERSION) {
        return false;
    }

    tokens.clear();
    if (req.n_tokens > 0 &&
            (req.cmd == SPLIT_GEN_CMD_PREFILL || req.cmd == SPLIT_GEN_CMD_DECODE)) {
        tokens.resize((size_t) req.n_tokens);
        if (!split_tcp_recv_all(fd, tokens.data(), tokens.size() * sizeof(int32_t))) {
            return false;
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
