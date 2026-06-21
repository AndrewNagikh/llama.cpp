#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Wire format for hidden-state transfer between split inference processes.
//
// Byte order: native (little-endian on x86/ARM). Single-machine localhost only.
//
// Message layout:
//   split_tcp_header  (fixed size)
//   float payload     [header.n_tokens * header.n_embd]
//
// float layout: row-major per token, same as llama batch.embd / llama_get_embeddings().

constexpr uint32_t SPLIT_TCP_MAGIC   = 0x48505453; // 'SPTH'
constexpr uint32_t SPLIT_TCP_VERSION = 1;

#pragma pack(push, 1)
struct split_tcp_header {
    uint32_t magic;
    uint32_t version;
    int32_t  n_tokens;
    int32_t  n_embd;
    int32_t  layer_end; // exclusive upper bound of sender layer range
};
#pragma pack(pop)

static_assert(sizeof(split_tcp_header) == 20, "split_tcp_header size mismatch");

#pragma pack(push, 1)
struct split_gen_hidden_meta {
    int32_t pos_start;
    int32_t include_logits;
};
#pragma pack(pop)

struct split_tcp_hidden_msg {
    split_tcp_header header;
    split_gen_hidden_meta meta{};
    std::vector<float> data;
};

struct split_tcp_result_file {
    int32_t  n_vocab;
    int32_t  n_tokens;
    int32_t  n_embd;
    int32_t  layer_start;
    int32_t  layer_end;
    double   ms_recv;
    double   ms_decode;
    double   ms_total;
    // followed by float logits[n_vocab]
};

struct split_tcp_perf {
    double ms_compute = 0.0;
    double ms_send    = 0.0;
    double ms_total   = 0.0;
    size_t nbytes     = 0;
};

int  split_tcp_listen(int port);
int  split_tcp_accept(int listen_fd);
int  split_tcp_connect(const char * host, int port);

bool split_tcp_send_all(int fd, const void * data, size_t size);
bool split_tcp_recv_all(int fd, void * data, size_t size);

bool split_tcp_send_hidden(int fd, int32_t n_tokens, int32_t n_embd, int32_t layer_end, const float * data);
bool split_tcp_recv_hidden(int fd, split_tcp_hidden_msg & msg);

bool split_tcp_write_result(const char * path, const split_tcp_result_file & meta, const float * logits);
bool split_tcp_read_result(const char * path, split_tcp_result_file & meta, std::vector<float> & logits);

// --- autoregressive split generation protocol (Task 3) ---

constexpr uint32_t SPLIT_GEN_MAGIC   = 0x47454E53; // 'SNEG'
constexpr uint32_t SPLIT_GEN_VERSION = 1;

constexpr uint32_t SPLIT_AB_MAGIC = 0x42415350; // 'PSAB'

enum split_gen_cmd : uint32_t {
    SPLIT_GEN_CMD_RESET    = 1,
    SPLIT_GEN_CMD_PREFILL  = 2,
    SPLIT_GEN_CMD_DECODE   = 3,
    SPLIT_GEN_CMD_SHUTDOWN = 4,
};

enum split_ab_cmd : uint32_t {
    SPLIT_AB_CMD_RESET   = 1,
    SPLIT_AB_CMD_HIDDEN  = 2,
    SPLIT_AB_CMD_SHUTDOWN = 3,
};

#pragma pack(push, 1)
struct split_gen_a_req {
    uint32_t magic;
    uint32_t version;
    uint32_t cmd;
    int32_t  n_tokens;
    int32_t  pos_start;
    int32_t  layer_end;
    int32_t  include_logits;
};

struct split_gen_a_resp {
    uint32_t magic;
    uint32_t version;
    int32_t  token_id;
    int32_t  include_logits;
    int32_t  n_vocab;
    double   ms_a_compute;
    double   ms_ab_xfer;
    double   ms_b_compute;
};

struct split_gen_b_resp {
    uint32_t magic;
    int32_t  token_id;
    int32_t  include_logits;
    int32_t  n_vocab;
    double   ms_compute;
};
#pragma pack(pop)

bool split_gen_send_req(int fd, split_gen_cmd cmd, int32_t n_tokens, int32_t pos_start,
        int32_t layer_end, int32_t include_logits, const int32_t * tokens);
bool split_gen_recv_req(int fd, split_gen_a_req & req, std::vector<int32_t> & tokens);

bool split_gen_send_resp(int fd, const split_gen_a_resp & resp, const float * logits, int32_t n_vocab);
bool split_gen_recv_resp(int fd, split_gen_a_resp & resp, std::vector<float> * logits);

bool split_ab_send_cmd(int fd, split_ab_cmd cmd);
bool split_ab_send_hidden(int fd, int32_t n_tokens, int32_t n_embd, int32_t layer_end,
        int32_t pos_start, int32_t include_logits, const float * data);
bool split_ab_send_reset(int fd);
bool split_ab_send_shutdown(int fd);

bool split_ab_recv_cmd(int fd, split_ab_cmd & cmd);
bool split_ab_recv_hidden(int fd, split_tcp_hidden_msg & msg);
bool split_ab_send_b_resp(int fd, const split_gen_b_resp & resp, const float * logits, int32_t n_vocab);
bool split_ab_recv_b_resp(int fd, split_gen_b_resp & resp, std::vector<float> * logits);

// --- 3-node autoregressive split generation (Task 4) ---

constexpr uint32_t SPLIT_GEN3_VERSION = 1;

#pragma pack(push, 1)
struct split_gen3_c_resp {
    uint32_t magic;
    int32_t  token_id;
    int32_t  n_vocab;
    double   ms_compute;
    double   ms_sample;
};

struct split_gen3_mid_resp {
    uint32_t magic;
    int32_t  token_id;
    int32_t  n_vocab;
    double   ms_b_compute;
    double   ms_bc_xfer;
    double   ms_c_compute;
    double   ms_c_sample;
};

struct split_gen3_a_resp {
    uint32_t magic;
    uint32_t version;
    int32_t  token_id;
    int32_t  include_logits;
    int32_t  n_vocab;
    double   ms_a_compute;
    double   ms_ab_xfer;
    double   ms_b_compute;
    double   ms_bc_xfer;
    double   ms_c_compute;
    double   ms_c_sample;
};
#pragma pack(pop)

bool split_gen3_send_a_resp(int fd, const split_gen3_a_resp & resp, const float * logits, int32_t n_vocab);
bool split_gen3_recv_a_resp(int fd, split_gen3_a_resp & resp, std::vector<float> * logits);

bool split_gen3_send_mid_resp(int fd, const split_gen3_mid_resp & resp);
bool split_gen3_recv_mid_resp(int fd, split_gen3_mid_resp & resp);

bool split_gen3_send_c_resp(int fd, const split_gen3_c_resp & resp);
bool split_gen3_recv_c_resp(int fd, split_gen3_c_resp & resp);
