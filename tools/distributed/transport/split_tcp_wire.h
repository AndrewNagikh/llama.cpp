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

void split_tcp_init();
void split_tcp_close(int fd);

int  split_tcp_listen(int port);
int  split_tcp_listen_host(const char * host, int port);
int  split_tcp_accept(int listen_fd);
int  split_tcp_connect(const char * host, int port);
int  split_tcp_connect_retry(const char * host, int port, int retries, int delay_ms);

bool split_tcp_send_all(int fd, const void * data, size_t size);
bool split_tcp_recv_all(int fd, void * data, size_t size);
void split_tcp_set_timeouts(int fd, int timeout_ms);

bool split_tcp_send_hidden(int fd, int32_t n_tokens, int32_t n_embd, int32_t layer_end, const float * data);
bool split_tcp_recv_hidden(int fd, split_tcp_hidden_msg & msg);

bool split_tcp_write_result(const char * path, const split_tcp_result_file & meta, const float * logits);
bool split_tcp_read_result(const char * path, split_tcp_result_file & meta, std::vector<float> & logits);

// --- autoregressive split generation protocol (Task 3) ---

constexpr uint32_t SPLIT_GEN_MAGIC   = 0x47454E53; // 'SNEG'
constexpr uint32_t SPLIT_GEN_VERSION = 1;

constexpr uint32_t SPLIT_AB_MAGIC = 0x42415350; // 'PSAB'

enum split_gen_cmd : uint32_t {
    SPLIT_GEN_CMD_RESET         = 1,
    SPLIT_GEN_CMD_PREFILL       = 2,
    SPLIT_GEN_CMD_DECODE        = 3,
    SPLIT_GEN_CMD_SHUTDOWN      = 4,
    SPLIT_GEN_CMD_PREFILL_HIDDEN = 5,
    SPLIT_GEN_CMD_DECODE_HIDDEN  = 6,
    SPLIT_GEN_CMD_PROTO_NEGOTIATE = 7,
    SPLIT_GEN_CMD_DRAIN_PENDING   = 8,
    // Speculative verify wave (Task 19): tokens carry [anchor, draft_1..draft_k];
    // entry decodes the batch and forwards hidden + the draft ids downstream,
    // final verifies per position and returns accepted_count + corrected token.
    SPLIT_GEN_CMD_VERIFY          = 9,
};

enum split_ab_cmd : uint32_t {
    SPLIT_AB_CMD_RESET   = 1,
    SPLIT_AB_CMD_HIDDEN  = 2,
    SPLIT_AB_CMD_SHUTDOWN = 3,
    // Draft token ids for the verify wave that follows as the next HIDDEN
    // on this connection. Middle stages forward it unchanged.
    SPLIT_AB_CMD_VERIFY_IDS = 4,
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
bool split_gen_send_hidden_req(int fd, split_gen_cmd cmd, int32_t n_tokens, int32_t n_embd,
        int32_t pos_start, int32_t layer_end, const float * hidden);
bool split_gen_recv_req(int fd, split_gen_a_req & req, std::vector<int32_t> & tokens,
        std::vector<float> * hidden = nullptr, int32_t * n_embd_out = nullptr);

bool split_gen_send_resp(int fd, const split_gen_a_resp & resp, const float * logits, int32_t n_vocab);
bool split_gen_recv_resp(int fd, split_gen_a_resp & resp, std::vector<float> * logits);

bool split_ab_send_cmd(int fd, split_ab_cmd cmd);
bool split_ab_send_hidden(int fd, int32_t n_tokens, int32_t n_embd, int32_t layer_end,
        int32_t pos_start, int32_t include_logits, const float * data);
bool split_ab_send_reset(int fd);
bool split_ab_send_shutdown(int fd);

// Verify-wave draft ids (Task 19). Sent immediately before the wave's HIDDEN
// message on the same connection; the receiver associates them by pos_start.
bool split_ab_send_verify_ids(int fd, int32_t pos_start, const int32_t * ids, int32_t n);
bool split_ab_recv_verify_ids(int fd, int32_t & pos_start, std::vector<int32_t> & ids);

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
    // Verify waves only (SPLIT_GEN_CMD_VERIFY): number of draft tokens whose
    // target argmax matched; token_id is then the corrected token sampled at
    // the first rejected position. -1 for ordinary decode waves.
    int32_t  accepted_count;
};

struct split_gen3_mid_resp {
    uint32_t magic;
    int32_t  token_id;
    int32_t  n_vocab;
    double   ms_b_compute;
    double   ms_bc_xfer;
    double   ms_c_compute;
    double   ms_c_sample;
    int32_t  accepted_count;
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
    int32_t  accepted_count;
};

struct split_proto_negotiate_resp {
    uint32_t magic;
    uint32_t version;
    uint32_t agreed_protocol;
    uint32_t server_max_protocol;
    uint32_t status;
};

constexpr uint32_t SPLIT_GENQ_MAGIC = 0x514E4547; // 'GENQ' queue ack
constexpr uint32_t SPLIT_GENT_MAGIC = 0x544E4547; // 'GENT' token ready

struct split_gen_queue_ack {
    uint32_t magic;
    uint32_t version;
    int32_t  queue_depth;
    int32_t  wave_id;
    int32_t  status;
};

struct split_gen_token_ready {
    uint32_t magic;
    uint32_t version;
    int32_t  token_id;
    int32_t  pos_start;
    int32_t  wave_id;
};
#pragma pack(pop)

bool split_gen3_send_a_resp(int fd, const split_gen3_a_resp & resp, const float * logits, int32_t n_vocab);
bool split_gen3_recv_a_resp(int fd, split_gen3_a_resp & resp, std::vector<float> * logits);

bool split_gen3_send_mid_resp(int fd, const split_gen3_mid_resp & resp);
bool split_gen3_recv_mid_resp(int fd, split_gen3_mid_resp & resp);

bool split_gen3_send_c_resp(int fd, const split_gen3_c_resp & resp);
bool split_gen3_recv_c_resp(int fd, split_gen3_c_resp & resp);

bool split_gen_send_queue_ack(int fd, int32_t queue_depth, int32_t wave_id, int32_t status = 0);
bool split_gen_recv_queue_ack(int fd, split_gen_queue_ack & ack);

bool split_gen_send_token_ready(int fd, int32_t token_id, int32_t pos_start, int32_t wave_id);
bool split_gen_recv_token_ready(int fd, split_gen_token_ready & ready);

bool split_gen_peek_ctrl_magic(int fd, uint32_t & magic_out);
