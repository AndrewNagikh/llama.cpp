#include "verification/hidden_transport_verify.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const int32_t n_tokens = argc >= 2 ? std::stoi(argv[1]) : 4;
    const int32_t n_embd   = argc >= 3 ? std::stoi(argv[2]) : 128;

    std::vector<float> data((size_t) n_tokens * n_embd);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i) * 0.001f;
    }

    setenv("LLAMA_DIST_TRANSPORT_DUMP", "1", 1);
    setenv("LLAMA_DIST_TRACE_DIR", "/tmp/hidden_transport_verify", 1);

    const auto result = verify_hidden_tcp_loopback(n_tokens, n_embd, data.data());
    printf("{\"ok\":%s,\"tcp_memcmp_ok\":%s,\"message\":\"%s\"}\n",
            result.ok ? "true" : "false",
            result.tcp_memcmp_ok ? "true" : "false",
            result.message.c_str());
    return result.ok ? 0 : 1;
}
