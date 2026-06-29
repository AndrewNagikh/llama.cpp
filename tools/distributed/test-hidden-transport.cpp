#include "verification/hidden_transport_verify.h"

#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    setenv("LLAMA_DIST_TRANSPORT_DUMP", "1", 1);
    setenv("LLAMA_DIST_TRACE_DIR", "/tmp/test_hidden_transport", 1);

    std::vector<float> data(512, 0.5f);
    const auto result = verify_hidden_tcp_loopback(2, 256, data.data());
    assert(result.tcp_memcmp_ok);
    printf("test-hidden-transport: OK %s\n", result.message.c_str());
    return 0;
}
