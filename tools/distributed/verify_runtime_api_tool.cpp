#include "verification/runtime_api_verify.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL [N_TOKENS]\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    const int32_t n_tokens  = argc >= 3 ? std::stoi(argv[2]) : 1;

    const auto result = verify_runtime_hidden_api(model, n_tokens);
    printf("{\"ok\":%s,\"diff_offset\":%zu,\"message\":\"%s\"}\n",
            result.ok ? "true" : "false",
            result.diff_offset,
            result.message.c_str());
    return result.ok ? 0 : 1;
}
