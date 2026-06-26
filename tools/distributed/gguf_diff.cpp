#include "verification/gguf_diff.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s ORIGINAL.gguf MATERIALIZED.gguf\n", argv[0]);
        return 1;
    }
    const auto result = gguf_diff_files(argv[1], argv[2]);
    printf("%s\n", result.to_json().dump(2).c_str());
    return result.passed ? 0 : 1;
}
