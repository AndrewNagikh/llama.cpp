#include "gguf_inspector.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s PATH.gguf\n", argv[0]);
        return 1;
    }
    const std::string json = gguf_inspect_file_json(argv[1]);
    printf("%s\n", json.c_str());
    return 0;
}
