#include "runtime/runtime_config.h"

#include <cstdio>
#include <cstdlib>

static int test_runtime_config_flags() {
    unsetenv("DIST_RUNTIME_LAYER_FIRST");
    unsetenv("DIST_RUNTIME_FORCE_MATERIALIZE");
    unsetenv("DIST_RUNTIME_BIND_ONLY");
    if (runtime_layer_first_enabled() || runtime_force_materialize() || runtime_bind_only_request()) {
        fprintf(stderr, "test-runtime-config: flags should default off\n");
        return 1;
    }

    setenv("DIST_RUNTIME_LAYER_FIRST", "1", 1);
    setenv("DIST_RUNTIME_FORCE_MATERIALIZE", "1", 1);
    setenv("DIST_RUNTIME_BIND_ONLY", "1", 1);
    if (!runtime_layer_first_enabled() || !runtime_force_materialize() || !runtime_bind_only_request()) {
        fprintf(stderr, "test-runtime-config: flags should be on\n");
        return 1;
    }

    unsetenv("DIST_RUNTIME_LAYER_FIRST");
    unsetenv("DIST_RUNTIME_FORCE_MATERIALIZE");
    unsetenv("DIST_RUNTIME_BIND_ONLY");
    return 0;
}

int main() {
    if (test_runtime_config_flags() != 0) {
        fprintf(stderr, "test-runtime-config: FAILED\n");
        return 1;
    }
    printf("test-runtime-config: OK\n");
    return 0;
}
