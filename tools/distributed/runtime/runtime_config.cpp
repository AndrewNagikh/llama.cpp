#include "runtime_config.h"

#include <cstdlib>

namespace {

bool env_flag(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr) {
        return false;
    }
    return v[0] == '1' && v[1] == '\0';
}

} // namespace

bool runtime_layer_first_enabled() {
    return env_flag("DIST_RUNTIME_LAYER_FIRST");
}

bool runtime_force_materialize() {
    return env_flag("DIST_RUNTIME_FORCE_MATERIALIZE");
}

bool runtime_bind_only_request() {
    return env_flag("DIST_RUNTIME_BIND_ONLY");
}
