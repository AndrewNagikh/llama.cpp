#include "verification/context_params_verify.h"

#include <cassert>
#include <cstdio>

int main() {
    context_params_diff_report empty{};
    assert(!empty.match);
    assert(context_params_diff_json(empty).find("diffs") != std::string::npos);
    printf("test-context-params: OK\n");
    return 0;
}
