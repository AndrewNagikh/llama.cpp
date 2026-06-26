#include "orchestrator/install_planner/install_planner.h"
#include "test_install_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_manifest_with_layer_ranges(4, 50 * 1024 * 1024, 8192);
    const layer_byte_range range = manifest_layer_byte_range(manifest, 2);

    if (range.offset != 8192 + 2 * 50 * 1024 * 1024) {
        fprintf(stderr, "test-download-operation: bad offset %llu\n",
                (unsigned long long) range.offset);
        return 1;
    }
    if (range.length != 50 * 1024 * 1024) {
        fprintf(stderr, "test-download-operation: bad length %llu\n",
                (unsigned long long) range.length);
        return 1;
    }

    const desired_model_layout desired = make_desired_layout("m", { { "a", 2 } });
    const actual_model_layout actual = make_actual_layout("m", {});
    const coverage_report coverage = make_coverage("m", { 2 }, {}, 1);

    const auto result = build_install_plan(manifest, desired, actual, coverage);
    if (!result.success || result.plan.operations.size() != 1) {
        fprintf(stderr, "test-download-operation: expected one download op\n");
        return 1;
    }

    const auto & op = result.plan.operations.front();
    if (op.download.tensor_offset != range.offset ||
            op.download.tensor_length != range.length) {
        fprintf(stderr, "test-download-operation: operation range mismatch\n");
        return 1;
    }

    printf("test-download-operation: OK offset=%llu length=%llu\n",
            (unsigned long long) op.download.tensor_offset,
            (unsigned long long) op.download.tensor_length);
    return 0;
}
