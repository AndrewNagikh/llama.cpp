#pragma once

#include "split_gen_common.h"

static constexpr int SPLIT_GEN3_MAX_NEW = 32;

struct split_gen3_layout {
    const char * name;
    int layer_a_end;
    int layer_b_start;
    int layer_b_end;
    int layer_c_start;
};

static constexpr split_gen3_layout SPLIT_GEN3_LAYOUTS[] = {
    { "A", 5, 5, 10, 10 },
    { "B", 3, 3,  8,  8 },
    { "C", 8, 8, 12, 12 },
};

static constexpr int SPLIT_GEN3_N_LAYOUTS = (int) (sizeof(SPLIT_GEN3_LAYOUTS) / sizeof(SPLIT_GEN3_LAYOUTS[0]));

struct split_gen3_step_perf {
    double ms_a       = 0.0;
    double ms_ab      = 0.0;
    double ms_b       = 0.0;
    double ms_bc      = 0.0;
    double ms_c       = 0.0;
    double ms_sample  = 0.0;
    double ms_total   = 0.0;
};
