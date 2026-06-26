#pragma once

#include "synchronization_executor.h"

// HTTP Range download executor — fetches byte ranges and stores layer blobs.

class http_range_download_executor : public synchronization_executor {
public:
    executor_result execute(
            const install_operation & operation,
            layer_store & store) override;
};
