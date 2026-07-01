#pragma once

#include "synchronization_executor.h"

#include <vector>

// HTTP Range download executor — fetches byte ranges and stores layer blobs.

class http_range_download_executor : public synchronization_executor {
public:
    executor_result execute(
            const install_operation & operation,
            layer_store & store) override;

    // Network fetch only (safe to run concurrently).
    static bool fetch_body(
            const download_operation & dl,
            std::vector<uint8_t> & out,
            std::string & error);

    // Store + verify a fetched body (serialize access to layer_store).
    static executor_result store_body(
            const install_operation & operation,
            layer_store & store,
            std::vector<uint8_t> body);
};
