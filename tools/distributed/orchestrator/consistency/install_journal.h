#pragma once

#include "coverage/coverage.h"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

// Task 9.9 — blob state transition journal (MISSING→DOWNLOADING→VERIFYING→READY→DELETED).

struct blob_state_event {
    std::string timestamp;
    std::string model_id;
    std::string node_id;
    std::string blob_id;
    std::string tensor_name;
    install_state from_state = install_state::missing;
    install_state to_state   = install_state::missing;
    std::string reason;
};

class install_journal {
public:
    explicit install_journal(std::filesystem::path root_dir);

    void set_root_dir(std::filesystem::path root_dir);

    void record(const blob_state_event & event);
    void record_transition(
            const std::string & model_id,
            const std::string & node_id,
            const std::string & blob_id,
            const std::string & tensor_name,
            install_state from_state,
            install_state to_state,
            const std::string & reason);

    std::vector<blob_state_event> read_events(const std::string & model_id) const;

    // Assert no READY→PARTIAL/MISSING without an intervening DELETE/DOWNLOAD event.
    bool assert_no_spurious_ready_regression(
            const std::string & model_id,
            std::string & error) const;

    nlohmann::json to_json(const std::string & model_id) const;

private:
    std::filesystem::path journal_path(const std::string & model_id) const;

    std::filesystem::path root_dir_;
    mutable std::mutex mu_;
};
