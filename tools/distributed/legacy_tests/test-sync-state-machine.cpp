#include "orchestrator/consistency/install_journal.h"

#include <cstdio>
#include <filesystem>

#if !defined(_WIN32)
#include <unistd.h>
#endif

static std::filesystem::path make_temp_journal_root(const char * suffix) {
    return std::filesystem::temp_directory_path() /
            (std::string("dist-journal-") + suffix + "-" + std::to_string(getpid()));
}

int main() {
    const auto root = make_temp_journal_root("sm");
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    install_journal journal(root);
    const std::string model_id = "sync-sm";

    journal.record_transition(
            model_id, "node-a", "embedding", "token_embd.weight",
            install_state::missing, install_state::downloading, "start download");
    journal.record_transition(
            model_id, "node-a", "embedding", "token_embd.weight",
            install_state::downloading, install_state::verifying, "download complete");
    journal.record_transition(
            model_id, "node-a", "embedding", "token_embd.weight",
            install_state::verifying, install_state::ready, "checksum ok");

    std::string err;
    if (!journal.assert_no_spurious_ready_regression(model_id, err)) {
        fprintf(stderr, "test-sync-state-machine: valid chain rejected: %s\n", err.c_str());
        return 1;
    }

    journal.record_transition(
            model_id, "node-a", "embedding", "token_embd.weight",
            install_state::ready, install_state::missing, "unexpected wipe");

    if (journal.assert_no_spurious_ready_regression(model_id, err)) {
        fprintf(stderr, "test-sync-state-machine: expected spurious regression detection\n");
        return 1;
    }

    const auto delete_root = make_temp_journal_root("delete");
    std::filesystem::remove_all(delete_root);
    std::filesystem::create_directories(delete_root);

    install_journal delete_journal(delete_root);
    delete_journal.record_transition(
            model_id, "node-a", "layer:0", "blk.0.attn.weight",
            install_state::missing, install_state::ready, "install complete");
    delete_journal.record_transition(
            model_id, "node-a", "layer:0", "blk.0.attn.weight",
            install_state::ready, install_state::missing, "DELETE misplaced blob");
    if (!delete_journal.assert_no_spurious_ready_regression(model_id, err)) {
        fprintf(stderr, "test-sync-state-machine: allowed DELETE regression rejected: %s\n",
                err.c_str());
        return 1;
    }

    printf("test-sync-state-machine: OK\n");
    return 0;
}
