#include "install_journal.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

std::string iso_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream os;
    os << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ");
    return os.str();
}

} // namespace

install_journal::install_journal(std::filesystem::path root_dir)
        : root_dir_(std::move(root_dir)) {}

void install_journal::set_root_dir(std::filesystem::path root_dir) {
    std::lock_guard<std::mutex> lock(mu_);
    root_dir_ = std::move(root_dir);
}

std::filesystem::path install_journal::journal_path(const std::string & model_id) const {
    std::error_code ec;
    std::filesystem::create_directories(root_dir_, ec);
    return root_dir_ / (model_id + ".jsonl");
}

void install_journal::record(const blob_state_event & event) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto path = journal_path(event.model_id);
    std::ofstream out(path, std::ios::app);
    if (!out) {
        return;
    }
    out << nlohmann::json{
        { "timestamp", event.timestamp },
        { "model_id", event.model_id },
        { "node_id", event.node_id },
        { "blob_id", event.blob_id },
        { "tensor_name", event.tensor_name },
        { "from", install_state_to_string(event.from_state) },
        { "to", install_state_to_string(event.to_state) },
        { "reason", event.reason },
    }.dump() << '\n';
}

void install_journal::record_transition(
        const std::string & model_id,
        const std::string & node_id,
        const std::string & blob_id,
        const std::string & tensor_name,
        install_state from_state,
        install_state to_state,
        const std::string & reason) {
    blob_state_event event;
    event.timestamp   = iso_timestamp_now();
    event.model_id    = model_id;
    event.node_id     = node_id;
    event.blob_id     = blob_id;
    event.tensor_name = tensor_name;
    event.from_state  = from_state;
    event.to_state    = to_state;
    event.reason      = reason;
    record(event);
}

std::vector<blob_state_event> install_journal::read_events(const std::string & model_id) const {
    std::vector<blob_state_event> events;
    const auto path = journal_path(model_id);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return events;
    }
    std::ifstream in(path);
    if (!in) {
        return events;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        try {
            const nlohmann::json j = nlohmann::json::parse(line);
            blob_state_event event;
            event.timestamp   = j.value("timestamp", "");
            event.model_id    = j.value("model_id", model_id);
            event.node_id     = j.value("node_id", "");
            event.blob_id     = j.value("blob_id", "");
            event.tensor_name = j.value("tensor_name", "");
            event.from_state  = install_state_from_string(j.value("from", "MISSING"));
            event.to_state    = install_state_from_string(j.value("to", "MISSING"));
            event.reason      = j.value("reason", "");
            events.push_back(std::move(event));
        } catch (...) {
            continue;
        }
    }
    return events;
}

bool install_journal::assert_no_spurious_ready_regression(
        const std::string & model_id,
        std::string & error) const {
    const auto events = read_events(model_id);
    for (size_t i = 1; i < events.size(); ++i) {
        const auto & prev = events[i - 1];
        const auto & cur  = events[i];
        if (prev.to_state != install_state::ready) {
            continue;
        }
        if (prev.blob_id != cur.blob_id || prev.tensor_name != cur.tensor_name ||
                prev.node_id != cur.node_id) {
            continue;
        }
        if (cur.to_state == install_state::missing ||
                cur.to_state == install_state::downloading) {
            const bool allowed = cur.from_state == install_state::ready &&
                    (cur.reason.find("delete") != std::string::npos ||
                     cur.reason.find("DELETE") != std::string::npos ||
                     cur.reason.find("reset") != std::string::npos ||
                     cur.reason.find("repair") != std::string::npos);
            if (!allowed) {
                error = "spurious READY regression for " + cur.blob_id + "/" + cur.tensor_name +
                        " on " + cur.node_id + " without delete/repair reason";
                return false;
            }
        }
    }
    return true;
}

nlohmann::json install_journal::to_json(const std::string & model_id) const {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto & event : read_events(model_id)) {
        arr.push_back({
            { "timestamp", event.timestamp },
            { "node_id", event.node_id },
            { "blob_id", event.blob_id },
            { "tensor_name", event.tensor_name },
            { "from", install_state_to_string(event.from_state) },
            { "to", install_state_to_string(event.to_state) },
            { "reason", event.reason },
        });
    }
    return arr;
}
