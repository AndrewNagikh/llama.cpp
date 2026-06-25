#include "dist_common.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

static json capacity_for_nodes(const std::vector<dist_node_info> & nodes) {
    double ram_gb  = 0.0;
    double vram_gb = 0.0;
    for (const auto & n : nodes) {
        if (!n.online) {
            continue;
        }
        ram_gb  += dist_bytes_to_gb(n.memory.total_ram_bytes);
        vram_gb += dist_bytes_to_gb(n.memory.total_vram_bytes);
    }
    return {
        { "cluster", {
            { "ram_gb", ram_gb },
            { "vram_gb", vram_gb },
        }}
    };
}

int main() {
    std::vector<dist_node_info> nodes(2);
    nodes[0].node_id = "node-a";
    nodes[0].online  = true;
    nodes[0].memory.total_ram_bytes  = 32ULL * 1024 * 1024 * 1024;
    nodes[0].memory.total_vram_bytes = 12ULL * 1024 * 1024 * 1024;

    nodes[1].node_id = "node-b";
    nodes[1].online  = true;
    nodes[1].memory.total_ram_bytes  = 16ULL * 1024 * 1024 * 1024;
    nodes[1].memory.total_vram_bytes = 0;

    httplib::Server svr;
    svr.Get("/capacity", [&nodes](const httplib::Request &, httplib::Response & res) {
        res.set_content(capacity_for_nodes(nodes).dump(), "application/json");
    });

    const int port = 29999;
    std::thread server_thread([&]() {
        svr.listen("127.0.0.1", port);
    });

    // Wait for server to start.
    for (int i = 0; i < 50; ++i) {
        httplib::Client probe("127.0.0.1", port);
        probe.set_connection_timeout(0, 100);
        if (probe.Get("/capacity")) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(1, 0);
    auto res = cli.Get("/capacity");

    svr.stop();
    server_thread.join();

    if (!res || res->status != 200) {
        fprintf(stderr, "test-capacity-api: /capacity request failed\n");
        return 1;
    }

    json body;
    try {
        body = json::parse(res->body);
    } catch (...) {
        fprintf(stderr, "test-capacity-api: invalid JSON response\n");
        return 1;
    }

    if (!body.contains("cluster")) {
        fprintf(stderr, "test-capacity-api: missing cluster key\n");
        return 1;
    }
    const double ram_gb  = body["cluster"].value("ram_gb", -1.0);
    const double vram_gb = body["cluster"].value("vram_gb", -1.0);

    printf("test-capacity-api: ram_gb=%.1f vram_gb=%.1f\n", ram_gb, vram_gb);

    if (ram_gb < 47.0 || ram_gb > 49.0) {
        fprintf(stderr, "test-capacity-api: unexpected ram_gb\n");
        return 1;
    }
    if (vram_gb < 11.0 || vram_gb > 13.0) {
        fprintf(stderr, "test-capacity-api: unexpected vram_gb\n");
        return 1;
    }

    printf("test-capacity-api: OK\n");
    return 0;
}
