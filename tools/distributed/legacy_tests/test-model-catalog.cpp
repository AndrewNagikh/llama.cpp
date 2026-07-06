// Test model catalog and store functionality

#include "model_catalog.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>

static void test_model_catalog() {
    printf("Testing model_catalog...\n");
    
    model_catalog catalog;
    
    // Test default initialization
    auto models = catalog.get_models();
    assert(models.size() >= 1);
    assert(models[0].id == "llama-3.2-1b");
    assert(models[0].n_layers == 16);
    assert(models[0].n_embd == 2048);
    
    // Test find_model
    const auto * model = catalog.find_model("llama-3.2-1b");
    assert(model != nullptr);
    assert(model->display_name == "Llama 3.2 1B Instruct Q4_K_M");
    
    const auto * missing = catalog.find_model("nonexistent");
    assert(missing == nullptr);
    
    // Test install job creation
    std::string job_id = catalog.create_install_job("llama-3.2-1b");
    assert(!job_id.empty());
    
    auto * job = catalog.get_install_job(job_id);
    assert(job != nullptr);
    assert(job->model_id == "llama-3.2-1b");
    assert(job->status == install_status::unknown);
    
    // Test job status updates
    catalog.update_job_status(job_id, install_status::downloading, "", 0.5);
    assert(job->status == install_status::downloading);
    assert(job->progress == 0.5);
    
    catalog.complete_job(job_id);
    assert(job->status == install_status::ready);
    assert(job->progress == 1.0);
    
    printf("model_catalog tests passed\n");
}

static void test_model_store() {
    printf("Testing model_store...\n");
    
    model_store store;
    
    // Test empty store
    auto models = store.get_local_models();
    assert(models.empty());
    
    // Test model paths
    std::string models_dir = store.get_models_dir();
    assert(!models_dir.empty());
    
    std::string model_path = store.get_model_path("test-model");
    assert(model_path.find("test-model") != std::string::npos);
    assert(model_path.find("model.gguf") != std::string::npos);
    
    // Test add model
    installed_model test_model;
    test_model.model_id = "test-model";
    test_model.local_path = "/tmp/test.gguf";
    test_model.size_bytes = 1234567;
    test_model.ready = true;
    test_model.installed_ms = 1000000;
    
    bool added = store.add_model(test_model);
    assert(added);
    
    // Test find model
    const auto * found = store.find_model("test-model");
    assert(found != nullptr);
    assert(found->model_id == "test-model");
    assert(found->size_bytes == 1234567);
    assert(found->ready == true);
    
    // Test get models
    auto all_models = store.get_local_models();
    assert(all_models.size() == 1);
    assert(all_models[0].model_id == "test-model");
    
    // Test remove model
    bool removed = store.remove_model("test-model");
    assert(removed);
    
    auto after_remove = store.get_local_models();
    assert(after_remove.empty());
    
    printf("model_store tests passed\n");
}

static void test_placement_planner() {
    printf("Testing placement_planner...\n");
    
    model_catalog catalog;
    
    std::vector<std::string> node_ids = {"node-a", "node-b", "node-c"};
    std::vector<uint64_t> node_memory = {8*1024*1024*1024ULL, 4*1024*1024*1024ULL, 2*1024*1024*1024ULL}; // 8GB, 4GB, 2GB
    
    auto placements = catalog.plan_placement("llama-3.2-1b", node_ids, node_memory);
    
    assert(placements.size() == 3);
    
    // Check layer distribution (16 layers across 3 nodes)
    int total_layers = 0;
    for (const auto & placement : placements) {
        assert(placement.layer_start >= 0);
        assert(placement.layer_end > placement.layer_start);
        total_layers += (placement.layer_end - placement.layer_start);
        
        printf("Node %s: layers [%d, %d), memory: %zu MB\n",
               placement.node_id.c_str(), 
               placement.layer_start, 
               placement.layer_end,
               placement.estimated_memory / (1024*1024));
    }
    
    assert(total_layers == 16);
    
    // Verify no gaps or overlaps
    int expected_start = 0;
    for (const auto & placement : placements) {
        assert(placement.layer_start == expected_start);
        expected_start = placement.layer_end;
    }
    
    printf("placement_planner tests passed\n");
}

static void test_persistence() {
    printf("Testing persistence...\n");
    
    const std::string test_catalog_path = "/tmp/test_catalog.json";
    const std::string test_store_path = "/tmp/test_store.json";
    
    // Clean up from previous runs
    std::filesystem::remove(test_catalog_path);
    std::filesystem::remove(test_store_path);
    
    // Test catalog save/load
    {
        model_catalog catalog;
        bool saved = catalog.save_catalog(test_catalog_path);
        assert(saved);
        assert(std::filesystem::exists(test_catalog_path));
        
        // Verify file contains expected content
        std::ifstream file(test_catalog_path);
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        assert(content.find("llama-3.2-1b") != std::string::npos);
        assert(content.find("models") != std::string::npos);
    }
    
    // Test store save/load
    {
        model_store store;
        
        installed_model test_model;
        test_model.model_id = "persist-test";
        test_model.local_path = "/tmp/persist.gguf";
        test_model.ready = true;
        
        store.add_model(test_model);
        
        bool saved = store.save_state(test_store_path);
        assert(saved);
        assert(std::filesystem::exists(test_store_path));
        
        // Verify file contains expected content
        std::ifstream file(test_store_path);
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        assert(content.find("persist-test") != std::string::npos);
    }
    
    // Clean up
    std::filesystem::remove(test_catalog_path);
    std::filesystem::remove(test_store_path);
    
    printf("persistence tests passed\n");
}

int main() {
    printf("=== Model Catalog Tests ===\n");
    
    test_model_catalog();
    test_model_store();
    test_placement_planner();
    test_persistence();
    
    printf("\n✅ All model catalog tests passed!\n");
    return 0;
}