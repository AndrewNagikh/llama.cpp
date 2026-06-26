#pragma once

#include "architecture/architecture_descriptor.h"
#include "layer_store.h"

#include <string>
#include <vector>

bool store_has_required_blobs(
        const layer_store & store,
        const architecture_descriptor & desc,
        const std::vector<std::string> & required_blobs);

bool verify_required_blobs(
        const layer_store & store,
        const architecture_descriptor & desc,
        const std::vector<std::string> & required_blobs,
        std::string & err);

bool materialize_descriptor_tensors(
        const layer_store & store,
        const architecture_descriptor & desc,
        const std::vector<std::string> & required_blobs,
        std::vector<uint8_t> & file,
        std::string & err);
