#!/bin/bash

# Quick determinism test: same prompt should produce same tokens
MODEL="/home/warpvoin/models/llama-3.2-1b-instruct-q4_k_m.gguf"

echo "=== Testing determinism ==="
echo "Running same prompt 3 times with full model..."

for i in {1..3}; do
    echo -n "Run $i: "
    timeout 30 ./build/bin/test-partial-forward "$MODEL" 2>&1 | grep "logits\[0\.\.3\]" | tail -1
done

echo
echo "If logits are identical → deterministic"
echo "If logits differ → non-deterministic (problem with RNG seed)"