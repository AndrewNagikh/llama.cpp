# Distributed runtime: orchestrator + node_agent + split_gen3 workers
# Build context: llama.cpp repository root (see docker-compose.yml)

ARG UBUNTU_VERSION=24.04
FROM docker.io/ubuntu:${UBUNTU_VERSION} AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-14 g++-14 cmake build-essential git libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14

WORKDIR /src
COPY . .

RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_DISTRIBUTED=ON \
    -DGGML_NATIVE=OFF \
    && cmake --build build -j"$(nproc)" \
        --target orchestrator node_agent split_gen3_a split_gen3_b split_gen3_c \
        mono_reference verify_decode_loop_parity verify_hidden_pipeline verify_logits_pipeline \
        verify_layer_equivalence verify_hidden_transport verify_runtime_api \
        verify_final_runtime verify_final_logits verify_decode_graph verify_context_params

RUN mkdir -p /out/lib \
    && find build -name "*.so*" -exec cp -P {} /out/lib \; \
    && cp build/bin/orchestrator build/bin/node_agent \
       build/bin/split_gen3_a build/bin/split_gen3_b build/bin/split_gen3_c \
       build/bin/mono_reference build/bin/verify_decode_loop_parity \
       build/bin/verify_hidden_pipeline build/bin/verify_logits_pipeline \
       build/bin/verify_layer_equivalence build/bin/verify_hidden_transport \
       build/bin/verify_runtime_api build/bin/verify_final_runtime \
       build/bin/verify_final_logits build/bin/verify_decode_graph \
       build/bin/verify_context_params /out/

FROM docker.io/ubuntu:${UBUNTU_VERSION} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libssl3 curl python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /out/lib/ /app/
COPY --from=build /out/orchestrator /out/node_agent \
     /out/split_gen3_a /out/split_gen3_b /out/split_gen3_c \
     /out/mono_reference /out/verify_decode_loop_parity \
     /out/verify_hidden_pipeline /out/verify_logits_pipeline \
     /out/verify_layer_equivalence /out/verify_hidden_transport \
     /out/verify_runtime_api /out/verify_final_runtime /out/verify_final_logits \
     /out/verify_decode_graph /out/verify_context_params /app/
COPY tools/distributed/download_model.py /app/download_model.py
COPY tools/distributed/docker/merge_runtime_trace.py /app/merge_runtime_trace.py
COPY tools/distributed/docker/analyze_root_cause.py /app/analyze_root_cause.py
COPY tools/distributed/docker/run_layer_equivalence.py /app/run_layer_equivalence.py
COPY tools/distributed/docker/run_runtime_state_verify.py /app/run_runtime_state_verify.py
COPY tools/distributed/docker/run_final_runtime_verify.py /app/run_final_runtime_verify.py
COPY tools/distributed/docker/collect_traces.sh /app/collect_traces.sh

ENV LD_LIBRARY_PATH=/app
ENV MODELS_DIR=/data/models
ENV PATH=/app:$PATH

RUN mkdir -p /data/models
VOLUME /data/models

HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD curl -sf http://127.0.0.1:${HEALTH_PORT:-9000}/health || exit 1
