FROM rust:1.94.0-slim-bookworm AS builder

# Install required tools for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    pkg-config \
    wget \
    build-essential

RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb \
    sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb \
    sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Create and empty project and build deps to have this steps cached.
RUN USER=root cargo new --bin mongo-rust-vector-search-llm
WORKDIR /mongo-rust-vector-search-llm

# Copy Cargo files and empty src folders to build only dependencies
COPY Cargo.toml Cargo.lock ./

# --release or empty
ARG TARGET=--release

# Set CUDA paths so the Rust compiler can find them
ENV PATH="/usr/local/cuda-12.2/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_COMPUTE_CAP=86

# Build only dependencies
RUN cargo build $TARGET && \
    rm -rf .git* src/* && \
    rm -f target/{debug,release}/mongo-rust-vector-search-llm* target/{debug,release}/deps/mongo-rust-vector-search-llm*

# Build app with deps already downloaded and compiled.
COPY src src
RUN touch src/main.rs && cargo build $TARGET && \
    (cp target/debug/mongo-rust-vector-search-llm . || cp target/release/mongo-rust-vector-search-llm .)

# Create final image
FROM debian:bookworm-slim

WORKDIR /mongo-rust-vector-search-llm

COPY --from=builder /mongo-rust-vector-search-llm/mongo-rust-vector-search-llm .

# Copy model and tokenizer files
COPY models/tokenizer.json ./models/
COPY models/tokenizer_config.json ./models/
COPY models/config.json ./models/
COPY models/model.safetensors ./models/

ENTRYPOINT ["/mongo-rust-vector-search-llm/mongo-rust-vector-search-llm"]
