FROM rust:1.94.0-slim-bookworm AS builder

# Install required tools for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    pkg-config \
    build-essential

# Create and empty project and build deps to have this steps cached.
RUN USER=root cargo new --bin mongo-rust-vector-search-llm
WORKDIR /mongo-rust-vector-search-llm

# Copy Cargo files and empty src folders to build only dependencies
COPY Cargo.toml Cargo.lock ./

# --release or empty
ARG TARGET=--release

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
