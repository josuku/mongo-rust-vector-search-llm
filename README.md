# Mongo Rust Vector Search LLM

## Install Git LFS

```sh
sudo apt update
sudo apt install git-lfs
git lfs install
```

## Download model locally

Only config.json, model.safetensors, tokenizer_config.json and tokenizer.json are required. Other files can be deleted

```sh
git clone https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
mv paraphrase-multilingual-MiniLM-L12-v2 models
cd models
git lfs pull
```

## Create mongodb Search Index from mongo db compass (index name will be important)

```json
{
  "fields": [{
    "name": "vector_index",
    "type": "vector",
    "path": "embedding",
    "numDimensions": 384,
    "similarity": "cosine"
  }]
}
```

## Duplicate and edit config.yaml

```sh
cp config-example.yaml config.yaml
vi config.yaml
```

## Run

```sh
cargo run config.yaml

cargo run config.yaml --release # better performance
```

## Build in docker

```sh
docker build -t mongo-rust-vector-search-llm .
```

## Run in docker

```sh
docker run --rm -it mongo-rust-vector-search-llm
```
