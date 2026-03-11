# Mongo Rust Vector Search LLM

Example made in Rust of vector search in Mongo using LLM for get embeddings

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

## Edit main.rs constants

- MONGO_URI: uri with mongo connection string
- MONGO_VECTOR_SEARCH_INDEX: name of the index created in previous step. If it doesn't match, vector search should not work
- MONGO_DATABASE: name of the mongo database
- MONGO_COLLECTION: name of the mongo collection where to search for data (and where search index is created)
- SCORE_PRECISION: min score to show in results. > 0.7 acceptable

## Run

```sh
cargo run
```

## Build in docker

```sh
docker build -t mongo-rust-vector-search-llm .
```

## Run in docker

```sh
docker run --rm -it mongo-rust-vector-search-llm
```
