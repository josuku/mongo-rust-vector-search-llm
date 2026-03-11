use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::VarBuilderArgs;
use candle_transformers::models::bert::{BertModel, Config};
use mongodb::{
    bson::{doc, Bson},
    Client, Collection,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io::{self, Write},
};
use tokenizers::Tokenizer;

const MONGO_URI: &str = "mongodb+srv://mongo_user:mongo_password@mongo_url/";
const MONGO_VECTOR_SEARCH_INDEX: &str = "vector_search";
const MONGO_DATABASE: &str = "default";
const MONGO_COLLECTION: &str = "descriptions";
const SCORE_PRECISION: f32 = 0.6f32;

#[derive(Debug, Serialize, Deserialize)]
struct Item {
    description: String,
    embedding: Vec<f32>,
}

struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    fn new() -> Result<Self> {
        let device = Device::Cpu;
        // Load tokenizer
        let tokenizer = Tokenizer::from_file("models/tokenizer.json")
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        // Load config
        let config: Config = serde_json::from_str(&std::fs::read_to_string("models/config.json")?)?;

        // Load safetensors weights into HashMap<String, Tensor>
        let weights: HashMap<String, Tensor> =
            candle_core::safetensors::load("models/model.safetensors", &device)?;

        // Wrap into VarBuilderArgs
        let vb = VarBuilderArgs::from_tensors(weights, DType::F32, &device);

        // Load BERT model
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|v| *v as i64).collect();
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;

        let seq_len = input.dims()[1];
        let token_type_ids = Tensor::zeros(&[1, seq_len], DType::I64, &self.device)?;
        let attention_mask = Tensor::ones(&[1, seq_len], DType::F32, &self.device)?;

        let embeddings = self
            .model
            .forward(&input, &token_type_ids, Some(&attention_mask))?;
        let pooled = embeddings.mean(1)?;
        Ok(pooled.flatten_all()?.to_vec1()?)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::with_uri_str(MONGO_URI).await?;
    let db = client.database(MONGO_DATABASE);
    let collection = db.collection::<Item>(MONGO_COLLECTION);

    let embedder = Embedder::new()?;

    loop {
        println!("\n=== MENU ===");
        println!("1) Add new text");
        println!("2) Search text");
        println!("3) Exit");

        print!("Select option: ");
        io::stdout().flush()?;

        let mut option = String::new();
        io::stdin().read_line(&mut option)?;

        match option.trim() {
            "1" => add_text_to_db(&embedder, &collection).await?,
            "2" => search_text(&embedder, &collection).await?,
            "3" => {
                println!("exiting application!!!");
                break;
            }
            _ => println!("Invalid option"),
        }
    }

    Ok(())
}

async fn add_text_to_db(embedder: &Embedder, collection: &Collection<Item>) -> anyhow::Result<()> {
    print!("Enter text to store: ");
    io::stdout().flush()?;

    let mut text = String::new();
    io::stdin().read_line(&mut text)?;
    let text = text.trim();

    let embedding = embedder.embed(text)?;

    let item = Item {
        description: text.to_string(),
        embedding,
    };

    collection.insert_one(item).await?;

    println!("✅ Stored successfully");
    Ok(())
}

async fn search_text(embedder: &Embedder, collection: &Collection<Item>) -> anyhow::Result<()> {
    print!("Enter search text: ");
    io::stdout().flush()?;

    let mut text = String::new();
    io::stdin().read_line(&mut text)?;
    let text = text.trim();

    let embedding = embedder.embed(text)?;

    let query_vector: Vec<Bson> = embedding.iter().map(|v| Bson::Double(*v as f64)).collect();

    let pipeline = vec![
        doc! {
            "$vectorSearch": {
                "index": MONGO_VECTOR_SEARCH_INDEX,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 5
            }
        },
        doc! {
            "$project": {
                "description": 1,
                "score": { "$meta": "vectorSearchScore" }
            }
        },
        doc! {
            "$match": {
                "score": { "$gt": SCORE_PRECISION }
            }
        },
    ];

    let mut cursor = collection.aggregate(pipeline).await?;
    println!("\n✅ Results:\n");

    while cursor.advance().await? {
        let doc = cursor.deserialize_current()?;
        println!(":{:?} {:?}", doc["score"], doc["description"]);
    }

    Ok(())
}
