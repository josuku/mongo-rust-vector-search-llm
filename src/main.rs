use crate::config::SearchEngineType;
use crate::store::mongo_store::MongoStore;
use crate::store::Store;
use crate::{embedder::Embedder, store::qdrant_store::QdrantStore};
use anyhow::Result;
use log::warn;
use mongodb::bson::oid::ObjectId;
use std::io::{self, Write};

mod config;
mod embedder;
mod store;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("{} CONFIG_FILE", args[0]);
        return Ok(());
    }

    let config_content = std::fs::read_to_string(&args[1]).expect("Could not read config");
    let config: config::Config =
        serde_yaml::from_str(&config_content).expect("Could not parse config");

    let mongo_store = MongoStore::new(&config.mongo, &config.search.minscore).await;
    let qdrant_store = QdrantStore::new(&config.qdrant, &config.search.minscore).await?;

    let mut embedder = Embedder::new(config.hardware, config.model)?;

    loop {
        println!("\n=== MENU ===");
        println!("1) Add new text");
        println!("2) Search text");
        println!("3) Update empty embeddings in collection");
        println!("4) Exit");

        print!("Select option: ");
        io::stdout().flush()?;

        let mut option = String::new();
        io::stdin().read_line(&mut option)?;

        match option.trim() {
            "1" => add_text_to_db(&mut embedder, &mongo_store, &qdrant_store).await?,
            "2" => {
                search_text(
                    &mut embedder,
                    &mongo_store,
                    &qdrant_store,
                    &config.search.engine,
                )
                .await?
            }
            "3" => {
                fill_empty_embeddings(
                    &mut embedder,
                    &mongo_store,
                    &qdrant_store,
                    &config.search.engine,
                )
                .await?
            }
            "4" => {
                println!("exiting application!!!");
                break;
            }
            _ => println!("Invalid option"),
        }
    }

    Ok(())
}

async fn add_text_to_db(
    embedder: &mut Embedder,
    mongo_store: &MongoStore,
    qdrant_store: &QdrantStore,
) -> anyhow::Result<()> {
    print!("Enter text to store: ");
    io::stdout().flush()?;

    let mut text = String::new();
    io::stdin().read_line(&mut text)?;
    let text = text.trim();

    let embedding = embedder.embed(text)?;

    let id = ObjectId::new().to_string();

    let _ = mongo_store
        .add(&id, Some(text.to_string()), &embedding)
        .await;
    let _ = qdrant_store.add(&id, None, &embedding).await;

    println!("Stored successfully");
    Ok(())
}

async fn search_text(
    embedder: &mut Embedder,
    mongo_store: &MongoStore,
    qdrant_store: &QdrantStore,
    search_engine: &SearchEngineType,
) -> anyhow::Result<()> {
    print!("Enter search text: ");
    io::stdout().flush()?;

    let mut text = String::new();
    io::stdin().read_line(&mut text)?;
    let text = text.trim();

    let embedding = embedder.embed(text)?;

    let results = if *search_engine == SearchEngineType::Mongo {
        mongo_store.search(&embedding).await?
    } else {
        let mut qdrant_results = qdrant_store.search(&embedding).await?;
        for qdrant_result in &mut qdrant_results {
            if let Some(mongo_doc) = mongo_store.get(&qdrant_result._id).await? {
                qdrant_result.description = mongo_doc.description.unwrap_or_default();
            }
        }
        qdrant_results
    };

    println!("Results:\n");
    for result in results {
        println!(
            ":{:?} {:?} id:{:?}",
            result.score, result.description, result._id
        );
    }

    Ok(())
}

async fn fill_empty_embeddings(
    embedder: &mut Embedder,
    mongo_store: &MongoStore,
    qdrant_store: &QdrantStore,
    search_engine: &SearchEngineType,
) -> anyhow::Result<()> {
    if *search_engine == SearchEngineType::Qdrant && !qdrant_store.is_empty().await? {
        warn!("qdrant already has data");
        return Ok(());
    }

    let mut total_documents = 0;
    let mut total_time = std::time::Duration::new(0, 0);
    let documents = mongo_store.get_all().await?;

    for doc in documents {
        // Only compute embeddings if field is empty
        if *search_engine == SearchEngineType::Mongo && doc.embeddings.is_some() {
            continue;
        }

        total_documents += 1;

        let start = std::time::Instant::now();
        let embedding = embedder.embed(&doc.description.clone().unwrap_or_default())?;
        let elapsed = start.elapsed();
        total_time += elapsed;

        // println!(
        //     "Computed embedding for '{:?}', time spent: {:.2?} count:{}",
        //     doc.description, elapsed, total_documents
        // );

        if *search_engine == SearchEngineType::Mongo {
            mongo_store.update(&doc.id, &embedding).await?;
        } else {
            qdrant_store.add(&doc.id, None, &embedding).await?
        }
    }

    println!("total docs:{} time:{:.2?}", total_documents, total_time);
    embedder.print_times(true);
    Ok(())
}
