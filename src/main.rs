use crate::embedder::Embedder;
use crate::store::mongo_store::MongoStore;
use crate::store::Store;
use anyhow::Result;
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

    let mongo_store = MongoStore::new(&config.mongo, config.minscore).await;

    let mut embedder = Embedder::new(config.hardware)?;

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
            "1" => add_text_to_db(&mut embedder, &mongo_store).await?,
            "2" => search_text(&mut embedder, &mongo_store).await?,
            "3" => fill_empty_embeddings(&mut embedder, &mongo_store).await?,
            "4" => {
                println!("exiting application!!!");
                break;
            }
            _ => println!("Invalid option"),
        }
    }

    Ok(())
}

async fn add_text_to_db(embedder: &mut Embedder, mongo_store: &MongoStore) -> anyhow::Result<()> {
    print!("Enter text to store: ");
    io::stdout().flush()?;

    let mut text = String::new();
    io::stdin().read_line(&mut text)?;
    let text = text.trim();

    let embedding = embedder.embed(text)?;

    let _ = mongo_store.create(Some(text.to_string()), embedding).await;

    println!("Stored successfully");
    Ok(())
}

async fn search_text(embedder: &mut Embedder, mongo_store: &MongoStore) -> anyhow::Result<()> {
    print!("Enter search text: ");
    io::stdout().flush()?;

    let mut text = String::new();
    io::stdin().read_line(&mut text)?;
    let text = text.trim();

    let embedding = embedder.embed(text)?;

    let results = mongo_store.search(embedding).await?;

    println!("Results:\n");
    for result in results {
        println!(":{:?} {:?}", result.score, result.description);
    }

    Ok(())
}

async fn fill_empty_embeddings(
    embedder: &mut Embedder,
    mongo_store: &MongoStore,
) -> anyhow::Result<()> {
    let mut total_documents = 0;
    let mut total_time = std::time::Duration::new(0, 0);
    let documents = mongo_store.get_all().await?;
    for doc in documents {
        // Only compute embeddings if field is empty
        if doc.embeddings.is_some() {
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

        mongo_store.update(doc.id, embedding).await?;
    }

    println!("total docs:{} time:{:.2?}", total_documents, total_time);
    embedder.print_times(true);
    Ok(())
}
