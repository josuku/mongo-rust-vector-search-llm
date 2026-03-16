use crate::{
    config::MongoConfig,
    store::{Store, StoreItem, StoreSearchResult},
};
use futures_util::TryStreamExt;
use mongodb::{
    bson::{doc, oid::ObjectId},
    Client, Collection,
};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Serialize, Deserialize)]
struct MongoItem {
    _id: ObjectId,
    description: String,
    embeddings: Option<Vec<f32>>,
}

pub struct MongoStore {
    collection: Collection<MongoItem>,
    index_name: String,
    min_score: f32,
}

impl MongoStore {
    pub async fn new(config: &MongoConfig, min_score: f32) -> Self {
        let client = Client::with_uri_str(config.uri.clone())
            .await
            .expect("Cannot open mongo client");
        let db = client.database(&config.database);
        let collection = db.collection::<MongoItem>(&config.collection);

        MongoStore {
            collection,
            index_name: config.index.clone(),
            min_score,
        }
    }
}

impl Store for MongoStore {
    async fn create(
        &self,
        description: Option<String>,
        embeddings: Vec<f32>,
    ) -> anyhow::Result<()> {
        let item = MongoItem {
            _id: ObjectId::new(),
            description: description.unwrap_or_default(),
            embeddings: Some(embeddings),
        };
        self.collection.insert_one(item).await?;
        Ok(())
    }

    async fn update(&self, id: String, embeddings: Vec<f32>) -> anyhow::Result<()> {
        let filter = doc! { "_id": ObjectId::from_str(&id)? };
        let update = doc! { "$set": { "embeddings": embeddings } };
        self.collection.update_one(filter, update).await?;
        Ok(())
    }

    async fn _get(&self, id: String) -> anyhow::Result<Option<StoreItem>> {
        let filter = doc! { "_id": ObjectId::from_str(&id)? };
        let mut cursor = self.collection.find(filter).await?;
        if let Some(doc) = cursor.try_next().await? {
            Ok(Some(StoreItem {
                id: doc._id.to_string(),
                embeddings: doc.embeddings,
                description: Some(doc.description),
            }))
        } else {
            Ok(None)
        }
    }

    async fn get_all(&self) -> anyhow::Result<Vec<StoreItem>> {
        let mut result = Vec::new();
        let mut cursor = self.collection.find(doc! {}).await?;
        while let Some(doc) = cursor.try_next().await? {
            result.push(StoreItem {
                id: doc._id.to_string(),
                embeddings: doc.embeddings,
                description: Some(doc.description),
            });
        }
        Ok(result)
    }

    async fn search(&self, query_vector: Vec<f32>) -> anyhow::Result<Vec<StoreSearchResult>> {
        let mut result = Vec::new();
        let pipeline = vec![
            doc! {
                "$vectorSearch": {
                    "index": self.index_name.clone(),
                    "path": "embeddings",
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
                    "score": { "$gt": self.min_score }
                }
            },
        ];
        // println!("pipeline:{:?}", pipeline);

        let mut cursor = self.collection.aggregate(pipeline).await?;

        while cursor.advance().await? {
            let doc = cursor.deserialize_current()?;
            println!(":{:?} {:?}", doc["score"], doc["description"]);

            result.push(StoreSearchResult {
                _id: doc["id"].to_string(),
                description: doc["description"].to_string(),
                score: doc.get("score").and_then(|b| b.as_f64()).map(|f| f as f32),
            });
        }
        Ok(result)
    }
}
