use std::collections::HashMap;

use crate::{
    config::QdrantConfig,
    store::{Store, StoreItem, StoreSearchResult},
};
use log::warn;
use qdrant_client::{
    qdrant::{
        value::Kind, vectors_config::Config, CreateCollection, Distance, HnswConfigDiff, PointId,
        PointStruct, ScoredPoint, SearchParams, SearchPoints, UpsertPoints, Value, VectorParams,
        Vectors, VectorsConfig,
    },
    Qdrant,
};
use uuid::Uuid;

pub struct QdrantStore {
    client: Qdrant,
    collection: String,
    min_score: f32,
}

impl QdrantStore {
    pub async fn new(config: &QdrantConfig, min_score: &f32) -> anyhow::Result<Self> {
        let client = Qdrant::from_url(&config.uri).build()?;

        let _ = create_collection_if_not_exists(&client, &config.collection).await;

        Ok(QdrantStore {
            client,
            collection: config.collection.to_string(),
            min_score: *min_score,
        })
    }

    pub async fn is_empty(&self) -> anyhow::Result<bool> {
        let info = self.client.collection_info(self.collection.clone()).await?;

        let points = info.result.and_then(|r| r.points_count).unwrap_or(0);

        Ok(points == 0)
    }
}

impl Store for QdrantStore {
    async fn add(
        &self,
        id: &str,
        description: Option<String>,
        embeddings: &[f32],
    ) -> anyhow::Result<()> {
        let point = make_qdrant_point(id, description, embeddings)?;

        self.client
            .upsert_points(UpsertPoints {
                collection_name: self.collection.clone(),
                points: vec![point],
                ..Default::default()
            })
            .await?;

        Ok(())
    }

    async fn update(&self, _id: &str, _embeddings: &[f32]) -> anyhow::Result<()> {
        todo!("TODO not implemented");
    }

    async fn get(&self, _id: &str) -> anyhow::Result<Option<StoreItem>> {
        todo!("TODO not implemented");
    }

    async fn get_all(&self) -> anyhow::Result<Vec<StoreItem>> {
        todo!("TODO not implemented");
    }

    async fn search(&self, query_vector: &[f32]) -> anyhow::Result<Vec<StoreSearchResult>> {
        let search_start = std::time::Instant::now();
        let search_result = self
            .client
            .search_points(SearchPoints {
                collection_name: self.collection.clone(),
                vector: query_vector.to_vec(),
                limit: 5,
                with_payload: Some(true.into()),
                params: Some(SearchParams {
                    hnsw_ef: Some(128), // higher = more accurate
                    ..Default::default()
                }),
                score_threshold: Some(self.min_score),
                ..Default::default()
            })
            .await?;
        println!("search time:{:.2?}", search_start.elapsed());

        let mut result = Vec::new();
        for scored_point in search_result.result {
            // println!(":{:?} {:?}", result.score, result.description);
            let mongo_id = scored_point_field_to_string(&scored_point, "mongo_id");

            if let Some(_id) = mongo_id {
                result.push(StoreSearchResult {
                    _id,
                    description: "".to_string(),
                    score: Some(scored_point.score),
                });
            } else {
                warn!("cannot get mongo_id from qdrant point");
            }
        }

        Ok(result)
    }
}

async fn create_collection_if_not_exists(client: &Qdrant, collection: &str) -> anyhow::Result<()> {
    if client.collection_exists(collection).await? {
        return Ok(());
    }

    client
        .create_collection(CreateCollection {
            collection_name: collection.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: 384,
                    distance: Distance::Cosine.into(),
                    hnsw_config: Some(HnswConfigDiff {
                        m: Some(16),
                        ef_construct: Some(200),
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        })
        .await?;

    Ok(())
}

fn make_qdrant_point(
    id: &str,
    _description: Option<String>,
    embeddings: &[f32],
) -> anyhow::Result<PointStruct> {
    let uuid = Uuid::new_v4();
    let mut payload = HashMap::new();
    payload.insert(
        "mongo_id".to_string(),
        Value {
            kind: Some(Kind::StringValue(id.to_string())),
        },
    );
    Ok(PointStruct {
        id: Some(PointId::from(uuid.to_string())),
        vectors: Some(Vectors::from(embeddings)),
        payload,
    })
}

fn scored_point_field_to_string(scored_point: &ScoredPoint, field: &str) -> Option<String> {
    if let Some(value) = scored_point.payload.get(field) {
        if let Some(Kind::StringValue(field_value)) = &value.kind {
            Some(field_value.clone())
        } else {
            None
        }
    } else {
        None
    }
}
