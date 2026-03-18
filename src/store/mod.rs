pub mod mongo_store;
pub mod qdrant_store;

pub struct StoreItem {
    pub id: String,
    pub embeddings: Option<Vec<f32>>,
    pub description: Option<String>,
}

pub struct StoreSearchResult {
    pub _id: String,
    pub description: String,
    pub score: Option<f32>,
}

pub trait Store {
    async fn add(
        &self,
        id: &str,
        description: Option<String>,
        embeddings: &[f32],
    ) -> anyhow::Result<()>;
    async fn update(&self, id: &str, embeddings: &[f32]) -> anyhow::Result<()>;
    async fn get(&self, id: &str) -> anyhow::Result<Option<StoreItem>>;
    async fn get_all(&self) -> anyhow::Result<Vec<StoreItem>>;
    async fn search(&self, query: &[f32]) -> anyhow::Result<Vec<StoreSearchResult>>;
}
