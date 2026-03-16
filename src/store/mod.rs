pub mod mongo_store;

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
    async fn create(&self, description: Option<String>, embeddings: Vec<f32>)
        -> anyhow::Result<()>;
    async fn update(&self, id: String, embeddings: Vec<f32>) -> anyhow::Result<()>;
    async fn _get(&self, id: String) -> anyhow::Result<Option<StoreItem>>;
    async fn get_all(&self) -> anyhow::Result<Vec<StoreItem>>;
    async fn search(&self, query: Vec<f32>) -> anyhow::Result<Vec<StoreSearchResult>>;
}
