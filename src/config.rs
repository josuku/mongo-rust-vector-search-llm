use serde::Deserialize;

#[derive(Deserialize, Default)]
pub struct Config {
    pub mongo: MongoConfig,
    pub qdrant: QdrantConfig,
    pub hardware: HardwareType,
    pub search: SearchEngineConfig,
    pub model: ModelConfig,
}

#[derive(Deserialize, Default)]
pub struct MongoConfig {
    pub uri: String,
    pub database: String,
    pub collection: String,
    pub index: String,
}

#[derive(Deserialize, Default)]
pub struct QdrantConfig {
    pub uri: String,
    pub collection: String,
}

#[derive(Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HardwareType {
    #[default]
    Cpu,
    Cuda,
}
impl From<&str> for HardwareType {
    fn from(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "cpu" => HardwareType::Cpu,
            "cuda" => HardwareType::Cuda,
            _ => HardwareType::Cpu,
        }
    }
}

#[derive(Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SearchEngineType {
    #[default]
    Qdrant,
    Mongo,
}
impl From<&str> for SearchEngineType {
    fn from(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "qdrant" => SearchEngineType::Qdrant,
            "mongo" => SearchEngineType::Mongo,
            _ => SearchEngineType::Qdrant,
        }
    }
}

#[derive(Deserialize, Default)]
pub struct SearchEngineConfig {
    pub minscore: f32,
    pub engine: SearchEngineType,
}

#[derive(Deserialize, Default)]
pub struct ModelConfig {
    pub tokenizer: String,
    pub config: String,
    pub safetensors: String,
}
