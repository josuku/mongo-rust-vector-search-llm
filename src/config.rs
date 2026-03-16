use serde::Deserialize;

#[derive(Deserialize, Default)]
pub struct Config {
    pub mongo: MongoConfig,
    pub hardware: HardwareType,
    pub minscore: f32,
}

#[derive(Deserialize, Default)]
pub struct MongoConfig {
    pub uri: String,
    pub database: String,
    pub collection: String,
    pub index: String,
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
