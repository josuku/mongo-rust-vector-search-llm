use crate::config::HardwareType;
use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::VarBuilderArgs;
use candle_transformers::models::bert::{BertModel, Config};
use std::{collections::HashMap, time::Duration};
use tokenizers::Tokenizer;

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    pub tokenization_time: Duration,
    pub inferencing_time: Duration,
    pub pooling_time: Duration,
}

impl Embedder {
    pub fn new(hardware: HardwareType) -> anyhow::Result<Self> {
        let device = get_device(hardware)?;
        let tokenizer = load_tokenizer()?;
        let model = load_model(&device)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            tokenization_time: Duration::new(0, 0),
            inferencing_time: Duration::new(0, 0),
            pooling_time: Duration::new(0, 0),
        })
    }

    pub fn embed(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        let tokens = self.tokenization(text)?;
        let embeddings = self.model_inferencing(tokens)?;
        let result = self.pooling(embeddings)?;
        Ok(result)
    }

    fn tokenization(&mut self, text: &str) -> anyhow::Result<Tensor> {
        let tok_start = std::time::Instant::now();
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|v| *v as i64).collect();
        let tokens = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        self.tokenization_time += tok_start.elapsed();
        Ok(tokens)
    }

    fn model_inferencing(&mut self, tokens: Tensor) -> anyhow::Result<Tensor> {
        let inf_start = std::time::Instant::now();
        let seq_len = tokens.dims()[1];
        let token_type_ids = Tensor::zeros(&[1, seq_len], DType::I64, &self.device)?;
        let attention_mask = Tensor::ones(&[1, seq_len], DType::F32, &self.device)?;

        let embeddings = self
            .model
            .forward(&tokens, &token_type_ids, Some(&attention_mask))?;
        self.inferencing_time += inf_start.elapsed();
        Ok(embeddings)
    }

    fn pooling(&mut self, embeddings: Tensor) -> anyhow::Result<Vec<f32>> {
        let pool_start = std::time::Instant::now();
        let pooled = embeddings.mean(1)?;
        let result = pooled.flatten_all()?.to_vec1()?;
        self.pooling_time += pool_start.elapsed();
        Ok(result)
    }

    pub fn print_times(&mut self, clear: bool) {
        println!(
            "tokenization:{:.2?} inference:{:.2?} pooling:{:.2?}",
            self.tokenization_time, self.inferencing_time, self.pooling_time
        );
        if clear {
            self.clear_times();
        }
    }

    pub fn clear_times(&mut self) {
        self.tokenization_time = Duration::new(0, 0);
        self.inferencing_time = Duration::new(0, 0);
        self.pooling_time = Duration::new(0, 0);
    }
}

fn get_device(hardware: HardwareType) -> anyhow::Result<Device> {
    let device = if hardware == HardwareType::Cuda {
        Device::cuda_if_available(0)?
    } else {
        Device::Cpu
    };

    if Device::is_cpu(&device) {
        println!("Using CPU");
    } else if Device::is_cuda(&device) {
        println!("Using CUDA");
    }

    Ok(device)
}

fn load_tokenizer() -> anyhow::Result<Tokenizer> {
    Tokenizer::from_file("models/tokenizer.json").map_err(|e| anyhow::anyhow!(e.to_string()))
}

fn load_model(device: &Device) -> anyhow::Result<BertModel> {
    let config: Config = serde_json::from_str(&std::fs::read_to_string("models/config.json")?)?;
    let weights: HashMap<String, Tensor> =
        candle_core::safetensors::load("models/model.safetensors", device)?;
    let vb = VarBuilderArgs::from_tensors(weights, DType::F32, device);
    Ok(BertModel::load(vb, &config)?)
}
