use anyhow::Result;
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::{fs::File, path::PathBuf};

use super::model::ModelBackend;

pub struct CandleModel {
    vocab_size: usize,
    device: Device,
    model: ModelWeights,
}

impl CandleModel {
    pub fn new(vocab_size: usize, model_path: &str) -> Result<Self> {
        let device = Device::Cpu;

        let path = PathBuf::from(model_path);
        let mut file = File::open(&path)?;

        println!("Loading GGUF model from {}", path.display());

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(path.clone()))?;

        println!("Loaded GGUF metadata.");
        println!("Tensor count: {}", content.tensor_infos.len());

        let model = ModelWeights::from_gguf(content, &mut file, &device)?;

        println!("Candle model loaded on CPU.");

        Ok(Self {
            vocab_size,
            device,
            model,
        })
    }
}

impl ModelBackend for CandleModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn forward(&mut self, tokens: &[u32], cache_len: usize) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            anyhow::bail!("Cannot run model.forward on empty token list.");
        }

        let input_tokens: Vec<u32>;
        let position: usize;

        if cache_len == 0 {
            // First pass: process the whole prompt.
            input_tokens = tokens.to_vec();
            position = 0;
        } else {
            // Later passes: process only the newest token.
            input_tokens = vec![*tokens.last().unwrap()];
            position = tokens.len().saturating_sub(1);
        }

        let input = Tensor::new(input_tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        let logits = self.model.forward(&input, position)?;
        let logits = logits.squeeze(0)?;

        let logits_vec = logits.to_vec1::<f32>()?;

        Ok(logits_vec)
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "smollm2-360m-q8-candle"
    }
}