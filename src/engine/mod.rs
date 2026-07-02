pub mod cache;
pub mod metrics;
pub mod model;
pub mod sampler;
pub mod tokenizer;
pub mod candle_backend;
pub mod gguf;

use anyhow::Result;

use cache::KvCache;
use metrics::GenerationMetrics;
use model::{FakeModel, ModelBackend};
use candle_backend::CandleModel;
use sampler::{Sampler, SamplingConfig};
use tokenizer::RealTokenizer;
use std::time::Instant;

#[derive(Clone, Debug)]
pub enum BackendKind {
    Fake,
    Candle,
}

pub struct InferenceEngine {
    tokenizer: RealTokenizer,
    model: Box<dyn ModelBackend>,
    cache: KvCache,
}

pub struct GenerationResult {
    pub text: String,
    pub metrics: GenerationMetrics,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub model_name: String,
}

impl InferenceEngine {
    pub fn new(tokenizer_path: &str) -> Result<Self> {
        Self::new_with_backend(tokenizer_path, BackendKind::Fake, None)
    }

    pub fn new_with_backend(
        tokenizer_path: &str,
        backend: BackendKind,
        model_path: Option<&str>,
    ) -> Result<Self> {
        let tokenizer = RealTokenizer::from_file(tokenizer_path)?;
        let vocab_size = tokenizer.vocab_size();

        let model: Box<dyn ModelBackend> = match backend {
            BackendKind::Fake => Box::new(FakeModel::new(vocab_size)),
            BackendKind::Candle => {
                let model_path = model_path.unwrap_or("models/smollm2-360m-instruct-q8_0.gguf");
                Box::new(CandleModel::new(vocab_size, model_path)?)
            }
        };

        Ok(Self {
            tokenizer,
            model,
            cache: KvCache::new(),
        })
    }

    pub fn format_chat_prompt(user_message: &str) -> String {
        format!(
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_message
        )
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<GenerationResult> {
        self.generate_with_sampling(prompt, max_tokens, SamplingConfig::greedy())
    }

    pub fn generate_with_sampling(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: SamplingConfig,
    ) -> Result<GenerationResult> {
        self.cache.reset();
        self.model.reset()?;

        let mut tokens = self.tokenizer.encode(prompt)?;
        let prompt_len = tokens.len();
        let eos_token_ids = self.tokenizer.eos_token_ids();

        let mut metrics = GenerationMetrics::start();
        metrics.set_prompt_tokens(prompt_len);

        let decode_start = Instant::now();

        for step in 0..max_tokens {
            let forward_start = Instant::now();

            let logits = self.model.forward(&tokens, self.cache.seq_len)?;

            let forward_time = forward_start.elapsed();

            if step == 0 {
                metrics.record_prefill_time(forward_time);
            }

            let next_token = Sampler::sample(&logits, &sampling)? as u32;

            tokens.push(next_token);

            if step == 0 {
                self.cache.update(prompt_len);
            }

            self.cache.update(1);
            metrics.set_cache_position(self.cache.seq_len);
            metrics.record_token();

            if eos_token_ids.contains(&next_token) {
                break;
            }
        }

        metrics.record_decode_time(decode_start.elapsed());

        let generated_tokens = &tokens[prompt_len..];
        let mut text = self.tokenizer.decode(generated_tokens)?;

        let stop_strings = [
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
            "<eos>",
            "<|im_start|>",
            "\nUser:",
            "\nuser:",
            "\nQuestion:",
            "\nOptions:",
        ];

        for stop in stop_strings {
            if let Some(index) = text.find(stop) {
                text.truncate(index);
            }
        }

        let completion_tokens = generated_tokens.len();
        let total_tokens = tokens.len();

        Ok(GenerationResult {
            text,
            metrics,
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens,
            model_name: self.model.name().to_string(),
        })
    }

    pub fn generate_stream_with_sampling<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: SamplingConfig,
        mut on_token: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(&str),
    {
        self.cache.reset();
        self.model.reset()?;

        let mut tokens = self.tokenizer.encode(prompt)?;
        let prompt_len = tokens.len();
        let eos_token_ids = self.tokenizer.eos_token_ids();

        let mut metrics = GenerationMetrics::start();
        metrics.set_prompt_tokens(prompt_len);

        let decode_start = Instant::now();
        let mut generated_text = String::new();

        for step in 0..max_tokens {
            let forward_start = Instant::now();

            let logits = self.model.forward(&tokens, self.cache.seq_len)?;

            let forward_time = forward_start.elapsed();

            if step == 0 {
                metrics.record_prefill_time(forward_time);
            }

            let next_token = Sampler::sample(&logits, &sampling)? as u32;

            tokens.push(next_token);

            if step == 0 {
                self.cache.update(prompt_len);
            }

            self.cache.update(1);
            metrics.set_cache_position(self.cache.seq_len);
            metrics.record_token();

            if eos_token_ids.contains(&next_token) {
                break;
            }

            let piece = self.tokenizer.decode(&[next_token])?;

            let stop_strings = [
                "<|endoftext|>",
                "<|im_end|>",
                "</s>",
                "<eos>",
                "<|im_start|>",
                "\nUser:",
                "\nuser:",
                "\nQuestion:",
                "\nOptions:",
            ];

            let mut should_stop = false;
            let mut clean_piece = piece.clone();

            for stop in stop_strings {
                if let Some(index) = clean_piece.find(stop) {
                    clean_piece.truncate(index);
                    should_stop = true;
                    break;
                }
            }

            if !clean_piece.is_empty() {
                on_token(&clean_piece);
                generated_text.push_str(&clean_piece);
            }

            if should_stop {
                break;
            }
        }

        metrics.record_decode_time(decode_start.elapsed());

        let completion_tokens = tokens.len().saturating_sub(prompt_len);
        let total_tokens = tokens.len();

        Ok(GenerationResult {
            text: generated_text,
            metrics,
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens,
            model_name: self.model.name().to_string(),
        })
    }
}