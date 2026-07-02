pub mod cache;
pub mod candle_backend;
pub mod gguf;
pub mod metrics;
pub mod rag;
pub mod model;
pub mod sampler;
pub mod tokenizer;

use anyhow::Result;
use std::time::Instant;

use cache::KvCache;
use candle_backend::CandleModel;
use metrics::GenerationMetrics;
use model::{FakeModel, ModelBackend};
use sampler::{Sampler, SamplingConfig};
use tokenizer::RealTokenizer;

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
        Self::format_chat_prompt_with_template(user_message, "smollm")
    }

    pub fn format_chat_prompt_with_template(user_message: &str, template: &str) -> String {
        let messages = vec![("user".to_string(), user_message.to_string())];
        Self::format_messages_with_template(&messages, template)
    }

    pub fn format_messages_with_template(
        messages: &[(String, String)],
        template: &str,
    ) -> String {
        match template {
            "llama3" => Self::format_llama3_messages(messages),
            "smollm" | _ => Self::format_smollm_messages(messages),
        }
    }

    fn format_llama3_messages(messages: &[(String, String)]) -> String {
        let mut prompt = String::new();

        prompt.push_str("<|begin_of_text|>");

        let has_system = messages
            .iter()
            .any(|(role, _)| role.eq_ignore_ascii_case("system"));

        if !has_system {
            prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
            prompt.push_str(
                "You are a helpful AI assistant. Answer the user's question directly and concisely.",
            );
            prompt.push_str("<|eot_id|>");
        }

        for (role, content) in messages {
            let role = Self::normalize_chat_role(role);

            prompt.push_str("<|start_header_id|>");
            prompt.push_str(role);
            prompt.push_str("<|end_header_id|>\n\n");
            prompt.push_str(content);
            prompt.push_str("<|eot_id|>");
        }

        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

        prompt
    }

    fn format_smollm_messages(messages: &[(String, String)]) -> String {
        let mut prompt = String::new();

        let has_system = messages
            .iter()
            .any(|(role, _)| role.eq_ignore_ascii_case("system"));

        if !has_system {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(
                "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Answer the user's question directly and concisely.",
            );
            prompt.push_str("<|im_end|>\n");
        }

        for (role, content) in messages {
            let role = Self::normalize_chat_role(role);

            prompt.push_str("<|im_start|>");
            prompt.push_str(role);
            prompt.push('\n');
            prompt.push_str(content);
            prompt.push_str("<|im_end|>\n");
        }

        prompt.push_str("<|im_start|>assistant\n");

        prompt
    }

    fn normalize_chat_role(role: &str) -> &'static str {
        if role.eq_ignore_ascii_case("system") {
            "system"
        } else if role.eq_ignore_ascii_case("assistant") {
            "assistant"
        } else {
            "user"
        }
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.tokenizer.encode(text)?.len())
    }

    pub fn ensure_context_limit(
        &self,
        prompt: &str,
        max_context_tokens: Option<usize>,
    ) -> Result<()> {
        let Some(max_context_tokens) = max_context_tokens else {
            return Ok(());
        };

        let prompt_tokens = self.count_tokens(prompt)?;

        if prompt_tokens > max_context_tokens {
            anyhow::bail!(
                "Prompt has {} tokens, which exceeds max_context_tokens={}.\nReduce chat history or increase the context limit.",
                prompt_tokens,
                max_context_tokens
            );
        }

        Ok(())
    }

    pub fn truncate_messages_to_context(
        &self,
        messages: &[(String, String)],
        template: &str,
        max_context_tokens: usize,
    ) -> Result<Vec<(String, String)>> {
        let system_messages: Vec<(String, String)> = messages
            .iter()
            .filter(|(role, _)| role.eq_ignore_ascii_case("system"))
            .cloned()
            .collect();

        let non_system_messages: Vec<(String, String)> = messages
            .iter()
            .filter(|(role, _)| !role.eq_ignore_ascii_case("system"))
            .cloned()
            .collect();

        let mut kept: Vec<(String, String)> = Vec::new();

        for message in non_system_messages.iter().rev() {
            let mut candidate = system_messages.clone();

            let mut reversed_kept = kept.clone();
            reversed_kept.push(message.clone());
            reversed_kept.reverse();

            candidate.extend(reversed_kept);

            let prompt = Self::format_messages_with_template(&candidate, template);
            let token_count = self.count_tokens(&prompt)?;

            if token_count <= max_context_tokens {
                kept.push(message.clone());
            } else {
                break;
            }
        }

        kept.reverse();

        let mut final_messages = system_messages;
        final_messages.extend(kept);

        let final_prompt = Self::format_messages_with_template(&final_messages, template);
        let final_tokens = self.count_tokens(&final_prompt)?;

        if final_tokens > max_context_tokens {
            anyhow::bail!(
                "Could not fit even the latest message/system prompt into max_context_tokens={}. Final prompt has {} tokens.",
                max_context_tokens,
                final_tokens
            );
        }

        Ok(final_messages)
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

        let cache_info = self.model.cache_info();
        metrics.set_backend_cache_info(
            cache_info.supports_kv_cache,
            cache_info.cache_owner,
            cache_info.cache_description,
        );

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
                self.cache.start_prefill(prompt_len);
            }

            self.cache.push_decode_token();
            metrics.set_cache_position(self.cache.seq_len);
            metrics.record_token();

            if eos_token_ids.contains(&next_token) {
                break;
            }
        }

        metrics.set_cache_breakdown(self.cache.prefill_tokens, self.cache.decode_tokens);
        metrics.record_decode_time(decode_start.elapsed());

        let generated_tokens = &tokens[prompt_len..];
        let mut text = self.tokenizer.decode(generated_tokens)?;

        Self::clean_stop_strings(&mut text);

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

        let cache_info = self.model.cache_info();
        metrics.set_backend_cache_info(
            cache_info.supports_kv_cache,
            cache_info.cache_owner,
            cache_info.cache_description,
        );

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
                self.cache.start_prefill(prompt_len);
            }

            self.cache.push_decode_token();
            metrics.set_cache_position(self.cache.seq_len);
            metrics.record_token();

            if eos_token_ids.contains(&next_token) {
                break;
            }

            let piece = self.tokenizer.decode(&[next_token])?;
            let mut clean_piece = piece.clone();

            let should_stop = Self::clean_stop_strings(&mut clean_piece);

            if !clean_piece.is_empty() {
                on_token(&clean_piece);
                generated_text.push_str(&clean_piece);
            }

            if should_stop {
                break;
            }
        }

        metrics.set_cache_breakdown(self.cache.prefill_tokens, self.cache.decode_tokens);
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

    fn clean_stop_strings(text: &mut String) -> bool {
        let stop_strings = [
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
            "<eos>",
            "<|im_start|>",
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|begin_of_text|>",
            "<|reserved_special_token",
            "\nUser:",
            "\nuser:",
            "\nQuestion:",
            "\nOptions:",
            "\nassistant",
            "\nAssistant",
            "assistant\n",
            "Assistant\n",
        ];

        for stop in stop_strings {
            if let Some(index) = text.find(stop) {
                text.truncate(index);
                return true;
            }
        }

        false
    }
}
