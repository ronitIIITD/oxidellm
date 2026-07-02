use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct GenerationMetrics {
    pub start_time: Instant,
    pub first_token_time: Option<Duration>,
    pub generated_tokens: usize,

    pub prompt_tokens: usize,
    pub prefill_time: Option<Duration>,
    pub decode_time: Option<Duration>,
    pub cache_position: usize,
}

impl GenerationMetrics {
    pub fn start() -> Self {
        Self {
            start_time: Instant::now(),
            first_token_time: None,
            generated_tokens: 0,

            prompt_tokens: 0,
            prefill_time: None,
            decode_time: None,
            cache_position: 0,
        }
    }

    pub fn set_prompt_tokens(&mut self, prompt_tokens: usize) {
        self.prompt_tokens = prompt_tokens;
    }

    pub fn record_prefill_time(&mut self, duration: Duration) {
        self.prefill_time = Some(duration);
    }

    pub fn record_decode_time(&mut self, duration: Duration) {
        self.decode_time = Some(duration);
    }

    pub fn set_cache_position(&mut self, cache_position: usize) {
        self.cache_position = cache_position;
    }

    pub fn record_token(&mut self) {
        self.generated_tokens += 1;

        if self.first_token_time.is_none() {
            self.first_token_time = Some(self.start_time.elapsed());
        }
    }

    pub fn total_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn tokens_per_second(&self) -> f64 {
        let secs = self.total_time().as_secs_f64();

        if secs == 0.0 {
            return 0.0;
        }

        self.generated_tokens as f64 / secs
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        let Some(decode_time) = self.decode_time else {
            return 0.0;
        };

        let secs = decode_time.as_secs_f64();

        if secs == 0.0 {
            return 0.0;
        }

        self.generated_tokens as f64 / secs
    }

    pub fn print(&self) {
        println!("\n--- Metrics ---");
        println!("Prompt tokens: {}", self.prompt_tokens);
        println!("Generated tokens: {}", self.generated_tokens);
        println!("Cache position: {}", self.cache_position);

        if let Some(prefill_time) = self.prefill_time {
            println!("Prefill time: {:.2}s", prefill_time.as_secs_f64());
        }

        if let Some(decode_time) = self.decode_time {
            println!("Decode time: {:.2}s", decode_time.as_secs_f64());
            println!(
                "Decode tokens/sec: {:.2}",
                self.decode_tokens_per_second()
            );
        }

        println!("Total time: {:.2}s", self.total_time().as_secs_f64());
        println!("Tokens/sec: {:.2}", self.tokens_per_second());

        if let Some(first_token_time) = self.first_token_time {
            println!(
                "Time to first token: {:.2}s",
                first_token_time.as_secs_f64()
            );
        }
    }
}