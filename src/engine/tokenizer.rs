use anyhow::Result;
use tokenizers::Tokenizer;

pub struct RealTokenizer {
    tokenizer: Tokenizer,
}

impl RealTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;

        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let text = self
            .tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;

        Ok(text)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        let candidates = [
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
            "<eos>",
        ];

        candidates
            .iter()
            .filter_map(|token| self.token_to_id(token))
            .collect()
    }
}