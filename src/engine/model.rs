use anyhow::Result;

pub trait ModelBackend: Send {
    fn vocab_size(&self) -> usize;

    fn forward(&mut self, tokens: &[u32], cache_len: usize) -> Result<Vec<f32>>;

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &'static str;
}

pub struct FakeModel {
    vocab_size: usize,
}

impl FakeModel {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl ModelBackend for FakeModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn forward(&mut self, tokens: &[u32], _cache_len: usize) -> Result<Vec<f32>> {
        let mut logits = vec![0.0; self.vocab_size];

        let last = tokens.last().copied().unwrap_or(0) as usize;
        let next = (last + 1) % self.vocab_size;

        logits[next] = 1.0;

        Ok(logits)
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "oxidellm-fake-model"
    }
}