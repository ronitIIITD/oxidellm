pub struct KvCache {
    pub seq_len: usize,
}

impl KvCache {
    pub fn new() -> Self {
        Self { seq_len: 0 }
    }

    pub fn update(&mut self, new_tokens: usize) {
        self.seq_len += new_tokens;
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}