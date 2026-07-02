#[derive(Clone, Debug)]
pub struct KvCache {
    pub seq_len: usize,
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub prefill_done: bool,
}

impl KvCache {
    pub fn new() -> Self {
        Self {
            seq_len: 0,
            prefill_tokens: 0,
            decode_tokens: 0,
            prefill_done: false,
        }
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        self.prefill_tokens = 0;
        self.decode_tokens = 0;
        self.prefill_done = false;
    }

    pub fn start_prefill(&mut self, prompt_tokens: usize) {
        self.prefill_tokens = prompt_tokens;
        self.seq_len = prompt_tokens;
        self.prefill_done = true;
    }

    pub fn push_decode_token(&mut self) {
        self.decode_tokens += 1;
        self.seq_len += 1;
    }

    pub fn is_prefill_done(&self) -> bool {
        self.prefill_done
    }
}