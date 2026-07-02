# OxideLLM

OxideLLM is a Rust-native local LLM inference engine built from scratch as a systems project. It can load a real tokenizer, inspect GGUF model files, run quantized local inference through Candle and expose both CLI and HTTP APIs for text generation.

The goal of this project is not to wrap an existing Python server. The goal is to understand and build the core parts of an inference stack in Rust: tokenization, model loading, sampling, streaming, metrics and API serving.

---

## Features

- Real `tokenizer.json` loading
- Prompt tokenization into model token IDs
- GGUF model inspection
- Quantized GGUF model loading
- Candle-based CPU inference
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p sampling
- CLI text generation
- CLI streaming token-by-token generation
- OpenAI-style HTTP chat/completions API
- HTTP streaming through Server-Sent Events
- Explicit model path and tokenizer path configuration
- Basic prefill/decode metrics
- Backend abstraction for fake and real model backends

---

## Why OxideLLM?

Most LLM demos hide the actual inference pipeline behind Python libraries or hosted APIs. OxideLLM is built to make the pipeline visible.

A prompt goes through:

```text
text prompt
→ tokenizer.json
→ token IDs
→ GGUF model weights
→ Candle tensors
→ forward pass
→ logits
→ sampler
→ decoded text
→ CLI or HTTP response