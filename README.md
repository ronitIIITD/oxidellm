# OxideLLM

OxideLLM is a Rust-native local LLM inference engine built to run quantized language models on-device with a clean CLI and HTTP API.

The project supports GGUF model loading through Candle, tokenization, sampling, chat templates, streaming responses, context-window management and a basic local Retrieval-Augmented Generation pipeline.

## Why OxideLLM?

Most LLM apps depend heavily on Python stacks and remote APIs. OxideLLM explores what a lightweight local inference runtime can look like in Rust, with a focus on control, performance visibility and systems-level design.

It is built as a learning-focused but practical local LLM engine with:

- GGUF model loading
- Candle CPU inference
- Tokenization
- Greedy and sampled generation
- Chat-style prompting
- Streaming token output
- HTTP endpoints
- Context length checks
- Auto-truncation for chat history
- Debug metadata for prompts and token counts
- Local keyword-based RAG over user documents

## Features

### Local Inference

OxideLLM can load local GGUF models and generate responses from the command line.

```powershell
cargo run --release -- ask `
  --prompt "Explain TCP in one sentence." `
  --max-tokens 60 `
  --backend candle `
  --chat-template llama3 `
  --temperature 0.3 `
  --top-k 40 `
  --top-p 0.9 `
  --tokenizer-path models\llama3.2-1b-tokenizer.json `
  --model-path models\llama3.2-1b-instruct-q4_k_m.gguf