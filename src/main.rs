use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::io::{self, Write};

mod api;
mod engine;

use engine::sampler::SamplingConfig;
use engine::tokenizer::RealTokenizer;
use engine::{BackendKind, InferenceEngine};

#[derive(Clone, Debug, ValueEnum)]
enum BackendArg {
    Fake,
    Candle,
}

impl From<BackendArg> for BackendKind {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Fake => BackendKind::Fake,
            BackendArg::Candle => BackendKind::Candle,
        }
    }
}

#[derive(Parser)]
#[command(name = "oxidellm")]
#[command(about = "Rust-native local LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Generate {
        #[arg(long)]
        prompt: String,

        #[arg(long, default_value_t = 32)]
        max_tokens: usize,

        #[arg(long, value_enum, default_value_t = BackendArg::Fake)]
        backend: BackendArg,

        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        #[arg(long)]
        top_k: Option<usize>,

        #[arg(long, default_value_t = 1.0)]
        top_p: f32,

        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer_path: String,

        #[arg(long)]
        model_path: Option<String>,
    },

    Ask {
        #[arg(long)]
        prompt: String,

        #[arg(long, default_value_t = 32)]
        max_tokens: usize,

        #[arg(long, value_enum, default_value_t = BackendArg::Candle)]
        backend: BackendArg,

        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        #[arg(long, default_value_t = 40)]
        top_k: usize,

        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer_path: String,

        #[arg(long)]
        model_path: Option<String>,

        #[arg(long, default_value = "smollm")]
        chat_template: String,

        #[arg(long)]
        max_context_tokens: Option<usize>,
    },

    AskStream {
        #[arg(long)]
        prompt: String,

        #[arg(long, default_value_t = 64)]
        max_tokens: usize,

        #[arg(long, value_enum, default_value_t = BackendArg::Candle)]
        backend: BackendArg,

        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        #[arg(long, default_value_t = 40)]
        top_k: usize,

        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer_path: String,

        #[arg(long)]
        model_path: Option<String>,

        #[arg(long, default_value = "smollm")]
        chat_template: String,

        #[arg(long)]
        max_context_tokens: Option<usize>,
    },

    RagIndex {
        #[arg(long)]
        docs: String,

        #[arg(long, default_value = "rag/index.json")]
        index: String,

        #[arg(long, default_value_t = 220)]
        chunk_size_words: usize,

        #[arg(long, default_value_t = 40)]
        chunk_overlap_words: usize,
    },

    RagAsk {
        #[arg(long)]
        question: String,

        #[arg(long)]
        docs: Option<String>,

        #[arg(long, default_value = "rag/index.json")]
        index: String,

        #[arg(long, default_value_t = 4)]
        top_chunks: usize,

        #[arg(long, default_value_t = 220)]
        chunk_size_words: usize,

        #[arg(long, default_value_t = 40)]
        chunk_overlap_words: usize,

        #[arg(long, default_value_t = 128)]
        max_tokens: usize,

        #[arg(long, value_enum, default_value_t = BackendArg::Candle)]
        backend: BackendArg,

        #[arg(long, default_value_t = 0.3)]
        temperature: f32,

        #[arg(long, default_value_t = 40)]
        top_k: usize,

        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer_path: String,

        #[arg(long)]
        model_path: Option<String>,

        #[arg(long, default_value = "smollm")]
        chat_template: String,

        #[arg(long)]
        max_context_tokens: Option<usize>,

        #[arg(long, default_value_t = false)]
        print_context: bool,
    },

    Bench {
        #[arg(long)]
        prompt: String,

        #[arg(long, default_value_t = 128)]
        max_tokens: usize,

        #[arg(long, value_enum, default_value_t = BackendArg::Fake)]
        backend: BackendArg,

        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer_path: String,

        #[arg(long)]
        model_path: Option<String>,
    },

    Tokenize {
        #[arg(long)]
        prompt: String,

        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer_path: String,
    },

    Serve {
        #[arg(long, default_value_t = 3000)]
        port: u16,

        #[arg(long, value_enum, default_value_t = BackendArg::Fake)]
        backend: BackendArg,

        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer_path: String,

        #[arg(long)]
        model_path: Option<String>,
    },

    CandleSmoke,

    InspectGguf {
        #[arg(long, default_value = "models/smollm2-360m-instruct-q8_0.gguf")]
        model_path: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            prompt,
            max_tokens,
            backend,
            temperature,
            top_k,
            top_p,
            tokenizer_path,
            model_path,
        } => {
            let mut engine = InferenceEngine::new_with_backend(
                &tokenizer_path,
                backend.into(),
                model_path.as_deref(),
            )?;

            println!("Prompt: {}", prompt);
            println!("Generated:\n");

            let sampling = SamplingConfig::new(temperature, top_k, top_p);
            let result = engine.generate_with_sampling(&prompt, max_tokens, sampling)?;

            println!("{}", result.text);
            result.metrics.print();
        }

        Commands::Ask {
            prompt,
            max_tokens,
            backend,
            temperature,
            top_k,
            top_p,
            tokenizer_path,
            model_path,
            chat_template,
            max_context_tokens,
        } => {
            let mut engine = InferenceEngine::new_with_backend(
                &tokenizer_path,
                backend.into(),
                model_path.as_deref(),
            )?;

            let chat_prompt =
                InferenceEngine::format_chat_prompt_with_template(&prompt, &chat_template);

            engine.ensure_context_limit(&chat_prompt, max_context_tokens)?;

            let sampling = SamplingConfig::new(temperature, Some(top_k), top_p);

            println!("User: {}", prompt);
            println!("Assistant:\n");

            let result = engine.generate_with_sampling(&chat_prompt, max_tokens, sampling)?;

            println!("{}", result.text.trim());
            result.metrics.print();
        }

        Commands::AskStream {
            prompt,
            max_tokens,
            backend,
            temperature,
            top_k,
            top_p,
            tokenizer_path,
            model_path,
            chat_template,
            max_context_tokens,
        } => {
            let mut engine = InferenceEngine::new_with_backend(
                &tokenizer_path,
                backend.into(),
                model_path.as_deref(),
            )?;

            let chat_prompt =
                InferenceEngine::format_chat_prompt_with_template(&prompt, &chat_template);

            engine.ensure_context_limit(&chat_prompt, max_context_tokens)?;

            let sampling = SamplingConfig::new(temperature, Some(top_k), top_p);

            println!("User: {}", prompt);
            println!("Assistant:\n");

            let result = engine.generate_stream_with_sampling(
                &chat_prompt,
                max_tokens,
                sampling,
                |token| {
                    print!("{}", token);
                    io::stdout().flush().unwrap();
                },
            )?;

            println!();
            result.metrics.print();
        }

        Commands::RagIndex {
            docs,
            index,
            chunk_size_words,
            chunk_overlap_words,
        } => {
            println!("Building RAG index...");
            println!("Docs: {}", docs);
            println!("Index: {}", index);

            let rag_index = engine::rag::build_index(
                &docs,
                chunk_size_words,
                chunk_overlap_words,
            )?;

            engine::rag::save_index(&rag_index, &index)?;

            println!("RAG index saved.");
            println!("Chunks indexed: {}", rag_index.chunks.len());
            println!("Chunk size words: {}", rag_index.chunk_size_words);
            println!("Chunk overlap words: {}", rag_index.chunk_overlap_words);
        }

        Commands::RagAsk {
            question,
            docs,
            index,
            top_chunks,
            chunk_size_words,
            chunk_overlap_words,
            max_tokens,
            backend,
            temperature,
            top_k,
            top_p,
            tokenizer_path,
            model_path,
            chat_template,
            max_context_tokens,
            print_context,
        } => {
            let rag_index = match engine::rag::load_index(&index) {
                Ok(index_data) => index_data,
                Err(load_err) => {
                    let Some(docs_path) = docs else {
                        anyhow::bail!(
                            "Could not load RAG index at '{}': {}\nRun rag-index first or pass --docs to build it automatically.",
                            index,
                            load_err
                        );
                    };

                    println!("Could not load index. Building a new one from docs...");
                    println!("Docs: {}", docs_path);
                    println!("Index: {}", index);

                    let built_index = engine::rag::build_index(
                        &docs_path,
                        chunk_size_words,
                        chunk_overlap_words,
                    )?;

                    engine::rag::save_index(&built_index, &index)?;
                    built_index
                }
            };

            let retrieved = engine::rag::retrieve(&rag_index, &question, top_chunks);

            if retrieved.is_empty() {
                anyhow::bail!("No relevant chunks found for question: {}", question);
            }

            let context = engine::rag::format_retrieved_context(&retrieved);

            if print_context {
                println!("Retrieved context:\n");
                println!("{}", context);
                println!("---");
            }

            let rag_user_prompt = format!(
                "Answer the question using only the provided context.\n\
If the answer is not present in the context, say: I don't know from the provided context.\n\n\
Context:\n{}\n\
Question:\n{}\n\n\
Answer:",
                context,
                question
            );

            let chat_prompt = InferenceEngine::format_chat_prompt_with_template(
                &rag_user_prompt,
                &chat_template,
            );

            let mut engine = InferenceEngine::new_with_backend(
                &tokenizer_path,
                backend.into(),
                model_path.as_deref(),
            )?;

            engine.ensure_context_limit(&chat_prompt, max_context_tokens)?;

            let sampling = SamplingConfig::new(temperature, Some(top_k), top_p);

            println!("Question: {}", question);
            println!("Top chunks used: {}", retrieved.len());
            println!("Assistant:\n");

            let result = engine.generate_with_sampling(&chat_prompt, max_tokens, sampling)?;

            println!("{}", result.text.trim());
            result.metrics.print();
        }

        Commands::Bench {
            prompt,
            max_tokens,
            backend,
            tokenizer_path,
            model_path,
        } => {
            let mut engine = InferenceEngine::new_with_backend(
                &tokenizer_path,
                backend.into(),
                model_path.as_deref(),
            )?;

            let result = engine.generate(&prompt, max_tokens)?;
            result.metrics.print();
        }

        Commands::Tokenize {
            prompt,
            tokenizer_path,
        } => {
            let tokenizer = RealTokenizer::from_file(&tokenizer_path)?;
            let tokens = tokenizer.encode(&prompt)?;
            let decoded = tokenizer.decode(&tokens)?;

            println!("Prompt: {}", prompt);
            println!("Tokenizer path: {}", tokenizer_path);
            println!("Vocab size: {}", tokenizer.vocab_size());
            println!("Token count: {}", tokens.len());
            println!("Token IDs: {:?}", tokens);
            println!("Decoded: {}", decoded);
        }

        Commands::Serve {
            port,
            backend,
            tokenizer_path,
            model_path,
        } => {
            api::server::run(port, backend.into(), tokenizer_path, model_path).await?;
        }

        Commands::CandleSmoke => {
            use candle_core::{Device, Tensor};

            let device = Device::Cpu;

            let a = Tensor::randn(0f32, 1f32, (2, 3), &device)?;
            let b = Tensor::randn(0f32, 1f32, (3, 4), &device)?;

            let c = a.matmul(&b)?;

            println!("Candle smoke test passed.");
            println!("Result tensor:");
            println!("{c}");
        }

        Commands::InspectGguf { model_path } => {
            engine::gguf::inspect_gguf(&model_path)?;
        }
    }

    Ok(())
}