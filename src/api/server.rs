use anyhow::Result;
use axum::{
    extract::State,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, net::SocketAddr, sync::Arc};
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

use crate::engine::sampler::SamplingConfig;
use crate::engine::{BackendKind, InferenceEngine};

struct AppState {
    engine: Mutex<InferenceEngine>,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub top_k: Option<usize>,

    #[serde(default)]
    pub top_p: Option<f32>,
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub text: String,
    pub usage: Option<Usage>,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub top_k: Option<usize>,

    #[serde(default)]
    pub top_p: Option<f32>,
}

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct AssistantMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
struct StreamToken {
    token: String,
}

async fn health() -> &'static str {
    "ok"
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let max_tokens = req.max_tokens.unwrap_or(32);

    let sampling = SamplingConfig::new(
        req.temperature.unwrap_or(0.0),
        req.top_k,
        req.top_p.unwrap_or(1.0),
    );

    let mut engine = state.engine.lock().await;

    let result = match engine.generate_with_sampling(&req.prompt, max_tokens, sampling) {
        Ok(result) => result,
        Err(err) => {
            return Json(CompletionResponse {
                text: format!("Generation error: {}", err),
                usage: None,
            });
        }
    };

    Json(CompletionResponse {
        text: result.text,
        usage: Some(Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.total_tokens,
        }),
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    let max_tokens = req.max_tokens.unwrap_or(32);

    let user_message = req
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| message.content.clone())
        .unwrap_or_else(|| "".to_string());

    let prompt = InferenceEngine::format_chat_prompt(&user_message);

    let sampling = SamplingConfig::new(
        req.temperature.unwrap_or(0.7),
        Some(req.top_k.unwrap_or(40)),
        req.top_p.unwrap_or(0.9),
    );

    let mut engine = state.engine.lock().await;

    let result = match engine.generate_with_sampling(&prompt, max_tokens, sampling) {
        Ok(result) => result,
        Err(err) => {
            return Json(ChatCompletionResponse {
                id: "chatcmpl-oxidellm-local".to_string(),
                object: "chat.completion".to_string(),
                model: "oxidellm-error".to_string(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: AssistantMessage {
                        role: "assistant".to_string(),
                        content: format!("Generation error: {}", err),
                    },
                    finish_reason: "error".to_string(),
                }],
                usage: None,
            });
        }
    };

    Json(ChatCompletionResponse {
        id: "chatcmpl-oxidellm-local".to_string(),
        object: "chat.completion".to_string(),
        model: result.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant".to_string(),
                content: result.text,
            },
            finish_reason: "length".to_string(),
        }],
        usage: Some(Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.total_tokens,
        }),
    })
}

async fn chat_completions_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let max_tokens = req.max_tokens.unwrap_or(32);

    let user_message = req
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| message.content.clone())
        .unwrap_or_else(|| "".to_string());

    let prompt = InferenceEngine::format_chat_prompt(&user_message);

    let sampling = SamplingConfig::new(
        req.temperature.unwrap_or(0.7),
        Some(req.top_k.unwrap_or(40)),
        req.top_p.unwrap_or(0.9),
    );

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    let state_clone = state.clone();

    tokio::spawn(async move {
        let mut engine = state_clone.engine.lock().await;

        let result = engine.generate_stream_with_sampling(
            &prompt,
            max_tokens,
            sampling,
            |token| {
                let payload = StreamToken {
                    token: token.to_string(),
                };

                let json = match serde_json::to_string(&payload) {
                    Ok(json) => json,
                    Err(_) => return,
                };

                let _ = tx.blocking_send(Ok(Event::default().data(json)));
            },
        );

        if let Err(err) = result {
            let payload = StreamToken {
                token: format!("Generation error: {}", err),
            };

            if let Ok(json) = serde_json::to_string(&payload) {
                let _ = tx.blocking_send(Ok(Event::default().data(json)));
            }
        }

        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(ReceiverStream::new(rx))
}

pub async fn run(
    port: u16,
    backend: BackendKind,
    tokenizer_path: String,
    model_path: Option<String>,
) -> Result<()> {
    let engine = InferenceEngine::new_with_backend(
        &tokenizer_path,
        backend,
        model_path.as_deref(),
    )?;

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/completions/stream", post(chat_completions_stream))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    println!("OxideLLM server running on http://{}", addr);
    println!("Health endpoint: http://{}/health", addr);
    println!("Completion endpoint: POST http://{}/v1/completions", addr);
    println!("Chat endpoint: POST http://{}/v1/chat/completions", addr);
    println!(
        "Chat stream endpoint: POST http://{}/v1/chat/completions/stream",
        addr
    );

    axum::serve(listener, app).await?;

    Ok(())
}