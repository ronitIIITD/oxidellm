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

    #[serde(default)]
    pub chat_template: Option<String>,

    #[serde(default)]
    pub max_context_tokens: Option<usize>,

    #[serde(default)]
    pub auto_truncate: Option<bool>,

    #[serde(default)]
    pub return_debug_prompt: Option<bool>,

    #[serde(default)]
    pub return_debug: Option<bool>,
}

#[derive(Deserialize, Clone)]
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

    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_prompt: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<ChatDebugInfo>,
}

#[derive(Serialize)]
pub struct ChatDebugInfo {
    pub prompt_tokens: usize,
    pub max_context_tokens: Option<usize>,
    pub messages_before_truncation: usize,
    pub messages_after_truncation: usize,
    pub auto_truncate_requested: bool,
    pub auto_truncated: bool,
    pub chat_template: String,
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

#[derive(Deserialize)]
pub struct RagCompletionRequest {
    pub question: String,

    #[serde(default)]
    pub index: Option<String>,

    #[serde(default)]
    pub top_chunks: Option<usize>,

    pub max_tokens: Option<usize>,

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub top_k: Option<usize>,

    #[serde(default)]
    pub top_p: Option<f32>,

    #[serde(default)]
    pub chat_template: Option<String>,

    #[serde(default)]
    pub max_context_tokens: Option<usize>,

    #[serde(default)]
    pub return_context: Option<bool>,

    #[serde(default)]
    pub return_debug: Option<bool>,
}

#[derive(Deserialize)]
pub struct RagChatCompletionRequest {
    pub messages: Vec<ChatMessage>,

    #[serde(default)]
    pub index: Option<String>,

    #[serde(default)]
    pub top_chunks: Option<usize>,

    pub max_tokens: Option<usize>,

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub top_k: Option<usize>,

    #[serde(default)]
    pub top_p: Option<f32>,

    #[serde(default)]
    pub chat_template: Option<String>,

    #[serde(default)]
    pub max_context_tokens: Option<usize>,

    #[serde(default)]
    pub auto_truncate: Option<bool>,

    #[serde(default)]
    pub return_context: Option<bool>,

    #[serde(default)]
    pub return_debug_prompt: Option<bool>,

    #[serde(default)]
    pub return_debug: Option<bool>,
}

#[derive(Serialize)]
pub struct RagCompletionResponse {
    pub answer: String,
    pub sources: Vec<RagSource>,
    pub usage: Option<Usage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<RagDebugInfo>,
}

#[derive(Serialize)]
pub struct RagChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub sources: Vec<RagSource>,
    pub usage: Option<Usage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_prompt: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<RagChatDebugInfo>,
}

#[derive(Serialize)]
pub struct RagSource {
    pub source: String,
    pub chunk_index: usize,
    pub score: f32,
}

#[derive(Serialize)]
pub struct RagDebugInfo {
    pub index_path: String,
    pub retrieved_chunks: usize,
    pub prompt_tokens: usize,
    pub max_context_tokens: Option<usize>,
    pub chat_template: String,
}

#[derive(Serialize)]
pub struct RagChatDebugInfo {
    pub index_path: String,
    pub retrieval_query: String,
    pub retrieved_chunks: usize,
    pub prompt_tokens: usize,
    pub max_context_tokens: Option<usize>,
    pub messages_before_truncation: usize,
    pub messages_after_truncation: usize,
    pub auto_truncate_requested: bool,
    pub auto_truncated: bool,
    pub chat_template: String,
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

fn context_error_response(err: impl std::fmt::Display) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: "chatcmpl-oxidellm-local".to_string(),
        object: "chat.completion".to_string(),
        model: "oxidellm-error".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant".to_string(),
                content: format!("Context error: {}", err),
            },
            finish_reason: "error".to_string(),
        }],
        usage: None,
        debug_prompt: None,
        debug: None,
    }
}

fn generation_error_response(err: impl std::fmt::Display) -> ChatCompletionResponse {
    ChatCompletionResponse {
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
        debug_prompt: None,
        debug: None,
    }
}

fn rag_chat_error_response(
    message: String,
    sources: Vec<RagSource>,
    context: Option<String>,
) -> RagChatCompletionResponse {
    RagChatCompletionResponse {
        id: "chatcmpl-oxidellm-rag-local".to_string(),
        object: "chat.completion".to_string(),
        model: "oxidellm-error".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant".to_string(),
                content: message,
            },
            finish_reason: "error".to_string(),
        }],
        sources,
        usage: None,
        context,
        debug_prompt: None,
        debug: None,
    }
}

fn latest_user_message(messages: &[ChatMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role.eq_ignore_ascii_case("user"))
        .map(|message| message.content.clone())
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    let max_tokens = req.max_tokens.unwrap_or(32);
    let chat_template = req.chat_template.as_deref().unwrap_or("smollm");
    let auto_truncate = req.auto_truncate.unwrap_or(false);
    let return_debug_prompt = req.return_debug_prompt.unwrap_or(false);
    let return_debug = req.return_debug.unwrap_or(false);

    let mut messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|message| (message.role.clone(), message.content.clone()))
        .collect();

    let messages_before_truncation = messages.len();
    let mut auto_truncated = false;

    let sampling = SamplingConfig::new(
        req.temperature.unwrap_or(0.7),
        Some(req.top_k.unwrap_or(40)),
        req.top_p.unwrap_or(0.9),
    );

    let mut engine = state.engine.lock().await;

    if let Some(max_context_tokens) = req.max_context_tokens {
        if auto_truncate {
            match engine.truncate_messages_to_context(
                &messages,
                chat_template,
                max_context_tokens,
            ) {
                Ok(truncated_messages) => {
                    auto_truncated = truncated_messages.len() < messages.len();
                    messages = truncated_messages;
                }
                Err(err) => {
                    return Json(context_error_response(err));
                }
            }
        }

        let prompt_check = InferenceEngine::format_messages_with_template(
            &messages,
            chat_template,
        );

        if let Err(err) = engine.ensure_context_limit(&prompt_check, Some(max_context_tokens)) {
            return Json(context_error_response(err));
        }
    }

    let prompt = InferenceEngine::format_messages_with_template(
        &messages,
        chat_template,
    );

    let prompt_tokens = match engine.count_tokens(&prompt) {
        Ok(count) => count,
        Err(err) => {
            return Json(context_error_response(err));
        }
    };

    let debug_prompt = if return_debug_prompt {
        Some(prompt.clone())
    } else {
        None
    };

    let debug = if return_debug {
        Some(ChatDebugInfo {
            prompt_tokens,
            max_context_tokens: req.max_context_tokens,
            messages_before_truncation,
            messages_after_truncation: messages.len(),
            auto_truncate_requested: auto_truncate,
            auto_truncated,
            chat_template: chat_template.to_string(),
        })
    } else {
        None
    };

    let result = match engine.generate_with_sampling(&prompt, max_tokens, sampling) {
        Ok(result) => result,
        Err(err) => {
            return Json(generation_error_response(err));
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
        debug_prompt,
        debug,
    })
}

async fn send_stream_token(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    token: String,
) {
    let payload = StreamToken { token };

    if let Ok(json) = serde_json::to_string(&payload) {
        let _ = tx.send(Ok(Event::default().data(json))).await;
    }
}

async fn chat_completions_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let max_tokens = req.max_tokens.unwrap_or(32);
    let chat_template = req.chat_template.as_deref().unwrap_or("smollm");
    let auto_truncate = req.auto_truncate.unwrap_or(false);

    let mut messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|message| (message.role.clone(), message.content.clone()))
        .collect();

    let sampling = SamplingConfig::new(
        req.temperature.unwrap_or(0.7),
        Some(req.top_k.unwrap_or(40)),
        req.top_p.unwrap_or(0.9),
    );

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    let state_clone = state.clone();

    {
        let engine = state.engine.lock().await;

        if let Some(max_context_tokens) = req.max_context_tokens {
            if auto_truncate {
                match engine.truncate_messages_to_context(
                    &messages,
                    chat_template,
                    max_context_tokens,
                ) {
                    Ok(truncated_messages) => {
                        messages = truncated_messages;
                    }
                    Err(err) => {
                        send_stream_token(
                            &tx,
                            format!("Context error: {}", err),
                        )
                        .await;

                        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                        return Sse::new(ReceiverStream::new(rx));
                    }
                }
            }

            let prompt_check = InferenceEngine::format_messages_with_template(
                &messages,
                chat_template,
            );

            if let Err(err) = engine.ensure_context_limit(&prompt_check, Some(max_context_tokens)) {
                send_stream_token(
                    &tx,
                    format!("Context error: {}", err),
                )
                .await;

                let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                return Sse::new(ReceiverStream::new(rx));
            }
        }
    }

    let prompt = InferenceEngine::format_messages_with_template(
        &messages,
        chat_template,
    );

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

async fn rag_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagCompletionRequest>,
) -> Json<RagCompletionResponse> {
    let index_path = req
        .index
        .clone()
        .unwrap_or_else(|| "rag/index.json".to_string());

    let top_chunks = req.top_chunks.unwrap_or(4);
    let max_tokens = req.max_tokens.unwrap_or(128);
    let chat_template = req.chat_template.as_deref().unwrap_or("smollm");
    let return_context = req.return_context.unwrap_or(false);
    let return_debug = req.return_debug.unwrap_or(false);

    let rag_index = match crate::engine::rag::load_index(&index_path) {
        Ok(index) => index,
        Err(err) => {
            return Json(RagCompletionResponse {
                answer: format!("RAG index error: {}", err),
                sources: Vec::new(),
                usage: None,
                context: None,
                debug: None,
            });
        }
    };

    let retrieved = crate::engine::rag::retrieve(
        &rag_index,
        &req.question,
        top_chunks,
    );

    if retrieved.is_empty() {
        return Json(RagCompletionResponse {
            answer: "I don't know from the provided context.".to_string(),
            sources: Vec::new(),
            usage: None,
            context: None,
            debug: None,
        });
    }

    let context = crate::engine::rag::format_retrieved_context(&retrieved);

    let sources: Vec<RagSource> = retrieved
        .iter()
        .map(|scored| RagSource {
            source: scored.chunk.source.clone(),
            chunk_index: scored.chunk.chunk_index,
            score: scored.score,
        })
        .collect();

    let rag_user_prompt = format!(
        "Answer the question using only the provided context.\n\
If the answer is not present in the context, say: I don't know from the provided context.\n\n\
Context:\n{}\n\
Question:\n{}\n\n\
Answer:",
        context,
        req.question
    );

    let prompt = InferenceEngine::format_chat_prompt_with_template(
        &rag_user_prompt,
        chat_template,
    );

    let sampling = SamplingConfig::new(
        req.temperature.unwrap_or(0.3),
        Some(req.top_k.unwrap_or(40)),
        req.top_p.unwrap_or(0.9),
    );

    let mut engine = state.engine.lock().await;

    if let Err(err) = engine.ensure_context_limit(&prompt, req.max_context_tokens) {
        return Json(RagCompletionResponse {
            answer: format!("Context error: {}", err),
            sources,
            usage: None,
            context: if return_context { Some(context) } else { None },
            debug: None,
        });
    }

    let prompt_tokens = match engine.count_tokens(&prompt) {
        Ok(count) => count,
        Err(err) => {
            return Json(RagCompletionResponse {
                answer: format!("Token count error: {}", err),
                sources,
                usage: None,
                context: if return_context { Some(context) } else { None },
                debug: None,
            });
        }
    };

    let result = match engine.generate_with_sampling(&prompt, max_tokens, sampling) {
        Ok(result) => result,
        Err(err) => {
            return Json(RagCompletionResponse {
                answer: format!("Generation error: {}", err),
                sources,
                usage: None,
                context: if return_context { Some(context) } else { None },
                debug: None,
            });
        }
    };

    Json(RagCompletionResponse {
        answer: result.text.trim().to_string(),
        sources,
        usage: Some(Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.total_tokens,
        }),
        context: if return_context { Some(context) } else { None },
        debug: if return_debug {
            Some(RagDebugInfo {
                index_path,
                retrieved_chunks: retrieved.len(),
                prompt_tokens,
                max_context_tokens: req.max_context_tokens,
                chat_template: chat_template.to_string(),
            })
        } else {
            None
        },
    })
}

async fn rag_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagChatCompletionRequest>,
) -> Json<RagChatCompletionResponse> {
    let index_path = req
        .index
        .clone()
        .unwrap_or_else(|| "rag/index.json".to_string());

    let top_chunks = req.top_chunks.unwrap_or(4);
    let max_tokens = req.max_tokens.unwrap_or(128);
    let chat_template = req.chat_template.as_deref().unwrap_or("smollm");
    let auto_truncate = req.auto_truncate.unwrap_or(false);
    let return_context = req.return_context.unwrap_or(false);
    let return_debug_prompt = req.return_debug_prompt.unwrap_or(false);
    let return_debug = req.return_debug.unwrap_or(false);

    let retrieval_query = match latest_user_message(&req.messages) {
        Some(message) => message,
        None => {
            return Json(rag_chat_error_response(
                "RAG chat error: no user message found.".to_string(),
                Vec::new(),
                None,
            ));
        }
    };

    let rag_index = match crate::engine::rag::load_index(&index_path) {
        Ok(index) => index,
        Err(err) => {
            return Json(rag_chat_error_response(
                format!("RAG index error: {}", err),
                Vec::new(),
                None,
            ));
        }
    };

    let retrieved = crate::engine::rag::retrieve(
        &rag_index,
        &retrieval_query,
        top_chunks,
    );

    if retrieved.is_empty() {
        return Json(RagChatCompletionResponse {
            id: "chatcmpl-oxidellm-rag-local".to_string(),
            object: "chat.completion".to_string(),
            model: "oxidellm-rag".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant".to_string(),
                    content: "I don't know from the provided context.".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            sources: Vec::new(),
            usage: None,
            context: None,
            debug_prompt: None,
            debug: None,
        });
    }

    let context = crate::engine::rag::format_retrieved_context(&retrieved);

    let sources: Vec<RagSource> = retrieved
        .iter()
        .map(|scored| RagSource {
            source: scored.chunk.source.clone(),
            chunk_index: scored.chunk.chunk_index,
            score: scored.score,
        })
        .collect();

    let system_rag_message = format!(
        "You are a retrieval-augmented assistant. Answer using only the provided context.\n\
If the answer is not present in the context, say: I don't know from the provided context.\n\n\
Retrieved context:\n{}",
        context
    );

    let mut messages: Vec<(String, String)> = Vec::new();

    messages.push(("system".to_string(), system_rag_message));

    for message in &req.messages {
        if message.role.eq_ignore_ascii_case("system") {
            let merged_system = format!(
                "Additional system instruction from the user/application:\n{}",
                message.content
            );
            messages.push(("system".to_string(), merged_system));
        } else {
            messages.push((message.role.clone(), message.content.clone()));
        }
    }

    let messages_before_truncation = messages.len();
    let mut auto_truncated = false;

    let sampling = SamplingConfig::new(
        req.temperature.unwrap_or(0.3),
        Some(req.top_k.unwrap_or(40)),
        req.top_p.unwrap_or(0.9),
    );

    let mut engine = state.engine.lock().await;

    if let Some(max_context_tokens) = req.max_context_tokens {
        if auto_truncate {
            match engine.truncate_messages_to_context(
                &messages,
                chat_template,
                max_context_tokens,
            ) {
                Ok(truncated_messages) => {
                    auto_truncated = truncated_messages.len() < messages.len();
                    messages = truncated_messages;
                }
                Err(err) => {
                    return Json(rag_chat_error_response(
                        format!("Context error: {}", err),
                        sources,
                        if return_context { Some(context) } else { None },
                    ));
                }
            }
        }

        let prompt_check = InferenceEngine::format_messages_with_template(
            &messages,
            chat_template,
        );

        if let Err(err) = engine.ensure_context_limit(&prompt_check, Some(max_context_tokens)) {
            return Json(rag_chat_error_response(
                format!("Context error: {}", err),
                sources,
                if return_context { Some(context) } else { None },
            ));
        }
    }

    let prompt = InferenceEngine::format_messages_with_template(
        &messages,
        chat_template,
    );

    let prompt_tokens = match engine.count_tokens(&prompt) {
        Ok(count) => count,
        Err(err) => {
            return Json(rag_chat_error_response(
                format!("Token count error: {}", err),
                sources,
                if return_context { Some(context) } else { None },
            ));
        }
    };

    let debug_prompt = if return_debug_prompt {
        Some(prompt.clone())
    } else {
        None
    };

    let debug = if return_debug {
        Some(RagChatDebugInfo {
            index_path,
            retrieval_query: retrieval_query.clone(),
            retrieved_chunks: retrieved.len(),
            prompt_tokens,
            max_context_tokens: req.max_context_tokens,
            messages_before_truncation,
            messages_after_truncation: messages.len(),
            auto_truncate_requested: auto_truncate,
            auto_truncated,
            chat_template: chat_template.to_string(),
        })
    } else {
        None
    };

    let result = match engine.generate_with_sampling(&prompt, max_tokens, sampling) {
        Ok(result) => result,
        Err(err) => {
            return Json(rag_chat_error_response(
                format!("Generation error: {}", err),
                sources,
                if return_context { Some(context) } else { None },
            ));
        }
    };

    Json(RagChatCompletionResponse {
        id: "chatcmpl-oxidellm-rag-local".to_string(),
        object: "chat.completion".to_string(),
        model: result.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant".to_string(),
                content: result.text.trim().to_string(),
            },
            finish_reason: "length".to_string(),
        }],
        sources,
        usage: Some(Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.total_tokens,
        }),
        context: if return_context { Some(context) } else { None },
        debug_prompt,
        debug,
    })
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
        .route("/v1/rag/completions", post(rag_completions))
        .route("/v1/rag/chat/completions", post(rag_chat_completions))
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
    println!("RAG endpoint: POST http://{}/v1/rag/completions", addr);
    println!(
        "RAG chat endpoint: POST http://{}/v1/rag/chat/completions",
        addr
    );

    axum::serve(listener, app).await?;

    Ok(())
}