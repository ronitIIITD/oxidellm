use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagChunk {
    pub id: usize,
    pub source: String,
    pub chunk_index: usize,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagIndex {
    pub chunk_size_words: usize,
    pub chunk_overlap_words: usize,
    pub chunks: Vec<RagChunk>,
}

#[derive(Clone, Debug)]
pub struct ScoredChunk {
    pub chunk: RagChunk,
    pub score: f32,
}

pub fn build_index(
    docs_path: &str,
    chunk_size_words: usize,
    chunk_overlap_words: usize,
) -> Result<RagIndex> {
    let docs = load_documents(docs_path)?;

    let mut chunks = Vec::new();
    let mut next_id = 0usize;

    for (path, text) in docs {
        let source = path.display().to_string();
        let doc_chunks = chunk_text(&text, chunk_size_words, chunk_overlap_words);

        for (chunk_index, chunk_text) in doc_chunks.into_iter().enumerate() {
            chunks.push(RagChunk {
                id: next_id,
                source: source.clone(),
                chunk_index,
                text: chunk_text,
            });

            next_id += 1;
        }
    }

    Ok(RagIndex {
        chunk_size_words,
        chunk_overlap_words,
        chunks,
    })
}

pub fn save_index(index: &RagIndex, index_path: &str) -> Result<()> {
    if let Some(parent) = Path::new(index_path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let json = serde_json::to_string_pretty(index)?;
    fs::write(index_path, json)?;
    Ok(())
}

pub fn load_index(index_path: &str) -> Result<RagIndex> {
    let json = fs::read_to_string(index_path)?;
    let index: RagIndex = serde_json::from_str(&json)?;
    Ok(index)
}

pub fn retrieve(index: &RagIndex, query: &str, top_k: usize) -> Vec<ScoredChunk> {
    let query_terms = tokenize(query);

    if query_terms.is_empty() {
        return Vec::new();
    }

    let query_set: HashSet<String> = query_terms.iter().cloned().collect();

    let mut scored = Vec::new();

    for chunk in &index.chunks {
        let score = score_chunk(&query_terms, &query_set, &chunk.text);

        if score > 0.0 {
            scored.push(ScoredChunk {
                chunk: chunk.clone(),
                score,
            });
        }
    }

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    scored.truncate(top_k);
    scored
}

pub fn format_retrieved_context(chunks: &[ScoredChunk]) -> String {
    let mut context = String::new();

    for (i, scored) in chunks.iter().enumerate() {
        context.push_str(&format!(
            "[{}] Source: {}\n{}\n\n",
            i + 1,
            scored.chunk.source,
            scored.chunk.text
        ));
    }

    context
}

fn load_documents(docs_path: &str) -> Result<Vec<(PathBuf, String)>> {
    let root = PathBuf::from(docs_path);

    if !root.exists() {
        anyhow::bail!("Docs path does not exist: {}", docs_path);
    }

    let mut files = Vec::new();
    collect_files(&root, &mut files)?;

    let mut docs = Vec::new();

    for file in files {
        if !is_supported_text_file(&file) {
            continue;
        }

        let text = fs::read_to_string(&file)?;

        if !text.trim().is_empty() {
            docs.push((file, text));
        }
    }

    if docs.is_empty() {
        anyhow::bail!(
            "No supported text documents found in {}. Use .txt, .md, .rs, .py, .json, .toml, .csv, .html or .css files.",
            docs_path
        );
    }

    Ok(docs)
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        files.push(path.to_path_buf());
        return Ok(());
    }

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();

        if child.is_dir() {
            collect_files(&child, files)?;
        } else {
            files.push(child);
        }
    }

    Ok(())
}

fn is_supported_text_file(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };

    matches!(
        ext.to_ascii_lowercase().as_str(),
        "txt" | "md" | "rs" | "py" | "json" | "toml" | "csv" | "html" | "css" | "js" | "ts"
    )
}

fn chunk_text(
    text: &str,
    chunk_size_words: usize,
    chunk_overlap_words: usize,
) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return Vec::new();
    }

    let chunk_size_words = chunk_size_words.max(50);
    let chunk_overlap_words = chunk_overlap_words.min(chunk_size_words / 2);

    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < words.len() {
        let end = (start + chunk_size_words).min(words.len());
        let chunk = words[start..end].join(" ");

        chunks.push(chunk);

        if end == words.len() {
            break;
        }

        start = end.saturating_sub(chunk_overlap_words);
    }

    chunks
}

fn score_chunk(
    query_terms: &[String],
    query_set: &HashSet<String>,
    chunk_text: &str,
) -> f32 {
    let chunk_terms = tokenize(chunk_text);

    if chunk_terms.is_empty() {
        return 0.0;
    }

    let mut term_counts: HashMap<String, usize> = HashMap::new();

    for term in chunk_terms {
        *term_counts.entry(term).or_insert(0) += 1;
    }

    let mut score = 0.0f32;

    for term in query_terms {
        if let Some(count) = term_counts.get(term) {
            score += 2.0 + (*count as f32).ln();
        }
    }

    let chunk_set: HashSet<String> = term_counts.keys().cloned().collect();
    let overlap = query_set.intersection(&chunk_set).count();

    score += overlap as f32;

    let query_lower = query_terms.join(" ");
    let chunk_lower = chunk_text.to_ascii_lowercase();

    if !query_lower.is_empty() && chunk_lower.contains(&query_lower) {
        score += 10.0;
    }

    score
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.to_ascii_lowercase())
        .filter(|part| !is_stopword(part))
        .collect()
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "the"
            | "a"
            | "an"
            | "and"
            | "or"
            | "but"
            | "if"
            | "then"
            | "else"
            | "to"
            | "of"
            | "in"
            | "on"
            | "for"
            | "with"
            | "as"
            | "by"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "this"
            | "that"
            | "these"
            | "those"
            | "it"
            | "its"
            | "from"
            | "at"
            | "into"
            | "about"
            | "what"
            | "who"
            | "when"
            | "where"
            | "why"
            | "how"
    )
}