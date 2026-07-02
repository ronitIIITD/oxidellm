use anyhow::Result;
use candle_core::quantized::gguf_file;
use std::{fs::File, path::PathBuf};

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{size_in_bytes} B")
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2} KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2} MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2} GB", size_in_bytes as f64 / 1e9)
    }
}

pub fn inspect_gguf(path: &str) -> Result<()> {
    let model_path = PathBuf::from(path);
    let mut file = File::open(&model_path)?;

    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| e.with_path(model_path.clone()))?;

    let mut total_size_in_bytes = 0usize;

    for (_, tensor) in content.tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size_in_bytes +=
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }

    println!("GGUF inspection successful.");
    println!("Model path: {}", model_path.display());
    println!("Tensor count: {}", content.tensor_infos.len());
    println!("Estimated tensor storage: {}", format_size(total_size_in_bytes));

    println!();
    println!("First 20 tensors:");

    for (index, (name, tensor)) in content.tensor_infos.iter().take(20).enumerate() {
        println!(
            "{:02}. {} | shape: {:?} | dtype: {:?}",
            index + 1,
            name,
            tensor.shape,
            tensor.ggml_dtype
        );
    }

    Ok(())
}