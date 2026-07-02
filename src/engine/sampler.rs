use anyhow::Result;
use rand::RngExt;

#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: Some(1),
            top_p: 1.0,
        }
    }

    pub fn new(temperature: f32, top_k: Option<usize>, top_p: f32) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
        }
    }
}

pub struct Sampler;

impl Sampler {
    pub fn greedy(logits: &[f32]) -> usize {
        let mut best_index = 0;
        let mut best_value = f32::NEG_INFINITY;

        for (index, &value) in logits.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_index = index;
            }
        }

        best_index
    }

    pub fn sample(logits: &[f32], config: &SamplingConfig) -> Result<usize> {
        if logits.is_empty() {
            anyhow::bail!("Cannot sample from empty logits.");
        }

        if config.temperature <= 0.0 || config.top_k == Some(1) {
            return Ok(Self::greedy(logits));
        }

        let temperature = config.temperature.max(1e-6);
        let top_p = config.top_p.clamp(0.0, 1.0);

        let mut candidates: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(index, &logit)| (index, logit / temperature))
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(k) = config.top_k {
            let k = k.max(1).min(candidates.len());
            candidates.truncate(k);
        }

        let max_logit = candidates[0].1;

        let mut probs: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|(index, logit)| (index, (logit - max_logit).exp()))
            .collect();

        let total: f32 = probs.iter().map(|(_, prob)| *prob).sum();

        if total <= 0.0 || !total.is_finite() {
            return Ok(Self::greedy(logits));
        }

        for (_, prob) in probs.iter_mut() {
            *prob /= total;
        }

        if top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut keep = 0;

            for (_, prob) in probs.iter() {
                cumulative += *prob;
                keep += 1;

                if cumulative >= top_p {
                    break;
                }
            }

            probs.truncate(keep.max(1));

            let new_total: f32 = probs.iter().map(|(_, prob)| *prob).sum();

            if new_total > 0.0 {
                for (_, prob) in probs.iter_mut() {
                    *prob /= new_total;
                }
            }
        }

        let mut rng = rand::rng();
        let sample: f32 = rng.random();

        let mut cumulative = 0.0;

        for (index, prob) in probs {
            cumulative += prob;

            if sample <= cumulative {
                return Ok(index);
            }
        }

        Ok(Self::greedy(logits))
    }
}