use std::os::fd::AsRawFd;

use base64::Engine;
use candle_core::Tensor;

use fish_speech_core::lm::generate::generate_blocking;
use fish_speech_core::lm::sampling::SamplingArgs;
use fish_speech_core::text::clean::preprocess_text;
use fish_speech_core::text::prompt::PromptEncoder;

use crate::audio;
use crate::FishSpeechModel;

pub const SAMPLE_RATE: u32 = 44100;

/// Synthesis parameters.
pub struct SynthesisConfig {
    pub temp: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub max_new_tokens: usize,
}

// ---------------------------------------------------------------------------
// StdoutGuard: suppress stdout during fish_speech_core calls
// ---------------------------------------------------------------------------
// fish_speech_core contains unconditional println! macros in preprocess_text()
// and SingleBatchGenerator::next(). These pollute MCP's JSON-RPC stdout
// channel, causing parse failures. We redirect fd 1 to /dev/null for the
// duration of synthesis and restore it on Drop.

struct StdoutGuard {
    saved_fd: i32,
}

impl StdoutGuard {
    fn redirect() -> Option<Self> {
        use std::fs::File;

        let devnull = File::open("/dev/null").ok()?;
        let devnull_fd = devnull.as_raw_fd();

        // Save a copy of the real stdout fd
        let saved_fd = unsafe { libc::dup(1) };
        if saved_fd < 0 {
            return None;
        }

        // Point fd 1 at /dev/null
        let rc = unsafe { libc::dup2(devnull_fd, 1) };
        if rc < 0 {
            unsafe { libc::close(saved_fd) };
            return None;
        }

        Some(Self { saved_fd })
    }
}

impl Drop for StdoutGuard {
    fn drop(&mut self) {
        unsafe {
            libc::dup2(self.saved_fd, 1);
            libc::close(self.saved_fd);
        }
    }
}

// ---------------------------------------------------------------------------
// Full synthesis pipeline
// ---------------------------------------------------------------------------

pub fn synthesize(
    text: &str,
    voice_prompt: Option<&Tensor>,
    model: &mut FishSpeechModel,
    config: &SynthesisConfig,
) -> Result<String, String> {
    tracing::debug!(text = %text, has_voice = voice_prompt.is_some(), "starting synthesis");

    // Suppress stdout to prevent fish_speech_core println! from polluting
    // the MCP JSON-RPC channel. Our own output uses tracing (stderr).
    let _guard = StdoutGuard::redirect();

    // 1. Preprocess text into chunks.
    //    Without a voice prompt there is no conditioning to keep the speaker
    //    consistent across chunks — each chunk would pick a random voice.
    //    Merge everything into a single chunk to guarantee one voice.
    let chunks = {
        let raw = preprocess_text(text);
        if raw.is_empty() {
            return Err("text produced no processable chunks".into());
        }
        if voice_prompt.is_none() && raw.len() > 1 {
            tracing::debug!(
                original_chunks = raw.len(),
                "merging chunks (no voice prompt)"
            );
            vec![raw.join(" ")]
        } else {
            raw
        }
    };
    tracing::debug!(chunks = chunks.len(), "text preprocessed");

    // 2. Build prompt encoder
    let prompt_encoder = PromptEncoder::new(
        &model.tokenizer,
        &model.device,
        model.config.num_codebooks,
        model.lm_version,
    );

    // 3. Build conditioning from voice prompt (if any)
    let cached_speaker = match voice_prompt {
        Some(prompt_tensor) => {
            let conditioning = prompt_encoder
                .encode_conditioning_prompt("", prompt_tensor)
                .map_err(|e| format!("failed to encode voice conditioning: {e}"))?;
            Some(conditioning)
        }
        None => None,
    };

    // 4. Encode full sequence
    let sysprompt = Some("Speak out the provided text.".to_string());
    let (n_conditioning_tokens, encoded_chunks) = prompt_encoder
        .encode_sequence(chunks, sysprompt, cached_speaker, true)
        .map_err(|e| format!("failed to encode sequence: {e}"))?;

    tracing::debug!(
        n_chunks = encoded_chunks.len(),
        n_conditioning = n_conditioning_tokens,
        "sequence encoded"
    );

    // 5. Sampling args
    let sampling_args = SamplingArgs {
        temp: config.temp,
        top_p: config.top_p,
        top_k: config.top_k,
        repetition_penalty: config.repetition_penalty,
    };

    // 6. Generate VQ codes for each chunk and collect audio
    let mut all_pcm: Vec<f32> = Vec::new();

    for (i, chunk_prompt) in encoded_chunks.iter().enumerate() {
        tracing::debug!(chunk = i, "generating VQ codes");

        let semantic_tokens = generate_blocking(
            &mut model.lm,
            chunk_prompt,
            config.max_new_tokens,
            &sampling_args,
            false,
        )
        .map_err(|e| format!("generation failed for chunk {i}: {e}"))?;

        let n_tokens = semantic_tokens.dims()[0];
        if n_tokens >= config.max_new_tokens {
            tracing::warn!(
                chunk = i,
                tokens = n_tokens,
                max = config.max_new_tokens,
                "generation hit max_new_tokens limit — audio may be truncated"
            );
        }
        tracing::debug!(chunk = i, tokens = n_tokens, "VQ codes generated");

        // Decode VQ codes to audio via FireflyCodec
        let audio_tensor = model
            .codec
            .decode(&semantic_tokens.unsqueeze(0).map_err(|e| e.to_string())?)
            .map_err(|e| format!("codec decode failed for chunk {i}: {e}"))?;

        let pcm: Vec<f32> = audio_tensor
            .squeeze(0)
            .map_err(|e| e.to_string())?
            .squeeze(0)
            .map_err(|e| e.to_string())?
            .to_vec1::<f32>()
            .map_err(|e| format!("failed to extract PCM data: {e}"))?;

        all_pcm.extend_from_slice(&pcm);

        // Preserve conditioning KV cache, clear rest for next chunk
        if i < encoded_chunks.len() - 1 {
            model
                .lm
                .clear_slow_caches_until(n_conditioning_tokens)
                .map_err(|e| format!("failed to clear caches: {e}"))?;
        }
    }

    // Clear all caches after generation
    model.lm.clear_slow_layer_caches();

    if all_pcm.is_empty() {
        return Err("model produced no audio samples".into());
    }
    tracing::debug!(samples = all_pcm.len(), "inference complete");

    // 7. Convert f32 -> i16 -> OGG/Opus -> base64
    let pcm_i16 = audio::f32_to_i16(&all_pcm);
    let ogg_bytes = audio::encode_ogg(&pcm_i16, SAMPLE_RATE)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&ogg_bytes);

    tracing::info!(
        text_len = text.len(),
        samples = all_pcm.len(),
        ogg_bytes = ogg_bytes.len(),
        "synthesis complete"
    );

    Ok(b64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdout_guard_redirect_and_restore() {
        use std::io::Write;

        // In sandboxed environments (e.g. nix on macOS) fd redirection may fail
        let Some(guard) = StdoutGuard::redirect() else {
            return;
        };

        // While suppressed, println! should go to /dev/null (not panic)
        println!("this should go to /dev/null");
        std::io::stdout().flush().ok();

        // Drop restores stdout
        drop(guard);

        // After restore, stdout should work normally
        println!("stdout restored");
        std::io::stdout().flush().ok();
    }

    #[test]
    fn stdout_guard_is_reentrant() {
        for _ in 0..3 {
            let Some(guard) = StdoutGuard::redirect() else {
                return;
            };
            drop(guard);
        }
    }

    #[test]
    fn sample_rate_is_44100() {
        assert_eq!(SAMPLE_RATE, 44100);
    }
}
