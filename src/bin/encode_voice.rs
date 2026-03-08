//! Encode a reference audio file into a Fish Speech voice prompt (.npy).
//!
//! Usage:
//!   encode-voice --checkpoint-dir <path> --input voice.mp3 --output voice.npy

use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;

use fish_speech_core::audio;
use fish_speech_core::audio::functional;
use fish_speech_core::codec::{FireflyCodec, FireflyConfig};
use fish_speech_core::config::WhichFishVersion;

#[derive(Parser)]
#[command(
    name = "encode-voice",
    about = "Encode audio into a Fish Speech voice prompt (.npy)"
)]
struct Args {
    /// Path to model checkpoint directory
    #[arg(long)]
    checkpoint_dir: String,

    /// Input audio file (MP3, WAV, OGG, FLAC — any format supported by Symphonia)
    #[arg(long, short)]
    input: String,

    /// Output .npy file path
    #[arg(long, short)]
    output: String,

    /// Device: "cpu" or "cuda"
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let device = if args.device == "cuda" {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    eprintln!("Loading codec from {} ...", args.checkpoint_dir);
    let checkpoint_dir = PathBuf::from(&args.checkpoint_dir);
    let codec_path = checkpoint_dir.join("firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors");
    let cfg = FireflyConfig::get_config_for(WhichFishVersion::Fish1_5);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[codec_path], DType::F32, &device)? };
    let codec = FireflyCodec::load(cfg, vb, WhichFishVersion::Fish1_5)?;

    eprintln!("Loading audio: {} ...", args.input);
    let (audio_tensor, sr) = audio::load(&args.input, &device)?;
    eprintln!(
        "  sample rate: {} Hz, shape: {:?}",
        sr,
        audio_tensor.shape()
    );

    // Resample to codec sample rate (44100 Hz) if needed
    let audio_tensor = if sr != codec.sample_rate {
        eprintln!("  resampling {} -> {} Hz", sr, codec.sample_rate);
        functional::resample(&audio_tensor, sr, codec.sample_rate)?
    } else {
        audio_tensor
    };

    // audio shape: (1, samples) — add batch dim -> (1, 1, samples)
    let audio_batch = audio_tensor.unsqueeze(0)?;
    eprintln!("Encoding VQ codes ...");
    let codes = codec.encode(&audio_batch)?;

    // Remove batch dim -> (codebooks, frames)
    let codes = codes.squeeze(0)?;
    eprintln!("  codes shape: {:?}", codes.shape());

    // Save as .npy (U32 dtype, matching load_prompt_text expectations)
    let codes = codes.to_dtype(DType::U32)?;
    codes.write_npy(&args.output)?;

    let duration = audio_tensor.dim(1)? as f32 / codec.sample_rate as f32;
    eprintln!(
        "Saved voice prompt to {} ({:.1}s of audio -> {:?})",
        args.output,
        duration,
        codes.shape()
    );

    Ok(())
}
