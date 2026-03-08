use std::io;

/// Convert f32 samples (range -1.0..1.0) to i16 PCM with clamping.
pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| {
            let clamped = s.clamp(-1.0, 1.0);
            (clamped * 32767.0) as i16
        })
        .collect()
}

/// Encode i16 PCM samples as a WAV byte buffer (mono, 16-bit).
#[cfg(test)]
pub fn encode_wav(samples: &[i16], sample_rate: u32) -> Result<Vec<u8>, String> {
    let mut cursor = io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::new(&mut cursor, spec).map_err(|e| format!("WAV write error: {e}"))?;
    for &sample in samples {
        writer
            .write_sample(sample)
            .map_err(|e| format!("WAV sample write error: {e}"))?;
    }
    writer
        .finalize()
        .map_err(|e| format!("WAV finalize error: {e}"))?;
    Ok(cursor.into_inner())
}

/// Encode i16 PCM samples as OGG/Opus using native Rust libraries.
///
/// Opus requires 48 kHz input, so samples are resampled from `sample_rate`.
pub fn encode_ogg(samples: &[i16], sample_rate: u32) -> Result<Vec<u8>, String> {
    use ogg::writing::PacketWriteEndInfo;
    use opus::{Application, Channels, Encoder};

    const OPUS_RATE: u32 = 48_000;
    // 20 ms frames at 48 kHz = 960 samples
    const FRAME_SIZE: usize = 960;

    if samples.is_empty() {
        return Err("no samples to encode".into());
    }

    // Resample to 48 kHz if needed
    let resampled = if sample_rate == OPUS_RATE {
        samples.to_vec()
    } else {
        resample_i16(samples, sample_rate, OPUS_RATE)
    };

    let mut encoder = Encoder::new(OPUS_RATE, Channels::Mono, Application::Audio)
        .map_err(|e| format!("Opus encoder creation error: {e}"))?;

    let mut ogg_writer = ogg::writing::PacketWriter::new(io::Cursor::new(Vec::new()));
    let serial = 1u32;

    // Write OpusHead header (RFC 7845)
    let opus_head = build_opus_head(1, OPUS_RATE);
    ogg_writer
        .write_packet(
            opus_head,
            serial,
            PacketWriteEndInfo::EndPage,
            0, // granule position
        )
        .map_err(|e| format!("OGG write error: {e}"))?;

    // Write OpusTags header
    let opus_tags = build_opus_tags();
    ogg_writer
        .write_packet(opus_tags, serial, PacketWriteEndInfo::EndPage, 0)
        .map_err(|e| format!("OGG write error: {e}"))?;

    // Encode audio frames
    let mut granule_pos: u64 = 0;
    let total_frames = resampled.len().div_ceil(FRAME_SIZE);
    let mut encoded_buf = vec![0u8; 4000]; // max Opus packet size

    for (i, chunk) in resampled.chunks(FRAME_SIZE).enumerate() {
        let is_last = i + 1 == total_frames;

        // Pad last frame with silence if needed
        let frame = if chunk.len() < FRAME_SIZE {
            let mut padded = vec![0i16; FRAME_SIZE];
            padded[..chunk.len()].copy_from_slice(chunk);
            padded
        } else {
            chunk.to_vec()
        };

        let encoded_len = encoder
            .encode(&frame, &mut encoded_buf)
            .map_err(|e| format!("Opus encode error: {e}"))?;

        granule_pos += FRAME_SIZE as u64;

        let end_info = if is_last {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };

        ogg_writer
            .write_packet(
                encoded_buf[..encoded_len].to_vec(),
                serial,
                end_info,
                granule_pos,
            )
            .map_err(|e| format!("OGG write error: {e}"))?;
    }

    let result = ogg_writer.into_inner().into_inner();

    tracing::debug!(
        input_samples = samples.len(),
        ogg_bytes = result.len(),
        "OGG/Opus encoding complete"
    );

    Ok(result)
}

/// Build OpusHead packet per RFC 7845 section 5.1.
fn build_opus_head(channels: u8, sample_rate: u32) -> Vec<u8> {
    let mut head = Vec::with_capacity(19);
    head.extend_from_slice(b"OpusHead"); // magic
    head.push(1); // version
    head.push(channels); // channel count
    head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
    head.extend_from_slice(&sample_rate.to_le_bytes()); // input sample rate
    head.extend_from_slice(&0i16.to_le_bytes()); // output gain
    head.push(0); // channel mapping family
    head
}

/// Build minimal OpusTags packet per RFC 7845 section 5.2.
fn build_opus_tags() -> Vec<u8> {
    let vendor = b"fishspeech-mcp-server";
    let mut tags = Vec::new();
    tags.extend_from_slice(b"OpusTags");
    tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    tags.extend_from_slice(vendor);
    tags.extend_from_slice(&0u32.to_le_bytes()); // no user comments
    tags
}

/// Simple linear resampling for i16 samples.
fn resample_i16(samples: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_idx = i as f64 / ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;
        let sample = if idx + 1 < samples.len() {
            let a = samples[idx] as f64;
            let b = samples[idx + 1] as f64;
            (a * (1.0 - frac) + b * frac) as i16
        } else if idx < samples.len() {
            samples[idx]
        } else {
            0
        };
        output.push(sample);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_to_i16_clamps() {
        let samples = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let result = f32_to_i16(&samples);
        assert_eq!(result[0], -32767); // clamped to -1.0
        assert_eq!(result[1], -32767);
        assert_eq!(result[2], 0);
        assert_eq!(result[3], 32767);
        assert_eq!(result[4], 32767); // clamped to 1.0
    }

    #[test]
    fn encode_wav_roundtrip() {
        let samples: Vec<i16> = vec![0, 1000, -1000, 32767, -32768];
        let wav = encode_wav(&samples, 22050).unwrap();

        // Verify it's valid WAV by reading it back.
        let cursor = io::Cursor::new(wav);
        let mut reader = hound::WavReader::new(cursor).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 22050);
        assert_eq!(spec.bits_per_sample, 16);

        let decoded: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn encode_wav_empty() {
        let wav = encode_wav(&[], 44100).unwrap();
        let cursor = io::Cursor::new(wav);
        let reader = hound::WavReader::new(cursor).unwrap();
        assert_eq!(reader.len(), 0);
    }

    #[test]
    fn encode_ogg_produces_valid_output() {
        // Generate 1 second of 440 Hz sine wave at 22050 Hz
        let sample_rate = 22050u32;
        let samples: Vec<i16> = (0..sample_rate as usize)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (f64::sin(t * 440.0 * 2.0 * std::f64::consts::PI) * 16000.0) as i16
            })
            .collect();

        let ogg_bytes = encode_ogg(&samples, sample_rate).unwrap();

        // Verify OGG structure: starts with "OggS" magic
        assert!(ogg_bytes.len() > 100);
        assert_eq!(&ogg_bytes[..4], b"OggS");
    }

    #[test]
    fn resample_same_rate() {
        let samples: Vec<i16> = vec![100, 200, 300];
        let result = resample_i16(&samples, 22050, 22050);
        assert_eq!(result, samples);
    }

    #[test]
    fn resample_upsample() {
        let samples: Vec<i16> = vec![0, 1000, 0];
        let result = resample_i16(&samples, 22050, 44100);
        // Should roughly double the number of samples
        assert_eq!(result.len(), 6);
    }
}
