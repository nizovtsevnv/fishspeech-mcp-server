use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use fish_speech_core::text::prompt::load_prompt_text;

pub struct VoiceStore {
    voices: HashMap<String, Tensor>,
}

impl VoiceStore {
    pub fn load(dir: &Path, device: &Device, num_codebooks: usize) -> Result<Self, String> {
        let mut voices = HashMap::new();

        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("failed to read voices directory {}: {e}", dir.display()))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("failed to read directory entry: {e}"))?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("npy") {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| format!("invalid filename: {}", path.display()))?
                    .to_string();

                let tensor = load_prompt_text(&path, device, num_codebooks)
                    .map_err(|e| format!("failed to load voice '{}': {e}", name))?;

                tracing::info!(voice = %name, "loaded voice prompt");
                voices.insert(name, tensor);
            }
        }

        Ok(Self { voices })
    }

    pub fn empty() -> Self {
        Self {
            voices: HashMap::new(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.voices.get(name)
    }

    pub fn names(&self) -> Vec<&String> {
        self.voices.keys().collect()
    }

    pub fn len(&self) -> usize {
        self.voices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_store_empty() {
        let store = VoiceStore::empty();
        assert!(store.get("test").is_none());
        assert!(store.names().is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn voice_store_nonexistent_dir() {
        let result = VoiceStore::load(Path::new("/nonexistent/path"), &Device::Cpu, 8);
        assert!(result.is_err());
    }
}
