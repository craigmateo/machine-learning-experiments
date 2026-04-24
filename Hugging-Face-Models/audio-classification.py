from transformers.utils import logging
logging.set_verbosity_error()

from datasets import load_dataset, Audio
from transformers import pipeline
import librosa
import tempfile
import os

# Load dataset
dataset = load_dataset("ashraq/esc50", split="train[:10]")

# Disable automatic decoding to avoid TorchCodec/FFmpeg issues
dataset = dataset.cast_column("audio", Audio(decode=False))

sample = dataset[0]

print("Category:", sample["category"])
print("Target:", sample["target"])
print("Audio keys:", sample["audio"].keys())

# Save raw bytes to a temporary WAV file
audio_bytes = sample["audio"]["bytes"]

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    tmp.write(audio_bytes)
    tmp_path = tmp.name

# Load and resample manually to 48 kHz for CLAP
audio_array, sr = librosa.load(tmp_path, sr=48000, mono=True)

print("Loaded audio shape:", audio_array.shape)
print("Sampling rate:", sr)

# Build classifier
zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="laion/clap-htsat-unfused"
)

print("Model expected sampling rate:", zero_shot_classifier.feature_extractor.sampling_rate)

candidate_labels = [
    "Sound of a dog",
    "Sound of vacuum cleaner",
    "Sound of a child crying",
    "Sound of a bird singing",
    "Sound of an airplane"
]

result = zero_shot_classifier(
    audio_array,
    candidate_labels=candidate_labels
)

print("\nPredictions:")
for item in result:
    print(f"{item['label']}: {item['score']:.4f}")

# Clean up temp file
os.remove(tmp_path)