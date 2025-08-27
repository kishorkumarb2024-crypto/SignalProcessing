import librosa
import soundfile as sf
import numpy as np

# Load the audio file
audio, sr = librosa.load('data/aud_test.wav', sr=None)
print(f"Original audio shape: {audio.shape}")
print(f"Original sample rate: {sr}")

# Convert to mono if it's stereo (librosa.load does this by default, but just in case)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Normalize audio to range [-1, 1]
audio = audio / np.max(np.abs(audio))

# Optional: RMS normalization to control loudness (set to target RMS = 0.1)
rms = np.sqrt(np.mean(audio**2))
audio = audio / (rms / 0.1)

# Save the cleaned/normalized audio
sf.write('cleaned/aud_test_clean.wav', audio, sr)
print("✅ Cleaned audio saved to 'cleaned/aud_test_clean.wav'")

# Optional: Reload and verify
audio_clean, sr_clean = librosa.load('cleaned/aud_test_clean.wav', sr=None)
print(f"Cleaned audio shape: {audio_clean.shape}")
print(f"Cleaned sample rate: {sr_clean}")

# ✅ Compare stats
print(f"\n📊 Amplitude Comparison:")
print(f"Original max amplitude: {np.max(np.abs(audio))}")
print(f"Cleaned max amplitude: {np.max(np.abs(audio_clean))}")

rms_orig = np.sqrt(np.mean(audio**2))
rms_clean = np.sqrt(np.mean(audio_clean**2))
print(f"Original RMS: {rms_orig:.4f}")
print(f"Cleaned RMS: {rms_clean:.4f}")
