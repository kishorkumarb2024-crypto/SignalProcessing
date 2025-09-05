import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def load_raw_data():
    # Load the audio file
    audio, sr = librosa.load('data/aud_test.wav', sr=None)
    print(f"Original audio shape: {audio.shape}")
    print(f"Original sample rate: {sr}")
    return audio, sr

def mono(audio):
    # Convert to mono if it's stereo (librosa.load does this by default, but just in case)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    # Normalize audio to range [-1, 1]
    audio = audio / np.max(np.abs(audio))
    return audio

def normalize(audio):
    # Optional: RMS normalization to control loudness (set to target RMS = 0.1)
    rms = np.sqrt(np.mean(audio**2))
    audio = audio / (rms / 0.1)
    return audio

def save(audio, source):
    # Save the cleaned/normalized audio
    sf.write('cleaned/aud_test_clean.wav', audio, source)
    print("✅ Cleaned audio saved to 'cleaned/aud_test_clean.wav'")

def load():
    # Optional: Reload and verify
    audio_clean, sr_clean = librosa.load('cleaned/aud_test_clean.wav', sr=None)
    print(f"Cleaned audio shape: {audio_clean.shape}")
    print(f"Cleaned sample rate: {sr_clean}")
    return audio_clean, sr_clean

def compare(raw, clean):
    # Compare stats
    print(f"\n📊 Amplitude Comparison:")
    print(f"Original max amplitude: {np.max(np.abs(raw))}")
    print(f"Cleaned max amplitude: {np.max(np.abs(clean))}")
    rms_orig = np.sqrt(np.mean(raw**2))
    rms_clean = np.sqrt(np.mean(clean**2))
    print(f"Original RMS: {rms_orig:.4f}")
    print(f"Cleaned RMS: {rms_clean:.4f}")

def plot(mag_db, sr):
    plt.figure(figsize=(12, 6))  # sizing the image so that dimensions are 12x6
    librosa.display.specshow(mag_db, sr=sr, hop_length=512, x_axis="time", y_axis="log")
    plt.title("Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def stft(audio, sr):
    """
    Divide a longer time signal into shorter segments and compute Fourier transform
    """
    data = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
    mag_db = librosa.amplitude_to_db(np.abs(data), ref=np.max)
    plot(mag_db, sr)
    return data

def istft(data, sr):
    recon = librosa.istft(data, hop_length=512, win_length=2048)
    save_reconstructed(recon, sr)

def save_reconstructed(data, sr):
    sf.write("recon.wav", data, sr)
    print("🔊 Reconstructed audio saved to 'recon.wav'")

if __name__ == "__main__":
    audio_raw, sr = load_raw_data()
    audio_raw = mono(audio_raw)
    audio_raw = normalize(audio_raw)
    save(audio_raw, sr)  # Save cleaned audio
    audio_clean, sr = load()
    compare(audio_raw, audio_clean)
    decon = stft(audio_raw, sr)
    istft(decon, sr)
