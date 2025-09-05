import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.decomposition import NMF

def load_raw_data():
    audio, sr = librosa.load('data/aud_test.wav', sr=None)
    print(f"Original audio shape: {audio.shape}")
    print(f"Original sample rate: {sr}")
    return audio, sr

def mono(audio):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    audio = audio / np.max(np.abs(audio))
    return audio

def normalize(audio):
    rms = np.sqrt(np.mean(audio**2))
    audio = audio / (rms / 0.1)
    return audio

def save(audio, source):
    sf.write('cleaned/aud_test_clean.wav', audio, source)
    print("✅ Cleaned audio saved to 'cleaned/aud_test_clean.wav'")

def load():
    audio_clean, sr_clean = librosa.load('cleaned/aud_test_clean.wav', sr=None)
    print(f"Cleaned audio shape: {audio_clean.shape}")
    print(f"Cleaned sample rate: {sr_clean}")
    return audio_clean, sr_clean

def compare(raw, clean):
    print(f"\n📊 Amplitude Comparison:")
    print(f"Original max amplitude: {np.max(np.abs(raw))}")
    print(f"Cleaned max amplitude: {np.max(np.abs(clean))}")
    rms_orig = np.sqrt(np.mean(raw**2))
    rms_clean = np.sqrt(np.mean(clean**2))
    print(f"Original RMS: {rms_orig:.4f}")
    print(f"Cleaned RMS: {rms_clean:.4f}")

def plot(mag_db, sr):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mag_db, sr=sr, hop_length=512, x_axis="time", y_axis="log")
    plt.title("Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def stft(audio, sr):
    data = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
    mag_db = librosa.amplitude_to_db(np.abs(data), ref=np.max)
    plot(mag_db, sr)
    return data

def apply_nmf(data, sr, n_components=2):
    magnitude, phase = np.abs(data), np.angle(data)
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=500)
    W = model.fit_transform(magnitude)
    H = model.components_
    
    sources = []
    for i in range(n_components):
        source_mag = np.outer(W[:, i], H[i, :])
        source_stft = source_mag * np.exp(1j * phase)
        source_audio = librosa.istft(source_stft, hop_length=512, win_length=2048)
        filename = f"source_{i+1}.wav"
        sf.write(filename, source_audio, sr)
        print(f"🔊 saved {filename}")
        sources.append(source_audio)
    return sources

def istft(data, sr):
    # Instead of directly ISTFT here, apply NMF and reconstruct sources
    sources = apply_nmf(data, sr, n_components=2)  # Adjust n_components as needed
    return sources

def save_reconstructed(data, sr):
    sf.write("recon.wav", data, sr)
    print("🔊 Reconstructed audio saved to 'recon.wav'")

if __name__ == "__main__":
    audio_raw, sr = load_raw_data()
    audio_raw = mono(audio_raw)
    audio_raw = normalize(audio_raw)
    save(audio_raw, sr)
    audio_clean, sr = load()
    compare(audio_raw, audio_clean)
    decon = stft(audio_raw, sr)
    separated_sources = istft(decon, sr)
