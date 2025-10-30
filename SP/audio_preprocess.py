import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.decomposition import NMF, FastICA



def load_raw_data():
    """Load the raw audio file."""
    audio, sr = librosa.load('data/aud_test.wav', sr=None)
    print(f"Original audio shape: {audio.shape}")
    print(f"Original sample rate: {sr}")
    return audio, sr

def mono(audio):
    """Convert to mono and normalize amplitude."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio / np.max(np.abs(audio))
    return audio

def normalize(audio):
    """RMS normalization to target RMS=0.1"""
    rms = np.sqrt(np.mean(audio**2))
    audio = audio / (rms / 0.1)
    return audio

def save(audio, sr):
    """Save cleaned/normalized audio."""
    sf.write('cleaned/aud_test_clean.wav', audio, sr)

def load():
    """Reload cleaned audio to verify."""
    audio_clean, sr_clean = librosa.load('cleaned/aud_test_clean.wav', sr=None)
    print(f"Cleaned audio shape: {audio_clean.shape}")
    print(f"Cleaned sample rate: {sr_clean}")
    return audio_clean, sr_clean

def compare(raw, clean):
    """Compare amplitude and RMS before/after cleaning."""
    print(f"Original max amplitude: {np.max(np.abs(raw))}")
    print(f"Cleaned max amplitude: {np.max(np.abs(clean))}")
    rms_orig = np.sqrt(np.mean(raw**2))
    rms_clean = np.sqrt(np.mean(clean**2))
    print(f"Original RMS: {rms_orig:.4f}")
    print(f"Cleaned RMS: {rms_clean:.4f}")

def plot(mag_db, sr, title="Spectrogram"):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mag_db, sr=sr, hop_length=512, x_axis="time", y_axis="log")
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def stft(audio, sr):
    """Compute STFT and plot spectrogram."""
    data = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
    mag_db = librosa.amplitude_to_db(np.abs(data), ref=np.max)
    plot(mag_db, sr, title="Original Audio Spectrogram")
    return data

def apply_nmf(data, sr, n_components=2):
    """Apply NMF on magnitude spectrogram and reconstruct separated sources."""
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
        sources.append(source_audio)
    return sources

def istft(data, sr):
    """Wrapper: separate via NMF and reconstruct."""
    sources = apply_nmf(data, sr, n_components=2)
    return sources

def postprocess_sources(sources):
    """Normalize and enhance separated outputs."""
    processed = []
    for src in sources:
        src = src / np.max(np.abs(src))  # normalize amplitude
        src = librosa.effects.preemphasis(src)  # simple enhancement
        processed.append(src)
    return processed

def evaluate_sources(mix, sources, sr):
    """Evaluate separation using RMS and spectrograms."""
    for i, src in enumerate(sources):
        rms = np.sqrt(np.mean(src**2))
        print(f"Source {i+1} RMS: {rms:.4f}")

    for i, src in enumerate(sources):
        plt.figure(figsize=(12, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(src)), ref=np.max)
        librosa.display.specshow(D, sr=sr, hop_length=512, x_axis='time', y_axis='log')
        plt.title(f"Spectrogram - Separated Source {i+1}")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()

def separate_ica(file_path):
    """Apply ICA to stereo recordings (2-channel input)."""
    X, sr = sf.read(file_path)
    if X.ndim != 2:
        return
    X = X - np.mean(X, axis=0)
    ica = FastICA(n_components=2, random_state=0, max_iter=1000)
    S_sep = ica.fit_transform(X)
    sf.write("ica_source1.wav", S_sep[:, 0], sr)
    sf.write("ica_source2.wav", S_sep[:, 1], sr)
    return S_sep, sr

def enhance_speech(audio, sr):
    """
    Simple noise reduction using spectral gating (no external library).
    This simulates real-world speech enhancement (like in hearing aids or voice calls).
    """

    # Compute STFT
    stft_data = librosa.stft(audio)
    magnitude, phase = np.abs(stft_data), np.angle(stft_data)

    noise_mag = np.mean(magnitude[:, :int(sr * 0.5 / 512)], axis=1, keepdims=True)
    enhanced_mag = np.maximum(magnitude - noise_mag, 0.0)
    enhanced_stft = enhanced_mag * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft)

    sf.write('enhanced_speech.wav', enhanced_audio, sr)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max),
                             sr=sr, hop_length=512, y_axis='log', x_axis='time')
    plt.title("Before Enhancement")
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(enhanced_mag, ref=np.max),
                             sr=sr, hop_length=512, y_axis='log', x_axis='time')
    plt.title("After Enhancement")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    return enhanced_audio

if __name__ == "__main__":
    audio_raw, sr = load_raw_data()
    audio_raw = mono(audio_raw)
    audio_raw = normalize(audio_raw)
    save(audio_raw, sr)

    audio_clean, sr = load()
    compare(audio_raw, audio_clean)

    decon = stft(audio_raw, sr)
    separated_sources = istft(decon, sr)

    separated_sources = postprocess_sources(separated_sources)
    evaluate_sources(audio_raw, separated_sources, sr)

    enhanced_audio = enhance_speech(audio_raw, sr)

    print("Check generated files: 'source_1.wav', 'source_2.wav', 'enhanced_speech.wav', and 'cleaned/aud_test_clean.wav'.")
