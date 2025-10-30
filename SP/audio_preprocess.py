

import warnings
warnings.filterwarnings("ignore")

import soundfile as sf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.signal import savgol_filter



def load_audio():
    audio, sr = librosa.load('data/aud_test.wav', sr=None)
    print(f"🎵 Loaded audio: {audio.shape}, Sample Rate: {sr}")
    return audio, sr

def preprocess(audio):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio / np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    audio = audio / (rms / 0.1)
    return audio

def show_spec(audio, sr, title):
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()



def nmf_separate(audio, sr, n_components=3):
    stft_data = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
    mag, phase = np.abs(stft_data), np.angle(stft_data)

    model = NMF(n_components=n_components, init='nndsvda', random_state=42, max_iter=800)
    W = model.fit_transform(mag)
    H = model.components_
    WH = np.dot(W, H) + 1e-8  # full reconstruction magnitude

    sources = []
    for i in range(n_components):
        ratio = (W[:, [i]] @ H[[i], :]) / WH
        src_stft = ratio * stft_data
        src_audio = librosa.istft(src_stft, hop_length=512, win_length=2048)

        # Smooth high-frequency artifacts
        src_audio = savgol_filter(src_audio, 51, 3)
        src_audio = src_audio / np.max(np.abs(src_audio) + 1e-8)

        sf.write(f"source_{i+1}.wav", src_audio, sr)
        show_spec(src_audio, sr, f"Separated Source {i+1}")
        print(f"✅ Saved: source_{i+1}.wav")
        sources.append(src_audio)
    return sources



def enhance_speech(audio, sr):
    stft_data = librosa.stft(audio)
    mag, phase = np.abs(stft_data), np.angle(stft_data)
    noise_profile = np.mean(mag[:, :20], axis=1, keepdims=True)
    enhanced_mag = np.maximum(mag - noise_profile, 0)
    enhanced_stft = enhanced_mag * np.exp(1j * phase)
    enhanced = librosa.istft(enhanced_stft)
    enhanced = enhanced / np.max(np.abs(enhanced))
    sf.write("enhanced_speech.wav", enhanced, sr)
    show_spec(enhanced, sr, "Enhanced Speech (Noise Reduced)")
    print("✅ Enhanced speech saved as enhanced_speech.wav")
    return enhanced



def simple_sdr(reference, estimate):
    min_len = min(len(reference), len(estimate))
    ref = reference[:min_len]
    est = estimate[:min_len]
    noise = ref - est
    sdr = 10 * np.log10(np.sum(ref**2) / (np.sum(noise**2) + 1e-8))
    return sdr

def evaluate_quality(reference, sources):
    print("\n🎧 QUALITY EVALUATION (SDR in dB)")
    for i, src in enumerate(sources):
        sdr_val = simple_sdr(reference, src)
        print(f"Source {i+1}: SDR = {sdr_val:.2f} dB")
    print("✅ Evaluation complete.\n")



def process():
    audio, sr = load_audio()
    audio = preprocess(audio)
    show_spec(audio, sr, "Original Audio Spectrogram")

    sources = nmf_separate(audio, sr, n_components=3)
    enhanced = enhance_speech(audio, sr)
    evaluate_quality(audio, sources)

    print("🎶 Files saved: source_1.wav, source_2.wav, enhanced_speech.wav")

if __name__ == "__main__":
    process()

