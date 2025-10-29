"""
Performs frequency-based audio source separation using digital filters (Butterworth).
Splits input audio into multiple bands and saves them as separate .wav files.
Also provides visualization of spectrograms for each band.
"""

import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, b, a):
    return lfilter(b, a, data)

def separate_audio_by_frequency(input_wav: str, output_dir: str = "output_bands", fs: int = 44100, show_plots: bool = True):
    import os
    os.makedirs(output_dir, exist_ok=True)

    y, sr = sf.read(input_wav)
    if y.ndim > 1:
        print("Detected stereo audio — converting to mono (averaged channels).")
        y = np.mean(y, axis=1)
    if sr != fs:
        print(f"Warning: expected {fs} Hz, but file has {sr} Hz — resampling.")
        y = librosa.resample(y, orig_sr=sr, target_sr=fs)
        sr = fs

    print(f"Audio length: {len(y)/sr:.2f} seconds, Sample rate: {sr} Hz")

    bands = [
        ("sub_bass", 20, 60),
        ("bass", 60, 250),
        ("low_mid", 250, 1000),
        ("high_mid", 1000, 4000),
        ("presence", 4000, 10000),
        ("brilliance", 10000, 20000),
    ]

    separated_bands = {}
    for name, low, high in bands:
        print(f"Filtering {name}: {low}-{high} Hz")
        if low <= 20:
            b, a = butter_lowpass(high, sr, order=6)
        elif high >= 20000:
            b, a = butter_highpass(low, sr, order=6)
        else:
            b, a = butter_bandpass(low, high, sr, order=6)

        filtered = apply_filter(y, b, a)
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(filtered)) > 0:
            filtered = filtered / np.max(np.abs(filtered))

        separated_bands[name] = filtered

        out_path = os.path.join(output_dir, f"{name}.wav")
        sf.write(out_path, filtered, sr)
        print(f"   Saved: {out_path}")

    combined = np.zeros_like(y)
    for sig in separated_bands.values():
        combined += sig
    sf.write(os.path.join(output_dir, "recombined.wav"), combined, sr)

    if show_plots:
        fig, axs = plt.subplots(len(bands), 1, figsize=(10, 14), sharex=True)
        for i, (name, _, _) in enumerate(bands):
            y_band = separated_bands[name]
            S = librosa.amplitude_to_db(np.abs(librosa.stft(y_band, n_fft=1024, hop_length=256)), ref=np.max)
            img = librosa.display.specshow(
                S, sr=sr, hop_length=256, x_axis='time', y_axis='log', ax=axs[i]
            )
            axs[i].set_title(f"{name} band")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced frequency-based audio source separation.")
    parser.add_argument("--input", required=True, help="Path to input .wav file")
    parser.add_argument("--output", default="output_bands", help="Output folder for separated bands")
    parser.add_argument("--no-plots", action="store_true", help="Disable spectrogram plots")
    args = parser.parse_args()

    separate_audio_by_frequency(input_wav=args.input, output_dir=args.output, show_plots=not args.no_plots)
