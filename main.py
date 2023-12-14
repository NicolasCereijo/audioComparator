import glob
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from frechet_audio_distance import FrechetAudioDistance
from sklearn.metrics.pairwise import cosine_similarity

# Frechet repository: https://github.com/gudgud96/frechet-audio-distance
# Package torchvision needed

# PANN model
frechet = FrechetAudioDistance(
    model_name="pann",
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False
)

reference_path = "referenceFiles/"
compare_path = "fileToCompare/"
reference_files = glob.glob(reference_path + "*.wav")
compare_files = glob.glob(compare_path + "*.wav")
n_fft = 1024  # Fourier transform window size


def load_audio(file_path):
    audio_signal, sample_rate = librosa.load(file_path)
    return audio_signal, sample_rate


def split_audio_by_time(audio, samplerate, seconds_per_segment):
    segments = []
    total_duration = librosa.get_duration(y=audio)
    start_time = 0

    while start_time < total_duration:
        end_time = min(start_time + seconds_per_segment, total_duration)

        # Convert time to sample indices
        start_index = int(start_time * samplerate)
        end_index = int(end_time * samplerate)

        segment = audio[start_index:end_index]
        segments.append(segment)
        start_time += seconds_per_segment

    return segments


def calculate_mfcc(audio_signal, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=20, n_fft=n_fft)
    return mfcc


def compare_mfcc(audio_signal1, audio_signal2, sample_rate1, sample_rate2):
    mfcc1 = calculate_mfcc(audio_signal1, sample_rate1)
    mfcc2 = calculate_mfcc(audio_signal2, sample_rate2)

    # Ensure that both vectors have the same length
    min_length = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_length]
    mfcc2 = mfcc2[:, :min_length]

    # Reshape the MFCC matrices to vectors for cosine similarity calculation
    mfcc1_vector = mfcc1.reshape(-1)
    mfcc2_vector = mfcc2.reshape(-1)

    # Calculate cosine similarity
    similarity_percentage = cosine_similarity([mfcc1_vector], [mfcc2_vector])[0][0] * 100
    return similarity_percentage


def calculate_spectrogram(audio_signal):
    spectrogram = np.abs(librosa.stft(audio_signal, n_fft=n_fft))
    return spectrogram


def compare_spectrograms(audio_signal1, audio_signal2):
    # Calculate the spectrograms
    spectrogram1 = calculate_spectrogram(audio_signal1)
    spectrogram2 = calculate_spectrogram(audio_signal2)

    # Ensure that both spectrograms have the same dimensions
    min_frames = min(spectrogram1.shape[1], spectrogram2.shape[1])
    spectrogram1 = spectrogram1[:, :min_frames]
    spectrogram2 = spectrogram2[:, :min_frames]

    # Reshape the spectrogram matrices to vectors for cosine similarity calculation
    spectrogram1_vector = spectrogram1.reshape(-1)
    spectrogram2_vector = spectrogram2.reshape(-1)

    # Calculate cosine similarity
    similarity_percentage = cosine_similarity([spectrogram1_vector], [spectrogram2_vector])[0][0] * 100
    return similarity_percentage


def plot_audio_series(ax, audio_signal, color, label):
    series = pd.Series(audio_signal)
    series.plot(ax=ax, lw=1, color=color, label=label)
    return series.max(), series.min()


def plot_spectrogram(ax, audio_signal, sample_rate, title):
    spectrogram = np.abs(librosa.stft(audio_signal))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sample_rate, hop_length=512,
                             x_axis='time', y_axis='log', ax=ax)
    ax.set_title(title)


def main():
    audio_signal1, sample_rate1 = load_audio(reference_files[0])
    audio_signal2, sample_rate2 = load_audio(compare_files[0])

    print("Sample rate model: ", frechet.sample_rate)
    print("Sample rate audio 1: ", sample_rate1)
    print("Sample rate audio 2: ", sample_rate2)
    print("Frechet distance: ", frechet.score(reference_path, compare_path, dtype="float32"))

    sns.set_theme(style="white", palette=None)
    color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Overall comparison, MFCC and spectrograms
    # Compare MFCC and print the similarity percentage
    similarity_percentage_mfcc = compare_mfcc(audio_signal1, audio_signal2, sample_rate1, sample_rate2)
    print("Similarity Percentage (MFCC): {:.2f}%".format(similarity_percentage_mfcc))

    # Compare Spectrograms and print the similarity percentage
    similarity_percentage_spectrograms = compare_spectrograms(audio_signal1, audio_signal2)
    print("Similarity Percentage (Spectrograms): {:.2f}%".format(similarity_percentage_spectrograms))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plotting the audio series for audio_signal1 and audio_signal2
    plot_audio_series(axs[0, 0], audio_signal1, color_pal[0], label="Audio 1")
    y_max1, y_min1 = plot_audio_series(axs[0, 0], audio_signal2, color_pal[1], label="Audio 2")
    y_max2, y_min2 = plot_audio_series(axs[0, 1], audio_signal1, color_pal[0], label="Audio 1")

    # Adjusted y-axes for both subgraphs
    axs[0, 0].set_ylim(min(y_min1, y_min2), max(y_max1, y_max2))
    axs[0, 1].set_ylim(min(y_min1, y_min2), max(y_max1, y_max2))

    # Plotting the spectrogram for audio_signal1 and audio_signal2
    plot_spectrogram(axs[1, 0], audio_signal1, sample_rate1, 'Spectrogram for audio 1')
    plot_spectrogram(axs[1, 1], audio_signal2, sample_rate2, 'Spectrogram for audio 2')

    axs[0, 0].legend()
    plt.tight_layout()
    plt.show()

    # Split comparison, MFCC and spectrograms
    segments_audio1 = split_audio_by_time(audio_signal1, sample_rate1, 5)
    segments_audio2 = split_audio_by_time(audio_signal2, sample_rate2, 5)
    split_comparisons = []

    if len(segments_audio1) > len(segments_audio2):
        segments_audio1 = segments_audio1[:len(segments_audio2)]
    elif len(segments_audio1) < len(segments_audio2):
        segments_audio2 = segments_audio2[:len(segments_audio1)]

    for i in range(len(segments_audio1)):
        row = []

        for j in range(len(segments_audio1)):
            comparison = [compare_mfcc(segments_audio1[i], segments_audio2[j], sample_rate1, sample_rate2),
                          compare_spectrograms(segments_audio1[i], segments_audio2[j])]

            row.append(comparison)
        split_comparisons.append(row)

    print(split_comparisons)


if __name__ == "__main__":
    main()
