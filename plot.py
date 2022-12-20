import matplotlib.pyplot as plt
import librosa.display as lrd
import numpy as np
import librosa


def plot_waveform(amp, sr):
    """
    Plot a waveform of the given amplitudes
        amp: amplitudes as a numpy array
        sr: sample rate of the audio
    """
    plt.figure(figsize=(14, 5))
    lrd.waveshow(amp, sr=sr)

def plot_melspectogram(S, sr, fmax, save_fig=True):
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=fmax, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    if save_fig:
        fig.savefig('Plots/melfrequency_spectogram with fmax{f}.jpg'.format(f=int(fmax)))

def plot_hist(history, keys, legends, title, y_label, x_label):
    max_size = 0
    for key in keys:
        if max_size < len(history[key]):
            max_size = len(history[key])
        plt.plot(history[key])
    # plt.xticks(range(max_size))
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid()
    plt.legend(legends, loc='upper left')
    plt.show()