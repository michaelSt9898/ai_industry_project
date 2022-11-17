import librosa as lr
import soundfile as sf
import numpy as np

"""
Introduces some methods to perform preprocessing on audio data
"""

def slice_to_length(audiofile, sr, length_ms, pad=True):
    """
    slices the audiofile to the desired length from the beginning, discards the rest of the audio
        Parameters:
            audiofile: n-dimensional numpy array
            sr: sample rate of the provided audio
            length_ms: length of the sliced audio array
            pad: if True, audio is zero padded at the end if the provided audio is shorter than the length_ms
        returns: first length_ms milliseconds of the passed audiofile
    """
    length_index = int(sr*length_ms/1000)
    if pad and audiofile.shape[0] < length_index:
        return np.concatenate((audiofile, np.zeros((length_index - audiofile.shape[0], 2), dtype=np.float32)), axis=0)
    return audiofile[0:length_index, :]

def to_mono_channel(audiofile):
    """
    converts a audiofile with multiple channels to a single (mono) channel audio file
        Parameters:
            audiofile: n-dimensional numpy array
        returns: 1-dimensional numpy array
    """
    return lr.to_mono(audiofile.T).T

def resample(audiofile, old_sample_rate, new_sample_rate):
    """
    resamples the audiofile from old_sample_rate to the new_sample_rate
        Parameters:
            audiofile: n-dimensional numpy array
            old_sample_rate: sample rate of the provided audio file
            new_sample_rate: desired sampling rate
        returns: 1-dimensional numpy array with the new sample rate
    """
    return lr.resample(audiofile.T, orig_sr=old_sample_rate, target_sr=new_sample_rate).T

def channels(audiofile) -> int:
    """
    Get the number of channels of the audio file
        Parameters:
            audiofile: n-dimensional numpy array
        returns: nr of channels of the audiofile
    """
    return 1 if len(audiofile.shape) <= 1 else audiofile.shape[1]

def duration(audiofile, sample_rate) -> float:
    """
    gets the duration of the audio file in seconds
        Parameters:
            audiofile: n-dimensional numpy array
            sr: sampling_rate
        returns: duration of the audiofile in seconds
    """
    return audiofile.shape[0] / sample_rate

def test():
    audio1, sr1 = sf.read('wav_data/06 Deep House/Bam Bam Beat.wav', dtype='float32')
    audio2, sr2 = sf.read('wav_data/06 Deep House/Underground States Chord Layers 02.wav', dtype='float32')

    print(audio1.shape[0]/sr1)
    audio1 = slice_to_length(audio1, sr1, 5000)
    print(audio1.shape[0]/sr1)

    print(audio2.shape[0]/sr2)
    audio2 = slice_to_length(audio2, sr2, 5000)
    print(audio2.shape[0]/sr2)

    assert audio1.shape == audio2.shape