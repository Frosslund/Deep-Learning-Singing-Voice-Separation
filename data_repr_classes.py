# help classes for the musdb-dataset
import random

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.signal import decimate
from tqdm import tqdm

# globals
SEED = np.random.seed(36)


class Track_signal_repr():
    """ Representation of an individual tracks signal information """

    def __init__(self, samples, samples_size, sample_rate):
        self.samples_down, self.samples_size, self.rate_down = self.__downsample(samples, samples_size, sample_rate)

        if Dataset_repr.__init__.freq_repr == 'stft':
            self.win_length = 1024
            self.n_fft = 1024
            self.hop_length = int(self.win_length / 2)
            self.magnitude, self.phase = self.__compute_stft(self.samples_down)
        else:
            self.n_mels = 128
            self.magnitude, self.phase = self.__compute_log_mel(self.samples_down, self.rate_down)

        self.filter_bins, self.frames = self.magnitude.shape

    def __downsample(self, samples, samples_size, sample_rate):
        sampling_factor = 2
        downsampled = samples[np.arange(0, samples_size, sampling_factor)]
        downsampled_sample_size = len(downsampled)
        return downsampled, downsampled_sample_size, sample_rate/sampling_factor

    def __compute_stft(self, samples):
        """ Computes short-time fourier transform and returns magntiude- and phase spectrogram """
        stft = librosa.stft(samples, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        return np.abs(stft), np.angle(stft)

    def __compute_log_mel(self, samples):
        """ Creates log-Mel-spectrum representation of a track """
        mel_spec = librosa.feature.melspectrogram(y=samples, sr=self.rate_down, n_mels=self.n_mels)
        log_mel_spec = np.log(mel_spec + np.finfo(float).eps)
        return np.abs(log_mel_spec), np.angle(log_mel_spec)

    def get_frame_slice(self, idx, size):
        return np.expand_dims(self.magnitude[:, idx:idx+size], axis=2)

    def binary_mask(self, other_samples, samples_size, sample_rate):
        """ thresholds mask values to [0, 1] """
        other_down, _, _ = self.__downsample(other_samples, samples_size, sample_rate)
        if Dataset_repr.__init__.freq_repr == 'stft':
            other_magnitude, _ = self.__compute_stft(other_down)
        else:
            other_magnitude, _ = self.__compute_log_mel(other_down)
        self.magnitude = np.where(self.magnitude > other_magnitude, 1., 0.)

    def norm_spec(self):
        """ normalize spectogram """
        self.magnitude /= self.magnitude.max()


class Track_repr():
    """ Representation of an individual track """

    def __init__(self, idx, mixture, vocal, acc, drums, bass, other, sample_size, sample_rate):
        self.track_id = idx
        self.mixture_samples = mixture
        self.vocals_samples = vocal
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.accompaniment_samples = acc
        self.drums_samples = drums
        self.bass_samples = bass
        self.other_samples = other
        self.mixture = Track_signal_repr(self.mixture_samples, self.sample_size, self.sample_rate)
        self.vocals = Track_signal_repr(self.vocals_samples, self.sample_size, self.sample_rate)

        # create binary mask
        self.vocals.binary_mask(self.accompaniment_samples, self.sample_size, self.sample_rate)

        # normalize mixture spectogram
        self.mixture.norm_spec()


class Dataset_repr():
    """ Representation of a dataset containing training, validation and test set """

    def __init__(self, data, test_size=0.2, val_size=0.1, augment=[False, 'basic', 2], freq_repr='stft', cut_off=0):
        self.augment = augment[0]
        self.augment_style = augment[1]
        self.augment_factor = augment[2]
        Dataset_repr.__init__.freq_repr = freq_repr
        self.cut_off = cut_off
        self.test_size = test_size
        self.val_size = val_size
        # splitting data
        self.train_set, self.test_set, self.augmet_set = self.__split_data(data)
        # slice data
        train_slices_mixture, train_slices_vocal = self.get_slices(
            self.train_set, self.train_set[0].vocals.filter_bins)
        self.test_slices_mixture, self.test_slices_vocal = self.get_slices(
            self.test_set, self.train_set[0].vocals.filter_bins)
        # Shuffle of training slices
        train_slices_mixture, train_slices_vocal = shuffle(
            train_slices_mixture, train_slices_vocal, random_state=SEED)
        # Split train/validation
        self.train_slices_mixture, self.val_slices_mixture, self.train_slices_vocal, self.val_slices_vocal = train_test_split(
            train_slices_mixture, train_slices_vocal, test_size=self.val_size, random_state=SEED)

        if self.augment:
            # slice augmented data
            augmet_set_mixture, augmet_set_vocal = self.get_slices(
                self.augmet_set, self.train_set[0].vocals.filter_bins)
            # concat train and augment slices
            train_slices_mixture = np.concatenate((self.train_slices_mixture, augmet_set_mixture))
            train_slices_vocal = np.concatenate((self.train_slices_vocal, augmet_set_vocal))
            self.train_slices_mixture, self.train_slices_vocal = shuffle(
                train_slices_mixture, train_slices_vocal, random_state=SEED)

    def __split_data(self, data):
        track_list = []
        appendtrack = track_list.append
        for item in tqdm(data, desc='Processing original data'):

            tot = item.audio.shape[0]
            start = int(tot*(self.cut_off/2))

            name = item.name
            mixture = librosa.to_mono(item.audio[start:tot-start, :].T)
            vocals = librosa.to_mono(item.targets['vocals'].audio[start:tot-start, :].T)
            acc = librosa.to_mono(item.targets['accompaniment'].audio[start:tot-start, :].T)
            drums = librosa.to_mono(item.targets['drums'].audio[start:tot-start, :].T)
            bass = librosa.to_mono(item.targets['bass'].audio[start:tot-start, :].T)
            other = librosa.to_mono(item.targets['other'].audio[start:tot-start, :].T)
            sample_size = item.audio[start:tot-start, :].shape[0]
            sample_rate = item.rate
            appendtrack(Track_repr(name, mixture, vocals, acc, drums, bass, other, sample_size, sample_rate))

        # lists of Track objects
        train_set, test_set = train_test_split(track_list, test_size=self.test_size, shuffle=False)

        if self.augment:
            augment_set = []
            appendaug = augment_set.append
            for track in tqdm(train_set, desc='Augmentation'):
                augments = self.__augment(track)
                for augm_list in augments:
                    idx, mixture_augm, vocal_augm = augm_list
                    appendaug(Track_repr(idx,
                                         mixture_augm,
                                         vocal_augm,
                                         track.accompaniment_samples,
                                         track.drums_samples,
                                         track.bass_samples,
                                         track.other_samples,
                                         track.sample_size,
                                         track.sample_rate))
        return [train_set, test_set, augment_set] if self.augment else [train_set, test_set, None]

    def __augment(self, track):
        """ Augmentation of samples and corresponding vocals """

        def mix_stems(track):
            """ addition of random velocity stems """
            rand = random.sample([0, 0.2, 0.4, 0.6, 0.8, 1], 3)
            new_mix = track.vocals_samples + rand[0] * track.drums_samples + \
                rand[1] * track.bass_samples + rand[2] * track.other_samples
            return new_mix

        # borrowed from https://www.kaggle.com/huseinzol05/sound-augmentation-librosa#Streching
        def stretching(samples, stretch):
            input_length = samples.shape[0]
            stretching = samples.copy()
            stretching = librosa.effects.time_stretch(stretching.astype('float'), stretch)
            if len(stretching) > input_length:
                stretching = stretching[:input_length]
            else:
                stretching = np.pad(stretching, (0, max(0, input_length - len(stretching))), "constant")
            return stretching

        # borrowed from https://www.kaggle.com/huseinzol05/sound-augmentation-librosa#Streching
        def pitch(samples, sample_rate, pitch_pm):
            y_pitch = samples.copy()
            bins_per_octave = 12
            pitch_pm = pitch_pm
            pitch_change = pitch_pm * 2*(np.random.uniform())
            y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'),
                                                  sample_rate, n_steps=pitch_change,
                                                  bins_per_octave=bins_per_octave)
            return y_pitch

        augmentations = []
        for i in range(self.augment_factor):
            idx = track.track_id + ' augm_v' + str(i+1)
            mixture = track.mixture_samples if self.augment_style == 'basic' else mix_stems(track)
            stretch = random.randint(90, 110) / 100
            mixture_strech = stretching(mixture, stretch)
            vocal_strech = stretching(track.vocals_samples, stretch)
            pitch_pm = random.randint(-200, 200) / 100
            mixture_augm = pitch(mixture_strech, track.sample_rate, pitch_pm)
            vocal_augm = pitch(vocal_strech, track.sample_rate, pitch_pm)
            augmentations.append([idx, mixture_augm, vocal_augm])
        return augmentations

    def get_slices(self, dset, rows, frames=128, per_track=30, in_order=False):
        if in_order:
            per_track = int(dset[0].mixture.frames / frames)
            rows = set[0].mixture.filter_bins
        mixture_slices = np.zeros((len(dset)*per_track, rows, frames, 1))
        vocal_slices = np.zeros((len(dset)*per_track, rows, frames, 1))
        for i, track in tqdm(enumerate(dset), desc='Slicing frames'):
            for j in range(per_track):
                idx = random.randint(0, track.mixture.frames - frames) if not in_order else j*frames
                mixture_slices[i*per_track+j, :, :, :] = track.mixture.get_frame_slice(idx, frames)
                vocal_slices[i*per_track+j, :, :, :] = track.vocals.get_frame_slice(idx, frames)
        return mixture_slices, vocal_slices
