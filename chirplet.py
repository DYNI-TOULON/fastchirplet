# -*- coding: utf-8 -*-
"""
__author__ = "Virgil Tassan, Randall Baleistriero, Herve Glotin"
__maintainer__ = "Virgil Tassan"

To run an example :
    $ python example.py

Github link:
https://github.com/DYNI-TOULON/fastchirplet


"""
import numpy as np
from pylab import (arange, flipud, linspace, cos, pi, log, hanning,
                   ceil, log2, floor, empty_like, fft, ifft, fabs, exp, roll, convolve)


class FCT:
    """
    Attributes :
        _duration_longest_chirplet : duration of the longest chirplet in the bank of chirplets
        _num_octaves : the number of octaves
        _num_chirps_by_octave : the number of chirps by octave
        _polynome_degree : degree of the polynomial function
        _end_smoothing : define the size the output of the signal
        _samplerate : samplerate of the signal

    """
    def __init__(self,
                 duration_longest_chirplet=1,
                 num_octaves=5,
                 num_chirps_by_octave=10,
                 polynome_degree=0,
                 end_smoothing=0.001,
                 sample_rate=22050):
        """
        Args:
            duration_longest_chirplet : duration of the longest chirplet in the bank of chirplets
            num_octaves : the number of octaves
            num_chirps_by_octave : the number of chirps by octave
            polynome_degree : degree of the polynomial function
            end_smoothing : define the size the output of the signal
            sample_rate : samplerate of the signal
        """
        self._duration_longest_chirplet = duration_longest_chirplet

        self._num_octaves = num_octaves

        self._num_chirps_by_octave = num_chirps_by_octave

        self._polynome_degree = polynome_degree

        self._end_smoothing = end_smoothing

        # Samplerate of the signal. Has to be defined in advance.
        self._samplerate = sample_rate

        self._chirps = self.__init_chirplet_filter_bank()

    def __init_chirplet_filter_bank(self):
        """generate all the chirplets based on the attributes

        Returns :
            The bank of chirplets
        """
        num_chirps = self._num_octaves*self._num_chirps_by_octave

        #create a list of coefficients based on attributes
        lambdas = 2.0**(1+arange(num_chirps)/float(self._num_chirps_by_octave))

        #Low frequencies for a signal
        start_frequencies = (self._samplerate /lambdas)/2.0

        #high frequencies for a signal
        end_frequencies = self._samplerate /lambdas

        durations = 2.0*self._duration_longest_chirplet/flipud(lambdas)

        chirplets = list()
        for low_frequency, high_frequency, duration in zip(start_frequencies, end_frequencies, durations):
            chirplets.append(Chirplet(self._samplerate, low_frequency, high_frequency, duration, self._polynome_degree))
        return chirplets

    @property
    def time_bin_duration(self):
        """
        Return :
            The time bin duration

        """
        return self._end_smoothing*10

    def compute(self, input_signal):
        """compute the FCT on the given signal
        Args :
            input_signal : Array of an audio signal

        Returns :
            The Fast Chirplet Transform of the given signal

        """
        # keep the real length of the signal
        size_data = len(input_signal)

        nearest_power_2 = 2**(size_data-1).bit_length()

        # find the best power of 2
        # the signal must not be too short

        while nearest_power_2 <= self._samplerate*self._duration_longest_chirplet:
            nearest_power_2 *= 2

        # pad with 0 to have the right length of signal

        data = np.lib.pad(input_signal, (0, nearest_power_2-size_data), 'constant', constant_values=0)

        # apply the fct to the adapted length signal

        chirp_transform = apply_filterbank(data, self._chirps, self._end_smoothing)

        # resize the signal to the right length

        chirp_transform = resize_chirps(size_data, nearest_power_2, chirp_transform)

        return chirp_transform


def resize_chirps(size_data, size_power_2, chirps):
    """Resize the matrix of chirps to the length of the signal
    Args:
        size_data : number of samples of the audio signal
        size_power_2 : number of samples of the signal to apply the FCT
        chirps : the signal to resize
    Returns :
        Chirps with the correct length
    """
    size_chirps = len(chirps)
    ratio = size_data/size_power_2
    size = int(ratio*len(chirps[0]))

    resize_chirps = np.zeros((size_chirps, size))
    for i in range(0, size_chirps):
        resize_chirps[i] = chirps[i][0:size]
    return resize_chirps


class Chirplet:
    """chirplet class
    Attributes:
        _min_frequency : lowest frequency where the chirplet is applied
        _max_frequency : highest frequency where the chirplet is applied
        _duration : duration of the chirp
        _samplerate : samplerate of the signal
        _polynome_degree : degree of the polynome to generate the coefficients of the chirplet
        _filter_coefficients : coefficients applied to the signal
    """
    def __init__(self, samplerate, min_frequency, max_frequency, sigma, polynome_degree):

        """
        Args :
            samplerate : samplerate of the signal
            min_frequency : lowest frequency where the chirplet is applied
            max_frequency : highest frequency where the chirplet is applied
            duration : duration of the chirp
            polynome_degree : degree of the polynome to generate the coefficients of the chirplet
        """
        self._min_frequency = min_frequency

        self._max_frequency = max_frequency

        self._duration = sigma/10

        self._samplerate = samplerate

        self._polynome_degree = polynome_degree

        self._filter_coefficients = self.calcul_coefficients()


    def calcul_coefficients(self):
        """calculate coefficients for the chirplets
        Returns :
            apodization coeeficients
        """
        num_coeffs = linspace(0, self._duration, int(self._samplerate*self._duration))

        if self._polynome_degree:
            temp = (self._max_frequency-self._min_frequency)
            temp /= ((self._polynome_degree+1)*self._duration**self._polynome_degree)*num_coeffs**self._polynome_degree+self._min_frequency
            wave = cos(2*pi*num_coeffs*temp)
        else:
            temp = (self._min_frequency*(self._max_frequency/self._min_frequency)**(num_coeffs/self._duration)-self._min_frequency)
            temp *= self._duration/log(self._max_frequency/self._min_frequency)
            wave = cos(2*pi*temp)

        coeffs = wave*hanning(len(num_coeffs))**2

        return coeffs

    def smooth_up(self, input_signal, thresh_window, end_smoothing):
        """generate fast fourier transform from a signal and smooth it
        Params :
            input_signal : audio signal
            thresh_window : relative to the size of the windows
            end_smoothing : relative to the length of the output signal
        Returns :
            fast Fourier transform of the audio signal applied to a specific domain of frequencies
        """
        windowed_fft = build_fft(input_signal, self._filter_coefficients, thresh_window)
        return fft_smoothing(fabs(windowed_fft), end_smoothing)

def apply_filterbank(input_signal, chirplets, end_smoothing):
    """generate list of signal with chirplets
    Params :
        input_signal : audio signal
        chirplets : the chirplet bank
        end_smoothing : relative to the length of the output signal
    Returns :
        fast Fourier transform of the signal to all the frequency domain
    """
    fast_chirplet_transform = list()

    for chirplet in chirplets:
        chirp_line = chirplet.smooth_up(input_signal, 6, end_smoothing)
        fast_chirplet_transform.append(chirp_line)

    return np.array(fast_chirplet_transform)



def fft_smoothing(input_signal, sigma):
    """smooth the fast transform Fourier
    Params :
        input_signal : audio signal
        sigma : relative to the length of the output signal
    Returns :
        a shorter and smoother signal

    """
    size_signal = input_signal.size

    #shorten the signal
    new_size = int(floor(10.0*size_signal*sigma))
    half_new_size = new_size//2

    fftx = fft(input_signal)

    short_fftx = []
    for ele in fftx[:half_new_size]:
        short_fftx.append(ele)

    for ele in fftx[-half_new_size:]:
        short_fftx.append(ele)

    apodization_coefficients = generate_apodization_coeffs(half_new_size, sigma, size_signal)

    #apply the apodization coefficients
    short_fftx[:half_new_size] *= apodization_coefficients
    short_fftx[half_new_size:] *= flipud(apodization_coefficients)

    realifftxw = ifft(short_fftx).real
    return realifftxw

def generate_apodization_coeffs(num_coeffs, sigma, size):
    """generate apodization coefficients
    Params :
        num_coeffs : number of coefficients
        sigma : relative to the length of the output signal
        size : size of the signal
    Returns :
        apodization coefficients

    """
    apodization_coefficients = arange(num_coeffs)
    apodization_coefficients = apodization_coefficients**2
    apodization_coefficients = apodization_coefficients/(2*(sigma*size)**2)
    apodization_coefficients = exp(-apodization_coefficients)
    return apodization_coefficients

def fft_based(input_signal, filter_coefficients, boundary=0):
    """applied fft if the signal is too short to be splitted in windows
    Params :
        input_signal : the audio signal
        filter_coefficients : coefficients of the chirplet bank
        boundary : manage the bounds of the signal
    Returns :
        audio signal with application of fast Fourier transform
    """
    num_coeffs = filter_coefficients.size
    half_size = num_coeffs//2

    if boundary == 0:#ZERO PADDING
        input_signal = np.lib.pad(input_signal, (half_size, half_size), 'constant', constant_values=0)
        filter_coefficients = np.lib.pad(filter_coefficients, (0, input_signal.size-num_coeffs), 'constant', constant_values=0)
        newx = ifft(fft(input_signal)*fft(filter_coefficients))
        return newx[num_coeffs-1:-1]

    elif boundary == 1:#symmetric
        input_signal = concatenate([flipud(input_signal[:half_size]), input_signal, flipud(input_signal[half_size:])])
        filter_coefficients = np.lib.pad(filter_coefficients, (0, input_signal.size-num_coeffs), 'constant', constant_values=0)
        newx = ifft(fft(input_signal)*fft(filter_coefficients))
        return newx[num_coeffs-1:-1]

    else:#periodic
        return roll(ifft(fft(input_signal)*fft(filter_coefficients, input_signal.size)), -half_size).real


def build_fft(input_signal, filter_coefficients, threshold_windows=6, boundary=0):
    """generate fast transform fourier by windows
    Params :
        input_signal : the audio signal
        filter_coefficients : coefficients of the chirplet bank
        threshold_windows : calcul the size of the windows
        boundary : manage the bounds of the signal
    Returns :
        fast Fourier transform applied by windows to the audio signal

    """
    num_coeffs = filter_coefficients.size
    #print(n,boundary,M)
    half_size = num_coeffs//2
    signal_size = input_signal.size
    #power of 2 to apply fast fourier transform
    windows_size = 2**ceil(log2(num_coeffs*(threshold_windows+1)))
    number_of_windows = floor(signal_size//windows_size)

    if number_of_windows == 0:
        return fft_based(input_signal, filter_coefficients, boundary)

    windowed_fft = empty_like(input_signal)
    #pad with 0 to have a size in a power of 2
    windows_size = int(windows_size)

    zeropadding = np.lib.pad(filter_coefficients, (0, windows_size-num_coeffs), 'constant', constant_values=0)

    h_fft = fft(zeropadding)

    #to browse the whole signal
    current_pos = 0

    #apply fft to a part of the signal. This part has a size which is a power
    #of 2
    if boundary == 0:#ZERO PADDING

        #window is half padded with since it's focused on the first half
        window = input_signal[current_pos:current_pos+windows_size-half_size]
        zeropaddedwindow = np.lib.pad(window, (len(h_fft)-len(window), 0), 'constant', constant_values=0)
        x_fft = fft(zeropaddedwindow)

    elif boundary == 1:#SYMMETRIC
        window = concatenate([flipud(input_signal[:half_size]), input_signal[current_pos:current_pos+windows_size-half_size]])
        x_fft = fft(window)

    else:
        x_fft = fft(input_signal[:windows_size])

    windowed_fft[:windows_size-num_coeffs] = (ifft(x_fft*h_fft)[num_coeffs-1:-1]).real

    current_pos += windows_size-num_coeffs-half_size
    #apply fast fourier transofm to each windows
    while current_pos+windows_size-half_size <= signal_size:

        x_fft = fft(input_signal[current_pos-half_size:current_pos+windows_size-half_size])
        #Suppress the warning, work on the real/imagina
        windowed_fft[current_pos:current_pos+windows_size-num_coeffs] = (ifft(x_fft*h_fft)[num_coeffs-1:-1]).real
        current_pos += windows_size-num_coeffs
    # print(countloop)
    #apply fast fourier transform to the rest of the signal
    if windows_size-(signal_size-current_pos+half_size) < half_size:

        window = input_signal[current_pos-half_size:]
        zeropaddedwindow = np.lib.pad(window, (0, int(windows_size-(signal_size-current_pos+half_size))), 'constant', constant_values=0)
        x_fft = fft(zeropaddedwindow)
        windowed_fft[current_pos:] = roll(ifft(x_fft*h_fft), half_size)[half_size:half_size+windowed_fft.size-current_pos].real
        windowed_fft[-half_size:] = convolve(input_signal[-num_coeffs:], filter_coefficients, 'same')[-half_size:]
    else:

        window = input_signal[current_pos-half_size:]
        zeropaddedwindow = np.lib.pad(window, (0, int(windows_size-(signal_size-current_pos+half_size))), 'constant', constant_values=0)
        x_fft = fft(zeropaddedwindow)
        windowed_fft[current_pos:] = ifft(x_fft*h_fft)[num_coeffs-1:num_coeffs+windowed_fft.size-current_pos-1].real

    return windowed_fft
