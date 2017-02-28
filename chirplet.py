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
        _duration_longest_chirplet :
        _num_octaves :
        _num_chirps_by_octave :
        _polynome_degree :
        _end_smoothing :
        _samplerate :

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
        """generate all the chirplets from a given sample rate

        Returns: The bank of chirplets
        """
        num_chirps = self._num_octaves*self._num_chirps_by_octave
        lambdas = 2.0**(1+arange(num_chirps)/float(self._num_chirps_by_octave))
        #Low frequencies for a signal
        start_frequencies = (self._samplerate /lambdas)/2.0
        #high frequencies for a signal
        end_frequencies = self._samplerate /lambdas
        durations = 2.0*self._duration_longest_chirplet/flipud(lambdas)
        chirplets = list()
        for f0, f1, duration in zip(start_frequencies, end_frequencies, durations):
            chirplets.append(Chirplet(self._samplerate, f0, f1, duration, self._polynome_degree))
        return chirplets

    @property
    def time_bin_duration(self):
        """
        Return: the time bin duration

        """
        return self._end_smoothing*10

    def compute(self, input_signal):
        """compute the FCT on the given signal
        Args:
            input_signal : Array of an audio signal

        Returns:
            The Fast Chirplet Transform of the given signal

        """
        size_data = len(input_signal)

        nearest_power_2 = 2**(size_data-1).bit_length()

        while nearest_power_2 <= self._samplerate*self._duration_longest_chirplet:
            nearest_power_2 *= 2

        data = np.lib.pad(input_signal, (0, nearest_power_2-size_data), 'constant', constant_values=0)

        chirp_transform = apply_filterbank(data, self._chirps, self._end_smoothing)

        chirp_transform = resize_chirps(size_data, nearest_power_2, chirp_transform)

        return chirp_transform


def resize_chirps(size_data, nearest_power_2, chirps):
    """Resize the matrix of chirps to the length of the signal
    Args:
        size_data :
        nearest_power_2 :
        chirps :
    Returns :
        Chirps to the right width
    """
    size_chirps = len(chirps)
    ratio = size_data/nearest_power_2
    size = int(ratio*len(chirps[0]))

    tabfinal = np.zeros((size_chirps, size))
    for i in range(0, size_chirps):
        tabfinal[i] = chirps[i][0:size]
    return tabfinal


class Chirplet:
    """chirplet class
    Attributes:
        _min_frequency :
        _max_frequency :
        _duration :
        _samplerate :
        _ploynome_degree :
        _filter_coefficients :
    """
    def __init__(self, samplerate, min_frequency, max_frequency, sigma, polynome_degree):

        """
        Args :
            _samplerate : samplerate of the signal
            _min_frequency : lowest frequency where the chirplet is applied
            _max_frequency : highest frequency where the chirplet is applied
            _duration : duration of the chirp
            _polynome_degree : degree of the polynome to generate the coefficients of the chirplet
            _filter_coefficients : coefficients applied to the signal
        """
        self._min_frequency = min_frequency

        self._max_frequency = max_frequency

        self._duration = sigma/10

        self._samplerate = samplerate

        self._polynome_degree = polynome_degree

        self._filter_coefficients = self.calcul_coefficients()


    def calcul_coefficients(self):
        """calculate coefficients for the chirplets
        Returns:
            apodization coeeficients
        """
        num_coeffs = linspace(0, self._duration, int(self._samplerate*self._duration))

        if self._polynome_degree:
            wave = cos(2*pi*((self._max_frequency-self._min_frequency)/((self._polynome_degree+1)*self._duration**self._polynome_degree)*num_coeffs**self._polynome_degree+self._min_frequency)*num_coeffs)
        else:
            wave = cos(2*pi*((self._min_frequency*(self._max_frequency/self._min_frequency)**(num_coeffs/self._duration)-self._min_frequency)*self._duration/log(self._max_frequency/self._min_frequency)))

        coeffs = wave*hanning(len(num_coeffs))**2

        return coeffs

    def smooth_up(self, input_signal, sigma, end_smoothing):
        #generate fast fourier transform from a signal and smooth it
        """
        Params:
            input_signal :
            sigma :
            end_smoothing :
        Returns:
        """
        new_up = build_fft(input_signal, self._filter_coefficients, sigma)
        return fft_smoothing(fabs(new_up), end_smoothing)

def apply_filterbank(input_signal, chirplets, end_smoothing):
    """generate list of signal with chirplets
    Params :
        input_signal :
        chirplets :
        end_smoothing :
    Returns :
    """
    fast_chirplet_transform = list()

    for chirplet in chirplets:
        chirp_line = chirplet.smooth_up(input_signal, 6, end_smoothing)
        fast_chirplet_transform.append(chirp_line)

    return np.array(fast_chirplet_transform)



def fft_smoothing(input_signal, sigma):
    """smooth the fast transform fourier
    Params:
        input_signal :
        sigma :
    Returns:

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

    apodization_coefficients = generate_apodization_coefficients(half_new_size, sigma, size_signal)

    #apply the apodization coefficients
    short_fftx[:half_new_size] *= apodization_coefficients
    short_fftx[half_new_size:] *= flipud(apodization_coefficients)

    realifftxw = ifft(short_fftx).real
    return realifftxw

def generate_apodization_coefficients(num_coeffs, sigma, size):
    """generate apodization coefficients
    Params :
        num_coeffs :
        sigma :
        size :
    Returns :

    """
    apodization_coefficients = arange(num_coeffs)
    apodization_coefficients = apodization_coefficients**2
    apodization_coefficients = apodization_coefficients/(2*(sigma*size)**2)
    apodization_coefficients = exp(-apodization_coefficients)
    return apodization_coefficients

def fft_based(input_signal, coeffs, boundary=0):
    """
    Params :
        input_signal :
        coeffs :
        boundary :
    Returns :
    """
    M = coeffs.size
    half_size = M//2

    if boundary == 0:#ZERO PADDING
        input_signal = np.lib.pad(input_signal, (half_size, half_size), 'constant', constant_values=0)
        coeffs = np.lib.pad(coeffs, (0, input_signal.size-M), 'constant', constant_values=0)
        newx = ifft(fft(input_signal)*fft(coeffs))
        return newx[M-1:-1]

    elif boundary == 1:#symmetric
        input_signal = concatenate([flipud(input_signal[:half_size]), input_signal, flipud(input_signal[half_size:])])
        coeffs = np.lib.pad(coeffs, (0, input_signal.size-M), 'constant', constant_values=0)
        newx = ifft(fft(input_signal)*fft(coeffs))
        return newx[M-1:-1]

    else:#periodic
        return roll(ifft(fft(input_signal)*fft(coeffs, input_signal.size)), -half_size).real


def build_fft(input_signal, filter_coefficients, n=2, boundary=0):
    """generate fast transform fourier by windows
    Params :
        input_signal :
        filter_coefficients :
        n :
        boundary :
    Returns :

    """
    M = filter_coefficients.size
    #print(n,boundary,M)
    half_size = M//2
    signal_size = input_signal.size
    #power of 2 to apply fast fourier transform
    windows_size = 2**ceil(log2(M*(n+1)))
    number_of_windows = floor(signal_size//windows_size)
    if number_of_windows == 0:
        return fft_based(input_signal, filter_coefficients, boundary)

    output = empty_like(input_signal)
    #pad with 0 to have a size in a power of 2
    windows_size = int(windows_size)

    zeropadding = np.lib.pad(filter_coefficients, (0, windows_size-M), 'constant', constant_values=0)

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

    output[:windows_size-M] = (ifft(x_fft*h_fft)[M-1:-1]).real

    current_pos += windows_size-M-half_size
    #apply fast fourier transofm to each windows
    while current_pos+windows_size-half_size <= signal_size:

        x_fft = fft(input_signal[current_pos-half_size:current_pos+windows_size-half_size])
        #Suppress the warning, work on the real/imagina
        output[current_pos:current_pos+windows_size-M] = (ifft(x_fft*h_fft)[M-1:-1]).real
        current_pos += windows_size-M
    # print(countloop)
    #apply fast fourier transform to the rest of the signal
    if windows_size-(signal_size-current_pos+half_size) < half_size:

        window = input_signal[current_pos-half_size:]
        zeropaddedwindow = np.lib.pad(window, (0, int(windows_size-(signal_size-current_pos+half_size))), 'constant', constant_values=0)
        x_fft = fft(zeropaddedwindow)
        output[current_pos:] = roll(ifft(x_fft*h_fft), half_size)[half_size:half_size+output.size-current_pos].real
        output[-half_size:] = convolve(input_signal[-M:], filter_coefficients, 'same')[-half_size:]
    else:

        window = input_signal[current_pos-half_size:]
        zeropaddedwindow = np.lib.pad(window, (0, int(windows_size-(signal_size-current_pos+half_size))), 'constant', constant_values=0)
        x_fft = fft(zeropaddedwindow)
        output[current_pos:] = ifft(x_fft*h_fft)[M-1:M+output.size-current_pos-1].real

    return output
