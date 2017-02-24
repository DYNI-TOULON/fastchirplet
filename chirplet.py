import librosa
import os
import numpy as np
from pylab import *
import sys
from numpy.lib import pad
import joblib
from scipy.io.wavfile import read

class Chirplet:

	"""smallest time bin among the chirplet"""

	global _time_bins

	def __init__(self,samplerate,F0,F1,sigma,polynome_degree):

		"""lowest frequency where the chirplet is applied"""
		self.min_frequency  = F0

		"""highest frequency where the chirplet is applied"""
		self.max_frequency = F1

		"""duration of the chirp"""
		self.duration = sigma/10

		"""samplerate of the signal"""
		self.samplerate = samplerate

		"""degree of the polynome to generate the coefficients of the chirplet"""
		self.polynome_degree = polynome_degree

		"""coefficients applied to the signal"""
		self.filter_coefficients = self.calcul_coefficients()



	def _get_time_bins():
		return Chirplet._time_bins

	def _set_time_bins(value):
		Chirplet._time_bins = value


	def calcul_coefficients(self):
		"""calculate coefficients for the chirplets"""
		t = linspace(0,self.duration,int(self.samplerate*self.duration))

		if(self.polynome_degree):
			w=cos(2*pi*((self.max_frequency-self.min_frequency)/((self.polynome_degree+1)*self.duration**self.polynome_degree)*t**self.polynome_degree+self.min_frequency)*t)
		else:
			w=cos(2*pi*((self.min_frequency*(self.max_frequency/self.min_frequency)**(t/self.duration)-self.min_frequency)*self.duration/log(self.max_frequency/self.min_frequency)))
		
		coeffs = w*hanning(len(t))**2

		return coeffs

	def smooth_up(self,input_signal,sigma,end_smoothing):
	#generate fast fourier transform from a signal and smooth it

		new_up = build_fft(input_signal,self.filter_coefficients,sigma)
		return fft_smoothing(fabs(new_up),end_smoothing)


def compute(input_signal,
	save=False,
	duration_last_chirplet=1,
	num_octaves=5,
	num_chirps_by_octave=10,
	polynome_degree=0,
	end_smoothing=0.001,
	default_sample_rate=None,
	save_path=None):
	"""main function. Fast Chirplet Transform from a signal"""

	Chirplet._set_time_bins(end_smoothing*10)

	real_data, samplerate = librosa.load(input_signal,sr=default_sample_rate)

	#samplerate, data = read(input_signal)	

	size_data = len(real_data)

	nearest_power_2 = 2**(size_data-1).bit_length()

	while nearest_power_2 <= samplerate*duration_last_chirplet:
		nearest_power_2*=2

	data = np.lib.pad(real_data,(0,nearest_power_2-size_data),'constant',constant_values=0)
	# data = real_data

	chirplets = init_chirplet_filter_bank(samplerate,duration_last_chirplet,num_octaves,num_chirps_by_octave,polynome_degree)

	chirps = apply_filterbank(data,chirplets,end_smoothing)

	chirps = resize_chirps(size_data,nearest_power_2,chirps)

	if save:
		save_chirp(input_signal,chirps,save_path)

	return chirps


def save_chirp(path_file,chirps,save_path):
	"""save your chirp in the chosen directories"""
	if save_path:
		if not os.path.exists(save_path):
			os.makedirs(save_path)   
		joblib.dump(chirps,save_path+'/'+filename.split('.')[0]+'.jl')
	else:
		joblib.dump(chirps,path_file.split('.')[0]+'.jl')

def compute_folder(path_folder,
	save=True,
	duration_last_chirplet=1,
	num_octaves=5,
	num_chirps_by_octave=10,
	polynome_degree=3,
	end_smoothing=0.001,
	default_sample_rate=None,
	save_path=None,
	display_progress=False):

	"""apply the compute function to the audio files in the given path"""

	chirp_dict = dict()
	audiofiles = []
	for root, dirs, files in os.walk(path_folder):

		for file in files:

			if file.endswith(".wav"):
				audiofiles.append(os.path.join(root, file))

	lenwavefile = len(audiofiles)
	if display_progress:
		counter = 1
	for file in audiofiles:

		data = compute(file,
			save=save,
			duration_last_chirplet=duration_last_chirplet,
			num_octaves=num_octaves,
			num_chirps_by_octave=num_chirps_by_octave,
			polynome_degree=polynome_degree,
			end_smoothing=end_smoothing,
			default_sample_rate=default_sample_rate,
			save_path=save_path)

		chirp_dict[file] = data
		if display_progress:
			print(file,':',counter,'/',lenwavefile)
			counter += 1

	return chirp_dict


def resize_chirps(size_data,nearest_power_2,chirps):
	size_chirps = len(chirps)
	ratio = size_data/nearest_power_2
	size = int(ratio*len(chirps[0]))

	tabfinal = np.zeros((size_chirps,size))
	for i in range(0,size_chirps):
		tabfinal[i]=chirps[i][0:size]
	return tabfinal

def init_chirplet_filter_bank(samplerate,duration_last_chirplet,num_octaves,num_chirps_by_octave,polynome_degree):
	"""generate all the chirplets from a given sample rate"""

	lambdas            = 2.0**(1+arange(num_octaves*num_chirps_by_octave)/float(num_chirps_by_octave))
	#Low frequencies for a signal
	start_frequencies  = (samplerate /lambdas)/2.0
	#high frequencies for a signal
	end_frequencies    = samplerate /lambdas
	durations          = 2.0*duration_last_chirplet/flipud(lambdas)
	chirplets=list()
	for f0,f1,duration in zip(start_frequencies,end_frequencies,durations):
		chirplets.append(Chirplet(samplerate,f0,f1,duration,polynome_degree))
	return chirplets

def apply_filterbank(input_signal,chirplets,end_smoothing):
	"""generate list of signal with chirplets"""
	result=list()
	for chirplet in chirplets:
		result.append(chirplet.smooth_up(input_signal,6,end_smoothing))
	return array(result)



def fft_smoothing(input_signal,sigma):
	"""smooth the fast transform fourier"""
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

	apodization_coefficients = generate_apodization_coefficients(half_new_size,sigma,size_signal)
	#apply the apodization coefficients
	short_fftx[:half_new_size] *= apodization_coefficients
	#apply the apodization coefficients in a reverse list
	short_fftx[half_new_size:] *= flipud(apodization_coefficients)
	realifftxw = real(ifft(short_fftx))
	return realifftxw
	
def generate_apodization_coefficients(num_coeffs,sigma,size):
	"""generate apodization coefficients"""
	apodization_coefficients = arange(num_coeffs)
	apodization_coefficients = apodization_coefficients**2
	apodization_coefficients = apodization_coefficients/(2*(sigma*size)**2)
	apodization_coefficients = exp(-apodization_coefficients)
	return apodization_coefficients

def fft_based(input_signal,h,boundary=0):
	M=h.size
	half_size = M//2
	if(boundary==0):#ZERO PADDING
		input_signal=pad(input_signal,(half_size,half_size),'constant',
			constant_values=0)
		h=pad(h,(0,input_signal.size-M),'constant',constant_values=0)
		newx=ifft(fft(input_signal)*fft(h))
		return newx[M-1:-1]
	elif(boundary==1):#symmetric
		input_signal=concatenate([flipud(input_signal[:half_size]),input_signal,flipud(input_signal[half_size:])])
		h=pad(h,(0,x.size-M),'constant',constant_values=0)
		newx=ifft(fft(input_signal)*fft(h))
		return newx[M-1:-1]
	else:#periodic
		return real(roll(ifft(fft(input_signal)*fft(h,input_signal.size)),-half_size))


def build_fft(input_signal,filter_coefficients,n=2,boundary=0):
	"""generate fast transform fourier by windows"""
	M=filter_coefficients.size
	#print(n,boundary,M)
	half_size = M//2
	signal_size = input_signal.size
	#power of 2 to apply fast fourier transform
	windows_size=2**ceil(log2(M*(n+1)))
	number_of_windows=floor(signal_size//windows_size)
	if(number_of_windows==0):
		return fft_based(input_signal,filter_coefficients,boundary)

	output=empty_like(input_signal)
	#pad with 0 to have a size in a power of 2
	windows_size = int(windows_size)
	
	zeropadding = pad(filter_coefficients,(0,windows_size-M),'constant',constant_values=0)

	h_fft=fft(zeropadding)

	#to browse the whole signal
	current_pos=0

	#apply fft to a part of the signal. This part has a size which is a power 
	#of 2
	if(boundary==0):#ZERO PADDING
		#window is half padded with since it's focused on the first half
		window = input_signal[current_pos:current_pos+windows_size-half_size]
		zeropaddedwindow = pad(window,(len(h_fft)-len(window),0),'constant',constant_values=0)
		x_fft=fft(zeropaddedwindow)
	elif(boundary==1):#SYMMETRIC
		window = concatenate([flipud(input_signal[:half_size]),input_signal[current_pos:current_pos+windows_size-half_size]])
		x_fft=fft(window)
	else:
		x_fft=fft(input_signal[:windows_size])

	output[:windows_size-M]=(ifft(x_fft*h_fft)[M-1:-1]).real

	current_pos+=windows_size-M-half_size
	countloop = 0
	#apply fast fourier transofm to each windows 
	while(current_pos+windows_size-half_size<=signal_size):
		x_fft=fft(input_signal[current_pos-half_size:current_pos+windows_size-half_size])

		#Suppress the warning, work on the real/imagina
		output[current_pos:current_pos+windows_size-M]=(ifft(x_fft*h_fft)[M-1:-1]).real
		countloop+=1
		current_pos+=windows_size-M
	# print(countloop)
	#apply fast fourier transform to the rest of the signal
	if(windows_size-(signal_size-current_pos+half_size)<half_size):
		window = input_signal[current_pos-half_size:]
		zeropaddedwindow = pad(window,(0,int(windows_size-(signal_size-current_pos+half_size))),'constant',constant_values=0)
		x_fft=fft(zeropaddedwindow)
		output[current_pos:]=real(roll(ifft(x_fft*h_fft),half_size)[half_size:half_size+output.size-current_pos])
		output[-half_size:]=convolve(input_signal[-M:],filter_coefficients,'same')[-half_size:]
	else:
		window = input_signal[current_pos-half_size:]
		zeropaddedwindow = pad(window,(0,int(windows_size-(signal_size-current_pos+half_size))),'constant',constant_values=0)
		x_fft=fft(zeropaddedwindow)
		output[current_pos:]=real(ifft(x_fft*h_fft)[M-1:M+output.size-current_pos-1])
	return output


