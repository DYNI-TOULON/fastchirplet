import numpy as np
import chirplet as ch
import matplotlib.pyplot as plt
import librosa
import sys
import joblib
import os
import time


def compute_spectrogram(signal, sample_rate):
    """
        compute spectrogram from signal
        :param signal:
        :return: matrice representing the spectrum, frequencies corresponding and times
    """
    [spectrum, freqs, times] = plt.mlab.specgram(signal, NFFT=1024, Fs=sample_rate,
                                                 noverlap=512, window=np.hamming(1024))
    spectrum = 10. * np.log10(spectrum)

    return [spectrum, freqs, times]

def plotchirplet(tab1,audiopath):
	figure, axarr = plt.subplots(3, sharex=False)

	# title is not great => overlap 
	# plt.suptitle(audiopath[8:-4])

	data,sr = librosa.load(audiopath,sr=None)
	tabfinal=list(reversed(tab1))

	[spectrum, freqs, times] = compute_spectrogram(data, sr)

	index_frequency = np.argmax(freqs)
	mxf = freqs[index_frequency]



	axarr[0].matshow(tabfinal,
			                          origin='lower',
			                          extent=(0, times[-1],freqs[0], mxf),
			                          aspect='auto')
	axarr[0].axes.xaxis.set_ticks_position('bottom')
	axarr[0].set_ylabel("Frequency in Hz")
	axarr[0].xaxis.grid(which='major', color='Black',
	                             linestyle='-', linewidth=0.25)
	axarr[0].yaxis.grid(which='major', color='Black',
	                             linestyle='-', linewidth=0.25)
	axarr[0].set_yscale('log')

	axarr[0].set_title('chirplet')

	index_frequency = np.argmax(freqs)
	max_frequency = freqs[index_frequency]

	axarr[1].matshow(spectrum[0:index_frequency+1, :],
	                          origin='lower',
	                          extent=(times[0], times[-1],freqs[0], max_frequency),
	                          aspect='auto')
	axarr[1].axes.xaxis.set_ticks_position('bottom')
	axarr[1].set_ylabel("Frequency in Hz")
	axarr[1].xaxis.grid(which='major', color='Black',
	                             linestyle='-', linewidth=0.25)
	axarr[1].yaxis.grid(which='major', color='Black',
	                             linestyle='-', linewidth=0.25)

	axarr[1].set_title('spectrogram')


	time = np.linspace(0, len(data) / sr, num=len(data))
	axarr[2].set_xlim([0, time[-1]])
	axarr[2].plot(time, data)
	
	axarr[2].set_ylabel("Amplitude")

	axarr[2].axes.xaxis.set_ticks_position('bottom')
	axarr[2].set_ylabel("Intensity")
	axarr[2].xaxis.grid(which='major', color='Black',
	                             linestyle='-', linewidth=0.25)

	axarr[2].set_title('audiotrack')


	figure.tight_layout()

	figure.savefig(audiopath[:-3]+'png')
	plt.close('all')


def main(pathfile):	
	chirplet = []
	for root, dirs, files in os.walk(pathfile):

		for file in files:

			if file.endswith(".wav"):
				chirplet.append(os.path.join(root, file))

	for file in chirplet:
		start_time = time.time()
		chirps = ch.compute(file)
		joblib.dump(chirps,file[:-3]+'jl')
		plotchirplet(chirps,file)
		print("--- %s seconds ---" % (time.time() - start_time))






if __name__ == '__main__':
	start_time = time.time()
	main(sys.argv[1])
	print("--- %s seconds ---" % (time.time() - start_time))