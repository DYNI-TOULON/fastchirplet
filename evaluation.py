import numpy as np
import chirplet as ch
import matplotlib.pyplot as plt
import librosa
import sys
import joblib

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

def plotchirplet(tab1,audiopath,s):
	figure, axarr = plt.subplots(4, sharex=False)
	plt.suptitle(audiopath[8:-4])
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

	axarr[0].set_title('chirplet (time), s='+s)

	axarr[3].matshow(tabfinal,
			                          origin='lower',
			                         
			                          aspect='auto')
	axarr[3].axes.xaxis.set_ticks_position('bottom')
	axarr[3].set_ylabel("Frequency in Hz")
	axarr[3].xaxis.grid(which='major', color='Black',
	                             linestyle='-', linewidth=0.25)
	axarr[3].yaxis.grid(which='major', color='Black',
	                             linestyle='-', linewidth=0.25)

	axarr[3].set_title('chirplet (sample)')

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

	# axarr[2].plot(data)
	time = np.linspace(0, len(data) / sr, num=len(data))
	axarr[2].set_xlim([0, time[-1]])
	axarr[2].plot(time, data)
	
	axarr[2].set_ylabel("Amplitude")


	# axarr[2].matshow(data,
	#                           origin='lower',
	#                           extent=(times[0], times[-1],freqs[0], max_frequency),
	#                           aspect='auto')
	axarr[2].axes.xaxis.set_ticks_position('bottom')
	axarr[2].set_ylabel("Intensity")
	axarr[2].xaxis.grid(which='major', color='Black',
	                             linestyle='-', linewidth=0.25)

	# axarr[2].set_title('audiotrack')


	figure.tight_layout()

	figure.savefig(audiopath[:-3]+'png')
	# close('all')


def main(pathfile,s):
	chirps = ch.compute(pathfile,duration_last_chirplet=float(s))
	print(len(chirps),len(chirps[0]))
	joblib.dump(chirps,pathfile[:-3]+'jl')
	plotchirplet(chirps,pathfile,s)






if __name__ == '__main__':
	main(sys.argv[1],sys.argv[2])