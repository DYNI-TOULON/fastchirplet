import numpy as np
import csv
from pylab import *
import joblib
import os
from scipy.io.wavfile import read
import librosa

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

def plotchirplet2(tab1,samplerate,name,root,audiopath):
	figure, axarr = plt.subplots(3, sharex=False)

	tabfinal=list(reversed(tab1))
	[spectrum, freqs, times] = compute_spectrogram(tab1, samplerate)
	index_frequency = np.argmax(freqs)
	mxf = freqs[index_frequency]
	print(mxf)
	data,sr = librosa.load(audiopath,sr=None)
	print(len(data),sr)
	[spectrum, freqs, times] = compute_spectrogram(data, sr)

	axarr[0].matshow(tabfinal,
			                          origin='lower',
			                          extent=(0, times[-1], freqs[0], mxf),
			                          aspect='auto')

	axarr[0].axes.xaxis.set_ticks_position('bottom')
	axarr[0].set_ylabel("Frequency in Hz")
	axarr[0].xaxis.grid(which='major', color='Black',
	                             linestyle='-', linewidth=0.25)
	axarr[0].yaxis.grid(which='major', color='Black',
	                             linestyle='-', linewidth=0.25)

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


	# axarr[2].matshow(data,
	#                           origin='lower',
	#                           extent=(times[0], times[-1],freqs[0], max_frequency),
	#                           aspect='auto')
	axarr[2].axes.xaxis.set_ticks_position('bottom')
	axarr[2].set_ylabel("Intensity")
	axarr[2].xaxis.grid(which='major', color='Black',
	                             linestyle='-', linewidth=0.25)

	figure.tight_layout()

	if not os.path.exists(root):
		os.makedirs(root)
	figure.savefig(root+"/"+name+'.png')
	close('all')

# def save_chirp(path_file,chirps):
# 	if not os.path.exists("pkl"):
# 			os.makedirs("pkl")
# 	joblib.dump(chirps,'pkl/'+os.path.basename(path_file).split('.')[0]+'.pkl')
def main(path='.'):

	pklfiles = list()
	counter = 0

	for root, dirs, files in os.walk(path):

		for file in files:

			if file.endswith("jl"):
				pklfiles.append(os.path.join(root, file))
	num_files = len(pklfiles)
	for file in pklfiles:
		print(counter,'/',num_files,os.path.basename(file))
		counter += 1
		data = joblib.load(file)
		#pathfile = os.path.join(file)
		#save_chirp(file,output_up)
		audiopath = '/home/vtassan/git/generate_chirp/audio/corvus_corone.wav'
		print(audiopath)
		plotchirplet2(data,16000,os.path.basename(file).split('.')[0],path+'/spectro/',audiopath)


if __name__ == "__main__":
	if(len(sys.argv) > 1):
		main(sys.argv[1])
	else:
		main()

# csv = np.genfromtxt ('csv/'+name+'.csv', delimiter=",")

# csvN = (csv-mean)/std

# csvUN = csv*std+mean

# pkl = joblib.load("pkl/"+name+".pkl")

# subplot(311)
# imshow(pkl,aspect='auto')
# title('Increasing Chirps')

# subplot(311)
# imshow(csvNaspect='auto')
# title('Increasing Chirps')


# subplot(312)
# imshow(csv,aspect='auto')
# title('Increasing Chirps')

# subplot(313)
# imshow(csvUN,aspect='auto')
# title('Increasing Chirps')


# savefig(name+'.pdf')

# close('all')