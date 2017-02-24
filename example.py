import chirplet as ch
import time
import sys

# ch.compute("./audio/ID0134.wav",save=True)


def main(path='.',o,c):
	dico = ch.compute_folder(path,save=True,num_octaves=o,num_chirps_by_octave=c,default_sample_rate=16000)
	print(dico[0].shape)

if __name__ == "__main__":
	if(len(sys.argv) > 1):
		main(sys.argv[1],sys.argv[2],sys.argv[3])
	else:
		main()
# for key in dico.keys():
# 	print(key)

# print("--- %s seconds ---" % (time.time() - start_time))