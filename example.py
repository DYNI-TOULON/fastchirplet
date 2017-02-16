import chirplet as ch
import time
import sys

# ch.compute("./audio/ID0134.wav",save=True)


def main(path='.'):
	dico = ch.compute_folder(path,save=True,default_sample_rate=16000)


if __name__ == "__main__":
	if(len(sys.argv) > 1):
		main(sys.argv[1])
	else:
		main()
# for key in dico.keys():
# 	print(key)

# print("--- %s seconds ---" % (time.time() - start_time))