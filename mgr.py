import librosa
import librosa.display
import matplotlib.pyplot as plt

import os
from pathlib import Path
from glob import glob

# Directory where spectrogram image files will be saved
save_dir = './spectrograms'

# Dataset directory, from which audio files will be loaded
data_dir = './dataset'

# Grab genres inside data_dir
genre_dirs = glob(data_dir + '/*')
genres = []
for genre_dir in genre_dirs:
	genre = os.path.basename(genre_dir)
	genres.append( genre )

	# Create destination genre folder for spectrograms
	try:
		os.mkdir(save_dir + '/' + genre)
	except:
		print('Error creating directory for ' + genre + '. It might already exist.')

# Create spectrograms for each .wav file while keeping directory structure
for genre in genres:
	tracks = glob(data_dir + '/' + genre + '/*.wav')
	for track in tracks:
		x_full, sr = librosa.load(track) # Load audio file
		x = x_full[:int(x_full.shape[0]/3)] # Only keep first 1/3 (10s) of audio file

		X = librosa.stft(x)
		Xdb = librosa.amplitude_to_db(abs(X))
		plt.figure(figsize=(14, 5))
		librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
		plt.axis('off')
		plt.savefig(save_dir + '/' + genre + '/' + Path(track).stem + '.png')
		plt.close()
