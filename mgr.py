import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import (Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, 
                          Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D)
from keras.models import Model, load_model
from keras.utils import layer_utils
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from keras.initializers import glorot_uniform

import os
import gc
import shutil
import argparse
import random
from pathlib import Path
from glob import glob

parser = argparse.ArgumentParser(description='Mel Spectrogram / CNN Music Genre Recognition')

parser.add_argument('-s', '--spectrograms', action='store_true', dest='create',
										help='Create spectrograms')
parser.add_argument('-t', '--train', action='store_true', dest='train',
										help='Train the model')
parser.add_argument('-c', '--classify', nargs=1, dest='song',
										help='Get the genre of a given song')

args = parser.parse_args()
print(args)

# Directory where spectrogram image files will be saved
save_dir = './spectrograms5s'

# Dataset directory, from which audio files will be loaded
data_dir = './dataset'

# Defines how many parts dataset files will be split into
split = 3 # 6 parts - 30s / 6s - 5s per file
seconds = 5

class StateKeeper:
	def __init__(self):
		self.genres = []
		self.setGenres()

	def setGenres(self):
		genre_dirs = glob(data_dir + '/*')
		for genre_dir in genre_dirs:
			genre = os.path.basename(genre_dir)
			self.genres.append( genre )

	def getGenres(self):
		return self.genres

	def getTrackFiles(self, genre='*'):
		return glob(data_dir + '/' + genre + '/*.wav')

	def getSpectFiles(self, genre='*'):
		return glob(save_dir + '/' + genre + '/*.png')

	def createTrainTest(self):
		if(os.path.exists('./train')):
			shutil.rmtree('./train')
		if(os.path.exists('./test')):
			shutil.rmtree('./test')

		for genre in self.genres:
			files = self.getSpectFiles(genre)

			random.shuffle(files)
			test_files = files[:(10*split)]
			train_files = files[(10*split):]

			if not os.path.exists('./test/' + genre):
				os.makedirs('./test/' + genre, exist_ok=True)
			if not os.path.exists('./train/' + genre):
				os.makedirs('./train/' + genre, exist_ok=True)

			for file in test_files:
				dest_path = './test/' + genre + '/' + os.path.basename(file)
				shutil.copy(file, dest_path)
				print(dest_path)
			for file in train_files:
				dest_path = './train/' + genre + '/' + os.path.basename(file)
				shutil.copy(file, dest_path)
				print(dest_path)



def create_spectrograms():
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
			print('Directory ' + genre + ' was not created. It might already exist.')

	# Create spectrograms for each .wav file while keeping directory structure
	for genre in genres:
		tracks = glob(data_dir + '/' + genre + '/*.wav')
		for track in tracks:
			x_full, sr = librosa.load(track) # Load audio file
			x_split = np.array_split(x_full, split) # Split array into parts
			counter = 0
			for x in x_split:
				print(track)
				X = librosa.stft(x)
				Xdb = librosa.amplitude_to_db(abs(X))
				fig = plt.Figure()
				librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
				plt.axis('off')
				plt.savefig(save_dir + '/' + genre + '/' + Path(track).stem + '.' + f'{counter}' + '.png', bbox_inches='tight', pad_inches = 0)
				plt.cla()
				plt.clf()
				plt.close('all')
				plt.ioff()
				counter += 1
			gc.collect()


state = StateKeeper()

# Create model
def GenreModel(input_shape = (288,432,3),classes=10):
	X_input = Input(input_shape)

	# FEATURE EXTRACTION
	X = Conv2D(8,kernel_size=(3,3),strides=(1,1))(X_input)
	X = BatchNormalization(axis=3)(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((2,2))(X)

	X = Conv2D(16,kernel_size=(3,3),strides = (1,1))(X)
	X = BatchNormalization(axis=3)(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((2,2))(X)

	X = Conv2D(32,kernel_size=(3,3),strides = (1,1))(X)
	X = BatchNormalization(axis=3)(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((2,2))(X)

	X = Conv2D(64,kernel_size=(3,3),strides=(1,1))(X)
	X = BatchNormalization(axis=-1)(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((2,2))(X)

	X = Conv2D(128,kernel_size=(3,3),strides=(1,1))(X)
	X = BatchNormalization(axis=-1)(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((2,2))(X)

	# FLATTEN
	X = Flatten()(X)

	X = Dropout(rate=0.3)(X)

	X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

	model = Model(inputs=X_input,outputs=X,name='GenreModel')

	return model

def get_f1(y_true, y_pred): #taken from old keras source code
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	recall = true_positives / (possible_positives + K.epsilon())
	f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
	return f1_val

if args.create:
	create_spectrograms()

if args.train:
	classes = state.getGenres()
	state.createTrainTest()
	
	trdata = ImageDataGenerator(rescale=1./255)
	traindata = trdata.flow_from_directory(directory='./train', target_size=(288,432))
	tsdata = ImageDataGenerator(rescale=1./255)
	testdata = tsdata.flow_from_directory(directory='./test', target_size=(288,432))
	model = GenreModel(input_shape=(288,432,3),classes=10)
	opt = Adam(learning_rate=0.0005)
	model.compile(optimizer = opt,loss='categorical_crossentropy',metrics=['accuracy',get_f1]) 

	model.fit(traindata,epochs=20,validation_data=testdata)
	model.save('model.h5')

if args.song:
	model = load_model('model.h5', custom_objects = {'get_f1':get_f1})

	song_array, sr = librosa.load(args.song[0])
	song_timelen = song_array.shape[0] / sr # length of song in seconds
	split_count = song_timelen / 5 # how many parts to split into to get 5s parts
	song_split = np.array_split(song_array, split_count) # Split array into parts

	if not os.path.exists('./predict/'):
		os.makedirs('./predict/', exist_ok = True)

	result = {}
	for genre in state.getGenres():
		result[genre] = []

	counter = 0
	for x in song_split:
		X = librosa.stft(x)
		Xdb = librosa.amplitude_to_db(abs(X))
		fig = plt.Figure()
		librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
		plt.axis('off')
		plt.savefig('./predict/predict.' + f'{counter}' + '.png', bbox_inches='tight', pad_inches = 0)
		plt.cla()

		img = image.load_img('./predict/predict.'+ f'{counter}' +'.png', target_size = (288,432))
		img = image.img_to_array(img).astype('float32')/255
		img = np.expand_dims(img, axis = 0)

		predictions = model.predict(img)

		c = 0
		for p in predictions[0]:
			result[(state.getGenres()[c])].append(p)
			c += 1

		counter += 1

	for genre in result:
		result[genre] = np.average(result[genre])
		if(result[genre] > 0.0001):
			print(genre + ': ' + f'{result[genre]*100}' + '%')
