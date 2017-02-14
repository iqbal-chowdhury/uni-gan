'''Train a recurrent convolutional network on the IMDB sentiment
classification task.
Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function
import numpy as np
import sys

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers import Embedding, Dropout, Flatten
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
import pickle, random
import os
from pycocotools.coco import COCO
import sklearn
from sklearn.metrics import jaccard_similarity_score
import progressbar

# Embedding
max_features = 25000
maxlen = 35
embedding_size = 128

# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 256

# Training
batch_size = 30
nb_epoch = 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''
flowers_dataDir = 'Data/flowers'
dataRoot = 'Data'
dataDir = 'Data/mscoco'
tr_dataType = 'train2014'
tr_annFile = '%s/annotations_inst/instances_%s.json' % (dataDir, tr_dataType)

tr_image_list = []
tr_n_imgs = 0
tr_coco_captions = {}
tr_img_classes = {}

val_dataType = 'val2014'
val_annFile = '%s/annotations_inst/instances_%s.json' % (dataDir, val_dataType)

val_image_list = []
val_n_imgs = 0
val_coco_captions = {}
val_img_classes = {}


def load_data():
    tr_coco_captions = pickle.load(
        open(os.path.join(dataDir, 'train', 'coco_tv.pkl'), "rb"))
    # h1 = h5py.File(join(data_dir, 'flower_tc.hdf5'))
    tr_img_classes = pickle.load(
        open(os.path.join(dataDir, 'train', 'coco_tc.pkl'), "rb"))

    val_coco_captions = pickle.load(
        open(os.path.join(dataDir, 'val', 'coco_tv.pkl'), "rb"))
    # h1 = h5py.File(join(data_dir, 'flower_tc.hdf5'))
    val_img_classes = pickle.load(
        open(os.path.join(dataDir, 'val', 'coco_tc.pkl'), "rb"))

    n_classes = 80
    max_caps_len = 25

    tr_coco = COCO(tr_annFile)
    val_coco = COCO(val_annFile)
    tr_image_list = tr_coco.getImgIds()
    tr_n_imgs = len(tr_image_list)

    val_image_list = val_coco.getImgIds()
    val_n_imgs = len(val_image_list)

    return tr_coco_captions, tr_img_classes, val_coco_captions, \
           val_img_classes, \
           n_classes, max_caps_len, tr_image_list, tr_n_imgs, val_image_list, \
           val_n_imgs

def load_flowers_data():
    tr_coco_captions = pickle.load(
        open(os.path.join(flowers_dataDir, 'flower_tv.pkl'), "rb"))
    # h1 = h5py.File(join(data_dir, 'flower_tc.hdf5'))
    tr_img_classes = pickle.load(
        open(os.path.join(flowers_dataDir, 'flower_tc.pkl'), "rb"))

    n_classes = 102
    max_caps_len = 35

    tr_image_list = [key for key in tr_coco_captions]
    random.shuffle(tr_image_list)
    tr_n_imgs = len(tr_image_list)
    split_id = int(tr_n_imgs * 0.9)
    #print(split_id)
    val_image_list = tr_image_list[split_id : -1]
    val_n_imgs = len(val_image_list)
    #print(val_n_imgs)
    #tr_image_list = tr_image_list[0 : split_id]
    #tr_n_imgs = len(tr_image_list)
    #print(tr_n_imgs)
        

    return tr_coco_captions, tr_img_classes, tr_coco_captions, \
           tr_img_classes, n_classes, max_caps_len, tr_image_list, tr_n_imgs, val_image_list, \
           val_n_imgs
	

def get_batch(dataType='train', batch_size = 64, batch_no=0):
    X = []
    y = np.zeros((batch_size, 102))
    if dataType == 'train':
        #random.shuffle(tr_image_list)
        #batch_idx = np.random.randint(0, tr_n_imgs, size=batch_size)
        #imgs = np.take(tr_image_list, batch_idx)
        batch_idx = range(batch_no*batch_size, batch_no*batch_size + batch_size)
        imgs = np.take(tr_image_list, batch_idx)
        for i, img in enumerate(imgs):
            X.append(tr_coco_captions[img][np.random.randint(0,4)])
            #print('Maja')
            #print(tr_img_classes[img])
            #print(tr_img_classes[img].shape)
            y[i: ] = tr_img_classes[img]
        return np.array(X), y
    if dataType == 'val':
        random.shuffle(val_image_list)
        batch_idx = np.random.randint(0, val_n_imgs, size=batch_size)
        imgs = np.take(val_image_list, batch_idx)
        for i, img in enumerate(imgs) :
            X.append(val_coco_captions[img][np.random.randint(0,4)])
            y[i :] = val_img_classes[img]
        return np.array(X), y


def build_model():
	print('Build model...')

	model = Sequential()
	model.add(Embedding(max_features, embedding_size, input_length=maxlen))
	model.add(Dropout(0.25))
	model.add(Convolution1D(nb_filter=nb_filter,
		            filter_length=filter_length,
		            border_mode='valid',
		            activation='relu',
		            subsample_length=1))
	model.add(MaxPooling1D(pool_length=pool_length))
	model.add(BatchNormalization())
	#model.add(Dropout(0.25))
	model.add(Convolution1D(nb_filter=128,
		            filter_length=5,
		            border_mode='same',
		            activation='relu',
		            subsample_length=1))
	model.add(MaxPooling1D(pool_length=2))
	model.add(BatchNormalization())
	#model.add(Dropout(0.25))
	model.add(Convolution1D(nb_filter=256,
		            filter_length=3,
		            border_mode='valid',
		            activation='relu',
		            subsample_length=1))
	model.add(MaxPooling1D(pool_length=2))
	model.add(BatchNormalization())
	model.add(LSTM(lstm_output_size))
	#model.add(Flatten())
	#model.add(Dropout(0.5))
	#model.add(Dense(512))
	#model.add(Activation('sigmoid'))
	#model.add(Dropout(0.5))
	model.add(Dense(102))
	model.add(Activation('sigmoid'))

	model.compile(loss='categorical_crossentropy',
		  optimizer='adam',
		  metrics=['accuracy'])
	return model

def get_predict_model(model):
	pred_model = Sequential()
	pred_model.add(Embedding(max_features, embedding_size,
		             input_length=maxlen,
		             weights=model.layers[0].get_weights()))
	pred_model.add(Dropout(0.25))
	pred_model.add(Convolution1D(nb_filter=nb_filter,
		            filter_length=filter_length,
		            border_mode='valid',
		            activation='relu',
		            subsample_length=1,
		            weights=model.layers[2].get_weights()))
	pred_model.add(MaxPooling1D(pool_length=pool_length))
	pred_model.add(BatchNormalization(weights=model.layers[4].get_weights()))
	#pred_model.add(Dropout(0.25))
	pred_model.add(Convolution1D(nb_filter=128,
		            filter_length=5,
		            border_mode='same',
		            activation='relu',
		            subsample_length=1,
		            weights=model.layers[5].get_weights()))
	pred_model.add(MaxPooling1D(pool_length=2))
	pred_model.add(BatchNormalization(weights=model.layers[7].get_weights()))
	#pred_model.add(Dropout(0.25))
	pred_model.add(Convolution1D(nb_filter=256,
		            filter_length=3,
		            border_mode='valid',
		            activation='relu',
		            subsample_length=1,
		            weights=model.layers[8].get_weights()))
	pred_model.add(MaxPooling1D(pool_length=2))
	pred_model.add(BatchNormalization(weights=model.layers[10].get_weights()))
	pred_model.add(LSTM(lstm_output_size,
		                weights=model.layers[11].get_weights()))
	#pred_model.add(Flatten())
	#pred_model.add(Dropout(0.5))
	#pred_model.add(Dense(512, weights=model.layers[16].get_weights()))
	#pred_model.add(Activation('sigmoid'))

	pred_model.compile(loss='categorical_crossentropy', optimizer='adam')
	return pred_model

def get_predictions_samples(model):
	pred_model = get_predict_model(model)
	tr_coco_captions = pickle.load(open(os.path.join('Data', 'sample_caption_vectors.pkl'), "rb"))
	features_dict = {}
	preds = pred_model.predict(tr_coco_captions)	
	#bar = progressbar.ProgressBar(redirect_stdout=True,
	#	                          maxval=len(tr_image_list))
	#for i, img in enumerate(tr_image_list):
	#	caps = tr_coco_captions[img]
	#	
	#	features_dict[img] = preds
	#	bar.update(i)
	#bar.finish()
	pickle.dump(preds, open(os.path.join('Data',
		                                 'samples_features_dict.pkl'),
		                    "wb"))


def get_predictions(model):

    pred_model = get_predict_model(model)

    features_dict = {}
    bar = progressbar.ProgressBar(redirect_stdout=True,
                                          maxval=len(tr_image_list))
    for i, img in enumerate(tr_image_list):
        caps = tr_coco_captions[img]
        preds = pred_model.predict(caps)
        features_dict[img] = preds
        bar.update(i)
    bar.finish()
    pickle.dump(features_dict, open(os.path.join(flowers_dataDir,
                                                 'tr_features_dict.pkl'),
                                    "wb"))

def test(model, k=1):
    X_val, y_val = get_batch(dataType='val', batch_size=40000)
    y_pred = model.predict(X_val, batch_size=64, verbose=1)
    sim_sum = 0.0
    for i, (y_dash, y) in enumerate(zip(y_val, y_pred)):
        top_k = y.argsort()[-k:][::-1]
        y[top_k] = 1
        low_values_indices = y < 1
        y[low_values_indices] = 0
        sim_sum += jaccard_similarity_score(y_dash, y)
    score = sim_sum / 40000
    print('score for K(' + str(k) + ') : ' + str(score))


def train():
    model = None

    if os.path.exists(os.path.join(flowers_dataDir, "model.json")):
        with open(os.path.join(flowers_dataDir, "model.json"), 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
            adam = Adam(lr=0.00001)
            model.compile(loss='categorical_crossentropy',
                          optimizer=adam,
                          metrics=['accuracy'])
    else:
        model = build_model()
        model_json_string = model.to_json()
        with open(os.path.join(flowers_dataDir, "model.json"), "w") as f:
            f.write(model_json_string)

    if os.path.exists(os.path.join(flowers_dataDir, "model.h5")):
        model = load_model(os.path.join(flowers_dataDir, "model.h5"))

    for i in range(700):
        batch_no = 0
        epoch_bar = progressbar.ProgressBar(redirect_stdout=True,
                                      maxval=progressbar.UnknownLength)
        while batch_no * 128 + 128 < tr_n_imgs :
		X_train, y_train = get_batch(dataType='train', batch_size=128,
				         batch_no=batch_no)

		[loss, accuracy] = model.train_on_batch(X_train, y_train)
		batch_no += 1
		epoch_bar.update(batch_no)
		if batch_no % 20 == 0 and batch_no != 0:
		    print('\nTraining iteration ' , str(batch_no) + \
			'\ttraining loss: ' + str(loss) + '\ttraining accuracy: ' + str(accuracy))
		if batch_no % 50 == 0 and batch_no != 0:
		    X_val, y_val = get_batch(dataType='val', batch_size=800)
		    [loss, accuracy] = model.test_on_batch(X_val, y_val)
		    print('testing loss: ' + str(loss))
		    print('testing accuracy: ' + str(accuracy))
		    model.save(os.path.join(flowers_dataDir, 'model.h5'))
		    # save as JSON
		    print('Saving trained model')
	epoch_bar.finish()
    	model.save(os.path.join(flowers_dataDir, 'model.h5'))


print('Loading data...')
tr_coco_captions, tr_img_classes, val_coco_captions, val_img_classes, \
n_classes, max_caps_len, tr_image_list, tr_n_imgs, val_image_list, \
val_n_imgs = load_flowers_data()

if __name__ == '__main__':
    #train()
    
    model = None
    if os.path.exists(os.path.join(flowers_dataDir, "model.json")):
        with open(os.path.join(flowers_dataDir, "model.json"), 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
    else:
        print('Fucked up')
        sys.exit(0)

    if os.path.exists(os.path.join(flowers_dataDir, "model.h5")):
        model = load_model(os.path.join(flowers_dataDir, "model.h5"))
    else:
        print('Fucked up')
        sys.exit(0)

    
    #for i in range(10):
    #    test(model, k=i)
   
    #get_predictions(model)
    get_predictions_samples(model)
    
