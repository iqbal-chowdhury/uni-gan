import csv
import os
import traceback
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras import backend as K
import skimage
from skimage.transform import resize
from skimage import io
import random
from keras.models import Sequential, model_from_json, load_model
import pickle
import progressbar

dataRoot = 'Data/anp'
datarepo_path = os.path.join(dataRoot, "images")
target_file_path = os.path.join(dataRoot, "selectedanps.txt")
ser_loaded_data_path = os.path.join(dataRoot, "image_dict.pkl")
tr_image_ids = []
val_image_ids = []
image_dict = {}
target = []
one_hot_targets = []
n_targets = 0
n_instances = 0
labels = []

def one_hot_encode_str_lbl(lbl, target, one_hot_targets):
    '''
    Encodes a string label into one-hot encoding

    Example:
        input: "window"
        output: [0 0 0 0 0 0 1 0 0 0 0 0]
    the length would depend on the number of classes in the dataset. The
    above is just a random example.

    :param lbl: The string label
    :return: one-hot encoding
    '''
    idx = target.index(lbl)
    return one_hot_targets[idx]


def get_one_hot_targets(target_file_path):
    target = []
    one_hot_targets = []
    n_target = 0
    try:
        with open(target_file_path) as f:
            target = f.readlines()
            target = [t.strip('\n') for t in target]
            n_target = len(target)
    except IOError:
        print('Could not load the labels.txt file in the dataset. A '
              'dataset folder is expected in the "data/datasets" '
              'directory with the name that has been passed as an '
              'argument to this method. This directory should contain a '
              'file called labels.txt which contains a list of labels and '
              'corresponding folders for the labels with the same name as '
              'the labels.')
        traceback.print_stack()

    lbl_idxs = np.arange(n_target)
    one_hot_targets = np.zeros((n_target, n_target))
    one_hot_targets[np.arange(n_target), lbl_idxs] = 1

    return target, one_hot_targets, n_target

def load_data(data_dir, target_file_path, n_anps=200):
    data_path = os.path.join(data_dir, 'english.csv')
    labels = []
    image_ids = []
    selected_images = []
    image_dict = {}
    anp_dict = {}
    n_instances = 0
    n_selected_images = 0
    if os.path.exists(os.path.join(dataRoot, 'image_dict.pkl')) and \
        os.path.exists(os.path.join(dataRoot, 'labels.pkl')) and \
        os.path.exists(os.path.join(dataRoot, 'selected_anps.pkl')) and \
        os.path.exists(os.path.join(dataRoot, 'selected_images.pkl')) and \
        os.path.exists(os.path.join(dataRoot, 'anp_dict.pkl')) and \
        os.path.exists(os.path.join(dataRoot, 'image_ids.pkl')):
        print('Loading saved data objects. It might take some time so sit '
              'back and relax.')

        print('Pickling image_dict.pkl')
        image_dict = pickle.load(open(os.path.join(dataRoot, 'image_dict.pkl')))

        print('Pickling anp_dict.pkl')
        anp_dict = pickle.load(open(os.path.join(dataRoot, 'anp_dict.pkl')))

        print('Pickling selected_images.pkl')
        selected_images = pickle.load(
            open(os.path.join(dataRoot, 'selected_images.pkl')))

        print('Pickling selected_anps.pkl')
        selected_anps = pickle.load(
            open(os.path.join(dataRoot, 'selected_anps.pkl')))

        print('Pickling labels.pkl')
        labels = pickle.load(open(os.path.join(dataRoot, 'labels.pkl')))

        print('Pickling image_ids.pkl')
        image_ids = pickle.load(open(os.path.join(dataRoot, 'image_ids.pkl')))
        n_instances = len(image_dict.keys())
        n_selected_images = len(selected_images)
    else:

        with open(data_path, 'rb') as csvfile:
            rows = csv.DictReader(csvfile, delimiter=',', )
            print('Extracting Rows from ' + data_path)
            pklbar = progressbar.ProgressBar(redirect_stdout=True,
                                             maxval=progressbar.UnknownLength)
            for i, row in enumerate(rows):
                if i % 100 == 0 and i != 0:
                    pklbar.update(i)

                labels.append(row['ANP'])
                image_ids.append(i)
                image_dict[i] = {
                    'classlbl': row['ANP'],
                    'image_url': row['image_url']
                }
                if row['ANP'] not in anp_dict:
                    anp_dict[row['ANP']] = [i]
                else:
                    anp_dict[row['ANP']].append(i)

                n_instances += 1
        pklbar.finish()
        print('Selecting ' + str(n_anps) + 'ANPs')
        all_anps = anp_dict.keys()
        anps_idx_200 = np.random.randint(0, len(all_anps), size=n_anps)
        selected_anps = np.take(all_anps, anps_idx_200)
        for anp in selected_anps:
            selected_images = selected_images + anp_dict[anp]
        n_selected_images = len(selected_images)
        print('n_images_selected : ' + str(n_selected_images))


        print('Pickling selected_anps.pkl' )
        pickle.dump(selected_anps, open(os.path.join(dataRoot,
                                                       'selected_anps.pkl'),
                                          "wb"))

        print('Pickling selected_images.pkl')
        pickle.dump(selected_images, open(os.path.join(dataRoot,
                                                  'selected_images.pkl'),
                                    "wb"))

        print('Pickling image_dict.pkl')
        pickle.dump(image_dict, open(os.path.join(dataRoot, 'image_dict.pkl'),
                                     "wb"))

        print('Pickling anp_dict.pkl')
        pickle.dump(anp_dict, open(os.path.join(dataRoot, 'anp_dict.pkl'),
                                     "wb"))

        print('Pickling labels.pkl')
        pickle.dump(labels, open(os.path.join(dataRoot, 'labels.pkl'),
                                     "wb"))

        print('Pickling image_ids.pkl')
        pickle.dump(image_ids, open(os.path.join(dataRoot, 'image_ids.pkl'),
                                     "wb"))

    target = None
    n_targets = None
    one_hot_targets = None

    if not os.path.exists(target_file_path):
        with open(target_file_path, "w") as text_file:
            targets = selected_anps
            text_file.write('\n'.join(targets))
        target, one_hot_targets, n_target = get_one_hot_targets(
            target_file_path)
    else:
        target, one_hot_targets, n_target = get_one_hot_targets(
            target_file_path)

    return image_ids, image_dict, target, one_hot_targets, n_targets, \
                n_instances, n_selected_images, selected_images, anp_dict

def load_process_images(image_ids, new_shape=(244, 244, 3)):
    processed_images = []
    img = None
    for image_id in image_ids:
        if os.path.exists(os.path.join(datarepo_path, str(image_id) + '.png')):
            img = io.imread(os.path.join(datarepo_path, str(image_id) + '.png'))
        else:
            img = io.imread(image_dict[image_id]['image_url'])
            io.imsave(os.path.join(datarepo_path, str(image_id) + '.png'), img)
        img = resize(img, new_shape)
        processed_images.append(img.tolist())
    return processed_images

def process_lbls(image_ids):
    one_hot_encoded_lbls = []
    for image_id in image_ids:
        one_hot_encoded_lbls.append(
                    one_hot_encode_str_lbl(image_dict[image_id]['classlbl'],
                                           target, one_hot_targets).tolist())
    return one_hot_encoded_lbls

def get_batch(start, batch_size=64, dataType='train', new_shape=(224, 224, 3)):
    end = start + batch_size
    chk_len = len(tr_image_ids) if dataType == 'train' else len(val_image_ids)
    if end >= chk_len:
        return None, None
    batch_image_ids = None
    if dataType == 'train':
        batch_image_ids = tr_image_ids[start : end]
    else:
        batch_image_ids = val_image_ids[start: end]
    image_batches = load_process_images(batch_image_ids, new_shape)
    one_hot_encoded_lbls = process_lbls(batch_image_ids)
    return np.array(image_batches), np.array(one_hot_encoded_lbls)


def build_model(n_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(
        224,224,3))

    # add a global spatial average pooling layer
    x = base_model.output
    x = Flatten()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(5402, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(200, activation='softmax')(x)
    # let's add a fully-connected layer

    # and a logistic layer -- let's say we have 200 classes


    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def download_images(dwn_image_ids):
    img = None
    print('downloading images from urls. It will take a lot of time')

    downloadbar = progressbar.ProgressBar(redirect_stdout=True,
                                          maxval=len(dwn_image_ids))
    for cnt, image_id in enumerate(dwn_image_ids):
        if not os.path.exists(os.path.join(datarepo_path,
                                           str(image_id) + '.png')):
            try:
                img = io.imread(image_dict[image_id]['image_url'])
            except Exception as e:
                print(e)
                continue
            io.imsave(os.path.join(datarepo_path, str(image_id) + '.png'), img)
        if cnt%10 == 0 and cnt != 0:
            downloadbar.update(cnt)
    downloadbar.finish()

image_ids, image_dict, target, one_hot_targets, n_targets, \
n_instances, n_selected_images, selected_images, anp_dict = \
    load_data(dataRoot, target_file_path, n_anps=200)

if os.path.exists(os.path.join(dataRoot, 'train_ids.pkl')) and \
        os.path.exists(os.path.join(dataRoot, 'val_ids.pkl')):
    tr_image_ids = pickle.load(open(os.path.join(dataRoot, 'train_ids.pkl')))
    val_image_ids = pickle.load(open(os.path.join(dataRoot, 'val_ids.pkl')))
else:
    random.shuffle(selected_images)
    n_train_instances = int(n_selected_images * 0.9)

    tr_image_ids = selected_images[0 :n_train_instances]
    val_image_ids = selected_images[n_train_instances : -1]
    pickle.dump(tr_image_ids,
                open(os.path.join(dataRoot, 'train_ids.pkl'), "wb"))
    pickle.dump(val_image_ids,
                open(os.path.join(dataRoot, 'val_ids.pkl'), "wb"))

download_images(tr_image_ids)
download_images(val_image_ids)



def train():
    model = None
    batch_size=128
    if os.path.exists(os.path.join(dataRoot, "model.json")):
        with open(os.path.join(dataRoot, "model.json"), 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
            model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['accuracy'])
    else:
        model = build_model(n_targets)
        model_json_string = model.to_json()
        with open(os.path.join(dataRoot, "model.json"), "w") as f:
            f.write(model_json_string)

    if os.path.exists(os.path.join(dataRoot, "model.h5")):
        print('Loading model from previously saved weights')
        model = load_model(os.path.join(dataRoot, "model.h5"))

    for i in range(30):
        print('epoch: ' + str(i))
        batch_count = 0
        epochbar = progressbar.ProgressBar(redirect_stdout=True,
                                           max_value=progressbar.UnknownLength)
        while(1):
            start = batch_count*batch_size
            X_train, y_train = None, None
            try:
                X_train, y_train = get_batch(start, dataType='train',
                                             batch_size=64)
            except Exception as e:
                print('Error occurred while Getting Batch. Skipping this '
                      'batch.')
                traceback.print_stack()
                batch_count += 1
                continue
            if X_train is None or y_train is None:
                break
            [loss, accuracy] = model.train_on_batch(X_train, y_train)
            batch_count += 1
            epochbar.update(batch_count)
            if batch_count % 1 == 0 and batch_count != 0:
                print('Itr ' + str(batch_count) + '\ttraining loss: ' + str(loss) +
                '\ttraining accuracy: ' + str(accuracy))
            if batch_count % 100 == 0 and batch_count != 0:
                model.save(os.path.join(dataRoot, 'model.h5'))
                print('Saving trained model')
                try:
                    X_val, y_val = get_batch(0, dataType='val', batch_size=64)
                    [loss, accuracy] = model.test_on_batch(X_val, y_val)
                    print('testing loss: ' + str(loss))
                    print('testing accuracy: ' + str(accuracy))
                except Exception as e:
                    print('Error occurred while Getting Batch. Skipping this '
                          'batch.')
                    traceback.print_stack()
                    continue
                # save as JSON
        epochbar.finish()

        if i % 2 == 0 and i != 0:
            model.save(os.path.join(dataRoot, 'model.h5'))
            # save as JSON
            print('Saving trained model')
            try:
                X_val, y_val = get_batch(0, dataType='val', batch_size=64)
                [loss, accuracy] = model.test_on_batch(X_val, y_val)
                print('testing loss: ' + str(loss))
                print('testing accuracy: ' + str(accuracy))
            except Exception as e:
                print('Error occurred while Getting Batch. Skipping this '
                      'batch.')
                traceback.print_stack()
                continue

if __name__ == '__main__':
    train()
