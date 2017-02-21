# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from Utils.anp import MVSOCaffeNet as MyNet
import skimage
import skimage.io as io
import skimage.transform

import csv
import os
import traceback
import numpy as np
from skimage.transform import resize
import random
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
        print('Selecting ' + str(len(anp_dict.keys())) + 'ANPs')
        all_anps = anp_dict.keys()
        #anps_idx_200 = np.random.randint(0, len(all_anps), size=n_anps)
        #selected_anps = np.take(all_anps, anps_idx_200)
        selected_anps = all_anps
        for anp in selected_anps:
            selected_images = selected_images + anp_dict[anp]
        n_selected_images = len(selected_images)
        print('n_images_selected : ' + str(n_selected_images))

        print('Pickling selected_anps.pkl')
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
        batch_image_ids = tr_image_ids[start: end]
    else:
        batch_image_ids = val_image_ids[start: end]
    image_batches = load_process_images(batch_image_ids, new_shape)
    one_hot_encoded_lbls = process_lbls(batch_image_ids)
    return np.array(image_batches), np.array(one_hot_encoded_lbls)


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
        if cnt % 10 == 0 and cnt != 0:
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

    tr_image_ids = selected_images[0:n_train_instances]
    val_image_ids = selected_images[n_train_instances: -1]
    pickle.dump(tr_image_ids,
                open(os.path.join(dataRoot, 'train_ids.pkl'), "wb"))
    pickle.dump(val_image_ids,
                open(os.path.join(dataRoot, 'val_ids.pkl'), "wb"))

#download_images(tr_image_ids)
#download_images(val_image_ids)

def weight_variable(shape):
  #initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.get_variable("W", shape=shape,
           initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def train():
    batch_size = 128
    images = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    labels = tf.placeholder(tf.float32, [batch_size, 4421])
    net = MyNet({'data': images})
    fc7 = net.layers['fc7']
    W_fc8 = weight_variable([4096, 4421])
    b_fc8 = bias_variable([4421])
    fc8 = tf.matmul(fc7, W_fc8) + b_fc8
    pred = tf.nn.softmax(fc8)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    topFiver = tf.nn.in_top_k(pred, labels, 5)
    topTen = tf.nn.in_top_k(pred, labels, 10)
    top_five_acc = tf.reduce_mean(tf.cast(topFiver, tf.float32))
    top_ten_acc = tf.reduce_mean(tf.cast(topTen, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc8, labels), 0)
    opt = tf.train.AdamOptimizer(0.0001)
    train_op = opt.minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Load the data
        sess.run(tf.initialize_all_variables())
        net.load('./Utils/anp.npy', sess)
        for i in range(30):
            print('epoch: ' + str(i))
            batch_count = 0
            epochbar = progressbar.ProgressBar(redirect_stdout=True,
                                               max_value=progressbar.UnknownLength)
            while (1):
                start = batch_count * batch_size
                X_train, y_train = None, None
                try:
                    X_train, y_train = get_batch(start, dataType='train',
                                                 batch_size=128, new_shape=(227, 227, 3))
                except Exception as e:
                    print('Error occurred while Getting Batch. Skipping this '
                          'batch.')
                    traceback.print_stack()
                    batch_count += 1
                    continue
                if X_train is None or y_train is None:
                    break
                feed = {images: X_train, labels: y_train}
                #[loss, accuracy] = model.train_on_batch(X_train, y_train)
                trloss, np_pred, traccuracy, tr_top_five, tr_top_ten, _ = sess.run([loss, pred, accuracy, top_five_acc, top_ten_acc, train_op], feed_dict=feed)
                batch_count += 1
                epochbar.update(batch_count)
                if batch_count % 10 == 0 and batch_count != 0:
                    print('\nItr ' + str(batch_count) + '\ttraining loss: ' + str(
                        trloss) +
                          '\ttraining accuracy: ' + str(traccuracy) +
                          '\ttraining top 5 accuracy: ' + str(tr_top_five)+
                          '\ttraining top 10 accuracy: ' + str(tr_top_ten))
                if batch_count % 100 == 0 and batch_count != 0:
                    #model.save(os.path.join(dataRoot, 'model.ckpt'))
                    save_path = saver.save(sess, os.path.join(dataRoot, 'model.ckpt'))
                    print("Model saved in file: %s" % save_path)
                    try:
                        X_val, y_val = get_batch(0, dataType='val', batch_size=64)
                        val_feed = {images: X_val, labels: y_val}
                        #[loss, accuracy] = model.test_on_batch(X_val, y_val)
                        val_loss, val_np_pred, val_accuracy, val_top_five, val_top_ten = sess.run([loss, pred, accuracy, top_five_acc, top_ten_acc], feed_dict=val_feed)
                        print('\ntesting loss: ' + str(val_loss))
                        print('\ntesting accuracy: ' + str(val_accuracy)+
                          '\ttraining top 5 accuracy: ' + str(val_top_five)+
                          '\ttraining top 10 accuracy: ' + str(val_top_ten))
                    except Exception as e:
                        print('Error occurred while Getting Batch. Skipping this '
                              'batch.')
                        traceback.print_stack()
                        continue
                        # save as JSON
            epochbar.finish()

            if i % 2 == 0 and i != 0:
                #model.save(os.path.join(dataRoot, 'model.h5'))
                save_path = saver.save(sess, os.path.join(dataRoot, 'model.ckpt'))
                print("Model saved in file: %s" % save_path)
                # save as JSON
                try:
                    X_val, y_val = get_batch(0, dataType='val', batch_size=64)
                    val_feed = {images: X_val, labels: y_val}
                    # [loss, accuracy] = model.test_on_batch(X_val, y_val)
                    val_loss, val_np_pred, val_accuracy, val_top_five, val_top_ten = sess.run(
                        [loss, pred, accuracy, top_five_acc, top_ten_acc], feed_dict=val_feed)
                    print('\ntesting loss: ' + str(val_loss))
                    print('\ntesting accuracy: ' + str(val_accuracy) +
                          '\ttraining top 5 accuracy: ' + str(val_top_five) +
                          '\ttraining top 10 accuracy: ' + str(val_top_ten))
                except Exception as e:
                    print('Error occurred while Getting Batch. Skipping this '
                          'batch.')
                    traceback.print_stack()
                    continue


if __name__ == '__main__':
    train()

