# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from Utils.anp import MVSOCaffeNet as MyNet
import skimage
import skimage.io
import skimage.transform

batch_size = 1

images = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
net = MyNet({'data': images})


with tf.Session() as sess:
    # Load the data
    sess.run(tf.initialize_all_variables())
    net.load('./Utils/anp.npy', sess)
    np_images = skimage.io.imread(
        "/home/ayushman/Pictures/kutta.png")
    np_images = skimage.transform.resize(np_images, (227, 227))
    feed = {images: [np_images]}

    probs = sess.run(net.get_output(), feed_dict=feed)

    with open('/home/ayushman/projects/uni-gan/Data/conditioning_models/anp'
              '/mvso_detectors/english/english_label.txt', 'rb') as infile :
        class_labels = map(str.strip, infile.readlines())
    class_indices = np.argmax(probs, axis = 1)
    class_name = class_labels[class_indices[0]]
    confidence = round(probs[0, class_indices[0]] * 100, 2)
    print('{:20} {:30} {} %'.format("foola", class_name, confidence))

