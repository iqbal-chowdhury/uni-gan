import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os
import shutil
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

base_model = VGG16(weights='imagenet', include_top=True)
vggmodel = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

def main() :
	parser = argparse.ArgumentParser()
	
	
	parser.add_argument('--e_size', type=int, default=256,
	                    help='RNN Sequence encoder state size')
	
	parser.add_argument('--e_max_step', type=int, default=77,
	                    help='RNN Sequence encoder maximum time steps')
	
	parser.add_argument('--e_layers', type=int, default=2,
	                    help='RNN Sequence encoder number of layers')
	
	parser.add_argument('--batch_size', type=int, default=64,
	                    help='Batch Size')
	
	parser.add_argument('--image_size', type=int, default=224,
	                    help='Image Size a, a x a')
	

	parser.add_argument('--caption_vector_length', type=int, default=2400,
	                    help='Caption Vector Length')
	
	parser.add_argument('--data_dir', type=str, default="Data",
	                    help='Data Directory')
	
	parser.add_argument('--learning_rate', type=float, default=0.0002,
	                    help='Learning Rate')
	
	parser.add_argument('--beta1', type=float, default=0.5,
	                    help='Momentum for Adam Update')
	
	parser.add_argument('--epochs', type=int, default=600,
	                    help='Max number of epochs')
	
	parser.add_argument('--save_every', type=int, default=30,
	                    help='Save Model/Samples every x iterations over '
	                         'batches')
	
	parser.add_argument('--resume_model', type=str, default=None,
	                    help='Pre-Trained Model Path, to resume from')
	
	parser.add_argument('--data_set', type=str, default="flowers",
	                    help='Dat set: MS-COCO, flowers')
	
	args = parser.parse_args()
	
	loaded_data = load_training_data(args.data_dir, args.data_set)
	model_options = {
		'batch_size' : args.batch_size,
		'image_size' : args.image_size,
		'caption_vector_length' : args.caption_vector_length,
		'e_size' : args.e_size,
		'e_layers' : args.e_layers,
		'e_max_step' : args.e_max_step,
		'n_classes' : loaded_data['n_classes']
	}
	
	gan = model.GAN(model_options)
	input_tensors, variables, loss, outputs, checks = gan.build_encoder()
	
	d_optim = tf.train.AdamOptimizer(args.learning_rate,
	                                 beta1=args.beta1).minimize(
		loss['cost'], var_list=variables['e_vars'])
	
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()
	
	saver = tf.train.Saver()
	if args.resume_model :
		print(tf.train.latest_checkpoint(args.resume_model))
		saver.restore(sess, tf.train.latest_checkpoint(args.resume_model))
	
	for i in range(args.epochs) :
		batch_no = 0
		while batch_no * args.batch_size < loaded_data['data_length'] :
			real_images, wrong_images, caption_vectors, image_files, \
			real_classes, wrong_classes = get_training_batch(batch_no,
			                                                 args.batch_size,
			                                                 args.image_size,
			                                                 args.caption_vector_length,
			                                                 'train', args.data_dir,
			                                                 args.data_set, loaded_data)
			
			real_images = preprocess_input(real_images)
			fc2_features = vggmodel.predict(real_images)
			# DISCR UPDATE
			check_ts = [checks['caption_embeddings'],
			            checks['seq_outputs']]
			feed = {
				input_tensors['t_image_feat'].name : fc2_features,
				input_tensors['t_real_classes'].name : real_classes,
				input_tensors['keep_prob'].name : 0.5,
				input_tensors['train'].name : 1,
			}
			for c, d in zip(input_tensors['t_real_caption'], caption_vectors) :
				feed[c.name] = d
			
			# DISC UPDATE
			_, cost, acc, ce, so = sess.run(
				[d_optim, loss['cost'], outputs['accuracy']] + check_ts,
				feed_dict=feed)
			
			print "cost", cost
			print "acc", acc

			
			print batch_no, i, len(
				loaded_data['image_list']) / args.batch_size
			batch_no += 1
			if (batch_no % args.save_every) == 0 :
				print "Saving Images, Model"
				#save_for_vis(args.data_dir, real_images, gen, image_files)
				save_path = saver.save(sess,
				                       "Data/EncoderModels/latest_model_{}_temp.ckpt".format(
					                       args.data_set))
		if i % 5 == 0 :
			save_path = saver.save(sess,
			                       "Data/EncoderModels/model_after_{}_epoch_{"
			                       "}.ckpt".format(
				                       args.data_set, i))


def load_training_data(data_dir, data_set) :
	if data_set == 'flowers' :
		h = pickle.load(open(join(data_dir, 'flower_tv.pkl'), "rb"))
		# h1 = h5py.File(join(data_dir, 'flower_tc.hdf5'))
		h1 = pickle.load(open(join(data_dir, 'flower_tc.pkl'), "rb"))
		
		flower_captions = {}
		img_classes = {}
		n_classes = 0
		max_caps_len = 0
		
		for ds in h.iteritems() :
			flower_captions[ds[0]] = np.array(ds[1])
			if max_caps_len is 0 :
				max_caps_len = flower_captions[ds[0]].shape[1]
		
		for ds in h1.iteritems() :
			img_classes[ds[0]] = np.array(ds[1])
			if n_classes is 0 :
				n_classes = img_classes[ds[0]].shape[0]
		
		image_list = [key for key in flower_captions]
		image_list.sort()
		
		img_75 = int(len(image_list) * 0.75)
		training_image_list = image_list[0 :img_75]
		random.shuffle(training_image_list)
		
		return {
			'image_list' : training_image_list,
			'captions' : flower_captions,
			'data_length' : len(training_image_list),
			'classes' : img_classes,
			'n_classes' : n_classes,
			'max_caps_len' : max_caps_len
		}
	
	else :
		with open(join(data_dir, 'meta_train.pkl')) as f :
			meta_data = pickle.load(f)
		# No preloading for MS-COCO
		return meta_data


def save_for_vis(data_dir, real_images, generated_images, image_files) :
	shutil.rmtree(join(data_dir, 'samples'))
	os.makedirs(join(data_dir, 'samples'))
	
	for i in range(0, real_images.shape[0]) :
		real_image_255 = np.zeros((64, 64, 3), dtype=np.uint8)
		real_images_255 = (real_images[i, :, :, :])
		scipy.misc.imsave(join(data_dir, 'samples/{}_{}.jpg'.format(i,
		                                                            image_files[
			                                                            i].split(
			                                                            '/')[
			                                                            -1])),
		                  real_images_255)
		
		fake_image_255 = np.zeros((64, 64, 3), dtype=np.uint8)
		fake_images_255 = (generated_images[i, :, :, :])
		scipy.misc.imsave(join(data_dir, 'samples/fake_image_{}.jpg'.format(
			i)),
		                  fake_images_255)


def get_training_batch(batch_no, batch_size, image_size,
                       caption_vector_length, split, data_dir, data_set,
                       loaded_data=None) :
	if data_set == 'mscoco' :
		with h5py.File(join(data_dir,
		                    'tvs/' + split + '_tvs_' + str(batch_no))) as hf :
			caption_vectors = np.array(hf.get('tv'))
			caption_vectors = caption_vectors[:, 0 :caption_vector_length]
		with h5py.File(join(data_dir, 'tvs/' + split + '_tv_image_id_' + str(
				batch_no))) as hf :
			image_ids = np.array(hf.get('tv'))
		
		real_images = np.zeros((batch_size, 64, 64, 3))
		wrong_images = np.zeros((batch_size, 64, 64, 3))
		
		image_files = []
		for idx, image_id in enumerate(image_ids) :
			image_file = join(data_dir, '%s2014/COCO_%s2014_%.12d.jpg' % (
				split, split, image_id))
			image_array = image_processing.load_image_array(image_file,
			                                                image_size)
			real_images[idx, :, :, :] = image_array
			image_files.append(image_file)
		
		# TODO>> As of Now, wrong images are just shuffled real images.
		first_image = real_images[0, :, :, :]
		for i in range(0, batch_size) :
			if i < batch_size - 1 :
				wrong_images[i, :, :, :] = real_images[i + 1, :, :, :]
			else :
				wrong_images[i, :, :, :] = first_image
		
		z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
		
		return real_images, wrong_images, caption_vectors, z_noise, image_files
	
	if data_set == 'flowers' :
		real_images = np.zeros((batch_size, image_size, image_size, 3))
		wrong_images = np.zeros((batch_size, image_size, image_size, 3))
		# captions = np.zeros((batch_size, caption_vector_length))
		captions = np.zeros((batch_size, loaded_data['max_caps_len']))
		real_classes = np.zeros((batch_size, loaded_data['n_classes']))
		wrong_classes = np.zeros((batch_size, loaded_data['n_classes']))
		
		cnt = 0
		image_files = []
		for i in range(batch_no * batch_size,
		               batch_no * batch_size + batch_size) :
			idx = i % len(loaded_data['image_list'])
			image_file = join(data_dir,
			                  'flowers/jpg/' + loaded_data['image_list'][idx])
			image_array = image_processing.load_image_array(image_file,
			                                                image_size)
			real_images[cnt, :, :, :] = image_array
			
			# Improve this selection of wrong image
			wrong_image_id = random.randint(0,
			                                len(loaded_data['image_list']) - 1)
			wrong_image_file = join(data_dir,
			                        'flowers/jpg/' + loaded_data['image_list'][
				                        wrong_image_id])
			wrong_image_array = image_processing.load_image_array(wrong_image_file,
			                                                      image_size)
			wrong_images[cnt, :, :, :] = wrong_image_array
			
			wrong_classes[cnt, :] = loaded_data['classes'][loaded_data['image_list'][
				wrong_image_id]][0 :loaded_data['n_classes']]
			
			random_caption = random.randint(0, 4)
			captions[cnt, :] = \
				loaded_data['captions'][loaded_data['image_list'][idx]][
					random_caption][0 :loaded_data['max_caps_len']]
			
			real_classes[cnt, :] = \
				loaded_data['classes'][loaded_data['image_list'][idx]][
				0 :loaded_data['n_classes']]
			image_files.append(image_file)
			cnt += 1
		captions = tf_seq_reshape(batch_size, captions,
		                          loaded_data['max_caps_len'])
		
		
		return real_images, wrong_images, captions, image_files, \
		       real_classes, wrong_classes


def tf_seq_reshape(batch_size, captions, caps_max_len) :
	# Now we create batch-major vectors from the data selected above.
	batch_encoder_inputs = []
	
	# Batch encoder inputs are just re-indexed encoder_inputs.
	for length_idx in xrange(caps_max_len) :
		batch_encoder_inputs.append(
			np.array([[captions[batch_idx][length_idx]]
			          for batch_idx in xrange(batch_size)], dtype=np.float32))
	return batch_encoder_inputs


if __name__ == '__main__' :
	main()
