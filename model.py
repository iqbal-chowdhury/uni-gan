import tensorflow as tf
import tensorflow.contrib.slim as slim
#import GRUCell, DropoutWrapper,MultiRNNCell
import numpy as np
from Utils import ops
from Utils.anp import MVSOCaffeNet as MyNet


class GAN :
	'''
	OPTIONS
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	caption_vector_length : Caption Vector Length 2400
	batch_size : Batch Size 64
	'''

	def __init__(self, options) :
		self.options = options
		'''
		self.g_bn0 = ops.batch_norm(name = 'g_bn0')
		self.g_bn1 = ops.batch_norm(name = 'g_bn1')
		self.g_bn2 = ops.batch_norm(name = 'g_bn2')
		self.g_bn3 = ops.batch_norm(name = 'g_bn3')
		
		self.d_bn1 = ops.batch_norm(name = 'd_bn1')
		self.d_bn2 = ops.batch_norm(name = 'd_bn2')
		self.d_bn3 = ops.batch_norm(name = 'd_bn3')
		self.d_bn4 = ops.batch_norm(name = 'd_bn4')
		'''
	def build_model(self) :
		img_size = self.options['image_size']
		t_real_image = tf.placeholder('float32',
		                              [self.options['batch_size'],
		                                    img_size, img_size, 3],
		                              name = 'real_image')
		t_wrong_image = tf.placeholder('float32',
		                               [self.options['batch_size'],
		                                    img_size, img_size, 3],
		                               name = 'wrong_image')

		t_real_caption = tf.placeholder('float32',
		                     [self.options['batch_size'],
		                      self.options['caption_vector_length']],
		                                name='real_captions')

		t_attn_input_seq = [tf.placeholder('float32',
		                                 [self.options['batch_size'],
		                                  self.options['attn_word_feat_length']],
		                                 name = 'attn_caption_input' + str(i))
		                  for i in range(self.options['attn_time_steps'])]

		t_z = tf.placeholder('float32',
		                     [self.options['batch_size'],
		                      self.options['z_dim']], name='input_noise')

		t_real_classes = tf.placeholder('float32',
										[self.options['batch_size'],
										self.options['n_classes']], name='real_classes')

		t_wrong_classes = tf.placeholder('float32',
										[self.options['batch_size'],
										 self.options['n_classes']], name='wrong_classes')

		t_training = tf.placeholder(tf.bool, name='training')

		fake_image, attn_spn = self.generator(t_z, t_real_caption, t_training,
		                                      t_attn_input_seq,
		                                      self.options[
			                                      'attn_word_feat_length'],
		                                      self.options[
			                                      'attn_time_steps'])

		disc_real_image, disc_real_image_logits, disc_real_image_aux, \
			disc_real_image_aux_logits = self.discriminator(
				t_real_image, t_real_caption, self.options['n_classes'],
				t_training)

		disc_wrong_image, disc_wrong_image_logits, disc_wrong_image_aux, \
			disc_wrong_image_aux_logits  = self.discriminator(
				t_wrong_image, t_real_caption, self.options['n_classes'],
				t_training, reuse = True)

		disc_fake_image, disc_fake_image_logits, disc_fake_image_aux, \
			disc_fake_image_aux_logits  = self.discriminator(
				fake_image, t_real_caption, self.options['n_classes'],
				t_training, reuse = True)

		tf.get_variable_scope()._reuse = False
		
		#gt_gloss = ops.get_gt(self.options['batch_size'], t_real_classes,
		#                                                        1, 'gt_gloss')
		#gt_dloss1 = ops.get_gt(self.options['batch_size'], t_real_classes,
		#                                                        1, 'gt_dloss1')
		#gt_dloss2 = ops.get_gt(self.options['batch_size'], t_wrong_classes,
		#                                                        0, 'gt_dloss2')
		#gt_dloss3 = ops.get_gt(self.options['batch_size'], t_real_classes,
		#                                                        0, 'gt_dloss3')

		g_loss_1 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits,
			                                        tf.ones_like(disc_fake_image)))
		g_loss_2 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_aux_logits,
			                                        t_real_classes))

		d_loss1 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(disc_real_image_logits,
			                                        tf.ones_like(disc_real_image)))
		d_loss1_1 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(disc_real_image_aux_logits,
			                                        t_real_classes))
		d_loss2 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(disc_wrong_image_logits,
			                                        tf.zeros_like(disc_wrong_image)))
		d_loss2_1 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(disc_wrong_image_aux_logits,
			                                        t_wrong_classes))
		d_loss3 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.zeros_like(disc_fake_image)))

		d_loss = d_loss1 + d_loss1_1 + d_loss2 + d_loss2_1 + d_loss3 + g_loss_2
		
		g_loss = g_loss_1 + g_loss_2

		t_vars = tf.trainable_variables()

		print('List of all variables')
		for v in t_vars:
			print(v.name)
			print(v)

		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name or
		                                    'a_' in var.name]

		input_tensors = {
			't_real_image' : t_real_image,
			't_wrong_image' : t_wrong_image,
			't_real_caption' : t_real_caption,
			't_z' : t_z,
			't_real_classes' : t_real_classes,
			't_wrong_classes' : t_wrong_classes,
			't_training' : t_training,
			't_attn_input_seq' : t_attn_input_seq

		}

		variables = {
			'd_vars' : d_vars,
			'g_vars' : g_vars
		}

		loss = {
			'g_loss' : g_loss,
			'd_loss' : d_loss
		}

		outputs = {
			'generator' : fake_image
		}

		checks = {
			'd_loss1'                   : d_loss1,
			'd_loss2'                   : d_loss2,
			'd_loss3'                   : d_loss3,
			'disc_real_image_logits'    : disc_real_image_logits,
			'disc_wrong_image_logits'   : disc_wrong_image,
			'disc_fake_image_logits'    : disc_fake_image_logits,
			'attn_span'                  : attn_spn
		}

		return input_tensors, variables, loss, outputs, checks


	# GENERATOR IMPLEMENTATION based on :
	# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z, t_text_embedding, t_training, seq_outputs,
	              output_size, time_steps) :
		s = self.options['image_size']
		s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

		reduced_text_embedding = ops.lrelu(
			ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding'))
		z_concat = tf.concat(1, [t_z, reduced_text_embedding])
		z_ = ops.linear(z_concat, self.options['gf_dim'] * 8 * s16 * s16,
		                'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(slim.batch_norm(h0, is_training = t_training,
		                                scope="g_bn0"))

		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8,
		                       self.options['gf_dim'] * 4], name = 'g_h1')
		h1 = tf.nn.relu(slim.batch_norm(h1, is_training = t_training,
		                                scope="g_bn1"))


		h1_flat = tf.reshape(h1, [self.options['batch_size'], -1])
		h1_squeezed = ops.linear(h1_flat, output_size, 'g_h1_lin')
		attn_sum, attn_span = self.attention(h1_squeezed, seq_outputs,
		                                        output_size, time_steps)
		attn_sum = tf.expand_dims(attn_sum, 1)
		attn_sum = tf.expand_dims(attn_sum, 2)
		tiled_attn = tf.tile(attn_sum, [1, s8, s8, 1], name = 'tiled_attention')
		h1_concat = tf.concat(3, [h1, tiled_attn], name = 'h1_concat')

		h2 = ops.deconv2d(h1_concat, [self.options['batch_size'], s4, s4,
		                       self.options['gf_dim'] * 2], name = 'g_h2')
		h2 = tf.nn.relu(slim.batch_norm(h2, is_training = t_training,
		                                scope="g_bn2"))

		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2,
		                       self.options['gf_dim'] * 1], name = 'g_h3')
		h3 = tf.nn.relu(slim.batch_norm(h3, is_training = t_training,
		                                scope="g_bn3"))

		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3],
		                  name = 'g_h4')
		return (tf.tanh(h4) / 2. + 0.5), attn_span

	# GENERATOR IMPLEMENTATION based on :
	# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def seq_encoder(self, t_real_caption, embedding_size, state_size, e_layers) :

		lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size,
		                                            forget_bias = 1.0)
		#lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell,
		#                              output_keep_prob = e_dropout)
		lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * e_layers)
		# Backward direction cell
		lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size,
		                                            forget_bias = 1.0)
		#lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,
		#                              output_keep_prob = e_dropout)
		lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * e_layers)
		# Get lstm cell output
		outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell,
		                                            t_real_caption,
		                                            dtype = tf.float32,
		                                        scope = 'e_brnn')
		#print(len(outputs))
		#print(outputs[0].get_shape())
		output_size = outputs[0].get_shape()[1]
		time_steps = len(outputs)
		# try weight sharing later too
		concat_outs = tf.concat(1, outputs)
		#print(concat_outs)
		# add a
		# constant
		# initializer later
		caption_embeddings_logits = ops.linear(concat_outs, embedding_size,
		                                       'e_embeddings')
		#print(caption_embeddings_logits)
		preds = tf.sigmoid(caption_embeddings_logits)

		return preds, outputs, output_size, time_steps


	# DISCRIMINATOR IMPLEMENTATION based on :
	# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image, t_text_embedding, n_classes, t_training,
	                  reuse = False) :
		if reuse :
			tf.get_variable_scope().reuse_variables()

		h0 = ops.lrelu(
			ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv'))  # 64
		#print(h0)
		h1 = ops.lrelu(slim.batch_norm(ops.conv2d(h0,
		                                     self.options['df_dim'] * 2,
		                                     name = 'd_h1_conv'),
		                               reuse=reuse,
		                               is_training = t_training,
		                               scope = 'd_bn1'))  # 32
		#print("H1")
		#print(h1)
		h2 = ops.lrelu(slim.batch_norm(ops.conv2d(h1,
		                                     self.options['df_dim'] * 4,
		                                     name = 'd_h2_conv'),
		                               reuse=reuse,
		                               is_training = t_training,
		                               scope = 'd_bn2'))  # 16
		h3 = ops.lrelu(slim.batch_norm(ops.conv2d(h2,
		                                     self.options['df_dim'] * 6,
		                                     name = 'd_h3_conv'),
		                               reuse=reuse,
		                               is_training = t_training,
		                               scope = 'd_bn3'))  # 8

		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding,
		                                               self.options['t_dim'],
		                                               'd_embedding'))
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 2)
		tiled_embeddings = tf.tile(reduced_text_embeddings,
		                           [1, 8, 8, 1],
		                           name = 'tiled_embeddings')

		h3_concat = tf.concat(3, [h3, tiled_embeddings], name = 'h3_concat')
		h3_new = ops.lrelu(slim.batch_norm(ops.conv2d(h3_concat,
												self.options['df_dim'] * 8,
												name = 'd_h3_conv_new'),
		                                reuse=reuse,
		                                is_training = t_training,
		                                scope = 'd_bn4'))  # 4

		h3_flat = tf.reshape(h3_new, [self.options['batch_size'], -1])

		h4 = ops.linear(h3_flat, 1, 'd_h4_lin_rw')
		h4_aux = ops.linear(h3_flat, n_classes, 'd_h4_lin_ac')
		
		return tf.nn.sigmoid(h4), h4, tf.nn.sigmoid(h4_aux), h4_aux

	def attention(self, decoder_output, seq_outputs, output_size, time_steps) :
		ui = ops.attention(decoder_output, seq_outputs, output_size,
		                   time_steps, name = "a_attention")
		#print(len(ui))
		with tf.variable_scope('a_attention'):
			ui = tf.transpose(ui, [1, 0, 2])
			#print(ui)
			ai = tf.nn.softmax(ui)
			#print(ai)
			seq_outputs = tf.transpose(seq_outputs, [1, 0, 2])
			#print(seq_outputs)
			#print(tf.mul(seq_outputs, ai))
			d_dash = tf.reduce_sum(tf.mul(seq_outputs, ai), axis=1)
			#print(d_dash)
			return d_dash, ai
