import tensorflow as tf
import tensorflow.contrib.slim as slim
#import GRUCell, DropoutWrapper,MultiRNNCell
import numpy as np
from Utils import ops


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
		# t_real_caption = tf.placeholder('float32', [self.options[
		# 'batch_size'], self.options['caption_vector_length']],
		# name = 'real_caption_input')
		t_real_caption = [tf.placeholder('float32',
		                                [self.options['batch_size'], 1],
		                                name = 'real_caption_input' + str(i)) for i in range(self.options['e_max_step'])]
		
		#t_real_caption = tf.reshape(t_real_caption, [-1, 1])
		#t_real_caption = tf.split(0, self.options['e_max_step'], t_real_caption)

		t_z = tf.placeholder('float32',
		                     [self.options['batch_size'],
		                      self.options['z_dim']], name='input_noise')

		t_real_classes = tf.placeholder('float32',
										[self.options['batch_size'],
										self.options['n_classes']], name='real_classes')

		t_wrong_classes = tf.placeholder('float32',
										[self.options['batch_size'],
										 self.options['n_classes']], name='wrong_classes')

		e_dropout = tf.placeholder(tf.float32, name='dropout')

		caption_embeddings, seq_outputs, output_size, time_steps = \
			self.seq_encoder(t_real_caption,
			                 self.options['caption_vector_length'],
			                 self.options['e_size'],
			                 self.options['e_layers'],
			                 e_dropout)

		fake_image = self.generator(t_z, caption_embeddings)

		disc_real_image, disc_real_image_logits, attn_spn = self.discriminator(
			t_real_image, caption_embeddings,
			seq_outputs, output_size, time_steps, self.options['n_classes'])
		disc_wrong_image, disc_wrong_image_logits, attn_spn  = self.discriminator(
			t_wrong_image, caption_embeddings,
			seq_outputs, output_size,
			time_steps, self.options['n_classes'], reuse = True)
		disc_fake_image, disc_fake_image_logits, attn_spn  = self.discriminator(
			fake_image, caption_embeddings,
			seq_outputs, output_size,
			time_steps, self.options['n_classes'], reuse = True)

		tf.get_variable_scope()._reuse = False
		gt_gloss = ops.get_gt(self.options['batch_size'], t_real_classes,
		                                                        1, 'gt_gloss')
		gt_dloss1 = ops.get_gt(self.options['batch_size'], t_real_classes,
		                                                        1, 'gt_dloss1')
		gt_dloss2 = ops.get_gt(self.options['batch_size'], t_wrong_classes,
		                                                        0, 'gt_dloss2')
		gt_dloss3 = ops.get_gt(self.options['batch_size'], t_real_classes,
		                                                        0, 'gt_dloss3')

		g_loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(disc_fake_image_logits,
			                                        gt_gloss))

		d_loss1 = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(disc_real_image_logits,
			                                        gt_dloss1))
		d_loss2 = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(disc_wrong_image_logits,
			                                        gt_dloss2))
		d_loss3 = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(disc_fake_image_logits,
			                                        gt_dloss3))

		d_loss = d_loss1 + d_loss2 + d_loss3

		t_vars = tf.trainable_variables()
		'''
		for v in t_vars:
			print(v.name)
			print(v)
		'''
		d_vars = [var for var in t_vars if 'd_' in var.name or
		                                    'a_' in var.name or
								            'e_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name or 'e_' in var.name]

		input_tensors = {
			't_real_image' : t_real_image,
			't_wrong_image' : t_wrong_image,
			't_real_caption' : t_real_caption,
			't_z' : t_z,
			't_real_classes' : t_real_classes,
			't_wrong_classes' : t_wrong_classes,
			'e_dropout' : e_dropout
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
			'attn_span'                 : attn_spn
		}

		return input_tensors, variables, loss, outputs, checks

	def build_generator(self) :
		img_size = self.options['image_size']
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'],
		                                            self.options[
			                                            'caption_vector_length']],
		                                name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'],
		                                 self.options['z_dim']])
		fake_image = self.sampler(t_z, t_real_caption)

		input_tensors = {
			't_real_caption' : t_real_caption,
			't_z' : t_z
		}

		outputs = {
			'generator' : fake_image
		}

		return input_tensors, outputs

	# Sample Images for a text embedding
	def sampler(self, t_z, t_text_embedding) :
		tf.get_variable_scope().reuse_variables()

		s = self.options['image_size']
		s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

		reduced_text_embedding = ops.lrelu(
			ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding'))
		z_concat = tf.concat(1, [t_z, reduced_text_embedding])
		z_ = ops.linear(z_concat, self.options['gf_dim'] * 8 * s16 * s16,
		                'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(slim.batch_norm(h0, trainable = False, reuse=True, scope="g_bn0"))

		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8,
		                       self.options['gf_dim'] * 4], name = 'g_h1')
		h1 = tf.nn.relu(slim.batch_norm(h1, trainable = False, reuse=True, scope="g_bn1"))

		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4,
		                       self.options['gf_dim'] * 2], name = 'g_h2')
		h2 = tf.nn.relu(slim.batch_norm(h2, trainable = False, reuse=True, scope="g_bn2"))

		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2,
		                       self.options['gf_dim'] * 1], name = 'g_h3')
		h3 = tf.nn.relu(slim.batch_norm(h3, trainable = False, reuse=True, scope="g_bn3"))

		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3],
		                  name = 'g_h4')

		return (tf.tanh(h4) / 2. + 0.5)

	# GENERATOR IMPLEMENTATION based on :
	# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z, t_text_embedding) :
		s = self.options['image_size']
		s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

		reduced_text_embedding = ops.lrelu(
			ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding'))
		z_concat = tf.concat(1, [t_z, reduced_text_embedding])
		z_ = ops.linear(z_concat, self.options['gf_dim'] * 8 * s16 * s16,
		                'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(slim.batch_norm(h0, scope="g_bn0"))

		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8,
		                       self.options['gf_dim'] * 4], name = 'g_h1')
		h1 = tf.nn.relu(slim.batch_norm(h1, scope="g_bn1"))

		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4,
		                       self.options['gf_dim'] * 2], name = 'g_h2')
		h2 = tf.nn.relu(slim.batch_norm(h2, scope="g_bn2"))

		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2,
		                       self.options['gf_dim'] * 1], name = 'g_h3')
		h3 = tf.nn.relu(slim.batch_norm(h3, scope="g_bn3"))

		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3],
		                  name = 'g_h4')

		return (tf.tanh(h4) / 2. + 0.5)

	# GENERATOR IMPLEMENTATION based on :
	# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def seq_encoder(self, t_real_caption, embedding_size, state_size, e_layers,
	                e_dropout) :

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
		e_fcl0 = ops.lrelu(slim.batch_norm(ops.linear(concat_outs, 2048,'e_fcl_0'),
								    scope='e_bn0'))
		e_fcl1 = ops.lrelu(slim.batch_norm(ops.linear(e_fcl0, 1024, 'e_fcl_1'),
									scope='e_bn1'))
		caption_embeddings_logits = ops.linear(e_fcl1, embedding_size,
		                                       'e_embeddings')
		#print(caption_embeddings_logits)
		preds = tf.tanh(caption_embeddings_logits)

		return preds, outputs, output_size, time_steps


	# DISCRIMINATOR IMPLEMENTATION based on :
	# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image, t_text_embedding, seq_outputs,
	                  output_size, time_steps, n_classes, reuse = False) :
		if reuse :
			tf.get_variable_scope().reuse_variables()

		h0 = ops.lrelu(
			ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv'))  # 32
		#print(h0)
		h1 = ops.lrelu(slim.batch_norm(ops.conv2d(h0,
		                                     self.options['df_dim'] * 2,
		                                     name = 'd_h1_conv'),
		                               reuse=reuse,
		                               scope = 'd_bn1'))  # 16
		#print("H1")
		#print(h1)
		h2 = ops.lrelu(slim.batch_norm(ops.conv2d(h1,
		                                     self.options['df_dim'] * 4,
		                                     name = 'd_h2_conv'),
		                               reuse=reuse,
		                               scope = 'd_bn2'))  # 8
		h3 = ops.lrelu(slim.batch_norm(ops.conv2d(h2,
		                                     self.options['df_dim'] * 8,
		                                     name = 'd_h3_conv'),
		                               reuse=reuse,
		                               scope = 'd_bn3'))  # 4

		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding,
		                                               self.options['t_dim'],
		                                               'd_embedding'))
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 2)
		tiled_embeddings = tf.tile(reduced_text_embeddings,
		                           [1, 4, 4, 1],
		                           name = 'tiled_embeddings')

		h3_concat = tf.concat(3, [h3, tiled_embeddings], name = 'h3_concat')
		h3_new = ops.lrelu(slim.batch_norm(ops.conv2d(h3_concat,
												self.options['df_dim'] * 8,
												1, 1, 1, 1,
												name = 'd_h3_conv_new'),
		                               reuse=reuse,
		                               scope = 'd_bn4'))  # 4

		h3_flat = tf.reshape(h3_new, [self.options['batch_size'], -1])
		#print(h3_flat)
		h3_squeezed = ops.linear(h3_flat, output_size, 'd_h3_lin')
		#print(h3_squeezed)
		attn_sum, attn_span = self.attention(h3_squeezed, seq_outputs,
		                             output_size, time_steps)
		h3_attn = tf.concat(1, [h3_squeezed, attn_sum], name = 'h3_attn')
		h4 = ops.linear(h3_attn, 1 + n_classes, 'd_h4_lin')

		return tf.nn.sigmoid(h4), h4, attn_span

	def attention(self, decoder_output, seq_outputs, output_size, time_steps) :
		ui = ops.attention(decoder_output, seq_outputs, output_size,
		                   time_steps, name = "a_attention")
		#print(len(ui))
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
