import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import data_util as du

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--caption_file', type=str, default='Data/sample_captions.txt',
					   help='caption file')
	parser.add_argument('--data_dir', type=str, default='Data',
					   help='Data Directory')
	parser.add_argument('--dataset', type=str, default='flowers',
	                    help='dataset')
	parser.add_argument('--pad_len', type=int, default=77,
	                    help='Length of the sequences for padding')
	parser.add_argument('--vocab_size', type=int, default=40000,
	                    help='Size of the vocabulary')
	
	args = parser.parse_args()
	with open( args.caption_file ) as f:
		captions = f.read().split('\n')

	captions = [cap for cap in captions if len(cap) > 0]
	print captions
	#model = skipthoughts.load_model()
	#caption_vectors = skipthoughts.encode(model, captions)
	vocab_dir = os.path.join(args.data_dir, args.dataset)
	caps_path = os.path.join(args.data_dir, 'sample_caption_vectors.pkl')
	caps_vec = du.encode_text_to_ids(captions, vocab_dir, args.vocab_size, args.pad_len)
	
	if os.path.isfile(caps_path):
		os.remove(caps_path)
	pickle.dump(caps_vec, open(caps_path, "wb"))

if __name__ == '__main__':
	main()