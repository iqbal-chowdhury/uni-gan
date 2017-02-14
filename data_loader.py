import json
import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py
import data_util
import traceback
import pickle

# DID NOT TRAIN IT ON MS COCO YET
def save_caption_vectors_ms_coco(data_dir, split, batch_size) :
    meta_data = {}
    ic_file = join(data_dir, 'annotations/captions_{}2014.json'.format(split))
    with open(ic_file) as f :
        ic_data = json.loads(f.read())

    meta_data['data_length'] = len(ic_data['annotations'])
    with open(join(data_dir, 'meta_{}.pkl'.format(split)), 'wb') as f :
        pickle.dump(meta_data, f)

    model = skipthoughts.load_model()
    batch_no = 0
    print "Total Batches", len(ic_data['annotations']) / batch_size

    while batch_no * batch_size < len(ic_data['annotations']) :
        captions = []
        image_ids = []
        idx = batch_no
        for i in range(batch_no * batch_size, (batch_no + 1) * batch_size) :
            idx = i % len(ic_data['annotations'])
            captions.append(ic_data['annotations'][idx]['caption'])
            image_ids.append(ic_data['annotations'][idx]['image_id'])

        #print captions
        #print image_ids
        # Thought Vectors
        tv_batch = skipthoughts.encode(model, captions)
        h5f_tv_batch = h5py.File(join(data_dir, 'tvs/' + split + '_tvs_' + str(batch_no)),
                                 'w')
        h5f_tv_batch.create_dataset('tv', data = tv_batch)
        h5f_tv_batch.close()

        h5f_tv_batch_image_ids = h5py.File(
            join(data_dir, 'tvs/' + split + '_tv_image_id_' + str(batch_no)), 'w')
        h5f_tv_batch_image_ids.create_dataset('tv', data = image_ids)
        h5f_tv_batch_image_ids.close()

        print "Batches Done", batch_no, len(ic_data['annotations']) / batch_size
        batch_no += 1

def get_one_hot_targets(target_file_path):
	target = []
	one_hot_targets = []
	n_target = 0
	try :
		with open(target_file_path) as f :
			target = f.readlines()
			target = [t.strip('\n') for t in target]
			n_target = len(target)
	except IOError :
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

def save_caption_vectors_flowers(data_dir, vocab_size, recreate_vocab=True,
                                 part='text_c10', dt_range=(1, 103)) :
    import time

    img_dir = join(data_dir, 'flowers/jpg')
    #dataset_root_dir = join(data_dir, 'flowers')
    all_caps_dir = join(data_dir, 'flowers/all_captions.txt')
    vocab_path = os.path.join(data_dir, "flowers/vocab%d.txt" % vocab_size)
    target_file_path = os.path.join(data_dir, "flowers/allclasses_"+ part +".txt")
    caption_dir = join(data_dir, 'flowers/' + part)
    image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
    print image_files[300 :400]
    #print len(image_files)
    #image_captions = {img_file : [] for img_file in image_files}
    #image_classes = {img_file : None for img_file in image_files}
    image_captions = {}
    image_classes = {}
    class_dirs = []
    class_names = []

    if recreate_vocab and os.path.exists(all_caps_dir):
        os.remove(all_caps_dir)

    target, one_hot_targets, n_target = get_one_hot_targets(target_file_path)

    for i in range(dt_range[0], dt_range[1]) :
        class_dir_name = 'class_%.5d' % (i)
        class_dir = join(caption_dir, class_dir_name)
        class_names.append(class_dir_name)
        class_dirs.append(class_dir)
        onlyimgfiles = [f[0 :11] + ".jpg" for f in os.listdir(class_dir)
                                    if 'txt' in f]
        for img_file in onlyimgfiles:
            image_classes[img_file] = None

        for img_file in onlyimgfiles:
            image_captions[img_file] = []

    for class_dir, class_name in zip(class_dirs, class_names) :
        caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
        for i, cap_file in enumerate(caption_files) :
            if i%50 == 0:
                print(str(i) + ' captions extracted from' + str(class_dir))
            with open(join(class_dir, cap_file)) as f :
                str_captions = f.read()
                captions = str_captions.split('\n')
                if recreate_vocab :
                    with open(all_caps_dir, "a") as myfile :
                        myfile.write(str_captions)
            img_file = cap_file[0 :11] + ".jpg"
            # 5 captions per image
            image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0 :5]
            image_classes[img_file] = one_hot_encode_str_lbl(class_name,
                                                             target,
                                                             one_hot_targets)

    min_len, max_len, avg_len = 0, 0, 0.0
    min_len, max_len, avg_len = data_util.create_vocabulary(vocab_path,
                                                                all_caps_dir,
                                                                vocab_size,
                                                                normalize_digits = False)
    pad_len = 35
    #print len(image_captions)

    #model = skipthoughts.load_model()
    vocab, _ = data_util.initialize_vocabulary(vocab_path)
    encoded_captions = {}
    for i, img in enumerate(image_captions) :
        st = time.time()
        #encoded_captions[img] = skipthoughts.encode(model, image_captions[img])
        encoded_captions[img]  = data_util.data_to_token_ids(image_captions[img],
                                                            vocab,
                                                            normalize_digits = False)

        encoded_captions[img] = data_util.pad_data(encoded_captions[img], pad_len)

        print i, len(image_captions), img
        print "Seconds", time.time() - st

    ec_pkl_path = (join(data_dir, 'flower_tv.pkl'))
    pickle.dump(encoded_captions, open(ec_pkl_path, "wb"))

    fc_pkl_path = (join(data_dir, 'flower_tc.pkl'))
    pickle.dump(image_classes, open(fc_pkl_path, "wb"))

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type = str, default = 'train',
                        help = 'train/val')
    parser.add_argument('--data_dir', type = str, default = 'Data',
                        help = 'Data directory')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'Batch Size')
    parser.add_argument('--vocab_size', type = int, default = 40000,
                        help = 'Vocab Size')
    parser.add_argument('--recreate_vocab', type = bool, default = True,
                        help = 'Flag to recreate Vocabulary')
    parser.add_argument('--data_set', type = str, default = 'flowers',
                        help = 'Data Set : Flowers, MS-COCO')
    args = parser.parse_args()

    if args.data_set == 'flowers' :
        save_caption_vectors_flowers(args.data_dir, args.vocab_size,
                                     recreate_vocab = args.recreate_vocab,
                                     part='set_4', dt_range=(1, 103))
    else :
        save_caption_vectors_ms_coco(args.data_dir, args.split, args.batch_size)


if __name__ == '__main__' :
    main()
