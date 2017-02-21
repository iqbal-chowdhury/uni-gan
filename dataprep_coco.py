from pycocotools.coco import COCO
import pylab
import numpy as np
import os
import traceback
import keras
import pickle
import skipthoughts

dataRoot='Data/datasets/'
dataDir='Data/mscoco'
dataType='train2014'
valdataType='val2014'

annFile='%s/annotations_inst/instances_%s.json'%(dataDir,dataType)
annFile_caps = '%s/annotations_caps/captions_%s.json'%(dataDir,dataType)

valannFile='%s/annotations_inst/instances_%s.json'%(dataDir,valdataType)
valannFile_caps = '%s/annotations_caps/captions_%s.json'%(dataDir,valdataType)

model = skipthoughts.load_model()


coco=COCO(annFile)
coco_caps=COCO(annFile_caps)

valcoco=COCO(annFile)
valcoco_caps=COCO(annFile_caps)

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

def prepare_data(coco, mode='train'):
	all_caps_dir = os.path.join(dataRoot, mode ,'all_captions.txt')
	target_file_path = os.path.join(dataRoot, mode ,"allclasses.txt")
	if not os.path.exists(target_file_path):
		cats = coco.loadCats(coco.getCatIds())
		nms = [cat['name'] for cat in cats]
		with open(target_file_path, "w") as text_file:
		    text_file.write('\n'.join(nms))

	target, one_hot_targets, n_target = get_one_hot_targets(target_file_path)

	imgIds = coco.getImgIds()
	image_classes = {}

	if not os.path.exists(all_caps_dir):
		for i, imgid in enumerate(imgIds):
			if i%100 == 0:
				print(str(i) + ' Images loaded')
			annIds = coco.getAnnIds(imgIds=imgid)
			img_anns = coco.loadAnns(annIds)
			category_ids = []
			for i_anns in img_anns:
				category_ids.append(i_anns['category_id'])
			category_ids = set(category_ids)
			category_ids = list(category_ids)
			icats = coco.loadCats(category_ids)
			icatnms = [cat['name'] for cat in icats]
			lbl_k_hot = []
			for catnm in icatnms:
				lbl_k_hot.append(one_hot_encode_str_lbl(catnm,target,one_hot_targets))

			lbl_k_hot = np.sum(lbl_k_hot, axis=0)
			image_classes[imgid] = lbl_k_hot

			annIds_ = coco_caps.getAnnIds(imgIds=imgid)
			anns = coco_caps.loadAnns(annIds_)
			for ann in anns :
				#print(ann['caption'])
				with open(all_caps_dir, "a") as myfile :
					myfile.write(ann['caption'].lower() + '\n')

		if not os.path.exists(os.path.join(dataDir, mode, 'coco_tr_tc.pkl')):
			fc_pkl_path = (os.path.join(dataDir, mode, 'coco_tr_tc.pkl'))
			pickle.dump(image_classes, open(fc_pkl_path, "wb"))

	encoded_captions = {}

	if not os.path.exists(os.path.join(dataDir, mode, 'coco_tr_tv.pkl')):
		for i, imgid in enumerate(imgIds) :
			if i % 100 == 0 :
				print(str(i) + ' Images loaded')
			annIds_ = coco_caps.getAnnIds(imgIds=imgid)
			anns = coco_caps.loadAnns(annIds_)
			img_caps = [ann['caption'] for ann in anns]
			encoded_captions[imgid] = skipthoughts.encode(model, img_caps)
		ec_pkl_path = os.path.join(dataDir, mode, 'coco_tr_tv.pkl')
		pickle.dump(encoded_captions, open(ec_pkl_path, "wb"))

prepare_data(coco, mode = "train")
prepare_data(valcoco, mode = "val")

