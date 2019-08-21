import glob
import xml.etree.ElementTree as ET

import numpy as np
import os
from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "/home/cy/集群数据/data/labels/train_data/"
SIZE = 608
CLUSTERS = 9

def load_dataset(path):
	dataset = []
	txt_file = os.listdir(path)
	for txt in txt_file:
		with open(path + txt, 'r') as f:
			for annotation in f.readlines():
				annotation = annotation.split('\t')
				dataset.append([float(annotation[3]) * SIZE, float(annotation[4]) * SIZE])
	return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH)
out = kmeans(data, k=CLUSTERS)
# print(out)
original_anchor = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
original_anchor = np.array(original_anchor)
print("original accuracy: {:.2f}%".format(avg_iou(data, original_anchor) * 100))
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))