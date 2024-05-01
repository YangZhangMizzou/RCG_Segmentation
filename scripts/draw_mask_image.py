from PIL import Image
import glob
import numpy as np
import cv2
import os
import sys
from collections import Counter
import csv
import argparse
from core.utils.visualize import get_color_pallete

Image.MAX_IMAGE_PIXELS = 933120000
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)



parser = argparse.ArgumentParser(description='draw mask')
parser.add_argument('--pred_dir', '-p', default= './result', help='path to pred masks')
parser.add_argument('--gt_dir', '-g', default= '/media/yang/My Passport/RCG/RCG_dataset/RCG_dataset_ortho', help='path to gt masks')
args = parser.parse_args()

# root_dir = '/media/yang/My Passport/RCG/RCG_dataset/RCG_dataset_ortho'


def image_resize(image_dir):
	image_data = cv2.imread(image_dir)
	height, width, channels = image_data.shape
	ratio = max(height/500.0,width/500.0)
	resized = cv2.resize(image_data, (int(width/ratio),int(height/ratio)), interpolation = cv2.INTER_AREA)
	cv2.imwrite(image_dir.replace('.png','_resize.png'),resized)

def draw_border(image_data):
	h, w = image_data.shape
	start_point = (0, int(0.7*w))
	end_point = (h, int(0.7*w))
	image_data = cv2.line(image_data, start_point, end_point, (255), 5)
	return image_data


def get_metrics(result):
	temp = Counter(result.flatten())
	tp = temp[3]
	tn = temp[0]
	fp = temp[1]
	fn = temp[2]
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	f1 = 2*tp/(2*tp+fp+fn)
	accuracy = (tp+tn)/(tp+fp+tn+fn)
	return [round(100*precision,2),round(100*recall,2),round(100*f1,2),round(100*accuracy,2)]
	# return 'precision={:.2f}%,recall={:.2f}%,f1={:.2f}%,accuracy={:.2f}%'.format(100*precision,100*recall,100*f1,100*accuracy)


def draw_mask(pred_dir,gt_dir):
	pred_dir_list = glob.glob(args.pred_dir+'/*/*/*/prediction.png')
	fields = ['Image', 'Model', 'RCG Percentage', 'precision','recall','f1-score','Accuracy'] 
	datas = []
	for mask_dir in pred_dir_list:
		print('1')
		height = mask_dir.split('/')[-2]
		image_name = mask_dir.split('/')[-3]
		model_name = mask_dir.split('/')[-4]
		target_dir = '{}/{}/{}/RCG.gif'.format(args.gt_dir,image_name,height)
		target = np.array(Image.open(target_dir))
		mask = np.array(Image.open(mask_dir))

		# result = (merged_mask==target)
		result = mask + 2*target
		c_metrics = get_metrics(result)
		result = draw_border(result)
		mask = get_color_pallete(result,'RCG')
		painting_dit = mask_dir.replace('prediction.png','painting.png')
		mask.save(painting_dit)
		image_resize(painting_dit)

		h,w = target.shape
		# print('The image {} has {:.2f}% RCG, Model {} performance is training {}, test {} '.format(image_name, 100*np.sum(target/(h*w)),model_name,c_metrics,test_metrics))
		datas.append([image_name,model_name,round(100*np.sum(target/(h*w)),2),c_metrics[0],c_metrics[1],c_metrics[2],c_metrics[3]])

		with open('{}/{}'.format(args.pred_dir,'result.csv'), 'w') as csvfile: 
		    csvwriter = csv.writer(csvfile) 
		    csvwriter.writerow(fields) 
		    csvwriter.writerows(datas)