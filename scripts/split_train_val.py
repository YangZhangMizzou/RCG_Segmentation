import glob
import shutil
import random
import os

root_path = '/home/yuanfeng/torch_seg/awesome-semantic-segmentation-pytorch/datasets/RCG/RCG_Training_test_90/train'
image_root = root_path+'/RGB'
label_root = root_path+'/Label'
image_list = glob.glob(image_root+'/*.JPG')
val_image_list = random.sample(image_list,int(0.2*len(image_list)))
for image_dir in val_image_list:
	shutil.copy(image_dir,image_dir.replace('/train/','/val/'))
	label_dir = image_dir.replace('.JPG','_mask.gif').replace('/RGB','/Label')
	shutil.copy(label_dir,label_dir.replace('/train/','/val/'))
	os.remove(label_dir)
	os.remove(image_dir)

