import random
import os
import cv2
import argparse
from tqdm import tqdm
import json
import glob
import numpy as np 
import sys
import torch
import glob
from collections import Counter
import csv
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw


cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from core.utils.visualize import get_color_pallete
from core.models import get_model
from core.utils.score import SegmentationMetric
Image.MAX_IMAGE_PIXELS = 933120000

parser = argparse.ArgumentParser(description='image inference')
parser.add_argument('--path', '-p', default= './images', help='images path')
parser.add_argument('--altitude', '-a', default= 90, help='images height')
parser.add_argument('--checkpoint_dir', '-c', default= './checkpoint', help='images path')
parser.add_argument('--save_dir', '-s', default= './result', help='save path')
parser.add_argument('--model', '-m', default= 'fcn16s', help='model name')
parser.add_argument('--backbone', '-b', default= 'vgg16', help='model name')
parser.add_argument('--image_size',default= 512, help='crop_size')
parser.add_argument('--evaluate',default= True, help='if evaluate')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = args.image_size

def load_seg_model(backbone_name,model_name,model_dir):

	model_full = '{}_{}_{}'.format(backbone_name,model_name,'voc')
	model = get_model(model_full,local_rank = 0, pretrained=True, root=model_dir).to(device)
	return model

def sub_image_processing(image_data):

	image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB) #cv22pil
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])
	# image = Image.open(args.path).convert('RGB')
	sub_image_data = transform(image_data).unsqueeze(0).to(device)
	return sub_image_data

def get_sub_image(mega_image,overlap=0.2,ratio=1):
    #mage_image: original image
    #ratio: ratio * 512 counter the different heights of image taken
    #return: list of sub image and list fo the upper left corner of sub image
    coor_list = []
    sub_image_list = []
    w,h,c = mega_image.shape
    # if w<=512 and h<=512:
    #     return [mega_image],[[0,0]]
    size  = 512
    num_rows = int(w/int(size*(1-overlap)))
    num_cols = int(h/int(size*(1-overlap)))
    new_size = int(size*(1-overlap))
    for i in range(num_rows+1):
        if (i == num_rows):
            for j in range(num_cols+1):
                if (j==num_cols):
                    sub_image = mega_image[-size:,-size:,:]
                    coor_list.append([w-size,h-size])
                    sub_image_list.append (sub_image)
                else:
                    sub_image = mega_image[-size:,new_size*j:new_size*j+size,:]
                    coor_list.append([w-size,new_size*j])
                    sub_image_list.append (sub_image)
        else:
            for j in range(num_cols+1):
                if (j==num_cols):
                    sub_image = mega_image[new_size*i:new_size*i+size,-size:,:]
                    coor_list.append([new_size*i,h-size])
                    sub_image_list.append (sub_image)
                else:
                    sub_image = mega_image[new_size*i:new_size*i+size,new_size*j:new_size*j+size,:]
                    coor_list.append([new_size*i,new_size*j])
                    sub_image_list.append (sub_image)
    return sub_image_list,coor_list

def image_resize(mega_image,save_dir,scale = 10.0):

	height, width = mega_image.shape[0],mega_image.shape[1]
	# ratio = max(height/scale,width/scale)
	resized = cv2.resize(mega_image, (int(width/scale),int(height/scale)), interpolation = cv2.INTER_AREA)
	cv2.imwrite(save_dir,resized)

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

def compare_matrix(pred,gt):
    valid = gt!=0



def draw_mask(pred_dir,gt_dir):
    pred_dir_list = glob.glob(pred_dir+'/*/*/prediction.png')
    fields = ['Image', 'Model', 'RCG Percentage', 'precision','recall','f1-score','Accuracy'] 
    datas = []
    for mask_dir in pred_dir_list:
        #load images
        height = mask_dir.split('/')[-2]
        image_name = mask_dir.split('/')[-3]
        model_name = mask_dir.split('/')[-4]
        target_dir = '{}/{}/{}/RCG.gif'.format(gt_dir,image_name,height)
        image_dir = mask_dir.replace('prediction.png','image_resize.png')
        target = np.array(Image.open(target_dir))
        mask = np.array(Image.open(mask_dir))
        image = Image.open(image_dir)
        #get mask pallete
        result = mask + 2*target
        c_metrics = get_metrics(result)
        mask = get_color_pallete(result,'RCG')
        painting_dir = mask_dir.replace('prediction.png','painting.png')
        #Image resize
        new_size = (int(0.1*mask.width),int(0.1*mask.height))
        mask = mask.resize(new_size).convert("RGBA")
        image = image.resize(new_size).convert("RGBA")

        #merge two image layer 
        final =  Image.new("RGBA",new_size)
        trans_mask =  Image.new("RGBA",new_size)
        for x in range(mask.width):
            for y in range(mask.height):
                r,g,b,a = mask.getpixel((x,y))
                trans_mask.putpixel((x,y),(r,g,b,int(0.5*a)))
        final.paste(image,(0,0))
        final = Image.alpha_composite(final,trans_mask)

        #draw metrics
        info = "Red = FP,Green = FN and Yellow = TP\nPrecision = {}%\nRecall = {}%\nF1-score = {}%\nAccuracy = {}%".format(
            str(c_metrics[0]),str(c_metrics[1]),str(c_metrics[2]),str(c_metrics[3]))
        I1 = ImageDraw.Draw(final)
        I1.text((20, 20), info, fill=(255, 255, 255))
        final.save(painting_dir)

        h,w = target.shape
        # print('The image {} has {:.2f}% RCG, Model {} performance is training {}, test {} '.format(image_name, 100*np.sum(target/(h*w)),model_name,c_metrics,test_metrics))
        datas.append([image_name,model_name,round(100*np.sum(target/(h*w)),2),c_metrics[0],c_metrics[1],c_metrics[2],c_metrics[3]])
        with open('{}/{}'.format(pred_dir,'result.csv'), 'w') as csvfile: 
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(fields) 
            csvwriter.writerows(datas)

def inference_image(sub_image_data,model):

    with torch.no_grad():
        output = model(sub_image_data)

    pred = torch.argmax(output[0], 1).squeeze(0).cpu( ).data.numpy()
    return pred


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def split_train_test(mega_image):
	if len(mega_image.shape) == 3:
		h,w,c =mega_image.shape
		return mega_image[0:h,0:int(w*0.8),:],mega_image[0:h,int(w*0.8):w,:]
	else:
		h,w =mega_image.shape
		return mega_image[0:h,0:int(w*0.8)],mega_image[0:h,int(w*0.8):w]

def infer_train_test(image_dir,model,save_dir,mega_image):
    h,w,c = mega_image.shape
    result_map = np.zeros((h,w))
    sub_image_list,coor_list = get_sub_image(mega_image,overlap = 0,ratio = 1.0)
    for index,sub_image in enumerate(sub_image_list):
        sub_image = sub_image_processing(sub_image)
        with torch.no_grad():
        	output = model(sub_image)
        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, 'rcg')
        result_map[coor_list[index][0]:coor_list[index][0]+image_size,coor_list[index][1]:coor_list[index][1]+image_size] = mask

    result_map_pil = Image.fromarray(result_map.astype('uint8'))
    result_map_pil.save(save_dir+'/prediction.png')
    image_resize(np.uint8(255*result_map),save_dir+'/prediction_resize.png')
    image_resize(mega_image,save_dir+'/image_resize.png')
	


def main():

    model = load_seg_model(args.model,args.backbone,args.checkpoint_dir)
    model_name = '{}_{}'.format(args.model,args.backbone)
    model.eval()
    print('finish load model...')
    image_list = glob.glob(args.path+'/*/{}_p4/group1.JPG'.format(str(args.altitude)))
    print('There are '+str(len(image_list))+' images totally.')
    with tqdm(total = len(image_list)) as pbar:
        for image_dir in image_list:
            image_name = image_dir.split('/')[-3]
            height = image_dir.split('/')[-2]
            save_dir = '{}/{}/{}/{}'.format(args.save_dir,model_name,image_name,height)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            mega_image  = cv2.imread(image_dir)
            infer_train_test(image_dir,model,save_dir,mega_image)
            pbar.update(1)
    if args.evaluate:
        print('evaluating...')
        draw_mask('{}/{}'.format(args.save_dir,model_name),args.path)
main()