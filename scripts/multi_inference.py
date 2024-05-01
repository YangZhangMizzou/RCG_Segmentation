import os

combinations = [

	# ['resnet50','psp',2,0.001],
	# ['resnet50','deeplabv3',2,0.001],
	# ['resnet101','psp',2,0.001],
	# ['resnet101','deeplabv3',2,0.001],
	# ['vgg16','fcn32s',4,0.0005],
	# ['vgg16','fcn16s',4,0.0005],
	['vgg16','fcn8s',4,0.0005],
]

for combination in combinations:
	backbone = combination[0]
	model = combination[1]
	checkpoint_dir = './checkpoint/90_models'
	os.system('python ./scripts/infer_big_image.py --model {} --backbone {} --checkpoint_dir {}'.format(model,backbone,checkpoint_dir))