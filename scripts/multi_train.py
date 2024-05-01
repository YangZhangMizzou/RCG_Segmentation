import os

combinations = [

	# ['vgg16','fcn32s',4,0.0005],
	# ['vgg16','fcn16s',4,0.0005],
	# ['vgg16','fcn8s',4,0.0005],
	# ['vgg16','cgnet',4,0.005],
	# ['vgg16','espnet',4,0.005],
	# ['vgg16','lednet',4,0.005],

	# ['resnet18','bisenet',4,0.001],

	# ['resnet50','fcn',4,0.001],
	# ['resnet50','psp',2,0.001],
	# ['resnet50','deeplabv3',4,0.001],
	# ['resnet50','danet',2,0.001],
	# # ['resnet50','encnet',2,0.001],
	# # ['resnet50','icnet',2,0.001],
	# ['resnet50','dunet',4,0.001],
	# ['resnet50','ocnet',2,0.001],
	# ['resnet50','psanet',2,0.001],
	
	['resnet101','fcn',2,0.001],
	['resnet101','psp',2,0.001],
	['resnet101','deeplabv3',2,0.001],
	['resnet101','danet',2,0.001],
	# ['resnet101','encnet',2],
	# ['resnet101','icnet',2],
	['resnet101','dunet',2,0.001],
	['resnet101','ocnet',2,0.001],
	['resnet101','psanet',2,0.001],

	# ['resnet152','fcn',1],
	# ['resnet152','psp',1],
	# ['resnet152','deeplabv3',1],
	# ['resnet152','danet',1],
	# # ['resnet152','encnet',1],
	# # ['resnet152','icnet',1],
	# ['resnet152','dunet',1],
	# ['resnet152','ocnet',1],
	# ['resnet152','psanet',1],

	# ['densenet121','denseaspp',1],
	# ['densenet169','denseaspp',1],
	
]

for combination in combinations:
	backbone = combination[0]
	model = combination[1]
	batch_size = combination[2]
	lr = combination[3]
	os.system('python train.py --model {} --backbone {} --dataset rcg --lr {} --epochs 50 --batch-size {}'.format(model,backbone,lr,batch_size))