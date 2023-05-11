
For saving checkpoints,logfiles folders must be created earlier.
Download pretrained model and put it in pretrained folder and set path accordingly in args.

Main files are provided for rafdb , ferplus and affectnet datasets. All of these files are meant for reading the dataset. 

In algorithm folder:
	noisyfer_aug_new_arch.py file has code for training, evaluating.
	loss.py defines loss functions.
	randaug.py defines transformations for strong augmentation.
	transform.py has code for tranformations of weak augmentations as well as converting from numpy to tensor, normalize.

In common folder:
	utils.py defines accuracy function.

In model folder:
	cnn.py builds the models
	resnet.py has code for resnet models.
	


All the checkpoints are made available in link https://drive.google.com/drive/folders/11WW2O6Uc2P9MV3S33vjdfDJjbn-OsYvj?usp=sharing
