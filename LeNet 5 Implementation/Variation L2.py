import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import io
import os
import random

from PIL import Image
import matplotlib.pyplot as plt


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_image(tensor):
    plt.figure()
    plt.imshow(tensor.numpy().transpose(1, 2, 0))
    plt.show()


def find_mean_std(loader):

	sum, sumsq ,no =0,0,0

	for x,_ in loader: 
		sum += torch.mean(x, dim=[0,2,3])
		sumsq +=torch.mean(x**2, dim=[0,2,3])
		no +=1

	mean = sum/no
	std = (sumsq/no - mean**2)**(1/2)

	return mean,std


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


class NeuralNet(nn.Module):

	def __init__(self):

		super(NeuralNet, self).__init__()

		self.conv1 = nn.Conv2d(3,6,5)
		self.conv2 = nn.Conv2d(6,16,5)

		self.fc1 = nn.Linear(400,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self, x):

		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x,2)

		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x,2)

		x = torch.flatten(x,1)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x





# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


class my_dataset(Dataset):

	# def __init__(self, root_dir):

	# 	my_file = open(root_dir, "r")
	# 	content_list = my_file.read().split("\n")[:-1]
	# 	my_file.close()


    def __init__(self, txt_path='filelist.txt', img_dir='', transform=None):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """

        self.df = pd.read_csv(txt_path, sep=' ', index_col=False, header=None)
        self.img_names = self.df.iloc[:,0]
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.get_image_selector = True if img_dir.__contains__('tar') else False
        self.tf = tarfile.open(self.img_dir) if self.get_image_selector else None


    def get_image_from_tar(self, name):
        image = self.tf.extractfile(name)
        image = image.read()
        image = Image.open(io.BytesIO(image))
        return image

    def get_image_from_folder(self, name):

        image = Image.open(os.path.join(self.img_dir, name))
        return image

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

        if index == (self.__len__() - 1) and self.get_image_selector: 
            self.tf.close()

        if self.get_image_selector: 
            X = self.get_image_from_tar(self.img_names[index])
        else:
            X = self.get_image_from_folder(self.img_names[index])

        Y = self.df.iloc[index, 1]

        if self.transform is not None:
            X = self.transform(X)


        return X,Y



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



def data_preproc(train_dir,test_dir,valid_dir):

	transform = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Resize(32),
	    ])
	batch_size = 128

	#---------------------------------Load Train Set-----------------------------------

	trainset = my_dataset(train_dir,transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)

	m1,s1 = find_mean_std(trainloader)

	tr1 = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Resize(32),
	    transforms.Normalize(m1,s1)
	    ])

	trainset = my_dataset(train_dir,transform=tr1)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)


	#---------------------------------Load Test Set-----------------------------------

	testset = my_dataset(test_dir,transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
	                                         shuffle=False, num_workers=2)

	m2,s2 = find_mean_std(testloader)

	tr2 = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Resize(32),
	    transforms.Normalize(m2,s2)
	    ])

	testset = my_dataset(test_dir,transform=tr2)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
	                                         shuffle=False, num_workers=2)



	#---------------------------------Load valid Set-----------------------------------
	
	validset = my_dataset(valid_dir,transform=transform)
	validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
	                                         shuffle=False, num_workers=2)

	m3,s3 = find_mean_std(validloader)

	tr3 = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Resize(32),
	    transforms.Normalize(m3,s3)
	    ])


	validset = my_dataset(valid_dir,transform=tr3)
	validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
	                                         shuffle=False, num_workers=2)



	return trainloader,testloader,validloader


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def train_neural_net(trainloader,validloader,loss_func,ep=100,LR=0.004):

	net = NeuralNet()


	loss_list = []
	v_loss_list = []
	v_acc = []

	print('Training...')

	tb = SummaryWriter()

	for epoch in range(ep):  # loop over the dataset multiple times

		b_err = 0
		t_correct = 0

		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
			optimizer.zero_grad()

			outputs = net(inputs)
			loss = loss_func(outputs, labels)

			b_err += loss.item()
			t_correct+= get_num_correct(outputs, labels)

			loss.backward()
			optimizer.step()

		tb.add_scalar("Training Loss", b_err, epoch)
		tb.add_scalar("Correct Training Examples in Training out of 5000:", t_correct, epoch)
		tb.add_scalar("Training Accuracy", t_correct/ len(trainloader.dataset), epoch)


		print("Epoch:",epoch+1," | Avg Loss:",b_err/len(trainloader.dataset))
		loss_list.append(b_err/len(trainloader.dataset))


		if epoch%20 == 19:
			LR = LR/2
			print("Updated LR to ",LR)

		v_correct=0
		v_err=0
		with torch.no_grad():
			for i,data in enumerate(validloader):

				images, labels = data

				outputs = net(images)
				loss = loss_func(outputs, labels)

				v_err+=loss.item()
				v_correct+= get_num_correct(outputs, labels)


		v_loss_list.append(v_err/len(validloader.dataset))
		v_acc.append(v_correct*100/len(validloader.dataset))
		print("Epoch:",epoch+1," | Val Loss:",v_err/len(validloader.dataset),"Validation Acc:",v_correct*100/len(validloader.dataset))

		tb.add_scalar("Validation Loss", v_err, epoch)
		tb.add_scalar("Correct Examples in Training out of 3000:", v_correct, epoch)
		tb.add_scalar("Validation Accuracy", v_correct/ len(validloader.dataset), epoch)



	tb.close()

	PATH = './saved_models/mod.pth'
	torch.save(net.state_dict(), PATH)
	print('Finished Training')





	plt.plot(loss_list)
	plt.ylabel('Loss')
	plt.savefig('Training_Loss.jpg')
	plt.show()

	plt.plot(v_loss_list)
	plt.ylabel('Validation Loss')
	plt.savefig('Validation_Loss.jpg')
	plt.show()

	plt.plot(v_acc)
	plt.ylabel('Validation Accuracy')
	plt.savefig('Validation_Accuracy.jpg')
	plt.show()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def find_accuracy(trainloader,testloader,validloader):

	net = NeuralNet()
	PATH = './saved_models/mod.pth'
	net.load_state_dict(torch.load(PATH))


	correct = 0
	total = 0

	with torch.no_grad():
	    for data in testloader:
	        images, labels = data

	        outputs = net(images)

	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Correct=',correct,'/',total)
	print('Accuracy of the network on the 8000 test images: %d %%' % (100 * correct / total))


	correct = 0
	total = 0

	with torch.no_grad():
	    for data in trainloader:
	        images, labels = data

	        outputs = net(images)

	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Correct=',correct,'/',total)
	print('Accuracy of the network on the 5000 training images: %d %%' % (100 * correct / total))


	correct = 0
	total = 0

	with torch.no_grad():
	    for data in validloader:
	        images, labels = data

	        outputs = net(images)

	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Correct=',correct,'/',total)
	print('Accuracy of the network on the 3000 Validation images: %d %%' % (100 * correct / total))


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


def conf_mat(dataloader):

	n_class = 10

	net = NeuralNet()
	PATH = './saved_models/mod.pth'
	net.load_state_dict(torch.load(PATH))

	confusion_matrix = torch.zeros(n_class, n_class)

	with torch.no_grad():

	    for i, (inputs, classes) in enumerate(dataloader):

	        outputs = net(inputs)
	        _, preds = torch.max(outputs, 1)

	        for t, p in zip(classes.view(-1), preds.view(-1)):
	                confusion_matrix[t.long(), p.long()] += 1

	# print(confusion_matrix)
	# print(torch.sum(confusion_matrix,0))
	# print(torch.sum(confusion_matrix,1))
	px = pd.DataFrame(confusion_matrix.numpy())
	px.columns = ['pred_airplane', 'pred_bird', 'pred_car', 'pred_cat', 'pred_deer', 'pred_dog', 'pred_horse','pred_monkey', 'pred_ship', 'pred_truck']
	px.index = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse','monkey', 'ship', 'truck']
	print(px)



def classification_results(testloader,classes):


	net = NeuralNet()
	PATH = './saved_models/mod.pth'
	net.load_state_dict(torch.load(PATH))

	# prepare to count predictions for each class
	correct_pred = {classname: 0 for classname in classes}
	total_pred = {classname: 0 for classname in classes}


	all_preds = torch.tensor([])

	with torch.no_grad():
	    for data in testloader:
	        images, labels = data    

	        outputs = net(images)    
	        _, predictions = torch.max(outputs, 1)

	        all_preds = torch.cat(
	            (all_preds, predictions)
	            ,dim=0
	        	)

	        for label, prediction in zip(labels, predictions):
	            if label == prediction:
	                correct_pred[classes[label]] += 1
	            total_pred[classes[label]] += 1

	  
	print("-----------------------------------------------------------------")
	for classname, correct_count in correct_pred.items():
	    accuracy = 100 * float(correct_count) / total_pred[classname]
	    print("Accuracy for class ",classname," = ",correct_count,"/",total_pred[classname]," = ",accuracy,"%")
	print("-----------------------------------------------------------------")


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def show_some_images(dataloader,classes,num=4):


	net = NeuralNet()
	PATH = './saved_models/mod.pth'
	net.load_state_dict(torch.load(PATH))

	dataiter = iter(dataloader)
	images, labels = dataiter.next()

	outputs = net(images)
	_, predicted = torch.max(outputs, 1)
	
	r_idx = 5
	while(True and r_idx<len(images)):

		if(classes[predicted[r_idx]] != classes[labels[r_idx]] ):
			break;

		r_idx +=1 



	print("---------------------------------------------------")
	print("idx=",r_idx)
	print('GroundTruth:',classes[labels[r_idx]])

	print('Predicted: ',classes[predicted[r_idx]])
	print("---------------------------------------------------")

	imshow(torchvision.utils.make_grid(images[r_idx]))



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



if __name__=="__main__":

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	TRAIN_DIR = './splits/train.txt'
	TEST_DIR = './splits/test.txt'
	VAL_DIR = './splits/val.txt'
	epochs = 10
	lr = 0.001

	classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse','monkey', 'ship', 'truck')

	trainloader,testloader,validloader = data_preproc(TRAIN_DIR,TEST_DIR,VAL_DIR)

	loss_func = nn.CrossEntropyLoss()

	train_neural_net(trainloader,validloader,loss_func,epochs,lr)
	find_accuracy(trainloader,testloader,validloader)
	classification_results(testloader,classes)
	conf_mat(testloader)

	# show_some_images(testloader,classes,4)

	# tb = SummaryWriter()
	# model = NeuralNet()
	# images, labels = next(iter(trainloader))
	# grid = torchvision.utils.make_grid(images)
	# tb.add_image("images", grid)
	# tb.add_graph(model, images)
	# tb.close()