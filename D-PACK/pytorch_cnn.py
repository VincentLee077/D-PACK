"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
torchvision
matplotlib
"""
# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.cuda
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary
from matplotlib import cm
from sklearn.manifold import TSNE
from keras.utils import np_utils
import numpy as np
from sklearn.utils import shuffle
from pandas import DataFrame
from conf_plt import *
from sklearn.metrics import confusion_matrix, accuracy_score
import time

# torch.manual_seed(1)	# reproducible

# Hyper Parameters
EPOCH = 30	   # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
LR = 0.0001			  # learning rate
DOWNLOAD_MNIST = False

log_file = '1d_14class.out'
#arg_list = [[98,8],[198,4],[112,7]]
#arg_list = [[40,2],[40,3],[40,4],[40,5],[50,2],[50,3],[50,4],[50,5]
#			,[60,2],[60,3],[60,4],[60,5],[70,2],[70,3],[70,4],[70,5]
#			,[80,2],[80,3],[80,4],[80,5]]
#arg_list = [[40,2],[40,3]
#			,[60,2],[60,3]
#			,[80,2],[80,3]]

#arg_list = [[50,2],[50,3],[50,4],[50,5],[60,2],[60,3],[60,4],[60,5],[70,2],[70,3],[70,4],[70,5]
#			,[80,2],[80,3],[80,4],[80,5]]
arg_list = [[50,2]]
dict_14class = {0:'Outlook',1:'Facetime',2:'Skype',3:'SMB',4:'Gmail',5:'Weibo',6:'FTP'
				,7:'WorldOfWarcraft',8:'MySQL',9:'BitTorrent',10:'http',11:'syn',12:'udp',13:'ack'}
dict_2class = {0:'Benign', 1:'Malware'}
dict_20class = {0:'Outlook',1:'Facetime',2:'Skype',3:'SMB',4:'Gmail',5:'Weibo',6:'FTP'
				,7:'WorldOfWarcraft',8:'MySQL',9:'BitTorrent',10:'Miuref',11:'Shifu',12:'Tinba'
				,13:'Nsis-ay',14:'Neris',15:'Zeus',16:'Cridex',17:'Geodo',18:'Htbot',19:'Virut'}


benign_m = 0
attack_m = 0
#b_size = 2400
#m_size = 5990
b_size = 5997
m_size = 5997
NUM_classes=10
dict_20class = [dict_20class[i] for i in range(20)]
plot_14columns = [dict_14class[i] for i in range(14)]
plot_2columns = [dict_2class[i] for i in range(2)]
plot_10columns = [dict_20class[i] for i in range(10)]


## Mnist digits dataset
#if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
#	# not mnist dir or mnist is empyt dir
#	DOWNLOAD_MNIST = True
#
#train_data = torchvision.datasets.MNIST(
#	root='./mnist/',
#	train=True,									 # this is training data
#	transform=torchvision.transforms.ToTensor(),	# Converts a PIL.Image or numpy.ndarray to
#													# torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
#	download=True,
#)
#
## plot one example
#print(train_data.train_data.size())				 # (59970, 28, 28)
#print(train_data.train_labels.size())			   # (59970)
#plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
#plt.title('%i' % train_data.train_labels[0])
#plt.show()
#
## Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
#train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
#
## pick 2000 samples to speed up testing
#test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
#X_test = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
#test_y = test_data.test_labels[:2000]

class Session:
	
	def __init__(self, pshape, Best_acc_SD , CNN_confm, auto_confm, CNN_accuracy, auto_accuracy):
		self.pshape = pshape
		self.Best_acc_SD = Best_acc_SD
		self.CNN_confm = CNN_confm
		self.auto_confm = auto_confm
		self.CNN_accuracy = CNN_accuracy
		self.auto_accuracy = auto_accuracy
	def show_header(self):
		print("		  "+"{:15}".format('pshape')+"{:15}".format('Best_acc_SD')
				+"{:15}".format('CNN_accuracy')+"{:15}".format('auto_accuracy'))
	def show_contain(self):
		for x,y in self.__dict__.items() :
			if x not in  ['CNN_confm','auto_confm']:
				if type(y) is np.float64:
					print("{:15}".format(format(y,'.4f')),end='')
				else :
					print("{:15}".format(str(y)),end='')

def input_Data(Data_path, pshape, size):
	global img_shape
	global test_sample
	global arg_list
	global label
	
	Data=[]
	
	for flows in os.listdir(Data_path):
		tmp_read = np.load(Data_path+flows)
		print(flows)
		for i, flow in enumerate(tmp_read):
			if i >= size:
				break		
			img_Data=[]
			for j, pkt in enumerate(flow):
				if j >= pshape[1]:
					break
				tmp_=[]
				for k, value in enumerate(pkt):
					if k >= pshape[0]:
						break
					tmp_.append(value)
				if len(tmp_) < pshape[0]:
					tmp_.extend([0]*(pshape[0] - len(tmp_)))
				
				img_Data.extend(tmp_)
			img_Data=np.asarray(img_Data)
			img_Data.resize(img_shape[0],img_shape[1])
			Data.append(img_Data)
		print(len(Data))		
	return Data

def plot_with_labels(lowDWeights, labels):
	plt.cla()
	X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
	for x, y, s in zip(X, Y, labels):
		c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
	plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize Dense layer(1024 dim)'); plt.savefig('foo.png'); plt.pause(0.01)
	
def batch(iterable1,iterable2, n=1):
	if len(iterable1) != len(iterable2):
		raise Exception('The Data and Label size error')
	l = len(iterable1)
	for ndx in range(0, l, n):
		yield iterable1[ndx:min(ndx + n, l)], iterable2[ndx:min(ndx + n, l)]

class CNN_AUTO(nn.Module):
	def __init__(self,input_size):
		super(CNN_AUTO, self).__init__()
		self.conv1 = nn.Sequential(		 # input shape (1, 28, 28)
			nn.Conv1d(
				in_channels=1,			  # input height
				out_channels=32,			# n_filters
				kernel_size=6,			  # filter size
				stride=1,				   # filter movement/step
				padding=5,				  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),							  # output shape (16, 28, 28)
			nn.ReLU(),					  # activation
			nn.MaxPool1d(kernel_size=2),	# choose max value in 2x2 area, output shape (16, 14, 14)
		)
		input_size = (1,32,int(((input_size[1]-6+5*2)/1+1)/2))
		self.conv2 = nn.Sequential(		 # input shape (16, 14, 14)
			nn.Conv1d(32, 64, 6, 1, 5),	 # output shape (32, 14, 14)
			nn.ReLU(),					  # activation
			nn.MaxPool1d(kernel_size=2),				# output shape (32, 7, 7)
		)
		input_size = (1,64,int(((input_size[2]-6+5*2)/1+1)/2))
#		self,flatten = nn.Linear(np.prod(input_size[1:]), 10)
		self.dense1 = nn.Linear(np.prod(input_size[1:]), 1024)
#		self,dense1_1 = nn.Linear(1024, 10)
		self.dense2 = nn.Linear(1024, 25)
		self.cnn_out = nn.Linear(25, 10)	# fully connected layer, output 10 classes
		self.encoder_1 = nn.Linear(1024, 512)
		self.encoder_2 = nn.Linear(512, 256)
		self.decoder_1 = nn.Linear(256, 512)
		self.decoder_2 = nn.Linear(512, 1024)
		

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)	   # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		mid = self.dense1(x)
		x = self.dense2(mid)
		cnn_output = self.cnn_out(x)
		x = self.encoder_1(mid)
		x = self.encoder_2(x)
		x = self.decoder_1(x)
		x = self.decoder_2(x)
		return cnn_output, mid, x  # return mid for visualization

if __name__ == "__main__":
	
	DATAS = []	
	
	for pshape in arg_list:
		
		img_shape = (1,pshape[0]*pshape[1])
		
		Data = []
#		B_Data = []
#		M_Data = []
		Label = []
		
		
		B_data = input_Data('data/benign/deep_no_header_packet_structure/', pshape, b_size)
		M_data = input_Data('data/attack/mirai_no_header_packet_structure_1/', pshape, m_size)
		
		benign_m = len(B_data)
		attack_m = len(M_data)
		L=0
		
		for i in range(10):
#			B_Data.extend(B_data[i*int(b_size):(i+1)*int(b_size)])
			Label.extend([L]*int(b_size))
			L=L+1
#			Label.extend([L]*int(b_size))
#			L=L+1
#			B_Data.extend(M_B_Data[i*int(m_size):(i+1)*int(m_size)])
#			Label.extend([L]*int(m_size))
#			L=L+1
			
		for i in range(4):
#			M_Data.extend(M_data[ i*int(m_size):(i+1)*int(m_size)])
			Label.extend([L]*int(m_size))
			L=L+1
		
		
		Data.extend(B_data)
		Data.extend(M_data)
		Data = np.asarray(Data)
		B_data = np.asarray(B_data)
		M_data = np.asarray(M_data)
		Label = np.asarray(Label)
		
#		kf = KFold(n_splits=input_kf,shuffle=True)
#		
#		input_kf = int(input("How many times of k-folds:"))
#
#		kf = KFold(n_splits=input_kf,shuffle=True)
#		kf.get_n_splits(Input_data)
#		q = 0
#
#		X_train = list(range(input_kf));X_test = list(range(input_kf));y_train = list(range(input_kf)); y_test = list(range(input_kf)); Input_data1_predict = list(range(input_kf))
#
#		for train_index, test_index in kf.split(Input_data):
#			X_train[q], X_test[q] = Input_data[train_index], Input_data[test_index]
#			y_train[q], y_test[q] = Input_data1[train_index], Input_data1[test_index]
#
##		clf = KNeighborsClassifier(n_neighbors=5,algorithm='auto')
#			clf = RandomForestClassifier(n_estimators=11, criterion='gini', max_depth=None, min_samples_split=2,
#										 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
#										  max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
#										   bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
#											warm_start=False, class_weight=None)
#			clf.fit(X_train[q], y_train[q])
#			Input_data1_predict[q] = clf.predict(X_test[q])  # predict
#
#			q += 1
#			ss = np.array( X_train)
#			print(ss.shape)
		
#		np.random.shuffle(B_Data)
#		np.random.shuffle(label)
		
######				 plt ima			#####
#		for x in range(20):
#			for j,i in enumerate(range(6)):
#				plt.figure(num='deep_mirai',figsize=(15,15))
#				plt.subplot(1,6,j+1)
#				plt.title(Label[i+x*2400])
#				plt.imshow(np.reshape(B_Data[i+x*2400], [20, 50]), cmap='gray')
#				plt.axis('off')
#			plt.show()
#
######				  end			   #####
		
#		X_train = np.delete(Data,np.s_[::4],0).reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
#		y_train = np_utils.to_categorical(np.delete(Label[:59970],np.s_[:59970:9],0), num_classes=NUM_classes)
		X_train = np.delete(Data,np.s_[:59970:9],0).reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
		y_train = np.delete(Label,np.s_[:59970:9],0)
#		X_test = Data[::4].reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
		#y_test = np_utils.to_categorical(Label[:59970:9], num_classes=NUM_classes)
		X_test = Data[:59970:9].reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
		y_test = Label[:59970:9]
		
#		B_data = shuffle(B_data, random_state=0)
		B_test = B_data[:59970:3].reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
#		B_label = np_utils.to_categorical(Label[:59970], num_classes=NUM_classes)
#		
		M_test = M_data.reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
#		M_label = np_utils.to_categorical(Label[59970:], num_classes=NUM_classes)
		
		
		X_train, y_train = shuffle(X_train, y_train, random_state=0)
		X_test, y_test = shuffle(X_test, y_test, random_state=0)
		
		X_train = torch.from_numpy(X_train).cuda()
		y_train = torch.from_numpy(y_train).cuda()
		X_test = torch.from_numpy(X_test).cuda()
#		y_test = torch.from_numpy(y_test).cuda()
		
		B_test = torch.from_numpy(B_test).cuda()
		M_test = torch.from_numpy(M_test).cuda()
		
		print("\n===========================\n")
		print("Train_packet \t packet size : "+str(pshape[0])+"\t\tpacket count : "+str(pshape[1]))
		print("Classify to {} class (Deep_Benign vs. Four_Mirai)".format(NUM_classes))
		print("Train_shape \t: "+str(img_shape))
		print("Train sample \t: "+str(len(y_train)))
		print("Test sample \t: "+str(len(y_test)))
		
		print("\n===========================\n")	
		cnn_auto = CNN_AUTO([img_shape[0],img_shape[1]])
		print(cnn_auto)  # net architecture
#		summary(cnn,(1,80))
		cnn_auto.cuda()
		
		optimizer = torch.optim.Adam(cnn_auto.parameters(), lr=LR)   # optimize all cnn parameters
		
		# following function (plot_with_labels) is for visualization, can be ignored if not interested
		
		plt.ion()
		# training and testing
		for epoch in range(EPOCH):
			# gives batch data, normalize x when iterate train_loader
			epoch_start = time.clock()
			for step, (b_x, b_y) in enumerate(batch(X_train,y_train,BATCH_SIZE)):
				b_x = b_x.cuda()
				b_y = b_y.cuda()
				cnn_output = cnn_auto(b_x)[0]			   # cnn_auto output
				auto_input = cnn_auto(b_x)[1]
				auto_output = cnn_auto(b_x)[2]
				loss = nn.CrossEntropyLoss()(cnn_output, b_y) + nn.MSELoss()(auto_input, auto_output)  # cross entropy loss
				optimizer.zero_grad()		   # clear gradients for this training step
				loss.backward()				 # backpropagation, compute gradients
				optimizer.step()				# apply gradients
			
			print('Epoch: ', epoch, '\t| train loss: %.4f' % loss.data.item(), '\t time %.4f' % (time.clock() - epoch_start))
				
#				if step % 100 == 0:
#					tcnn_out,_,_ = cnn_auto(X_test)
#					pred_y = torch.max(tcnn_out, 1)[1].data.numpy()
#					accuracy = float((pred_y == y_test.data.numpy()).astype(int).sum()) / float(y_test.size(0))
#					print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
#					print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
					
#					# Visualization of trained flatten layer (T-SNE)
#					tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#					plot_only = 500
#					low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
#					labels = y_test.numpy()[:plot_only]
#					plot_with_labels(low_dim_embs, labels)
				
		plt.ioff()
		
		tcnn_out,_,_ = cnn_auto(X_test)
		tcnn_out = tcnn_out.cpu()
		pred_y = torch.max(tcnn_out, 1)[1].data.numpy()
		CNN_confm = confusion_matrix(y_test, pred_y)
		_,b_a_in,b_a_out = cnn_auto(B_test)
		_,m_a_in,m_a_out = cnn_auto(M_test)
		
		
		B_loss = np.array([nn.MSELoss()(b_a_in[i], b_a_out[i]).item() for i in range(len(B_test))])
		M_loss = np.array([nn.MSELoss()(m_a_in[i], m_a_out[i]).item() for i in range(len(M_test))])
		
		
		B_SD = np.std(B_loss,ddof=1)
		B_mean = B_loss.sum()/len(B_loss)
		
		
		#		plt.scatter([i*0.01+2.2 for i in range(20)], np.delete(Acc,0,0))
		#		plt.ylim(0.9755,0.9785)
		#		plt.show()
				
		Acc=[]
		
		for i in range(451):
			
			TP = len(B_loss[B_loss < B_mean + B_SD*(i*0.01 + 1.5)])
			FN = len(B_loss) - TP
			FP = len(M_loss[M_loss < B_mean + B_SD*(i*0.01 + 1.5)])
			TN = len(M_loss) - FP
		#	
		#			auto_confm = np.array([[TP,FN],[FP,TN]])
		#			
		#			df_cm = DataFrame(auto_confm, index=a_plot_columns, columns=a_plot_columns)
		#			
		#			pretty_plot_confusion_matrix(df_cm, fz=11, cmap='Oranges', figsize=[5,5]
		#								   , show_null_values=2)
			
			Acc.append((TP+TN)/(TP+TN+FP+FN))
		
		Max_index = np.argmax(Acc)
		TP = len(B_loss[B_loss < B_mean + B_SD*(Max_index*0.01 + 1.5)])
		FN = len(B_loss) - TP
		FP = len(M_loss[M_loss < B_mean + B_SD*(Max_index*0.01 + 1.5)])
		TN = len(M_loss) - FP
		
		auto_confm = np.array([[TP,FN],[FP,TN]])
		
		DATAS.append(Session(pshape,Max_index*0.01 + 1.5,CNN_confm,auto_confm,CNN_confm.diagonal().sum()/CNN_confm.sum(),auto_confm.diagonal().sum()/auto_confm.sum()))

	DATAS[0].show_header()
	for i in range(len(arg_list)):
		print("{}	 ".format("TEST"+str(i)+"\t"),end='')
		DATAS[i].show_contain()
		print()
		
		
		
