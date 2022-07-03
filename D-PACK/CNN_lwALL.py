#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:26:06 2018

@author: sykman
"""
import time
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
np.random.seed(2222)  # for reproducibility
import keras
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Conv1D, MaxPooling1D, Input
from keras.optimizers import Adam
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
from pandas import DataFrame
import seaborn as sn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from keras.models import model_from_json
from keras import backend as K
#from conf_plt import *

# download the mnist to the path '~/.keras/Datasets/' if it is the first time to be called
# training X shape (60000, 28x28), label shape (60000, ). test X shape (10000, 28x28), label shape (10000, )
log_file = '1d_14class.out'
#arg_list = [[98,8],[198,4],[112,7]]
#arg_list = [[40,2],[40,3],[40,4],[40,5],[50,2],[50,3],[50,4],[50,5]
#			,[60,2],[60,3],[60,4],[60,5],[70,2],[70,3],[70,4],[70,5]
#			,[80,2],[80,3],[80,4],[80,5]]
arg_list = [[40,2],[40,3]
			,[60,2],[60,3]
			,[80,2],[80,3]]
#arg_list = [[50,2],[50,3],[50,4],[50,5],[60,2],[60,3],[60,4],[60,5],[70,2],[70,3],[70,4],[70,5]
#			,[80,2],[80,3],[80,4],[80,5]]
#arg_list = [[70,2]]
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
b_size = 6000
m_size = 6000
NUM_classes=10
dict_20class = [dict_20class[i] for i in range(20)]
plot_14columns = [dict_14class[i] for i in range(14)]
plot_2columns = [dict_2class[i] for i in range(2)]
testtime=[]

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

def reinitLayers(model):
	session = K.get_session()
	for layer in model.layers: 
		if type(layer) == keras.engine.training.Model:
			reinitLayers(layer)
			continue
		print("LAYER::", layer.name)
		for v in layer.__dict__:
			v_arg = getattr(layer,v)
			if hasattr(v_arg,'initializer'):
				initializer_method = getattr(v_arg, 'initializer')
				initializer_method.run(session=session)
				print('reinitializing layer {}.{}'.format(layer.name, v))



		
def input_Data(Data_path, pshape, size):
	global img_shape
	global test_sample
	global arg_list
	global label
	
	Data=[]
	
	for flows in os.listdir(Data_path):
		tmp_read = np.load(Data_path+flows, allow_pickle=True)
		print(flows)
		test_start = time.clock()
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
		print(len(Data) / (time.clock() - test_start) ,'flows/second')
		testtime.append(len(Data) / (time.clock() - test_start))
	return Data

if __name__ == "__main__":
	
	DATAS = []	
	
	for pshape in arg_list:
        
		pshape = [60, 3]
		img_shape = [pshape[0]*pshape[1],1]
		
		Data = []
#		B_Data = []
#		M_Data = []
		Label = []
		
		
		B_data = input_Data('data/USTC_benign/', pshape, b_size)
		M_data = input_Data('data/USTC_attack/', pshape, m_size)
		
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
		y_train = np_utils.to_categorical(np.delete(Label[:60000],np.s_[:60000:9],0), num_classes=NUM_classes)
		
#		X_test = Data[::4].reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
		y_test = np_utils.to_categorical(Label[:60000:9], num_classes=NUM_classes)
		
		B_test = B_data[::3].reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
#		B_label = np_utils.to_categorical(Label[:60000], num_classes=NUM_classes)
#		
		M_test = M_data.reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
#		M_label = np_utils.to_categorical(Label[60000:], num_classes=NUM_classes)
		
		Data = Data.reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255

		print("\n===========================\n")
		print("Train_packet \t packet size : "+str(pshape[0])+"\t\tpacket count : "+str(pshape[1]))
		print("Classify to {} class (Deep_Benign vs. Four_Mirai)".format(NUM_classes))
		print("Train_shape \t: "+str(img_shape))
		print("Train sample \t: "+str(len(y_train)))
		print("Test sample \t: "+str(len(y_test)))
		
		print("\n===========================\n")
		
		
#		B_Data = []
#		M_Data = []
		
#########################		CNN		  #########################################
		
#		# Another way to build your CNN
#		model = Sequential()
#		
#		# Conv layer 1 output shape (32, 28, 28)
#		
#		model.add(Conv1D(
#			batch_input_shape=(None,img_shape[0],img_shape[1]),
#			filters=32,
#			kernel_size=5,
#			strides=1,
#			padding='same',	 # Padding method
#			B_Data_format='channels_first',
#		))
#		model.add(Activation('relu'))
#		
#		# Pooling layer 1 (max pooling) output shape (32, 14, 14)
#		model.add(MaxPooling1D(
#			pool_size=2,
#			strides=2,
#			padding='same',	# Padding method
#			B_Data_format='channels_first',
#		))
#		
#		# Conv layer 2 output shape (64, 14, 14)
#		model.add(Conv1D(64, 5, strides=1, padding='same', B_Data_format='channels_first'))
#		model.add(Activation('relu'))
#		
#		# Pooling layer 2 (max pooling) output shape (64, 7, 7)
#		model.add(MaxPooling1D(2, 2, 'same', B_Data_format='channels_first'))
#		
#		# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
#		model.add(Flatten())
#		model.add(Dense(1024))
#		model.add(Activation('relu'))
#		
#		# Fully connected layer 2 to shape (10) for 10 classes
#		model.add(Dense(NUM_classes))
#		model.add(Activation('softmax'))
#		
#		# Another way to define your optimizer
#		adam = Adam(lr=1e-4)
#		
#		# We add metrics to get more results you want to see
#		model.compile(optimizer=adam,
#					  loss='categorical_crossentropy',
#					  metrics=['accuracy'])
######################################################################################		
		
		
		inputs = Input(shape=(img_shape[0],img_shape[1]))
		CNN_con1 = Conv1D(
			filters=32,
			kernel_size=6,
			strides=1,
			padding='same',	 # Padding method
		)(inputs)
		
		CNN_con1_1 = BatchNormalization()(CNN_con1)
		CNN_con1_2 = Activation('relu')(CNN_con1_1)
		
		# Pooling layer 1 (max pooling) output shape (32, 14, 14)
		CNN_pool1 = MaxPooling1D(
			pool_size=2,
			strides=2,
			padding='same',	# Padding method
		)(CNN_con1_2)
		
		CNN_pool1_1 = BatchNormalization()(CNN_pool1)
		CNN_pool1_2 = Activation('relu')(CNN_pool1_1)
		
		# Conv layer 2 output shape (64, 14, 14)
		CNN_con2 = Conv1D(64, 6, strides=1, padding='same')(CNN_pool1_2)
		
		CNN_con2_1 = BatchNormalization()(CNN_con2)
		CNN_con2_2 = Activation('relu')(CNN_con2_1)
		
		# Pooling layer 2 (max pooling) output shape (64, 7, 7)
		CNN_pool2 = MaxPooling1D(2, 2, 'same')(CNN_con2_2)
				
		# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
		flatten = Flatten()(Activation('relu')(BatchNormalization()(CNN_pool2)))
		
		CNN_pre_1 = Dense(NUM_classes)(flatten)
		CNN_pre_1a = Activation('softmax')(CNN_pre_1)		
		CNN_pre_1m = Model(inputs,CNN_pre_1a)		
		
		CNN_dense1 = Dense(1024)(flatten)
		CNN_dense1_1 = Activation('relu')(BatchNormalization()(CNN_dense1))
		
		CNN_pre_2 = Dense(NUM_classes)(CNN_dense1_1)
		CNN_pre_2a = Activation('softmax')(CNN_pre_2)		
		CNN_pre_2m = Model(inputs,CNN_pre_2a)
		
		
		# Fully connected layer 2 to shape (10) for 10 classes
		CNN_dense2 = Dense(25)(CNN_dense1_1)
		CNN_dense2_1 = Activation('relu')(BatchNormalization()(CNN_dense2))
		
		CNN_dense2 = Dense(NUM_classes)(CNN_dense2_1)
		CNN_s = Activation('softmax')(CNN_dense2)
		
		
		CNN = Model(inputs,CNN_s)
		# Another way to define your optimizer
		adam = Adam(lr=1e-4)
		
		# We add metrics to get more results you want to see
		
#		CNN_cpre_1m.compile(optimizer=adam,
#					  loss='categorical_crossentropy',
#					  metrics=['accuracy'])	
#		CNN_cpre_2m.compile(optimizer=adam,
#					  loss='categorical_crossentropy',
#					  metrics=['accuracy'])	
		
		CNN_pre_1m.compile(optimizer=adam,
					  loss='categorical_crossentropy',
					  metrics=['accuracy'])
		CNN_pre_2m.compile(optimizer=adam,
					  loss='categorical_crossentropy',
					  metrics=['accuracy'])	
				
		CNN.compile(optimizer=adam,
					  loss='categorical_crossentropy',
					  metrics=['accuracy'])
		
		
		# Another way to train the model
		print('\n--------------------- CNN Flatten Training --------------------')
		CNN_pre_1m.fit(np.delete(Data[:60000],np.s_[:60000:9],0), y_train, epochs=5, batch_size=64,)
		print('\n--------------------- CNN Dense 1 Training --------------------')
		CNN_pre_2m.fit(np.delete(Data[:60000],np.s_[:60000:9],0), y_train, epochs=5, batch_size=64,)
#		print('\n-------------------- CNN Dense 1 Training --------------------')
#		CNN_pre_1m.fit(np.delete(Data[:60000],np.s_[:60000:9],0), y_train, epochs=5, batch_size=64,)
		print('\n------------------------ CNN Training ------------------------')
		CNN.fit(np.delete(Data[:60000],np.s_[:60000:9],0), y_train, epochs=20, batch_size=64,)
		
		
#		loss, accuracy = CNN.evaluate(X_test, y_test)
		
		print('\n------------------------ CNN Testing ------------------------')
		
		y=np.argmax(y_test, axis=1)
		pre_cls=CNN.predict(Data[:60000:9])
		pre_cls = np.argmax(pre_cls,axis=1)
		
		CNN_confm = confusion_matrix(y, pre_cls)
#df_cm = DataFrame(CNN_confm, index=plot_columns, columns=plot_columns)
#
#pretty_plot_confusion_matrix(df_cm, fz=11, cmap='Oranges', figsize=[12,12]
#					   , show_null_values=2)
#df_cm = DataFrame(DATAS[12].auto_confm, index=plot_2columns, columns=plot_2columns);pretty_plot_confusion_matrix(df_cm, fz=11, cmap='Oranges', figsize=[6,6], show_null_values=2)
		
#for i in range(NUM_classes):
#	TP = CNN_confm[i][i]
#	FN = sum([j for j in CNN_confm[i] ]) - TP
#	FP = sum([j for j in CNN_confm.transpose()[i] ]) - TP
#	TN = CNN_confm.sum() - TP - FN - FP
#	print(dict_14class[i],'\t',"{:.4f}".format((TP+TN)/(TP+TN+FP+FN)),'\t',"{:.4f}".format(TP/(TP+FP)),'\t',"{:.4f}".format(TP/(TP+FN)),'\t',"{:.4f}".format(FP/(FP+TN)))
		
		
#########################		Autoencoder		  #########################################
		
		CNN_mid = Model(inputs,CNN_dense1_1)
		CMM_mid_out = CNN_mid.predict(B_test)
		
		inputs_2 = Input(shape=(1024,))
		
		en_1 = Dense(512)(inputs_2)
		
		en_1_1 = Dense(1024)(en_1)
		auto_1 = Model(inputs_2,en_1_1)
		
		encoder = Dense(256)(en_1)
		
		encoder_1 = Dense(1024)(encoder)
		auto_2 = Model(inputs_2,encoder_1)
				 
#		encoder = Dense(28)(en_1_1)
#		encoder_1 = Activation('relu')(BatchNormalization()(encoder))
		
		de_1 = Dense(512)(encoder)
#		de_1_1 = Activation('relu')(BatchNormalization()(de_1))
		
		decoder_1 = Dense(1024)(de_1)
#		decoder = Activation('tanh')(decoder_1)
		
		autoencoder = Model(inputs_2, decoder_1)
		adam = Adam(lr=1e-4)
		auto_1.compile(optimizer=adam, loss='mse')
		auto_2.compile(optimizer=adam, loss='mse')
		autoencoder.compile(optimizer=adam, loss='mse')
		
		
		auto_1.fit(CMM_mid_out, CMM_mid_out,
				epochs=10,
				batch_size=64,
				shuffle=True)
		auto_2.fit(CMM_mid_out, CMM_mid_out,
				epochs=10,
				batch_size=64,
				shuffle=True)
		autoencoder.fit(CMM_mid_out, CMM_mid_out,
				epochs=50,
				batch_size=64,
				shuffle=True)
		
#		encoded_imgs = autoencoder.predict(CNN_mid.predict(Data))
#		
#		pca_result = PCA(n_components=2).fit_transform(encoded_imgs)
#		tsne_result = TSNE(n_components=2).fit_transform(encoded_imgs)
#		L = list(np.zeros(60000))
#		L.extend(list(np.ones(23960)))
#		
#		plt.scatter(pca_result[:, 0][:60000], pca_result[:, 1][:60000], c='purple')
#		plt.show()
#		plt.scatter(pca_result[:, 0][60000:], pca_result[:, 1][60000:], c='yellow')
#		plt.show()
#		plt.scatter(pca_result[:, 0], pca_result[:, 1], c=L)
#		plt.show()
#		
#		
#		plt.scatter(tsne_result[:, 0][:60000], tsne_result[:, 1][:60000], c='purple')
#		plt.show()
#		plt.scatter(tsne_result[:, 0][60000:], tsne_result[:, 1][60000:], c='yellow')
#		plt.show()
#		plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=L)
#		plt.show()
#		
#		
		B_sample = CNN_mid.predict(B_test)
		M_sample = CNN_mid.predict(M_test)
#		http = M_sample[:5990]
#		syn = M_sample[5990:5990*2]
#		udp = M_sample[5990*2:5990*3]
#		ack = M_sample[5990*3:]
#		
#		autoencoder.evaluate(http,http)
#		autoencoder.evaluate(syn,syn)
#		autoencoder.evaluate(udp,udp)
#		autoencoder.evaluate(ack,ack)
#		
		B_loss = np.array([autoencoder.evaluate(B_sample[i:i+1],B_sample[i:i+1],verbose=0) for i in range(len(B_test))])
		M_loss = np.array([autoencoder.evaluate(M_sample[i:i+1],M_sample[i:i+1],verbose=0) for i in range(len(M_test))])
		
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
		
		reinitLayers(CNN)
		reinitLayers(CNN_mid)
		reinitLayers(autoencoder)

#		
#		
#		auto_confm = np.array([[TP,FN],[FP,TN]])
#		
#		df_cm = DataFrame(auto_confm, index=a_plot_columns, columns=a_plot_columns)
#		
#		pretty_plot_confusion_matrix(df_cm, fz=11, cmap='Oranges', figsize=[5,5]
#							   , show_null_values=2)
		
#		encoded_imgs = autoencoder.predict(test)
#		plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
#		plt.colorbar()
#		plt.show()

		
#		plt.scatter(pca_result[:, 0], pca_result[:, 1], c=Label[::10])
#		plt.xlim(-10,15)
#		plt.ylim(-6,15)
#		plt.legend(dict_14class[0:14])
#		plt.show()



#########################		autoencoder		 #########################################
#		
#		# Another way to build your CNN
#		model = Sequential()
#		
#		# Conv layer 1 output shape (32, 28, 28)
#
#
#		model.add(Dense(128, input_shape=(img_shape[0],img_shape[1]), activation='relu'))
#		# Fully connected layer 2 to shape (10) for 10 classes
#		model.add(Dense(300, activation='sigmoid'))
#		
#		# Another way to define your optimizer
#		adam = Adam(lr=1e-4)
#		
#		# We add metrics to get more results you want to see
#		model.compile(optimizer=adam,
#					  loss='categorical_crossentropy',
#					  metrics=['accuracy'])
		
######################################################################################		
		

		
		
# Evaluate the model with the metrics we defined earlier
#		np.random.shuffle(Data)
#		np.random.shuffle(label)
#		
#		i = B_data[-10000:]
#		l = label[-10000:]
#		
#		unique, counts = np.unique(l, return_counts=True)
		
#		np.random.shuffle(X_test)
#		np.random.shuffle(y_test)
		

####################	   TEST DATAS	 ###########################
	DATAS[0].show_header()
    DATAS[0].show_contain()
	for i in range(len(arg_list)):
		print("{}	 ".format("TEST"+str(i)+"\t"),end='')
		DATAS[i].show_contain()
		print()
			
