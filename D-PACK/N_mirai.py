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
from conf_plt import *
import random

# download the mnist to the path '~/.keras/Datasets/' if it is the first time to be called
# training X shape (60000, 28x28), label shape (60000, ). test X shape (10000, 28x28), label shape (10000, )
log_file = '1d_14class.out'
#arg_list = [[98,8],[198,4],[112,7]]
#arg_list = [[40,2],[40,3],[40,4],[40,5],[50,2],[50,3],[50,4],[50,5]
#						,[60,2],[60,3],[60,4],[60,5],[70,2],[70,3],[70,4],[70,5]
#						,[80,2],[80,3],[80,4],[80,5]]
#arg_list = [[50,2],[50,3],[50,4],[50,5]
#						,[60,2],[60,3],[60,4],[60,5],[70,2],[70,3],[70,4],[70,5]
#						,[80,2],[80,3],[80,4],[80,5]]
arg_list = [[50,2],[50,3],[50,4],[50,5]]
#arg_list = [[50,2],[50,3],[50,4],[50,5],[60,2],[60,3],[60,4],[60,5],[70,2],[70,3],[70,4],[70,5]
#						,[80,2],[80,3],[80,4],[80,5]]
#arg_list = [[80,2]]
dict_14class = {0:'Outlook',1:'Facetime',2:'Skype',3:'SMB',4:'Gmail',5:'Weibo',6:'FTP'
								,7:'WorldOfWarcraft',8:'MySQL',9:'BitTorrent',10:'http',11:'syn',12:'udp',13:'ack'}
dict_2class = {0:'Benign', 1:'Malware'}
dict_20class = {0:'Outlook',1:'Facetime',2:'Skype',3:'SMB',4:'Gmail',5:'Weibo',6:'FTP'
								,7:'WorldOfWarcraft',8:'MySQL',9:'BitTorrent',10:'Miuref',11:'Shifu',12:'Tinba'
								,13:'Nsis-ay',14:'Neris',15:'Zeus',16:'Cridex',17:'Geodo',18:'Htbot',19:'Virut'}
dict_11class = {0: 'norm', 1: 'mirai', 2: 'vse', 3: 'dns', 4: 'udp', 5: 'http', 6: 'ack'
								, 7: 'greeth', 8: 'syn', 9: 'greip', 10: 'udpplain'}
dict_9class = {0: 'norm', 1: 'mirai', 2: 'vse', 3: 'dns', 4: 'udp', 5: 'http', 6: 'ack'
								, 7: 'syn', 8: 'greip'}
reverse_d9 = {v: k for k, v in dict_9class.items()}


benign_m = 0
attack_m = 0
#b_size = 2400
#m_size = 5990
b_size = 6000
m_size = 6000
NUM_classes=9
dict_20class = [dict_20class[i] for i in range(20)]
plot_14columns = [dict_14class[i] for i in range(14)]
plot_2columns = [dict_2class[i] for i in range(2)]
plot_11columns = [dict_11class[i] for i in range(11)]
plot_9columns = [dict_11class[i] for i in range(9)]

class Session:
	
	def __init__(self, pshape, Benign_loss, Benign_mean, Benign_sd, Benign_max, CNN_confm, auto_confm, CNN_accuracy, auto_accuracy):
		self.pshape = pshape
		self.Benign_loss = Benign_loss
		self.Benign_mean = Benign_mean
		self.Benign_sd = Benign_sd
		self.Benign_max = Benign_max
		self.CNN_confm = CNN_confm
		self.auto_confm = auto_confm
		self.CNN_accuracy = CNN_accuracy
		self.auto_accuracy = auto_accuracy
	def show_header(self):
		print(" "*8+"{:15}".format('pshape')
				+"{:15}".format('Benign_Mean')+"{:15}".format('Benign_max')
				+"{:15}".format('CNN_accuracy')+"{:15}".format('auto_accuracy'))
	def show_contain(self):
		for x,y in self.__dict__.items() :
			if x not in  ['CNN_confm','auto_confm','Benign_loss','Benign_sd']:
#				print(type(y),x)
				if type(y) is np.float64:
					print("{:15}".format(format(y,'.6f')),end='')
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



				
def input_Data(Data_path, pshape):
	global img_shape
	global label
	global Data
	
	for flows in sorted(os.listdir(Data_path)):
		if 'label' not in flows:
			for labels in os.listdir(Data_path):
				if labels.split('_label')[0] in flows.split('.npy')[0] and '_label' in labels and flows.split('.npy')[0] in labels.split('_label')[0]:
					label_read = np.load(Data_path+labels, allow_pickle=True)
					tmp_read = np.load(Data_path+flows, allow_pickle=True)
					print('-------------------')
					print(flows)
					for i, flow in enumerate(tmp_read):
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
						Label.append(label_read[i])
					print('data_len\t:'+str(len(Data)))
					print('label_len\t:'+str(len(Label)))
	
#	to_categorical
	
	return

if __name__ == "__main__":
		
	DATAS = []		
		
	for pshape in arg_list:
				
		img_shape = [pshape[0]*pshape[1],1]


#		B_Data = []
#		M_Data = []
		Data = []
		Label = []

		input_Data('data/Mirai_botnet/1-4/', pshape)

		mirai_num = 0
		syn_num = 0
		needed_arr = []

		for ind,va in enumerate(Label):
			if 'mirai' == va:
				if mirai_num < 76735:
					needed_arr.append(ind)
					mirai_num = mirai_num + 1
			elif 'syn' == va:
				if syn_num < 76735:
					needed_arr.append(ind)
					syn_num = syn_num + 1
			else:
				needed_arr.append(ind)
		
		Data = np.asarray(Data)
		Label = np.asarray(Label)
		Data = Data[needed_arr]
		Label = Label[needed_arr]
			
#				kf = KFold(n_splits=input_kf,shuffle=True)
#				
#				input_kf = int(input("How many times of k-folds:"))
#
#				kf = KFold(n_splits=input_kf,shuffle=True)
#				kf.get_n_splits(Input_data)
#				q = 0
#
#				X_train = list(range(input_kf));X_test = list(range(input_kf));y_train = list(range(input_kf)); y_test = list(range(input_kf)); Input_data1_predict = list(range(input_kf))
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
		
		Data = Data.reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
		B_test = np.asarray(Data[[index for index,value in enumerate(Label) if value == 'norm']])
		B_test_test = B_test[::9]
		B_test = np.delete(B_test,np.s_[::9],0)
		M_test = np.asarray(Data[[index for index,value in enumerate(Label) if value != 'norm']])
		M_test_test = M_test[::9]
		M_test = np.delete(M_test,np.s_[::9],0)
		
		Label = np_utils.to_categorical([reverse_d9[i] for i in Label], num_classes=NUM_classes)
		
		X_train = np.delete(Data,np.s_[::9],0)
		Y_train = np.delete(Label,np.s_[::9],0)
		X_test = Data[::9]
		Y_test = Label[::9]
		

		print("\n===========================\n")
		print("Train_packet \t packet size : "+str(pshape[0])+"\t\tpacket count : "+str(pshape[1]))
		print("Classify to {} class (Deep_Benign vs. Four_Mirai)".format(NUM_classes))
		print("Train_shape \t: "+str(img_shape))
		print("Train sample \t: "+str(len(Y_train)))
		print("Test sample \t: "+str(len(Y_test)))
		print("\n===========================\n")
				
				
#				B_Data = []
#				M_Data = []
				
#########################				CNN				  #########################################
				
#				# Another way to build your CNN
#				model = Sequential()
#				
#				# Conv layer 1 output shape (32, 28, 28)
#				
#				model.add(Conv1D(
#						batch_input_shape=(None,img_shape[0],img_shape[1]),
#						filters=32,
#						kernel_size=5,
#						strides=1,
#						padding='same',		 # Padding method
#						B_Data_format='channels_first',
#				))
#				model.add(Activation('relu'))
#				
#				# Pooling layer 1 (max pooling) output shape (32, 14, 14)
#				model.add(MaxPooling1D(
#						pool_size=2,
#						strides=2,
#						padding='same',		# Padding method
#						B_Data_format='channels_first',
#				))
#				
#				# Conv layer 2 output shape (64, 14, 14)
#				model.add(Conv1D(64, 5, strides=1, padding='same', B_Data_format='channels_first'))
#				model.add(Activation('relu'))
#				
#				# Pooling layer 2 (max pooling) output shape (64, 7, 7)
#				model.add(MaxPooling1D(2, 2, 'same', B_Data_format='channels_first'))
#				
#				# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
#				model.add(Flatten())
#				model.add(Dense(1024))
#				model.add(Activation('relu'))
#				
#				# Fully connected layer 2 to shape (10) for 10 classes
#				model.add(Dense(NUM_classes))
#				model.add(Activation('softmax'))
#				
#				# Another way to define your optimizer
#				adam = Adam(lr=1e-4)
#				
#				# We add metrics to get more results you want to see
#				model.compile(optimizer=adam,
#										  loss='categorical_crossentropy',
#										  metrics=['accuracy'])
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
		CNN_pre_1m.fit(X_train ,Y_train , epochs=5, batch_size=64,shuffle=True)
		print('\n--------------------- CNN Dense 1 Training --------------------')
		CNN_pre_2m.fit(X_train ,Y_train , epochs=5, batch_size=64,shuffle=True)
#		print('\n-------------------- CNN Dense 1 Training --------------------')
#		CNN_pre_1m.fit(np.delete(Data[:60000],np.s_[:60000:9],0), y_train, epochs=5, batch_size=64,)
		print('\n------------------------ CNN Training ------------------------')
		CNN.fit(X_train ,Y_train , epochs=20, batch_size=64,shuffle=True)
		
		
#		loss, accuracy = CNN.evaluate(X_test, y_test)
		
		print('\n------------------------ CNN Testing ------------------------')
		
		tru_cls=np.argmax(Y_test, axis=1)
		pre_cls=CNN.predict(X_test)
		pre_cls = np.argmax(pre_cls,axis=1)
		
		CNN_confm = confusion_matrix(tru_cls, pre_cls)
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
		
		autoencoder = Model(input=inputs_2, output=decoder_1)
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
		B_max = B_loss.max()
		
#		plt.scatter([i*0.01+2.2 for i in range(20)], np.delete(Acc,0,0))
#		plt.ylim(0.9755,0.9785)
#		plt.show()
				
		Acc=[]
		
		for i in range(801):
			
			TP = len(B_loss[B_loss < B_mean + B_SD*(i*0.01)])
			FN = len(B_loss) - TP
			FP = len(M_loss[M_loss < B_mean + B_SD*(i*0.01)])
			TN = len(M_loss) - FP
#	
#			auto_confm = np.array([[TP,FN],[FP,TN]])
#			
#			df_cm = DataFrame(auto_confm, index=a_plot_columns, columns=a_plot_columns)
#			
#			pretty_plot_confusion_matrix(df_cm, fz=11, cmap='Oranges', figsize=[5,5]
#								   , show_null_values=2)
			
			Acc.append((TP+TN)/(TP+TN+FP+FN))
			
		B_sample = CNN_mid.predict(B_test_test)
		M_sample = CNN_mid.predict(M_test_test)
		
		B_loss = np.array([autoencoder.evaluate(B_sample[i:i+1],B_sample[i:i+1],verbose=0) for i in range(len(B_sample))])
		M_loss = np.array([autoencoder.evaluate(M_sample[i:i+1],M_sample[i:i+1],verbose=0) for i in range(len(M_sample))])
		
#		Max_index = np.argmax(Acc)
		TP = len(B_loss[B_loss <= B_max])
		FN = len(B_loss) - TP
		FP = len(M_loss[M_loss <= B_max])
		TN = len(M_loss) - FP
		
		auto_confm = np.array([[TP,FN],[FP,TN]])
		
#		DATAS.append(Session(pshape,Max_index*0.01,CNN_confm,auto_confm,CNN_confm.diagonal().sum()/CNN_confm.sum(),auto_confm.diagonal().sum()/auto_confm.sum()))
		
		reinitLayers(CNN)
		reinitLayers(CNN_mid)
		reinitLayers(autoencoder)
		
		DATAS.append(Session(pshape,B_loss,B_mean,B_SD,B_max,CNN_confm,auto_confm,CNN_confm.diagonal().sum()/CNN_confm.sum(),auto_confm.diagonal().sum()/auto_confm.sum()))

	DATAS[0].show_header()
	for i in range(len(DATAS)):
		print("{}".format("TEST"+str(i)+"\t"),end='')
		DATAS[i].show_contain()
		print()
			