import torch
from scapy.all import *
import itertools
import torch.nn as nn
import numpy as np

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

def sortappend(a,b):
    L=PacketList() 
    while True:
        if len(a) == 0 : 
            L = L + b 
            break 
        elif len(b) == 0: 
            L = L + a 
            break 
        if a[0].time < b[0].time: 
            L = L + a[:1] 
            a=a[1:] 
        elif a[0].time > b[0].time: 
            L = L + b[:1] 
            b=b[1:] 
        else: 
            L = L + a[:1] 
            L = L + b[:1] 
            a=a[1:] 
            b=b[1:]
    return PacketList(L)

if __name__ == "__main__":
    
    # Reading pcap (which is just normal test pcap) into f_pkts with flow base list . U can use `tcpdump -i xxxx` to capture in another terminal
    # Can design to periodically read particular amount of newest packet.

    pkts = rdpcap("real_test.pcap")
    s_pkts = pkts.sessions()
    
    del_keys = []

    # Delete all Not TCP Session Maybe not ?

    for key in s_pkts.keys():
        if 'TCP' not in key:
            del_keys.append(key)
    for key in del_keys:
        s_pkts.__delitem__(key)

    # Sessions into flow, each flow is bidirectional and identified by 5-tuple
    
    f_pkts = []
    for i,j in itertools.combinations(s_pkts.keys(),2):
        if [p in j for p in i.split(' ')].count(False) == 0:
            f_pkts.append(sortappend(s_pkts[i],s_pkts[j]))
    del s_pkts
    del del_keys
    del pkts    

    # Loading model for (50,2) model classify Deeptraffic dataset

    cnn_auto = copy.deepcopy(torch.load('cnn_auto-1.pkl',map_location=torch.device('cpu'))) 

    # Processing flow data to input data

    data=[] 
    img_shape=(50,2) 
    for flow in f_pkts: 
        f = []
        for pkt in flow[:img_shape[1]]: 
            pkt_50 = [field for field in raw(pkt)] 
            pkt_50.extend([0]*img_shape[0]) 
            f.extend(pkt_50[:img_shape[0]]) 
        data.append(f)
    data = torch.FloatTensor(data).reshape(-1,1,img_shape[0]*img_shape[1])/255 
    
    # Feeding data to model and show the mid-output (cnn output) result

    tcnn_out,_,_ = cnn_auto(data)
    pred_y = torch.max(tcnn_out, 1)[1].data.numpy()
