from scapy.all import *
import itertools
import numpy as np
import random
import socket
import struct

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

def rand_mac():
    return "%02x:%02x:%02x:%02x:%02x:%02x" % (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
        )


if __name__ == "__main__":

    # =============================================================================
    # Processing benign data
    # =============================================================================
    benigns = ['WorldOfWarcraft', 'Weibo-1', 'SMB-1', 'Skype', 'Outlook', 'MySQL', 'Gmail', 'FTP', 'Facetime', 'BitTorrent']
    for benign in benigns:
        pkts = rdpcap("Benign/" + benign + ".pcap")
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
        
        # Processing flow data to input data
        data=[] 
        img_shape=(60,3) 
        for flow in f_pkts: 
            f = []
            for pkt in flow[:img_shape[1]]: 
                if(pkt.payload.name == 'ARP'):
                    #random IP
                    pkt.payload.psrc = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
                    pkt.payload.pdst = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
                elif(pkt.payload.name == 'IP'):
                    #random IP
                    pkt.payload.src = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
                    pkt.payload.dst = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
                #random MAC
                pkt.src = rand_mac()
                pkt.dst = rand_mac()
                #packet -> bytes
                pkt_50 = [field for field in raw(pkt)] 
                pkt_50.extend([0]*img_shape[0])
                f.extend(pkt_50[:img_shape[0]])
            #deal with pcaket<3
            if(img_shape[1]-len(flow) > 0):
                f.extend([0]*img_shape[0]*(img_shape[1]-len(flow)))
            data.append(f)
        np.save('self_normal/' + benign + '.npy', data)
        
        
    # =============================================================================
    # Processing malicious data
    # =============================================================================
    attacks = ['Zeus', 'Virut', 'Tinba', 'Shifu', 'Nsis-ay', 'Neris', 'Miuref', 'Geodo', 'Cridex']
    for attack in attacks:
        pkts = rdpcap("Malware/" + attack + ".pcap")
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
        
        # Processing flow data to input data
        data=[] 
        img_shape=(60,3) 
        for flow in f_pkts: 
            f = []
            for pkt in flow[:img_shape[1]]: 
                if(pkt.payload.name == 'ARP'):
                    #random IP
                    pkt.payload.psrc = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
                    pkt.payload.pdst = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
                elif(pkt.payload.name == 'IP'):
                    #random IP
                    pkt.payload.src = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
                    pkt.payload.dst = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
                #random MAC
                pkt.src = rand_mac()
                pkt.dst = rand_mac()
                #packet -> bytes
                pkt_50 = [field for field in raw(pkt)] 
                pkt_50.extend([0]*img_shape[0])
                f.extend(pkt_50[:img_shape[0]])
            #deal with pcaket<3
            if(img_shape[1]-len(flow) > 0):
                f.extend([0]*img_shape[0]*(img_shape[1]-len(flow)))
            data.append(f)
        np.save('self_abnormal/' + attack + '.npy', data)