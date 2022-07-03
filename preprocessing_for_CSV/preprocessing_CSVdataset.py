#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 08:05:24 2021

@author: vincent077
"""

from scapy.all import *
import csv

def forward(line):
    tuple5 = list()
    tuple5.append(line[4])
    tuple5.append(line[6])
    tuple5.append(line[3].upper())
    if(line[5] != ''):
        tuple5.append(int(line[5]))
    else:
        tuple5.append('')
    if(line[7] != ''):
        tuple5.append(int(line[7]))
    else:
        tuple5.append('')
    return tuple5

def retreat(line):
    tuple5 = list()
    tuple5.append(line[6])
    tuple5.append(line[4])
    tuple5.append(line[3].upper())
    if(line[7] != ''):
        tuple5.append(int(line[7]))
    else:
        tuple5.append('')
    if(line[5] != ''):
        tuple5.append(int(line[5]))
    else:
        tuple5.append('')
    return tuple5


if __name__ == "__main__":

    attacks = ['mining', 'ransomware', 'diskWipe', 'dos', 'rootkit', 'dataTheft']
    timestamps_start = [1637346522.045965,1646801192.327096, 1637318904.541754, 1637351173.494687, 1647343225.878973, 1637482155.944792]
    timestamps_end = [1637346688.080939,1646801373.197349,1637319093.515072, 1637351498.436468, 1647343419.802494, 1637482315.578169]
    host_address = {'mining':['192.168.1.11'], 
                    'ransomware':['192.168.1.11'], 'diskWipe':['192.168.1.11'],
                    'dos':['192.168.1.11'], 'rootkit':['192.168.1.11'], 'dataTheft':['192.168.1.11']}
    for attack in attacks:
        # =============================================================================
        # Get normal/abnormal 5-tuple information from CSV files
        # ============================================================================= 
        reader = csv.reader(open('data/label_traffic_'+ attack +'.csv', newline=''))
        title = next(reader)
        lines = list(reader)
        normal_flow = list()
        abnormal_flow = list()
        for line in lines:
            #normal 5-tuple information
            if(line[26] == '0' and line[3] == 'tcp'):
                normal_flow.append(forward(line))
                normal_flow.append(retreat(line))
                
            #abnormal 5-tuple information
            elif(line[26] == '1' and line[3] == 'tcp'):
                abnormal_flow.append(forward(line))
                abnormal_flow.append(retreat(line))
        
        
        # =============================================================================
        # generate normal/abnormal PCAP from original PCAP
        # =============================================================================
        pkts = rdpcap('data/traffic_' + attack + '.pcap')
        normal_pcap = []
        abnormal_pcap = []
        for pkt in pkts:
            if(pkt.time > timestamps_start[attacks.index(attack)] and pkt.time < timestamps_end[attacks.index(attack)]):
                tuple5 = list()
                if(pkt.payload.name == 'ARP' and (pkt.payload.psrc in host_address[attack] or pkt.payload.pdst in host_address[attack])):
                    tuple5.append(pkt.payload.psrc)
                    tuple5.append(pkt.payload.pdst)
                    tuple5.append(pkt.payload.name)
                    tuple5.append('')
                    tuple5.append('')
                elif(pkt.payload.name == 'IP' and (pkt.payload.src in host_address[attack] or pkt.payload.dst in host_address[attack])):
                    tuple5.append(pkt.payload.src)
                    tuple5.append(pkt.payload.dst)
                    tuple5.append(pkt.payload.payload.name)
                    tuple5.append(pkt.payload.payload.sport)
                    tuple5.append(pkt.payload.payload.dport)
                if(tuple5 in normal_flow):
                    normal_pcap.append(pkt)
                elif(tuple5 in abnormal_flow):
                    abnormal_pcap.append(pkt)
            
        wrpcap("preprocessing/normal_" + attack + ".pcap", normal_pcap)
        wrpcap("preprocessing/abnormal_" + attack + ".pcap", abnormal_pcap)