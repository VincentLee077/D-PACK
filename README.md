# D-PACK

## An Unsupervised Deep Learning Model for Early Network Traffic Anomaly Detection

### Introduction

In this work, we present an effective anomaly traffic detection mechanism, namely D-PACK, which consists of a CNN and an unsupervised deep learning model (e.g., Autoencoder) for auto-profiling the traffic patterns and filtering abnormal traffic. Notably, D-PACK inspects only the first few bytes of the first few packets in each flow for early detection.

![](https://i.imgur.com/gU2VI5T.png)


### Requirement

* Python: 3.6.12
* pytorch: 1.10.2


### Get started

Code organization:

* D-PACK: 重現D-PACK的preprocessing與model。
* preprocessing_for_CSV: D-PACK所使用的dataset一開始就已經分好normal.pcap與abnormal.pcap，然而現有的dataset大部分都是把PCAP檔以統計資料處理成CSV檔。為了應對這樣的情況，preprocessing_for_CSV示範了如何利用CSV檔dataset的資訊，將原始PCAP檔切分成normal.pcap與abnormal.pcap，進一步得以使用D-PACK做後續處裡。
* real_time_system_test: 新資料testing。


### Publications
* R.-H. Hwang, M.-C. Peng, C.-W. Huang, P.-C. Lin, and V.-L. Nguyen, “An unsupervised deep learning model for early network traffic anomaly detection,” IEEE Access, vol. 8, pp. 30 387–30 399, 2020.
* R.-H. Hwang, M.-C. Peng and C.-W. Huang, “Detecting IoT Malicious Traffic based on Autoencoder and Convolutional Neural Network,” 2019 IEEE Globecom IoTSEC Workshop, Waikoloa, Hawaii, USA, December 9 -13, 2019.