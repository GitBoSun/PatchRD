import numpy as np
import os

#path = '/home/bo/data/shapenet_processed2/02801938'
#path = '/home/bo/data/shapenet_processed/02828884'
path = '/home/bo/data/shapenet_processed/02958343'
path = '/home/bo/data/PCN/ShapeNetCompletion/train/partial/03001627/'

f1 = open('content_pcn_train.txt', 'w')
f2 = open('content_pcn_test.txt', 'w')

names = os.listdir(path)
data_len = len(names)
print(len(names))

data_len = 4000

for i in range(data_len):
    if i<int(5*data_len/6):
        f1.write('%s\n'%(names[i]))
    else:
        f2.write('%s\n'%(names[i]))
