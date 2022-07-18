import numpy as np
import os

#path = '/home/bo/data/shapenet_processed2/02801938'
#path = '/home/bo/data/shapenet_processed/02828884'
path = '/home/bo/data/shapenet_processed'

#f1 = open('content_bench_train.txt', 'w')
#f2 = open('content_bench_test.txt', 'w')
f1 = open('total_sub_train.txt', 'w')
f2 = open('total_sub_test.txt', 'w')

f0 = open('content_chair_train.txt', 'r')
lines = f0.readlines()
for l in lines:
    f1.write('03001627/%s\n'%(l.strip()))
f0.close()
f0 = open('content_chair_test.txt', 'r')
lines = f0.readlines()
for l in lines[0:200]:
    f2.write('03001627/%s\n'%(l.strip()))
f0.close()

f0 = open('content_table_test.txt', 'r')
lines = f0.readlines()
for l in lines:
    f1.write('04379243/%s\n'%(l.strip()))
f0.close()
f0 = open('content_table_test.txt', 'r')
lines = f0.readlines()
for l in lines[0:200]:
    f2.write('04379243/%s\n'%(l.strip()))
f0.close()


for cat in os.listdir(path):
    if cat in ['03001627', '04379243']:
        continue
    if not cat in ['02801938', '02828884', '03691459']:
        continue
    if not cat.startswith('0'):
        continue
    names = os.listdir(os.path.join(path, cat))
    data_len = len(names)

    for i in range(data_len):
        if i<int(4*data_len/5) and i<2000:
            f1.write('%s/%s\n'%(cat, names[i]))
        elif i>=int(4*data_len/5) and i<int(4*data_len/5)+200:
            f2.write('%s/%s\n'%(cat, names[i]))

f1.close()
f2.close()
