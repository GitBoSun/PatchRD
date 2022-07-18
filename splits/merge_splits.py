import os

cats = ['chair', 'plane', 'car', 'table', 'cabinet', 'lamp', 'boat', 'couch']
names = ['03001627', '02691156', '02958343', '04379243', '02933112', '03636649', '04530566', '04256520']
fout = open('content_total_test.txt', 'w')
for i, cat in enumerate(cats):
    fin = open("content_%s_test.txt"%(cat))
    dataset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    for model_name in dataset_names:
        fout.write("%s/%s\n"%(names[i], model_name))
fout.close()


