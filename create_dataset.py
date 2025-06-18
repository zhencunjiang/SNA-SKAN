import numpy as np
import os
import h5py
import pydicom
import random


def get_dcm_files(input_path):
    dcm_list = []
    for dirName, dirList, fileList in os.walk(input_path):
        for filename in fileList:
                dcm_list.append(os.path.join(dirName,filename))
    dcm_list.sort()
    return dcm_list


noisedatapath='/sna-skan/PKU37_OCT_Denoising/PKU37_OCT_Denoising/noisy'

noiselist=get_dcm_files(noisedatapath)
print(len(noiselist))

cleandatapath='/sna-skan/PKU37_OCT_Denoising/PKU37_OCT_Denoising/clean'
clean=get_dcm_files(cleandatapath)

print(len(clean))
# testlist=noiselist[0:10]
# noiselist=noiselist[10:]
random.shuffle(noiselist)
random.shuffle(clean)



noiselist_1=noiselist[0:1734]
noiselist_2=noiselist[int(0.25*len(noiselist)):]
cleanlist=[]
print(len(noiselist_1))
print(len(noiselist_2))
for i in range(60):
    cleanlist.extend(clean)
cleanlist=cleanlist[0:1734]
print(len(cleanlist))

file = open('/sna-skan/code/all_train_noisy.txt', 'w')
for item in noiselist_1:
    file.write(item + '\n')
file.close()



file = open('/sna-skan/code/all_train_clean.txt', 'w')
for item in cleanlist:
    file.write(item + '\n')
file.close()
