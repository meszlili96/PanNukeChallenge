#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:24:41 2020
@author: carotenuto
Script prepares predictions for metrics computing. Format is changed from
(types, inst) to (type1, type2, type3, type4, type5, background)
individual .mat or .npy files are summarized as one .npy file
"""

import scipy.io as sio
import numpy as np
import os

path_to_folder = "/home/ccurs011/HoverNet/linda_micronet/output_split2/v3.0/micronet/_proc"
file_format = ".mat"

# read all .mat or .npy files from directory

files = []
for i in os.listdir(path_to_folder):
    if file_format == ".npy":
        if i.endswith('.npy'):
            files.append(np.load(path_to_folder + "/" + i))
    if file_format == ".mat": #and i != "fold3_148.mat":
        if i.endswith('.mat'):
            mat_dict = sio.loadmat(path_to_folder + "/" + i)
            # add file number to dictionary (file name should be in format "fold3_105.mat" with 105 as the file number)
            x = i.split(".") 
            x = x[0].split("_")
            x = int(x[1])
            mat_dict['filenumber'] = x
            files.append(mat_dict)
            print("successfullly appended file " + i)

# sort files numerically by file number
files_sorted = sorted(files, key=lambda k: k['filenumber']) 

# transform format
result = np.zeros((len(files_sorted), 256, 256, 6))
for i, pred in enumerate(files_sorted):
    if file_format ==  ".npy":
        inst_map = pred[:,:,0]
        type_map = pred[:,:,1]
    if file_format ==  ".mat":
        inst_map = pred['inst_map']
        type_map = pred['type_map']
    # type channels 
    result[i,:,:,0] = np.where(type_map == 1, inst_map, result[i,:,:,0])
    result[i,:,:,1] = np.where(type_map == 2, inst_map, result[i,:,:,1])
    result[i,:,:,2] = np.where(type_map == 3, inst_map, result[i,:,:,2])
    result[i,:,:,3] = np.where(type_map == 4, inst_map, result[i,:,:,3])
    result[i,:,:,4] = np.where(type_map == 5, inst_map, result[i,:,:,4])
    # background channel
    result[i,:,:,5] = 1 # init as 1
    result[i,:,:,5] = np.where(inst_map != 0, 0, result[i,:,:,5]) # write 0 if value is not 0
print(result.shape)

# save as single .npy
output = "/home/ccurs011/TIA-Lab-Pannuke-metrics/fold3/mi_preds_split2/masks.npy"
np.save(output, result)
