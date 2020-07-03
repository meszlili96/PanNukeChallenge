#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 23:28:15 2020

script converts ground truth mask .npy for each image (inst_map, type_map) to  .mat for each image with (inst_centroid, inst_type)

@author: carotenuto
"""
import glob
import os
import shutil

import cv2
import numpy as np
import scipy.io as sio



#path_to_folder = "/home/ccurs011/HoverNet/PanTransform/fold3"

path_to_folder = "/Users/carotenuto/Downloads/fold3"

# read all .npy files from directory
files = []
for i in os.listdir(path_to_folder):
    if i.endswith('.npy'):
        files.append(np.load(path_to_folder + "/" + i))

def get_inst_centroid(inst_map):
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]: # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [(inst_moment["m10"] / inst_moment["m00"]),
                         (inst_moment["m01"] / inst_moment["m00"])]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)

result_dict_list= list()

for j, file in enumerate(files):
    inst_map = file[:,:,3]
    type_map = file[:,:,4]
    
    inst_centroid = get_inst_centroid(inst_map)
    inst_type = np.zeros((inst_centroid.shape[0], 1))
    for i in range(inst_centroid.shape[0]):
        x = int(round(inst_centroid[i,0]))
        y = int(round(inst_centroid[i,1]))
        inst_type[i] = type_map[y, x]
        
    result_dict = dict()
    result_dict['inst_centroid'] = inst_centroid
    result_dict['inst_type'] = inst_type
    result_dict['inst_map'] = inst_map
    result_dict_list.append(result_dict)
    sio.savemat("/home/ccurs011/HoverNet/PanTransform_centroids/fold3/fold3_" + str(j+1) + ".mat", result_dict)
    
    
    
    
