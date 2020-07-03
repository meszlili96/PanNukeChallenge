#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:10:55 2020

@author: carotenuto
"""

import numpy as np

masks = np.load("/projects/0/ismi2018/PanNuke/Fold-1/masks/fold1/masks.npy")
types = np.load("/projects/0/ismi2018/PanNuke/Fold-1/images/fold1/types.npy")

lost_ids = [1723, 1939, 1940, 1946, 1969, 1973, 1977, 1978, 1979, 2009, 2020, 2022]

for j, lost_id in enumerate(lost_ids):
    print(j)
    print(lost_id)
    masks = np.delete(masks, (lost_id-j), axis=0)
    types = np.delete(types, (lost_id-j), axis=0)

np.save("/home/ccurs011/TIA-Lab-Pannuke-metrics/fold1/gt/masks.npy", masks)
np.save("/home/ccurs011/TIA-Lab-Pannuke-metrics/fold1/gt/types.npy", types)
